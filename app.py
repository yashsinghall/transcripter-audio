import streamlit as st
import pandas as pd
import requests
import json
import os
import time
import logging
import mimetypes
import tempfile
import math
import random
from io import BytesIO
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any

# --- CONFIG ---
BASE_URL = "https://generativelanguage.googleapis.com"
UPLOAD_URL = "https://generativelanguage.googleapis.com/upload/v1beta/files"
MODEL_NAME = "gemini-2.5-flash"  # or "gemini-1.5-flash"

# chunk size for resumable upload (256 KB recommended; tune as needed)
RESUMABLE_CHUNK_SIZE = 256 * 1024

# Configure logging
logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("transcriber")

# --- UTILITIES & NETWORK HELPERS ---

def _sleep_with_jitter(base_seconds: float, attempt: int):
    jitter = random.uniform(0.5, 1.5)
    to_sleep = min(base_seconds * (2 ** attempt) * jitter, 30)
    time.sleep(to_sleep)

def make_request_with_retry(method: str, url: str, max_retries: int = 5, backoff_base: float = 0.5, **kwargs) -> requests.Response:
    """
    Robust wrapper for requests with exponential backoff + jitter.
    Will re-raise the last exception if all retries fail.
    """
    last_exc = None
    for attempt in range(max_retries):
        try:
            resp = requests.request(method, url, timeout=60, **kwargs)
            # Consider 429 and 5xx as transient retryable
            if resp.status_code == 429 or (500 <= resp.status_code < 600):
                logger.warning("Transient HTTP %s from %s (attempt %d). Retrying...", resp.status_code, url, attempt + 1)
                _sleep_with_jitter(backoff_base, attempt)
                continue
            return resp
        except requests.exceptions.RequestException as e:
            logger.warning("RequestException on %s %s: %s (attempt %d)", method, url, str(e), attempt + 1)
            last_exc = e
            _sleep_with_jitter(backoff_base, attempt)
    # all retries exhausted
    if last_exc:
        raise last_exc
    # if we got here without exception, return last response or raise
    raise Exception("make_request_with_retry: retries exhausted without a response")

# --- MIME & EXTENSION HANDLING ---

COMMON_AUDIO_MIME = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wave",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".webm": "audio/webm",
    ".flac": "audio/flac"
}

def detect_extension_and_mime(url_path: str, header_content_type: Optional[str]) -> (str, str):
    """
    Determine extension and mime type from URL path and header.
    Prioritize path extension, else header, else sane default.
    """
    _, ext = os.path.splitext(url_path or "")
    ext = ext.lower()
    if ext and ext in COMMON_AUDIO_MIME:
        return ext, COMMON_AUDIO_MIME[ext]
    # try header
    if header_content_type:
        # sometimes header includes charset; split
        ctype = header_content_type.split(";")[0].strip()
        guessed_ext = mimetypes.guess_extension(ctype)
        if guessed_ext:
            guessed_ext = guessed_ext.lower()
            guessed_mime = ctype
            return guessed_ext, guessed_mime
        # fallback to raw header mime if unknown extension
        return ".bin", ctype
    # last fallback
    return ".mp3", "audio/mpeg"

# --- UPLOAD PIPELINE (resumable, streaming, chunked) ---

def initiate_upload(api_key: str, display_name: str, mime_type: str, file_size: int) -> str:
    """
    Starts a resumable upload session. Returns the upload URL.
    """
    url = f"{UPLOAD_URL}?uploadType=resumable&key={api_key}"
    headers = {
        "Content-Type": "application/json; charset=UTF-8",
        "X-Goog-Upload-Protocol": "resumable",
        "X-Goog-Upload-Command": "start",
        "X-Goog-Upload-Header-Content-Length": str(file_size),
        "X-Goog-Upload-Header-Content-Type": mime_type,
    }
    payload = json.dumps({"file": {"display_name": display_name}})
    resp = make_request_with_retry("POST", url, headers=headers, data=payload)
    if resp.status_code not in (200, 201):
        logger.error("initiate_upload failed: %s %s", resp.status_code, resp.text)
        raise Exception(f"Init failed ({resp.status_code}): {resp.text}")
    upload_url = resp.headers.get("X-Goog-Upload-URL")
    if not upload_url:
        logger.error("No X-Goog-Upload-URL in response headers: %s", resp.headers)
        raise Exception("Failed to get upload URL from Google.")
    logger.debug("Initiated resumable upload: %s", upload_url)
    return upload_url

def _get_remote_offset(upload_url: str) -> int:
    """
    Query the upload URL to learn current committed offset. Returns integer offset.
    Uses a zero-byte PUT with special headers to get Range/Offset from server, or relies on 'Range' header in 308.
    """
    try:
        headers = {
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "query"
        }
        resp = make_request_with_retry("PUT", upload_url, headers=headers, data=b"")
        # Some servers may return 308 and include Range: bytes=0-12345 (last committed)
        if resp.status_code in (200, 201):
            # fully committed; offset == total size, but we cannot infer total here; return 0 sentinel
            return -1
        range_hdr = resp.headers.get("Range") or resp.headers.get("x-goog-upload-range")
        if range_hdr:
            # format "bytes=0-12345"
            try:
                s = range_hdr.split("=")[1].split("-")[1]
                return int(s) + 1
            except Exception:
                pass
        # fallback: no offset known, return 0
        return 0
    except Exception:
        # If query isn't supported, fallback to 0
        return 0

def upload_resumable_chunks(upload_url: str, file_path: str, chunk_size: int = RESUMABLE_CHUNK_SIZE) -> Dict[str, Any]:
    """
    Upload file in chunks to the upload_url using PUT with Content-Range.
    Returns parsed JSON metadata from the server on success.
    This function streams chunks from disk and never loads the whole file into RAM.
    """
    total_size = os.path.getsize(file_path)
    logger.info("Starting chunked upload: %s (%d bytes)", os.path.basename(file_path), total_size)

    # start offset (try to query remote)
    offset = _get_remote_offset(upload_url)
    if offset == -1:
        # remote claims "done" — attempt to GET metadata via the file resource instead; but here return error
        raise Exception("Upload already finalized on server (unexpected).")

    with open(file_path, "rb") as fh:
        fh.seek(offset)
        attempt = 0
        while offset < total_size:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            start = offset
            end = offset + len(chunk) - 1
            content_range = f"bytes {start}-{end}/{total_size}"
            headers = {
                "Content-Type": "application/octet-stream",
                "Content-Length": str(len(chunk)),
                "X-Goog-Upload-Protocol": "resumable",
                "X-Goog-Upload-Command": "upload",
                "X-Goog-Upload-Header-Content-Length": str(total_size),
                "X-Goog-Upload-Header-Content-Type": "application/octet-stream",
                "Content-Range": content_range,
            }

            try:
                # We use PUT for chunk upload (Google supports PUT with Content-Range)
                resp = make_request_with_retry("PUT", upload_url, headers=headers, data=chunk)
            except Exception as e:
                logger.warning("Chunk upload failed at offset %d: %s", offset, str(e))
                # attempt a few times locally before aborting
                attempt += 1
                if attempt >= 5:
                    raise
                _sleep_with_jitter(0.5, attempt)
                # reposition file handle and retry reading the same chunk
                fh.seek(offset)
                continue

            attempt = 0  # reset attempt on success

            if resp.status_code in (200, 201):
                # upload finalized; server should return JSON with file metadata
                try:
                    return resp.json().get("file", resp.json())
                except ValueError:
                    raise Exception("Upload finished but server returned non-JSON response.")
            elif resp.status_code in (308,):  # Resume Incomplete (308)
                # parse returned Range header to determine committed bytes
                range_hdr = resp.headers.get("Range")
                if range_hdr:
                    try:
                        committed_end = int(range_hdr.split("-")[1])
                        offset = committed_end + 1
                        fh.seek(offset)
                        logger.debug("Server acknowledged up to %d. Continuing from %d.", committed_end, offset)
                        continue
                    except Exception:
                        offset = end + 1
                        fh.seek(offset)
                        continue
                else:
                    # no Range header; assume our chunk committed
                    offset = end + 1
                    fh.seek(offset)
                    continue
            else:
                # Unexpected status; raise with body for debugging
                logger.error("Unexpected status during chunk upload: %s %s", resp.status_code, resp.text)
                raise Exception(f"Unexpected upload response ({resp.status_code}): {resp.text}")

    # If we reach here without server finalizing, attempt finalization with empty finalize command
    headers = {
        "X-Goog-Upload-Protocol": "resumable",
        "X-Goog-Upload-Command": "finalize"
    }
    resp = make_request_with_retry("PUT", upload_url, headers=headers, data=b"")
    if resp.status_code in (200, 201):
        return resp.json().get("file", resp.json())
    raise Exception(f"Upload did not finalize correctly: status {resp.status_code}: {resp.text}")

def upload_bytes(upload_url: str, file_path: str, mime_type: str) -> Dict[str, Any]:
    """
    High-level upload function.
    Streams the file in chunks and handles server responses.
    Returns file metadata dict on success.
    """
    # Use resumable chunked upload (safer for large files and concurrency)
    return upload_resumable_chunks(upload_url, file_path, chunk_size=RESUMABLE_CHUNK_SIZE)

# --- GOOGLE FILE STATUS POLLING ---

def wait_for_active(api_key: str, file_name: str, timeout_seconds: int = 300) -> bool:
    """
    Poll until Google marks the file ACTIVE or FAILED.
    Uses exponential backoff with cap.
    """
    url = f"{BASE_URL}/v1beta/{file_name}?key={api_key}"
    start = time.time()
    attempt = 0
    while True:
        resp = make_request_with_retry("GET", url)
        if resp.status_code != 200:
            logger.warning("Status poll: got %s. Retrying...", resp.status_code)
            attempt += 1
            _sleep_with_jitter(1, attempt)
        else:
            j = resp.json()
            state = j.get("state")
            logger.debug("Polled file %s state=%s", file_name, state)
            if state == "ACTIVE":
                return True
            if state == "FAILED":
                raise Exception(f"File processing failed: {j.get('processingError', j)}")
            # else still processing
            attempt += 1
            _sleep_with_jitter(1, attempt)

        if time.time() - start > timeout_seconds:
            raise Exception("Timed out waiting for file to become ACTIVE.")

# --- TRANSCRIPTION CALLS & VALIDATION ---

def generate_transcript(api_key: str, file_uri: str, mime_type: str, prompt: str) -> str:
    api_url = f"{BASE_URL}/v1beta/models/{MODEL_NAME}:generateContent?key={api_key}"

    # safetySettings left as in original but still validated server-side
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
    ]

    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"file_data": {"mime_type": mime_type, "file_uri": file_uri}}
            ]
        }],
        "safetySettings": safety_settings,
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 8192
        }
    }

    resp = make_request_with_retry("POST", api_url, json=payload, headers={"Content-Type": "application/json"})
    if resp.status_code != 200:
        logger.error("Transcription API returned %s: %s", resp.status_code, resp.text)
        return f"API ERROR {resp.status_code}: {resp.text}"

    try:
        body = resp.json()
    except ValueError:
        return "PARSE ERROR: Non-JSON response from transcription API."

    # Structured checks
    candidates = body.get("candidates") or []
    if not candidates:
        prompt_feedback = body.get("promptFeedback", {})
        if prompt_feedback and prompt_feedback.get("blockReason"):
            return f"BLOCKED: {prompt_feedback.get('blockReason')}"
        return "NO TRANSCRIPT (Empty Response)"
    # Safely access nested keys
    first = candidates[0]
    content = first.get("content", {})
    parts = content.get("parts", [])
    if not parts:
        return "NO TRANSCRIPT (No parts in response)"
    first_part = parts[0]
    text = first_part.get("text") or first_part.get("content") or ""
    return text

def delete_file(api_key: str, file_name: str):
    try:
        requests.delete(f"{BASE_URL}/v1beta/{file_name}?key={api_key}", timeout=20)
    except Exception as e:
        logger.warning("delete_file failed for %s: %s", file_name, str(e))

# --- WORKER (safe, streaming, ordered output) ---

def process_single_row(index: int, row: pd.Series, api_key: str, prompt_template: str, keep_remote: bool = False) -> Dict[str, Any]:
    """
    Processes a single row. Returns a dict with original index to allow stable ordering.
    """
    mobile = str(row.get("mobile_number", "Unknown"))
    audio_url = row.get("recording_url")
    result = {
        "index": index,
        "mobile_number": mobile,
        "recording_url": audio_url,
        "transcript": "",
        "status": "Pending",
        "error": None,
    }

    tmp_path = None
    file_info = None

    if not audio_url or not isinstance(audio_url, str):
        result.update({"status": "❌ Failed", "transcript": "", "error": "Missing or invalid recording_url"})
        return result

    try:
        parsed = urlparse(audio_url)
        path_ext = os.path.splitext(parsed.path)[1].lower()

        # Download file to temp location (streamed)
        r = make_request_with_retry("GET", audio_url, stream=True)
        if r.status_code != 200:
            raise Exception(f"Failed to download audio URL ({r.status_code})")

        header_ct = r.headers.get("content-type", "")
        ext, mime_type = detect_extension_and_mime(parsed.path, header_ct)

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    tmp.write(chunk)
            tmp_path = tmp.name

        # re-evaluate mime by file extension mapping (explicit, do not transcode)
        mime_type = COMMON_AUDIO_MIME.get(os.path.splitext(tmp_path)[1].lower(), mime_type or "audio/mpeg")
        file_size = os.path.getsize(tmp_path)

        # Prepare unique remote name
        cleaned_mobile = "".join(ch for ch in mobile if ch.isalnum())
        unique_name = f"rec_{cleaned_mobile}_{int(time.time())}{os.path.splitext(tmp_path)[1]}"

        # 1) Initiate
        upload_url = initiate_upload(api_key, unique_name, mime_type, file_size)

        # 2) Upload (streamed chunked/resumable)
        file_info = upload_bytes(upload_url, tmp_path, mime_type)

        # Validate file_info structure
        if not file_info or not isinstance(file_info, dict) or not file_info.get("name") or not file_info.get("uri"):
            raise Exception(f"Invalid file metadata returned: {file_info}")

        # 3) Wait for ACTIVE
        wait_for_active(api_key, file_info["name"])

        # 4) Generate transcript
        transcript = generate_transcript(api_key, file_info["uri"], mime_type, prompt_template)
        result["transcript"] = transcript
        if transcript.startswith("API ERROR") or transcript.startswith("PARSE ERROR") or transcript.startswith("BLOCKED"):
            result["status"] = "❌ Error"
        else:
            result["status"] = "✅ Success"

    except Exception as e:
        logger.exception("Processing failed for row %s: %s", index, str(e))
        result["transcript"] = f"SYSTEM ERROR: {str(e)}"
        result["status"] = "❌ Failed"
        result["error"] = str(e)

    finally:
        # cleanup local
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception as e:
                logger.warning("Failed to remove tmp file %s: %s", tmp_path, str(e))
        # cleanup remote unless user asked to keep it
        if file_info and isinstance(file_info, dict) and file_info.get("name") and not keep_remote:
            try:
                delete_file(api_key, file_info["name"])
            except Exception:
                logger.warning("Failed to delete remote file %s", file_info.get("name"))

    return result

# --- STREAMLIT UI ---

st.set_page_config(page_title="Gemini Call Transcriber Pro (Optimized)", layout="wide")
st.title(f"🎙️ Gemini Call Transcriber Pro ({MODEL_NAME}) — Optimized")

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Gemini API Key", type="password")
    max_workers = st.slider("Concurrency (Threads)", min_value=1, max_value=8, value=4,
                            help="Higher = faster but may hit API rate limits. Keep lower for large batches.")
    keep_remote = st.checkbox("Keep audio on Google (do not auto-delete)", value=False,
                              help="If you want to keep uploaded audio for debugging or reprocessing, enable this.")
    st.divider()
    language_mode = st.selectbox("Language", ["English (India)", "Hindi", "Mixed (Hinglish)"], index=2)
    lang_map = {
        "English (India)": "English (Indian accent)",
        "Hindi": "Hindi (Devanagari)",
        "Mixed (Hinglish)": "Mixed English and Hindi"
    }

uploaded_file = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])

if st.button("🚀 Start Batch Processing", type="primary"):
    if not api_key or not uploaded_file:
        st.error("Please enter API Key and upload a file containing 'recording_url' column.")
        st.stop()

    try:
        df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    if "recording_url" not in df.columns:
        st.error("Column 'recording_url' is missing.")
        st.stop()

    # Build prompt once
    prompt_template = f"""
Transcribe this audio in {lang_map[language_mode]}.
Requirements:
- Identify speakers (Speaker 1, Speaker 2).
- Add timestamps exactly in milliseconds (e.g. [0ms-2500ms]) at the start of every line.
- Do NOT use Minutes:Seconds format. Use raw milliseconds.
- Write exactly what is said.
- CRITICAL: Write ALL Hindi words in Hinglish (Latin script). Do NOT use Devanagari script.
- Keep English words in standard English.
"""

    total_rows = len(df)
    progress_bar = st.progress(0)
    status_text = st.empty()
    result_placeholder = st.empty()

    status_text.info(f"Starting processing with {max_workers} threads...")

    processed_results = []
    # Submit tasks with original index to preserve ordering
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_row, idx, row, api_key, prompt_template, keep_remote): idx
            for idx, row in df.iterrows()
        }

        completed = 0
        for future in as_completed(futures):
            res = future.result()
            processed_results.append(res)
            completed += 1
            progress_bar.progress(completed / total_rows)
            status_text.markdown(f"Processed **{completed}/{total_rows}** files.")
            # Show last 5 results
            live_df = pd.DataFrame(sorted(processed_results, key=lambda r: r["index"]))
            result_placeholder.dataframe(live_df[["mobile_number", "status", "transcript"]].tail(5),
                                         use_container_width=True)

    # final assembly & stable ordering by original index
    final_df = pd.DataFrame(sorted(processed_results, key=lambda r: r["index"]))
    final_df = final_df.drop(columns=["index"])

    st.success("Batch Processing Complete!")
    st.subheader("Final Results")
    st.dataframe(final_df, use_container_width=True)

    # Offer download (same as before)
    output = BytesIO()
    final_df.to_excel(output, index=False)
    st.download_button(
        label="📥 Download Full Transcript Excel",
        data=output.getvalue(),
        file_name=f"transcripts_{int(time.time())}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
