# app.py  — PART 1/4
# Core imports, config, network helpers, MIME detection, one-shot upload, polling, transcription API.

import streamlit as st
import pandas as pd
import requests
import json
import os
import time
import logging
import mimetypes
import tempfile
import random
from io import BytesIO
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any

# --- CONFIG ---
BASE_URL = "https://generativelanguage.googleapis.com"
UPLOAD_URL = "https://generativelanguage.googleapis.com/upload/v1beta/files"
MODEL_NAME = "gemini-2.5-flash"  # or "gemini-1.5-flash"

# streaming download chunk size (upload is streaming one-shot)
DOWNLOAD_CHUNK_SIZE = 8192

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
    Re-raises the last exception if all retries fail.
    """
    last_exc = None
    for attempt in range(max_retries):
        try:
            resp = requests.request(method, url, timeout=60, **kwargs)
            # Treat 429 and 5xx as transient
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
    Prioritize path extension, else header, else default.
    """
    _, ext = os.path.splitext(url_path or "")
    ext = ext.lower()
    if ext and ext in COMMON_AUDIO_MIME:
        return ext, COMMON_AUDIO_MIME[ext]
    # try header
    if header_content_type:
        ctype = header_content_type.split(";")[0].strip()
        guessed_ext = mimetypes.guess_extension(ctype) if ctype else None
        if guessed_ext:
            guessed_ext = guessed_ext.lower()
            guessed_mime = ctype
            return guessed_ext, guessed_mime
        # fallback to raw header mime if unknown extension
        return ".bin", ctype
    # last fallback
    return ".mp3", "audio/mpeg"

# --- UPLOAD PIPELINE (simple one-shot streaming upload — WORKING) ---

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

def upload_bytes(upload_url: str, file_path: str, mime_type: str) -> Dict[str, Any]:
    """
    Working one-shot streaming upload that streams file bytes and finalizes in one request.
    Falls back to PUT if POST returns 400 on some server variants.
    Returns file metadata dict on success.
    """
    file_size = os.path.getsize(file_path)
    headers = {
        "Content-Type": mime_type or "application/octet-stream",
        "Content-Length": str(file_size),
        "X-Goog-Upload-Offset": "0",
        "X-Goog-Upload-Command": "upload, finalize"
    }

    # Try POST first (streaming)
    with open(file_path, "rb") as f:
        resp = requests.post(upload_url, headers=headers, data=f, timeout=300)

    # Fallback: some endpoints expect PUT finalize
    if resp.status_code == 400:
        with open(file_path, "rb") as f:
            resp = requests.put(upload_url, headers=headers, data=f, timeout=300)

    if resp.status_code not in (200, 201):
        logger.error("Upload failed: %s %s", resp.status_code, resp.text)
        raise Exception(f"UPLOAD FAILED {resp.status_code}: {resp.text}")

    try:
        j = resp.json()
    except ValueError:
        raise Exception("Upload finished but server returned non-JSON response.")
    return j.get("file", j)

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
            attempt += 1
            _sleep_with_jitter(1, attempt)

        if time.time() - start > timeout_seconds:
            raise Exception("Timed out waiting for file to become ACTIVE.")

# --- TRANSCRIPTION CALLS & RESPONSE VALIDATION ---

def generate_transcript(api_key: str, file_uri: str, mime_type: str, prompt: str) -> str:
    """
    Calls Gemini generateContent for transcription. Returns the transcript text or structured error message.
    """
    api_url = f"{BASE_URL}/v1beta/models/{MODEL_NAME}:generateContent?key={api_key}"

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

    candidates = body.get("candidates") or []
    if not candidates:
        prompt_feedback = body.get("promptFeedback", {})
        if prompt_feedback and prompt_feedback.get("blockReason"):
            return f"BLOCKED: {prompt_feedback.get('blockReason')}"
        return "NO TRANSCRIPT (Empty Response)"
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
# app.py — PART 2/4
# Worker, prompt template, processing function, and result-collection/merging utilities.

# --- DIARIZATION-ENFORCEMENT PROMPT ---
def build_prompt(language_label: str) -> str:
    """
    Strong, enforced diarization prompt. Use build_prompt(lang_map[language_mode]).
    """
    return f"""
Transcribe this call in {language_label} exactly as spoken.

CRITICAL REQUIREMENTS — FOLLOW STRICTLY:
1. EVERY line MUST start with exactly one of these labels:
   - Speaker 1:
   - Speaker 2:
2. NEVER merge dialogue from two speakers in one line.
3. If you are unsure who is speaking, GUESS — but DO NOT leave the speaker label blank.
4. If the call sounds like a single-person monologue, STILL label every line as:
   Speaker 1: <text>
5. Do NOT summarize or improve the language. Write EXACTLY what was said.
6. Maintain natural turn-taking and break lines whenever the speaker changes.

TIMESTAMP RULES:
- Add timestamps at the start of EVERY line.
- Format MUST be: [0ms-2500ms]
- Use raw milliseconds only.
- No mm:ss format allowed.

LANGUAGE RULES:
- ALL Hindi words must be written in Hinglish (Latin script).
- NO Devanagari characters anywhere.
- English words should remain English.

STRICT FORMAT (DO NOT IGNORE):
- [timestamp] Speaker X: line of dialogue
- Only one speaker per line.
- Only one utterance per line.
- If two people speak at the same time, split into two separate lines with separate timestamps.

AUTO-CORRECTION:
- If any line is missing the speaker label, FIX IT and assign Speaker 1 or Speaker 2 based on your best guess.
- Do NOT output any unlabeled lines.

Return ONLY the transcript. No explanation.
"""

# --- WORKER / PROCESSING FUNCTION ---

def process_single_row(index: int, row: pd.Series, api_key: str, prompt_template: str, keep_remote: bool = False) -> Dict[str, Any]:
    """
    Processes a single row. Returns a dict with original index to allow stable ordering.
    Downloads audio (streamed), uploads (one-shot streamed), polls, transcribes.
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

        # Download file to temp location (streamed)
        r = make_request_with_retry("GET", audio_url, stream=True)
        if r.status_code != 200:
            raise Exception(f"Failed to download audio URL ({r.status_code})")

        header_ct = r.headers.get("content-type", "")
        ext, mime_type = detect_extension_and_mime(parsed.path, header_ct)

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            for chunk in r.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                if chunk:
                    tmp.write(chunk)
            tmp_path = tmp.name

        # Re-evaluate mime by extension mapping (explicit, do not transcode)
        mime_type = COMMON_AUDIO_MIME.get(os.path.splitext(tmp_path)[1].lower(), mime_type or "audio/mpeg")
        file_size = os.path.getsize(tmp_path)

        # Prepare unique remote name (sanitized)
        cleaned_mobile = "".join(ch for ch in mobile if ch.isalnum())
        unique_name = f"rec_{cleaned_mobile}_{int(time.time())}{os.path.splitext(tmp_path)[1]}"

        # 1) Initiate
        upload_url = initiate_upload(api_key, unique_name, mime_type, file_size)

        # 2) Upload (one-shot streamed)
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

# --- RESULT MERGE UTILITIES ---

def merge_results_with_original(df_original: pd.DataFrame, processed_results: list) -> pd.DataFrame:
    """
    Builds final DataFrame by merging processed results (which contain 'index') with original df columns.
    Preserves order, merges extra original columns like 'duration'.
    """
    results_df = pd.DataFrame(sorted(processed_results, key=lambda r: r["index"]))
    orig_reset = df_original.reset_index()  # column 'index' aligns with results_df.index values
    # Determine extra columns to merge (exclude 'index' and ones already present)
    existing_cols = set(results_df.columns)
    extra_cols = [c for c in orig_reset.columns if c not in ("index",) and c not in existing_cols]
    if extra_cols:
        merged = results_df.merge(orig_reset[["index"] + extra_cols], on="index", how="left")
    else:
        merged = results_df
    # drop internal index column
    if "index" in merged.columns:
        merged = merged.drop(columns=["index"])
    return merged

# --- STREAMLIT PROCESSING ENTRYPOINT (keeps old UI start logic) ---
# The actual UI and submission will be in Part 3; this file exposes the processing helper used by UI.
# app.py — PART 3/4
# Streamlit UI controls, processing loop, live preview + search/filter controls, and session state.

# --- UI Helpers & Small CSS for modern look and scrolling transcripts ---
BASE_CSS = """
<style>
/* Card look */
.call-card {
    border: 1px solid var(--border-color, #e6e6e6);
    border-radius: 10px;
    padding: 12px;
    margin-bottom: 12px;
    background: var(--card-bg, #fff);
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

/* Transcript scroll area */
.transcript-box {
    max-height: 320px;
    overflow: auto;
    padding: 8px;
    border-radius: 6px;
    background: var(--transcript-bg, #fafafa);
    border: 1px solid var(--border-color, #eee);
}

/* Speaker colors */
.speaker1 { color: #1f77b4; font-weight: 600; }
.speaker2 { color: #d62728; font-weight: 600; }
.other-speech { color: #333; }

/* compact meta row */
.meta-row { font-size: 13px; color: var(--meta-color, #666); margin-bottom: 8px; }

.dark-theme {
    --card-bg: #0f1115;
    --transcript-bg: #0b0c0f;
    --border-color: #222428;
    --meta-color: #9aa0a6;
    color: #e6eef3;
}
.light-theme {
    --card-bg: #ffffff;
    --transcript-bg: #fafafa;
    --border-color: #e6e6e6;
    --meta-color: #666666;
    color: #111;
}

/* small search box */
.search-box { margin-bottom: 10px; padding: 6px; border-radius: 6px; border: 1px solid var(--border-color, #eee); width:100%; }
</style>
"""

st.markdown(BASE_CSS, unsafe_allow_html=True)

# --- Initialize session state containers ---
if "processed_results" not in st.session_state:
    st.session_state.processed_results = []

if "final_df" not in st.session_state:
    st.session_state.final_df = pd.DataFrame()

if "orig_df" not in st.session_state:
    st.session_state.orig_df = pd.DataFrame()

# --- Sidebar: Config (keeps parity with previous layout) ---
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
    # Theme toggle
    theme_choice = st.radio("Theme", options=["Light", "Dark"], index=0, horizontal=True)
    st.caption("Use Dark theme if you prefer low-light UI.")

# Main uploader + start
uploaded_file = st.file_uploader("Upload Excel (.xlsx) with `recording_url` and optional columns (duration etc.)", type=["xlsx"])

# Search and filter controls (apply after processing)
col_search, col_filter = st.columns([2,1])
with col_search:
    global_search = st.text_input("Search transcripts (text)", placeholder="Search across transcripts...")
with col_filter:
    status_filter = st.selectbox("Status filter", options=["All", "✅ Success", "❌ Error", "❌ Failed"], index=0)

start_button = st.button("🚀 Start Batch Processing", type="primary")

# Live preview and processing placeholders
progress_bar = st.empty()
status_text = st.empty()
result_placeholder = st.empty()
live_preview_area = st.empty()

# Apply theme class to body container
theme_class = "dark-theme" if theme_choice == "Dark" else "light-theme"
st.markdown(f"<div class='{theme_class}'>", unsafe_allow_html=True)

# --- Start Processing Handler ---
if start_button:
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

    # Save original df into session state so Part 4 can access it
    st.session_state.orig_df = df.copy()

    # Build prompt once (use build_prompt helper)
    prompt_template = build_prompt(lang_map[language_mode])

    total_rows = len(df)
    progress_bar.progress(0.0)
    status_text.info(f"Starting processing with {max_workers} threads...")

    st.session_state.processed_results = []
    processed_results = st.session_state.processed_results  # local ref for speed

    # --- PARALLEL EXECUTION ---
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
            # update progress and status
            progress_bar.progress(completed / total_rows)
            status_text.markdown(f"Processed **{completed}/{total_rows}** files.")

            # Live merged preview including extra original columns
            live_results_df = pd.DataFrame(sorted(processed_results, key=lambda r: r["index"]))
            orig_reset = df.reset_index()
            extra_cols = [c for c in orig_reset.columns if c not in ("index", "recording_url", "mobile_number")]
            if extra_cols:
                preview_df = live_results_df.merge(orig_reset[["index"] + extra_cols], on="index", how="left")
            else:
                preview_df = live_results_df

            # apply status filter if any
            display_df = preview_df.copy()
            if status_filter != "All":
                display_df = display_df[display_df["status"] == status_filter]

            # show the last N rows (5)
            cols_to_show = ["mobile_number", "status", "transcript"] + extra_cols
            # ensure cols exist
            cols_to_show = [c for c in cols_to_show if c in display_df.columns]
            result_placeholder.dataframe(display_df[cols_to_show].tail(5), width='stretch')

           # After all done, merge fully into final_df and store in session_state for Part 4 (no forced rerun)
            final_df = merge_results_with_original(df, st.session_state.processed_results)
            st.session_state.final_df = final_df
            st.session_state.processing_done = True  # flag to indicate processing finished

            status_text.success("Batch Processing Complete!")
            # Do not call st.experimental_rerun() — instead rely on session_state to render final UI below.
            # Optionally give the user a direct button to jump to results
            st.info("Processing finished. Scroll down to the Transcript Browser, or click the button below.")
            if st.button("Show Results Now"):
                # no-op: pressing this will cause the app to re-run the script and show Part 4
                pass
    
# Close theme wrapper
st.markdown("</div>", unsafe_allow_html=True)
# app.py — PART 4/4
# Final transcript rendering: expandable panels, search/filters, colored speaker view, pagination & download.

# --- Final Transcript Viewer Helpers ---

def _wrap_line_html(line: str) -> str:
    """Return HTML-wrapped line with speaker color classes when label is present."""
    clean = line.strip()
    # detect speaker label at line start (case-insensitive)
    if clean.lower().startswith("speaker 1:"):
        # keep label and rest separated for styling
        return f"<div class='speaker1'>{st.escape(clean)}</div>"
    if clean.lower().startswith("speaker 2:"):
        return f"<div class='speaker2'>{st.escape(clean)}</div>"
    # fallback
    return f"<div class='other-speech'>{st.escape(clean)}</div>"

def colorize_transcript_html(text: str) -> str:
    """
    Convert transcript text into HTML block with colored speaker lines and safe escaping.
    Uses the CSS classes defined earlier.
    """
    if not isinstance(text, str) or not text.strip():
        return "<div class='other-speech'>No transcript</div>"
    lines = text.splitlines()
    wrapped = [_wrap_line_html(ln) for ln in lines if ln.strip() != ""]
    # join with small spacing
    return "<div>" + "".join(wrapped) + "</div>"

# --- Final UI (only render when final_df exists) ---
final_df = st.session_state.get("final_df", pd.DataFrame())

if final_df.empty:
    st.info("No processed transcripts to show yet. Upload and run processing to see results here.")
else:
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("## 🎛️ Transcript Browser")

    # Controls: search, status, speaker, page size
    col_a, col_b, col_c, col_d = st.columns([3,1,1,1])
    with col_a:
        search_q = st.text_input("Search transcripts (text)", value="", placeholder="search across transcripts or phone")
    with col_b:
        status_sel = st.selectbox("Status", options=["All", "✅ Success", "❌ Error", "❌ Failed"], index=0)
    with col_c:
        speaker_sel = st.selectbox("Speaker filter", options=["All", "Speaker 1", "Speaker 2"], index=0)
    with col_d:
        per_page = st.selectbox("Per page", options=[5, 10, 20, 50], index=1)

    # Filtering pipeline
    view_df = final_df.copy()

    # status filter
    if status_sel != "All":
        view_df = view_df[view_df["status"] == status_sel]

    # search filter (search in transcript, mobile_number, recording_url)
    if search_q and isinstance(search_q, str) and search_q.strip():
        q = search_q.strip().lower()
        mask = view_df["transcript"].fillna("").str.lower().str.contains(q) | \
               view_df["mobile_number"].fillna("").astype(str).str.lower().str.contains(q) | \
               view_df["recording_url"].fillna("").astype(str).str.lower().str.contains(q)
        view_df = view_df[mask]

    # speaker presence filter: basic heuristic (lines contain Speaker 1 / Speaker 2)
    if speaker_sel != "All":
        key = "speaker 1" if speaker_sel == "Speaker 1" else "speaker 2"
        mask = view_df["transcript"].fillna("").str.lower().str.contains(key)
        view_df = view_df[mask]

    total_items = len(view_df)
    st.markdown(f"**Showing {total_items} result(s)**")

    # Pagination
    pages = max(1, math.ceil(total_items / per_page))
    page_idx = st.number_input("Page", min_value=1, max_value=pages, value=1, step=1)
    start = (page_idx - 1) * per_page
    end = start + per_page
    page_df = view_df.iloc[start:end]

    # Download filtered view as Excel
    out_buf = BytesIO()
    page_df.to_excel(out_buf, index=False)
    st.download_button("📥 Download filtered results (current page)", data=out_buf.getvalue(),
                       file_name=f"transcripts_page{page_idx}_{int(time.time())}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # Render each transcript as an expander card for long text handling
    for idx, row in page_df.iterrows():
        header = f"{row.get('mobile_number','Unknown')} — {row.get('status','')}"
        with st.expander(header, expanded=False):
            # meta row
            meta_html = "<div class='meta-row'>"
            meta_html += f"URL: {st.escape(str(row.get('recording_url', '')))} &nbsp; | &nbsp; Status: {st.escape(str(row.get('status','')))}"
            # include duration / any extra columns
            extra_meta = []
            for col in row.index:
                if col not in ("mobile_number", "recording_url", "status", "transcript", "error"):
                    val = row.get(col)
                    extra_meta.append(f"{st.escape(str(col))}: {st.escape(str(val))}")
            if extra_meta:
                meta_html += " &nbsp; | &nbsp; " + " &nbsp; ".join(extra_meta)
            meta_html += "</div>"
            st.markdown(meta_html, unsafe_allow_html=True)

            # transcript box (scrollable)
            transcript_html = colorize_transcript_html(row.get("transcript", ""))
            st.markdown(f"<div class='transcript-box'>{transcript_html}</div>", unsafe_allow_html=True)

            # show error if present
            if row.get("error"):
                st.error(f"Error: {row.get('error')}")

    st.markdown("---")
    st.caption("Tip: Use the search box to quickly find words or phone numbers. Use speaker filter to view only calls mentioning a speaker label.")
