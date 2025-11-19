import streamlit as st
import pandas as pd
import requests
import json
from io import BytesIO
import tempfile
import os
import time
import logging
import mimetypes
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CONFIGURATION ---
BASE_URL = "https://generativelanguage.googleapis.com"
UPLOAD_URL = "https://generativelanguage.googleapis.com/upload/v1beta/files"
MODEL_NAME = "gemini-2.5-flash"  # Hardcoded Model Name (or 1.5-flash)

# Configure logging
logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()]
)

st.set_page_config(page_title="Gemini Call Transcriber Pro", layout="wide")

# --- NETWORK HELPERS ---

def make_request_with_retry(method, url, **kwargs):
    """Executes HTTP requests with exponential backoff."""
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = requests.request(method, url, **kwargs)
            
            if 200 <= response.status_code < 300:
                return response
            
            # Retry on Rate Limit (429) or Server Error (500+)
            if response.status_code == 429 or response.status_code >= 500:
                wait_time = (2 ** attempt) + 1
                time.sleep(wait_time)
                continue
                
            return response
            
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(2)
            
    return response

# --- CORE API FUNCTIONS ---

def initiate_upload(api_key, filename, mime_type, file_size):
    url = f"{UPLOAD_URL}?uploadType=resumable&key={api_key}"
    headers = {
        "Content-Type": "application/json; charset=UTF-8",
        "X-Goog-Upload-Protocol": "resumable",
        "X-Goog-Upload-Command": "start",
        "X-Goog-Upload-Header-Content-Length": str(file_size),
        "X-Goog-Upload-Header-Content-Type": mime_type
    }
    data = json.dumps({"file": {"display_name": filename}})
    
    response = make_request_with_retry("POST", url, headers=headers, data=data)
    if response.status_code != 200:
        raise Exception(f"Init failed ({response.status_code}): {response.text}")
        
    upload_url = response.headers.get('X-Goog-Upload-URL')
    if not upload_url:
        raise Exception("No upload URL returned from Google.")
    return upload_url

def upload_bytes(upload_url, file_path, mime_type):
    file_size = os.path.getsize(file_path)
    with open(file_path, 'rb') as f:
        file_bytes = f.read()

    headers = {
        "Content-Length": str(file_size),
        "Content-Type": mime_type,
        "X-Goog-Upload-Offset": "0",
        "X-Goog-Upload-Command": "upload, finalize"
    }
    
    response = requests.post(upload_url, headers=headers, data=file_bytes)
    if response.status_code == 400:
        response = requests.put(upload_url, headers=headers, data=file_bytes)

    if response.status_code != 200:
        raise Exception(f"Upload failed ({response.status_code}): {response.text}")
    
    return response.json().get("file", {})

def wait_for_active(api_key, file_name):
    url = f"{BASE_URL}/v1beta/{file_name}?key={api_key}"
    for _ in range(40): # Wait up to ~3 minutes
        response = make_request_with_retry("GET", url)
        if response.status_code != 200:
            time.sleep(5)
            continue
            
        state = response.json().get("state")
        if state == "ACTIVE":
            return True
        elif state == "FAILED":
            raise Exception(f"File processing failed on Google side.")
        time.sleep(5)
    raise Exception("Timed out waiting for file to become ACTIVE.")

def generate_transcript(api_key, file_uri, mime_type, prompt):
    api_url = f"{BASE_URL}/v1beta/models/{MODEL_NAME}:generateContent?key={api_key}"
    
    # Added Safety Settings to BLOCK_NONE to prevent filtered transcripts
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
    
    response = make_request_with_retry("POST", api_url, json=payload, headers={"Content-Type": "application/json"})
    
    if response.status_code != 200:
        return f"API ERROR {response.status_code}: {response.text}"
        
    try:
        candidates = response.json().get("candidates", [])
        if not candidates:
            # Check if it was a safety block despite our settings
            if response.json().get("promptFeedback", {}).get("blockReason"):
                return f"BLOCKED: {response.json()['promptFeedback']['blockReason']}"
            return "NO TRANSCRIPT (Empty Response)"
            
        return candidates[0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"PARSE ERROR: {str(e)}"

def delete_file(api_key, file_name):
    try:
        requests.delete(f"{BASE_URL}/v1beta/{file_name}?key={api_key}")
    except:
        pass

# --- WORKER FUNCTION (PARALLEL PROCESSING) ---

def process_single_row(row_data, api_key, prompt_template):
    """
    Independent function to process a single row. 
    Returns a dictionary with the result.
    """
    mobile = str(row_data.get('mobile_number', 'Unknown'))
    audio_url = row_data.get('recording_url')
    
    result = {
        "mobile_number": mobile,
        "recording_url": audio_url,
        "transcript": "",
        "status": "Pending"
    }

    tmp_path = None
    file_info = None

    try:
        # 1. Robust Extension Detection
        parsed_url = urlparse(audio_url)
        path_ext = os.path.splitext(parsed_url.path)[1]
        
        # Download
        r = requests.get(audio_url, stream=True, timeout=60)
        r.raise_for_status()
        
        content_type = r.headers.get('content-type', '')
        
        # Priority: URL Extension > Header > Default
        if path_ext:
            ext = path_ext
        else:
            ext = mimetypes.guess_extension(content_type) or ".mp3"
            
        # Create Temp File
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            for chunk in r.iter_content(chunk_size=8192):
                tmp.write(chunk)
            tmp_path = tmp.name
        
        mime_type = mimetypes.guess_type(tmp_path)[0] or "audio/mpeg"
        f_size = os.path.getsize(tmp_path)

        # 2. Upload
        unique_name = f"rec_{mobile}_{int(time.time())}{ext}"
        up_url = initiate_upload(api_key, unique_name, mime_type, f_size)
        file_info = upload_bytes(up_url, tmp_path, mime_type)
        
        # 3. Wait
        wait_for_active(api_key, file_info['name'])
        
        # 4. Transcribe
        transcript_text = generate_transcript(api_key, file_info['uri'], mime_type, prompt_template)
        
        result["transcript"] = transcript_text
        if "ERROR" in transcript_text or "BLOCKED" in transcript_text:
             result["status"] = "❌ Error"
        else:
             result["status"] = "✅ Success"

    except Exception as e:
        result["transcript"] = f"SYSTEM ERROR: {str(e)}"
        result["status"] = "❌ Failed"
        
    finally:
        # Cleanup Local
        if tmp_path and os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except: pass
        
        # Cleanup Cloud
        if file_info and 'name' in file_info:
            delete_file(api_key, file_info['name'])
            
    return result

# --- UI LOGIC ---

st.title(f"🎙️ Gemini Call Transcriber Pro ({MODEL_NAME})")
st.markdown("Batch transcription with parallel processing, safety bypass, and robust audio handling.")

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Gemini API Key", type="password")
    
    # Parallel Processing Control
    max_workers = st.slider("Concurrency (Threads)", min_value=1, max_value=8, value=4, 
                            help="Higher = Faster, but may hit API rate limits.")
    
    st.divider()
    language_mode = st.selectbox("Language", ["English (India)", "Hindi", "Mixed (Hinglish)"], index=2)
    lang_map = {
        "English (India)": "English (Indian accent)",
        "Hindi": "Hindi (Devanagari)",
        "Mixed (Hinglish)": "Mixed English and Hindi"
    }

uploaded_file = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])

if st.button("🚀 Start Batch Processing", type="primary", width='stretch'):
    if not api_key or not uploaded_file:
        st.error("Please enter API Key and Upload File.")
        st.stop()
        
    try:
        df = pd.read_excel(uploaded_file)
        if "recording_url" not in df.columns:
            st.error("Column 'recording_url' is missing.")
            st.stop()
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    # Construct Prompt Template
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

    # Status Containers
    progress_bar = st.progress(0)
    status_text = st.empty()
    result_placeholder = st.empty()
    
    processed_rows = []
    total_rows = len(df)
    
    status_text.info(f"Starting processing with {max_workers} threads...")

    # --- PARALLEL EXECUTION ---
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_row = {
            executor.submit(process_single_row, row, api_key, prompt_template): index 
            for index, row in df.iterrows()
        }
        
        completed_count = 0
        
        for future in as_completed(future_to_row):
            data = future.result()
            processed_rows.append(data)
            
            completed_count += 1
            progress_bar.progress(completed_count / total_rows)
            status_text.markdown(f"Processed **{completed_count}/{total_rows}** files.")
            
            # Live Result Update (Show last 5)
            live_df = pd.DataFrame(processed_rows)
            result_placeholder.dataframe(
                live_df[["mobile_number", "status", "transcript"]].tail(5), 
                use_container_width=True,
                hide_index=True
            )

    # Final Output
    final_df = pd.DataFrame(processed_rows)
    
    # Merge back with original DF to keep other columns if necessary, 
    # or just output the results. Here we output the clean results.
    
    st.success("Batch Processing Complete!")
    st.subheader("Final Results")
    st.dataframe(final_df, use_container_width=True)
    
    output = BytesIO()
    final_df.to_excel(output, index=False)
    st.download_button(
        label="📥 Download Full Transcript Excel",
        data=output.getvalue(),
        file_name=f"transcripts_{int(time.time())}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
        width='stretch'
    )
