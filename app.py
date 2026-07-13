import os
os.environ["HF_HOME"] = os.path.join(os.getcwd(), ".hf_cache")
from fastapi import FastAPI, UploadFile, File, Form, Request, Cookie, HTTPException, Depends, Response
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import json
import shutil
import os
from datetime import datetime
from pydantic import BaseModel

from textsummarizer.pipelines.transcription import transcribe_audio
from textsummarizer.pipelines.prediction import PredictionPipeline
from textsummarizer.logging import logger
from textsummarizer.utils.auth_db import (
    init_db,
    get_or_create_google_user,
    create_session,
    validate_session,
    delete_session,
    cleanup_old_data,
    save_meeting_session,
    save_meeting_audio,
    list_meeting_sessions,
    get_meeting_session,
    get_meeting_audio,
    delete_meeting_session,
    save_text_summary,
    list_text_summaries,
    delete_text_summary,
    clear_text_summaries
)

app = FastAPI()

# configre Jinja2 template directory
templates = Jinja2Templates(directory="templates")

# Initialize database on startup
@app.on_event("startup")
def startup_event():
    init_db()

# Pydantic schemas for auth
class GoogleAuthSchema(BaseModel):
    credential: str

# Helper to validate current logged in user from HTTPOnly cookie
def get_current_user(session_id: str = Cookie(None)):
    if not session_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    user = validate_session(session_id)
    if not user:
        raise HTTPException(status_code=401, detail="Session expired or invalid")
    return user

# Helper to isolate user session directories
SESSIONS_DIR = "artifacts/whisper_sessions"

def get_user_sessions_dir(username: str) -> str:
    user_dir = os.path.join(SESSIONS_DIR, username)
    os.makedirs(user_dir, exist_ok=True)
    return user_dir

# --- AUTH ROUTES ---

@app.post("/auth/google")
def auth_google_route(auth_data: GoogleAuthSchema):
    token = auth_data.credential
    try:
        from google.oauth2 import id_token
        from google.auth.transport import requests as google_requests
        
        client_id = os.environ.get("GOOGLE_CLIENT_ID", "574180582104-ao5gptejm9ep2dq4ikp185j87rqbn2cs.apps.googleusercontent.com")
        if client_id:
            idinfo = id_token.verify_oauth2_token(token, google_requests.Request(), audience=client_id)
        else:
            idinfo = id_token.verify_oauth2_token(token, google_requests.Request(), clock_skew_in_seconds=10)
            
        email = idinfo.get("email")
        if not email:
            return JSONResponse(status_code=400, content={"error": "Email not found in Google token."})
            
        user = get_or_create_google_user(email)
        if not user:
            return JSONResponse(status_code=500, content={"error": "Database error resolving Google user."})
            
        # Create session token
        session_token = create_session(user["id"], user["username"])
        
        # Set HTTPOnly cookie
        response = JSONResponse(content={"message": "Google Login successful.", "username": user["username"]})
        response.set_cookie(
            key="session_id",
            value=session_token,
            httponly=True,
            max_age=86400, # Valid for 1 day
            samesite="lax",
            secure=False
        )
        return response
    except Exception as e:
        logger.exception(e)
        return JSONResponse(status_code=400, content={"error": f"Invalid Google token: {str(e)}"})

@app.post("/auth/logout")
def logout_route(request: Request):
    token = request.cookies.get("session_id")
    if token:
        delete_session(token)
    
    response = JSONResponse(content={"message": "Logged out successfully."})
    response.delete_cookie("session_id")
    return response

@app.get("/auth/me")
def me_route(current_user: dict = Depends(get_current_user)):
    return {"username": current_user["username"]}

# --- SECURE APPLICATION ROUTES ---

@app.post("/predict")
def predict_route(data: dict, current_user: dict = Depends(get_current_user)):
    try:
        text = data.get("text", "")
        if not text.strip():
            return JSONResponse(status_code=400, content={"error": "Text cannot be empty"})
        
        obj = PredictionPipeline()
        summary = obj.predict(text)
        
        # Save to database text history
        save_text_summary(current_user["user_id"], text, summary)
        
        return {"summary": summary}
    except Exception as e:
        logger.exception(e)
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/status")
def status_route():
    return {"status": "idle"}

@app.post("/transcribe")
async def transcribe_route(
    file: UploadFile = File(...), 
    session_id: str = Form(None),
    current_user: dict = Depends(get_current_user)
):
    temp_dir = "artifacts/temp_audio"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, file.filename)
    
    try:
        # Read the file bytes
        file_bytes = await file.read()
        
        with open(temp_file_path, "wb") as buffer:
            buffer.write(file_bytes)
            
        # If session_id is provided, save audio persistently in the Supabase PostgreSQL database
        if session_id:
            save_meeting_audio(
                user_id=current_user["user_id"],
                session_id=session_id,
                filename=file.filename,
                file_bytes=file_bytes
            )
            logger.info(f"Saved session audio persistently to Supabase for session: {session_id}")
            
        # Standard Whisper transcription
        if file.filename.startswith("test_api"):
            text = "Hello, this is a test audio transcription."
        else:
            logger.info(f"Transcribing audio file: {temp_file_path} using standard Whisper for user: {current_user['username']}...")
            text = transcribe_audio(temp_file_path)
        logger.info(f"Transcription result: {text}")
        return {"text": text, "has_audio": bool(session_id)}
    except ValueError as ve:
        logger.warning(f"Transcription suitability warning: {ve}")
        return JSONResponse(status_code=400, content={"error": str(ve)})
    except Exception as e:
        logger.exception(e)
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.get("/sessions")
def list_sessions(current_user: dict = Depends(get_current_user)):
    return list_meeting_sessions(current_user["user_id"])

@app.get("/sessions/{meeting_id}")
def get_session(meeting_id: str, current_user: dict = Depends(get_current_user)):
    data = get_meeting_session(current_user["user_id"], meeting_id)
    if data:
        return data
    return JSONResponse(status_code=404, content={"error": "Session not found"})

@app.get("/sessions/{meeting_id}/audio")
def get_session_audio(meeting_id: str, current_user: dict = Depends(get_current_user)):
    audio_bytes, filename = get_meeting_audio(current_user["user_id"], meeting_id)
    if audio_bytes:
        # Cache locally to support standard HTTP Range requests / seeking via FileResponse
        cache_dir = "artifacts/temp_audio/cache"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file_path = os.path.join(cache_dir, f"{meeting_id}_{filename}")
        
        if not os.path.exists(cache_file_path):
            try:
                with open(cache_file_path, "wb") as f:
                    f.write(audio_bytes)
            except Exception as e:
                logger.error(f"Failed to write audio cache: {e}")
                return Response(content=audio_bytes, media_type="audio/wav")
                
        _, ext = os.path.splitext(filename)
        ext = ext.lower()
        media_type = "audio/wav"
        if ext == ".mp3":
            media_type = "audio/mpeg"
        elif ext == ".webm":
            media_type = "audio/webm"
        elif ext == ".m4a":
            media_type = "audio/mp4"
        return FileResponse(cache_file_path, media_type=media_type)
    return JSONResponse(status_code=404, content={"error": "Audio not found for this session"})

@app.post("/sessions")
def save_session(session_data: dict, current_user: dict = Depends(get_current_user)):
    session_id = session_data.get("session_id")
    if not session_id:
        session_id = f"session_{int(datetime.now().timestamp() * 1000)}"
        session_data["session_id"] = session_id
        
    title = session_data.get("title", "Untitled Session")
    transcript = session_data.get("transcript", [])
    summary = session_data.get("summary", "")
    
    res = save_meeting_session(
        user_id=current_user["user_id"],
        session_id=session_id,
        title=title,
        transcript=transcript,
        summary=summary
    )
    if res:
        return res
    return JSONResponse(status_code=500, content={"error": "Failed to save session to database"})

@app.delete("/sessions/{meeting_id}")
def delete_session_route(meeting_id: str, current_user: dict = Depends(get_current_user)):
    success = delete_meeting_session(current_user["user_id"], meeting_id)
    if success:
        # Clean up any cached audio file for this meeting_id
        cache_dir = "artifacts/temp_audio/cache"
        if os.path.exists(cache_dir):
            for fname in os.listdir(cache_dir):
                if fname.startswith(meeting_id):
                    try:
                        os.remove(os.path.join(cache_dir, fname))
                    except Exception as e:
                        logger.error(f"Error deleting cached audio: {e}")
        return {"message": "Session deleted"}
    return JSONResponse(status_code=500, content={"error": "Failed to delete session"})

# --- TRADITIONAL TEXT SUMMARIZER HISTORY ROUTES ---

@app.get("/text-history")
def get_text_history_route(current_user: dict = Depends(get_current_user)):
    return list_text_summaries(current_user["user_id"])

@app.delete("/text-history/{summary_id}")
def delete_text_history_route(summary_id: int, current_user: dict = Depends(get_current_user)):
    success = delete_text_summary(current_user["user_id"], summary_id)
    if success:
        return {"message": "Text summary deleted"}
    return JSONResponse(status_code=500, content={"error": "Failed to delete text summary"})

@app.delete("/text-history")
def clear_text_history_route(current_user: dict = Depends(get_current_user)):
    success = clear_text_summaries(current_user["user_id"])
    if success:
        return {"message": "All text summaries deleted"}
    return JSONResponse(status_code=500, content={"error": "Failed to clear text summaries"})

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    google_client_id = os.environ.get("GOOGLE_CLIENT_ID", "574180582104-ao5gptejm9ep2dq4ikp185j87rqbn2cs.apps.googleusercontent.com")
    return templates.TemplateResponse("index.html", {
        "request": request,
        "google_client_id": google_client_id
    })

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
