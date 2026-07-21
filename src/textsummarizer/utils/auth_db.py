import os
import psycopg2
from psycopg2.extras import RealDictCursor
import hashlib
import secrets
import json
from datetime import datetime, timedelta
from textsummarizer.logging import logger

def get_db_connection():
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        return psycopg2.connect(db_url, cursor_factory=RealDictCursor, connect_timeout=5)
    
    db_user = os.environ.get("DB_USER", "postgres.wbxaqhoqdewajnctumrb")
    db_pass = os.environ.get("DB_PASSWORD", "S1u2p3a4b5!@")
    db_host = os.environ.get("DB_HOST", "aws-0-ap-southeast-1.pooler.supabase.com")
    db_port = os.environ.get("DB_PORT", "6543")
    db_name = os.environ.get("DB_NAME", "postgres")
    
    conn = psycopg2.connect(
        host=db_host,
        database=db_name,
        user=db_user,
        password=db_pass,
        port=db_port,
        cursor_factory=RealDictCursor,
        connect_timeout=5
    )
    return conn

def init_db():
    logger.info("Initializing Supabase PostgreSQL authentication database...")
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Create users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(255) NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                token VARCHAR(255) PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                username VARCHAR(255) NOT NULL,
                expires_at TIMESTAMP NOT NULL
            )
        """)

        # Create meeting_sessions table to store voice transcripts and binary audio blobs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS meeting_sessions (
                id VARCHAR(255) PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                title VARCHAR(255) NOT NULL,
                transcript JSONB NOT NULL,
                summary TEXT,
                has_audio BOOLEAN DEFAULT FALSE,
                audio_data BYTEA,
                audio_filename VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create text_summaries table to store traditional text summarizer history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS text_summaries (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                input_text TEXT NOT NULL,
                summary_text TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Enable Row-Level Security (RLS) on all tables to prevent public access via PostgREST/Supabase REST API
        cursor.execute("ALTER TABLE users ENABLE ROW LEVEL SECURITY")
        cursor.execute("ALTER TABLE sessions ENABLE ROW LEVEL SECURITY")
        cursor.execute("ALTER TABLE meeting_sessions ENABLE ROW LEVEL SECURITY")
        cursor.execute("ALTER TABLE text_summaries ENABLE ROW LEVEL SECURITY")
        
        conn.commit()
        logger.info("Supabase PostgreSQL database initialized successfully.")
        
        # Run cleanup retention policy on startup
        cleanup_old_data()
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to initialize database: {e}")
        raise e
    finally:
        cursor.close()
        conn.close()

def hash_password(password: str, salt: str = None) -> tuple:
    """
    Hashes a password using PBKDF2-HMAC-SHA256 with 100,000 iterations.
    Returns: (password_hash_hex, salt_hex)
    """
    if salt is None:
        salt = secrets.token_hex(16)
    
    pwd_bytes = password.encode('utf-8')
    salt_bytes = salt.encode('utf-8')
    hash_bytes = hashlib.pbkdf2_hmac('sha256', pwd_bytes, salt_bytes, 100000)
    pwd_hash = hash_bytes.hex()
    
    return pwd_hash, salt

def register_user(username: str, password: str) -> bool:
    """
    Registers a new user. Returns True if successful, False if username already exists.
    """
    username = username.strip().lower()
    if not username or len(password) < 6:
        return False
        
    pwd_hash, salt = hash_password(password)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (username, password_hash, salt) VALUES (%s, %s, %s)",
            (username, pwd_hash, salt)
        )
        conn.commit()
        logger.info(f"User registered successfully: {username}")
        return True
    except psycopg2.IntegrityError:
        conn.rollback()
        logger.warning(f"Registration failed: Username '{username}' already exists.")
        return False
    except Exception as e:
        conn.rollback()
        logger.error(f"Error registering user: {e}")
        return False
    finally:
        cursor.close()
        conn.close()

def get_or_create_google_user(email: str) -> dict:
    """
    Finds a user by email/username. If they do not exist, registers them.
    Returns the user dict.
    """
    username = email.strip().lower()
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        if user:
            return dict(user)
            
        # Register a new user
        pwd_hash, salt = hash_password(secrets.token_hex(16))
        cursor.execute(
            "INSERT INTO users (username, password_hash, salt) VALUES (%s, %s, %s) RETURNING *",
            (username, pwd_hash, salt)
        )
        user = cursor.fetchone()
        conn.commit()
        logger.info(f"Google user registered successfully: {username}")
        return dict(user)
    except Exception as e:
        conn.rollback()
        logger.error(f"Error handling Google user: {e}")
        return None
    finally:
        cursor.close()
        conn.close()

def authenticate_user(username: str, password: str) -> dict:
    """
    Validates user credentials.
    Returns the user dict if authenticated, otherwise None.
    """
    username = username.strip().lower()
    conn = get_db_connection()
    cursor = conn.cursor()
    user = None
    try:
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
    except Exception as e:
        logger.error(f"Error authenticating user: {e}")
    finally:
        cursor.close()
        conn.close()
        
    if not user:
        return None
        
    # Verify hashed password using user's stored salt
    calculated_hash, _ = hash_password(password, user["salt"])
    if calculated_hash == user["password_hash"]:
        return dict(user)
    
    return None

def create_session(user_id: int, username: str, expiry_hours: int = 24) -> str:
    """
    Creates a new secure session token valid for a set amount of hours.
    """
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now() + timedelta(hours=expiry_hours)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO sessions (token, user_id, username, expires_at) VALUES (%s, %s, %s, %s)",
            (token, user_id, username, expires_at)
        )
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Error creating session: {e}")
        raise e
    finally:
        cursor.close()
        conn.close()
    return token

def validate_session(token: str) -> dict:
    """
    Checks if a session token is valid and not expired.
    Returns: User details if valid, None if invalid/expired.
    """
    if not token:
        return None
        
    conn = get_db_connection()
    cursor = conn.cursor()
    session = None
    try:
        cursor.execute("SELECT * FROM sessions WHERE token = %s", (token,))
        session = cursor.fetchone()
        
        if not session:
            return None
            
        # Check expiration
        expires_at = session["expires_at"]
        if isinstance(expires_at, str):
            expires_at = datetime.fromisoformat(expires_at)
            
        if datetime.now() > expires_at:
            # Delete expired session
            cursor.execute("DELETE FROM sessions WHERE token = %s", (token,))
            conn.commit()
            logger.info("Expired session token cleaned up.")
            return None
            
        return {
            "user_id": session["user_id"],
            "username": session["username"]
        }
    except Exception as e:
        conn.rollback()
        logger.error(f"Error validating session: {e}")
        return None
    finally:
        cursor.close()
        conn.close()

def delete_session(token: str):
    """
    Deletes a session token from the database (logout).
    """
    if not token:
        return
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM sessions WHERE token = %s", (token,))
        conn.commit()
        logger.info("Session token invalidated on logout.")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error deleting session: {e}")
    finally:
        cursor.close()
        conn.close()

def cleanup_old_data():
    """
    Cleans up meeting sessions and traditional text summaries that are older than 3 months.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        logger.info("Running 3-month data retention policy cleanup...")
        cursor.execute("DELETE FROM meeting_sessions WHERE created_at < NOW() - INTERVAL '3 months'")
        cursor.execute("DELETE FROM text_summaries WHERE created_at < NOW() - INTERVAL '3 months'")
        conn.commit()
        logger.info("Retention policy cleanup completed successfully.")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error executing retention policy cleanup: {e}")
    finally:
        cursor.close()
        conn.close()

def save_meeting_session(user_id: int, session_id: str, title: str, transcript: list, summary: str) -> dict:
    """
    Saves or updates a meeting session in the database.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Convert transcript to JSON string
        transcript_json = json.dumps(transcript)
        
        cursor.execute("""
            INSERT INTO meeting_sessions (id, user_id, title, transcript, summary, updated_at)
            VALUES (%s, %s, %s, %s, %s, NOW())
            ON CONFLICT (id) DO UPDATE SET
                title = EXCLUDED.title,
                transcript = EXCLUDED.transcript,
                summary = EXCLUDED.summary,
                updated_at = NOW()
            RETURNING id, title, transcript, summary, has_audio, created_at
        """, (session_id, user_id, title, transcript_json, summary))
        
        row = cursor.fetchone()
        conn.commit()
        
        # Run cleanup periodically on save
        cleanup_old_data()
        
        if row:
            res = dict(row)
            if isinstance(res["transcript"], str):
                res["transcript"] = json.loads(res["transcript"])
            return res
        return None
    except Exception as e:
        conn.rollback()
        logger.error(f"Error saving meeting session {session_id}: {e}")
        return None
    finally:
        cursor.close()
        conn.close()

def save_meeting_audio(user_id: int, session_id: str, filename: str, file_bytes: bytes) -> bool:
    """
    Saves or updates binary audio data associated with a meeting session.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO meeting_sessions (id, user_id, title, transcript, summary, has_audio, audio_data, audio_filename, updated_at)
            VALUES (%s, %s, %s, '[]'::jsonb, '', TRUE, %s, %s, NOW())
            ON CONFLICT (id) DO UPDATE SET
                audio_data = EXCLUDED.audio_data,
                audio_filename = EXCLUDED.audio_filename,
                has_audio = TRUE,
                updated_at = NOW()
        """, (session_id, user_id, f"Meeting Audio ({filename})", psycopg2.Binary(file_bytes), filename))
        
        conn.commit()
        logger.info(f"Saved meeting audio to database for session {session_id}")
        return True
    except Exception as e:
        conn.rollback()
        logger.error(f"Error saving meeting audio for session {session_id}: {e}")
        return False
    finally:
        cursor.close()
        conn.close()

def list_meeting_sessions(user_id: int) -> list:
    """
    Lists meeting sessions for a user (without transferring heavy audio data).
    """
    # Run cleanup periodically on list
    cleanup_old_data()
    
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT id, title, created_at, (summary IS NOT NULL AND summary != '') as has_summary, has_audio
            FROM meeting_sessions
            WHERE user_id = %s
            ORDER BY created_at DESC
        """, (user_id,))
        rows = cursor.fetchall()
        
        sessions = []
        for row in rows:
            sessions.append({
                "session_id": row["id"],
                "title": row["title"],
                "timestamp": row["created_at"].isoformat() if isinstance(row["created_at"], datetime) else row["created_at"],
                "has_summary": row["has_summary"],
                "has_audio": row["has_audio"]
            })
        return sessions
    except Exception as e:
        logger.error(f"Error listing meeting sessions for user {user_id}: {e}")
        return []
    finally:
        cursor.close()
        conn.close()

def get_meeting_session(user_id: int, session_id: str) -> dict:
    """
    Retrieves a single meeting session metadata and transcript turns.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT id, title, transcript, summary, has_audio, created_at
            FROM meeting_sessions
            WHERE user_id = %s AND id = %s
        """, (user_id, session_id))
        row = cursor.fetchone()
        if row:
            res = dict(row)
            res["session_id"] = res["id"]
            if isinstance(res["transcript"], str):
                res["transcript"] = json.loads(res["transcript"])
            elif isinstance(res["transcript"], (list, dict)):
                pass
            return res
        return None
    except Exception as e:
        logger.error(f"Error fetching meeting session {session_id}: {e}")
        return None
    finally:
        cursor.close()
        conn.close()

def get_meeting_audio(user_id: int, session_id: str) -> tuple:
    """
    Retrieves the raw binary audio data and filename for a meeting session.
    Returns: (audio_bytes, filename)
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT audio_data, audio_filename
            FROM meeting_sessions
            WHERE user_id = %s AND id = %s AND has_audio = TRUE
        """, (user_id, session_id))
        row = cursor.fetchone()
        if row and row["audio_data"]:
            return bytes(row["audio_data"]), row["audio_filename"]
        return None, None
    except Exception as e:
        logger.error(f"Error fetching meeting audio for session {session_id}: {e}")
        return None, None
    finally:
        cursor.close()
        conn.close()

def delete_meeting_session(user_id: int, session_id: str) -> bool:
    """
    Deletes a meeting session from the database.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            DELETE FROM meeting_sessions
            WHERE user_id = %s AND id = %s
        """, (user_id, session_id))
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        logger.error(f"Error deleting meeting session {session_id}: {e}")
        return False
    finally:
        cursor.close()
        conn.close()

def save_text_summary(user_id: int, input_text: str, summary_text: str) -> bool:
    """
    Saves a traditional text summary to the database.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO text_summaries (user_id, input_text, summary_text)
            VALUES (%s, %s, %s)
        """, (user_id, input_text, summary_text))
        conn.commit()
        logger.info(f"Saved text summary to database for user {user_id}")
        return True
    except Exception as e:
        conn.rollback()
        logger.error(f"Error saving text summary for user {user_id}: {e}")
        return False
    finally:
        cursor.close()
        conn.close()

def list_text_summaries(user_id: int) -> list:
    """
    Lists traditional text summaries for a user.
    """
    # Run cleanup periodically on list
    cleanup_old_data()
    
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT id, input_text, summary_text, created_at
            FROM text_summaries
            WHERE user_id = %s
            ORDER BY created_at DESC
        """, (user_id,))
        rows = cursor.fetchall()
        
        history = []
        for row in rows:
            input_snippet = row["input_text"].strip()
            first_line = input_snippet.split('\n')[0]
            if len(first_line) > 30:
                title = first_line[:30] + "..."
            else:
                title = first_line if first_line else "Dialogue Summary"
                
            history.append({
                "id": row["id"],
                "title": title,
                "input_text": row["input_text"],
                "summary_text": row["summary_text"],
                "timestamp": row["created_at"].isoformat() if isinstance(row["created_at"], datetime) else row["created_at"]
            })
        return history
    except Exception as e:
        logger.error(f"Error listing text summaries for user {user_id}: {e}")
        return []
    finally:
        cursor.close()
        conn.close()

def delete_text_summary(user_id: int, summary_id: int) -> bool:
    """
    Deletes a traditional text summary from the database.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            DELETE FROM text_summaries
            WHERE user_id = %s AND id = %s
        """, (user_id, summary_id))
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        logger.error(f"Error deleting text summary {summary_id}: {e}")
        return False
    finally:
        cursor.close()
        conn.close()

def clear_text_summaries(user_id: int) -> bool:
    """
    Deletes all traditional text summaries for a user from the database.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            DELETE FROM text_summaries
            WHERE user_id = %s
        """, (user_id,))
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        logger.error(f"Error clearing text summaries for user {user_id}: {e}")
        return False
    finally:
        cursor.close()
        conn.close()
