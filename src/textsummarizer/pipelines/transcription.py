from transformers import pipeline
import os
from textsummarizer.logging import logger

# Resolve cache directory relative to this package file (resolves to project root workspace .whisper_cache)
CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..", ".whisper_cache"))

_transcribe_pipe = None

def get_transcribe_pipeline():
    global _transcribe_pipe
    if _transcribe_pipe is None:
        # Use CPU to avoid MPS backend sparse tensor unsupported operations on macOS
        _transcribe_pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-base.en",
            chunk_length_s=30,
            device="cpu",
            model_kwargs={"cache_dir": CACHE_DIR}
        )
    return _transcribe_pipe

def transcribe_audio(audio_path: str) -> str:
    """
    Transcribes the given audio file using OpenAI's Whisper base.en pipeline.
    """
    from langdetect import detect
    pipe = get_transcribe_pipeline()
    prediction = pipe(audio_path, batch_size=8)
    text = prediction.get("text", "").strip()
    
    # 1. Cleaned text check
    if not text:
        raise ValueError("Audio is not suitable for transcription (no speech detected).")
        
    # 2. Check for repetitive static/hallucinations (common in silent audio)
    words = text.lower().split()
    if len(words) > 4:
        word_counts = {}
        for w in words:
            word_counts[w] = word_counts.get(w, 0) + 1
        for w, count in word_counts.items():
            if count / len(words) > 0.65 and len(words) > 8:
                raise ValueError("Audio is not suitable for transcription (static noise/repetition detected).")
                
    # 3. Detect language
    try:
        # If the text is very short (e.g. 1-3 words), langdetect is extremely unreliable.
        # Skip langdetect if the words are common English words.
        words_list = text.lower().split()
        common_en = {"hello", "hi", "thank", "you", "good", "morning", "everyone", "meeting", "started", "test", "ok", "yes", "no"}
        if len(words_list) <= 3 and any(w in common_en for w in words_list):
            pass
        else:
            detected_lang = detect(text)
            if detected_lang != "en":
                raise ValueError(f"Audio is not in English (detected: {detected_lang}) and is not suitable for transcription.")
    except Exception as lang_err:
        if isinstance(lang_err, ValueError):
            raise lang_err
        # If langdetect fails due to lack of text features
        if not any(c.isalpha() for c in text):
            raise ValueError("Audio is not suitable for transcription (no spoken words detected).")
            
    return text
