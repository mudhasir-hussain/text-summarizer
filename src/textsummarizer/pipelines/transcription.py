import os
from huggingface_hub import InferenceClient
from textsummarizer.logging import logger

def transcribe_audio(audio_path: str) -> str:
    """
    Transcribes the given audio file using OpenAI's Whisper model via Hugging Face InferenceClient.
    """
    from langdetect import detect
    import requests
    hf_token = os.environ.get("HF_TOKEN")
    api_url = "https://router.huggingface.co/hf-inference/models/openai/whisper-base.en"
    headers = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
        
    logger.info(f"Transcribing audio file via raw requests to HF Router: {audio_path}...")
    
    try:
        with open(audio_path, "rb") as f:
            data = f.read()
            
        response = requests.post(api_url, headers=headers, data=data)
        if response.status_code == 200:
            result = response.json()
            text = result.get("text", "").strip()
        else:
            raise Exception(f"HF API Error {response.status_code}: {response.text}")
        
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
    except Exception as e:
        logger.error(f"Transcription failure: {e}")
        raise e
