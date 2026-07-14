import os
from huggingface_hub import InferenceClient
from textsummarizer.logging import logger

def transcribe_audio(audio_path: str) -> str:
    """
    Transcribes the given audio file using OpenAI's Whisper model via Hugging Face InferenceClient.
    """
    from langdetect import detect
    
    hf_token = os.environ.get("HF_TOKEN")
    logger.info(f"Transcribing audio file via HF InferenceClient: {audio_path}...")
    
    try:
        with open(audio_path, "rb") as f:
            data = f.read()
            
        import json
        client = InferenceClient(api_key=hf_token)
        # Use raw client.post to bypass huggingface_hub's third-party provider resolution logic
        response = client.post(data=data, model="openai/whisper-base.en")
        result = json.loads(response)
        text = result.get("text", "").strip()
        
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
