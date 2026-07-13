# SumMeet: Secure Audio Transcription & Dialogue Summarization

SumMeet is a secure web dashboard that combines OpenAI's Whisper model (`whisper-base.en`) for offline speech-to-text transcription and a fine-tuned Google Pegasus model (`pegasus-samsum-model`) for dialogue summarization.

## Features
- **Real-time Microphone Transcription**: Speak and see live speech updates transcribed in real-time.
- **Audio File Upload**: Upload local audio files to transcribe dialogue turns.
- **Dialogue Summarization**: Condense speech transcripts into brief summaries using the Pegasus summarizer.
- **Save & Load Sessions**: Store and retrieve previous transcription meetings as JSON logs.
- **Model Training Pipeline**: Retrain the Pegasus summarizer model with data validation, data ingestion, and transformation stages.

## Installation
```bash
pip install -r requirements.txt
pip install -e .
```

## Running the Application
```bash
python app.py
```
Open `http://127.0.0.1:8080` in your web browser.
