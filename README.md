# SumMeet: AI-Powered Dialogue & Text Summarization Suite

SumMeet is a modern, responsive web application designed for audio transcription, dialogue analysis, and text summarization. The application features a dual-mode interface: a **Voice Session Summarizer** that captures live microphone inputs or uploaded audio files to generate dialogue-turn transcripts, and a **Traditional Text Summarizer** that condenses typed transcripts. 

All user sessions, transcripts, audio recordings, and text summaries are saved securely in a cloud database with automated data retention controls.

---

## Key Features

- **Microphone Transcription**: Live, real-time voice recording and speech-to-text transcription.
- **Audio Upload**: Support for uploading WAV, MP3, and other audio formats to transcribe conversational dialogues.
- **Dual Summarization Engine**:
  - **Dialogue Summarizer**: Specifically tuned for conversational dialogue patterns (turns, speakers) using custom weights.
  - **Traditional Text Summarizer**: For summarizing raw emails, documents, or typed transcripts.
- **Persistent History**: Sidebar panels to save, retrieve, view, and manage past meeting sessions and text summaries.
- **Secured Authentication**: Integrated Google OAuth2 login for secure user authentication.
- **Cloud Database Integration**: Database management using Supabase PostgreSQL, handling session state, audio bytes, and summary records.
- **Automated Data Retention**: Implements a rolling 3-month retention policy to automatically purge old logs and audio objects.

---

## Architecture Overview

The system is split into three main components:
1. **Frontend**: A single-page, responsive dashboard built with semantic HTML5, vanilla CSS3 (utilizing modern glassmorphism aesthetics), and custom JavaScript.
2. **Backend**: An asynchronous FastAPI web server handling request routing, session synchronization, database transactions, and client-side Google token validation.
3. **AI Core**: Offloads compute-heavy workloads (Whisper speech-to-text and fine-tuned Pegasus seq2seq summarizer) to the Hugging Face Serverless Inference API, ensuring a small memory footprint and rapid execution.

---

## Directory Structure

```text
├── app.py                  # Main FastAPI application server
├── setup.py                # Package metadata and installation configuration
├── requirements.txt        # Python dependency specifications
├── src/                    # Source code package
│   └── textsummarizer/
│       ├── logging/        # Logging config
│       ├── pipelines/      # Transcription & Prediction pipelines
│       └── utils/          # Database connection & authentication utility files
├── templates/              # HTML layout templates
└── static/                 # Static assets (images, styles)
```

---

## Environment Configuration

To run the application locally or in a cloud environment (e.g., Render), create an environment file or define the following environment variables:

| Variable | Description |
| :--- | :--- |
| `DB_HOST` | Supabase database host endpoint |
| `DB_PORT` | Supabase database port (usually `6543`) |
| `DB_NAME` | Supabase database name |
| `DB_USER` | Supabase database user name |
| `DB_PASSWORD` | Supabase database password |
| `PEGASUS_MODEL_ID` | Your custom Hugging Face model repository ID (e.g. `username/repo`) |
| `HF_TOKEN` | Hugging Face User Access Token (Read access) |
| `GOOGLE_CLIENT_ID` | Google OAuth Web client credential client ID |

---

## Local Development Setup

To run this project on your local machine for development:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mudhasir-hussain/text-summarizer.git
   cd text-summarizer
   ```

2. **Install dependencies**:
   It is recommended to use a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   pip install -e .
   ```

3. **Start the server**:
   ```bash
   python app.py
   ```
   Open your browser and navigate to `http://127.0.0.1:8080`.
