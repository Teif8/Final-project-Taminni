# 🩺 Taminni – Multimodal AI Health Assistant

**Taminni** is an AI-powered health assistant that helps users understand and interact with complex medical content from various sources like YouTube videos, voice notes, PDFs, and health images. It combines multiple AI models (OpenAI, Whisper, GPT-4o) with a clean web interface powered by FastAPI.

---

## 🚀 Features

- 🎙️ **Voice Note Transcription** – Convert spoken medical recordings into readable summaries.
- 📄 **PDF Summarization** – Simplify complex medical documents or lab results.
- 🎥 **YouTube Video QA** – Ask questions about health-related videos after auto-transcription.
- 🖼️ **Image Classification** – Upload general or skin condition images for AI analysis.
- 💬 **AI Chatbot** – Ask health-related questions conversationally.

---

## 🏗️ Project Architecture

```
Frontend (HTML/CSS/JS)
        ⬇
Backend (FastAPI + Python)
        ⬇
[ AI Models ]
  - Whisper (voice transcription)
  - OpenAI GPT-4 (chat, summarization)
  - GPT-4o (image diagnosis)
        ⬇
Output: User-friendly summaries, explanations, answers
```

---

## 🧪 Methodology

- Built modular utilities for PDF, voice, video, and image input
- Used prompt engineering to guide OpenAI model responses
- Applied GPT-4o for visual diagnostic support
- Used LangChain for RAG-based chatbot experiments
- Evaluated quality with custom prompts and QA datasets

---

## ⚙️ Installation

### 1. Clone the repo
```bash
git clone https://github.com/your-username/taminni.git
cd taminni
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3. Install requirements
```bash
pip install -r requirements.txt
```

### 4. Add `.env` file
Create a `.env` file based on `.env.example` and include your OpenAI key.

---

## ▶️ Run the App

```bash
uvicorn main:app --reload
```

Then open your browser at: [http://localhost:8000](http://localhost:8000)

---

## 📁 Folder Structure

```
taminni/
├── main.py                   # FastAPI app entrypoint
├── requirements.txt
├── .env                      # Secrets (not pushed)
├── static/                   # CSS/images
├── templates/                # HTML pages
├── utils/                    # Core logic for each modality
│   ├── pdf_utils.py
│   ├── audio_utils.py
│   ├── image_utils.py
│   └── chatbot_utils.py
├── deployment/               # Dockerfile, Procfile, start script
├── experiments/              # Prototypes, test notebooks
└── presentation/             # Final slides
```

---

## 📦 Deployment

See the `deployment/` folder for:
- `Dockerfile` for containerization
- `start.sh` for FastAPI launch
- `Procfile` for Heroku deployment
- `.env.example` for environment configuration

---

## 🧪 Experiments

Check the `/experiments` folder to explore:
- Prompt engineering tests
- Whisper model accuracy trials
- GPT-4o prompts
- LangChain RAG Q&A prototypes

---

## 📽️ Presentation

Presentation slides are included under `/presentation/Taminni.pdf`.

---

## 👥 Contributors

- **Deema Alsuawari** 
- **Teif Alharthi** 

---
