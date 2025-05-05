# main.py

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from utils import video_utils
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from utils.advice_utils import generate_advice
from utils.pdf_utils import summarize_pdf
from utils.audio_utils import transcribe_audio
from fastapi import FastAPI
from fastapi.responses import FileResponse
import os
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from utils.image_utils import classify_image
import shutil
from utils.image_utils import classify_image
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv




# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

app = FastAPI()

# === Static & templates ===
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    message: str

# === HTML Routes ===

@app.get("/")
@app.get("/index")
@app.get("/index.html")
def index():
    return FileResponse(os.path.join(TEMPLATES_DIR, "index.html"))

@app.get("/pdfVoice")
@app.get("/pdfVoice.html")
def pdf_voice():
    return FileResponse(os.path.join(TEMPLATES_DIR, "pdfVoice.html"))

@app.get("/videoTranscribe")
@app.get("/videoTranscribe.html")
def video_transcribe():
    return FileResponse(os.path.join(TEMPLATES_DIR, "videoTranscribe.html"))

@app.get("/imageClassification", response_class=HTMLResponse)
def image_classification_page(request: Request):
    return templates.TemplateResponse("imageClassification.html", {"request": request})
async def classify_uploaded_image(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = classify_image(file_path)
    os.remove(file_path)
    return {"result": result}

@app.get("/aiBot")
@app.get("/aiBot.html")
def ai_bot():
    return FileResponse(os.path.join(TEMPLATES_DIR, "aiBot.html"))


# === VIDEO PROCESSING ENDPOINTS ===

@app.post("/transcribe_video")
async def transcribe_video(youtube_url: str = Form(...)):
    audio_path = video_utils.download_audio_from_youtube(youtube_url)
    if not audio_path:
        return JSONResponse({"transcript": None, "error": "Audio download failed."}, status_code=400)

    transcript = video_utils.transcribe_audio(audio_path)
    if not transcript:
        return JSONResponse({"transcript": None, "error": "Transcription failed."}, status_code=500)

    # Cache the chain globally for the next question
    video_utils.video_qa_chain = video_utils.build_rag_chain_from_transcript(transcript)
    return JSONResponse({"transcript": transcript})

@app.post("/ask_video_question")
async def ask_video_question(question: str = Form(...)):
    if not video_utils.video_qa_chain:
        return JSONResponse({"answer": "Please transcribe a video first."}, status_code=400)

    result = video_utils.video_qa_chain.run(question)
    return JSONResponse({"answer": result})



# ===  PDF Analyzer  ===
@app.post("/analyze-pdf")
async def analyze_pdf(pdf_file: UploadFile = File(...)):
    os.makedirs("temp_files", exist_ok=True)
    file_path = f"temp_files/{pdf_file.filename}"

    try:
        # Save uploaded PDF
        with open(file_path, "wb") as f:
            f.write(await pdf_file.read())

        # Analyze it
        summary = summarize_pdf(file_path)
        advice = generate_advice(summary)

        return {"summary": summary, "advice": advice}
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


# ===  Audio Transcriber  ===
@app.post("/transcribe-audio")
async def transcribe_audio_route(audio_file: UploadFile = File(...)):
    os.makedirs("temp_files", exist_ok=True)
    file_path = f"temp_files/{audio_file.filename}"

    try:
        # Save uploaded audio
        with open(file_path, "wb") as f:
            f.write(await audio_file.read())

        # Transcribe it
        transcription = transcribe_audio(file_path)
        advice = generate_advice(transcription)

        return {"transcription": transcription, "advice": advice}
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
            
            
# ===  Image Classifier  ===            
@app.get("/imageClassification", response_class=HTMLResponse)
def image_classification_page(request: Request):
    return templates.TemplateResponse("imageClassification.html", {"request": request})

@app.post("/classify-image")
async def classify_uploaded_image(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = classify_image(file_path)
    os.remove(file_path)
    return {"result": result}


# ===  AI Bot  ===
@app.post("/ask")
async def ask_bot(msg: Message):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful health assistant."},
                {"role": "user", "content": msg.message}
            ]
        )
        answer = response.choices[0].message.content.strip()
        return JSONResponse(content={"response": answer})
    except Exception as e:
        return JSONResponse(content={"response": f"Error: {str(e)}"}, status_code=500)
