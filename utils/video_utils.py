import whisper
import tempfile
import yt_dlp
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load the model once globally
whisper_model = whisper.load_model("tiny")

video_qa_chain = None  # Cache for global access if needed

def download_audio_from_youtube(youtube_url: str) -> str:
    """
    Downloads audio from a YouTube URL and returns the path to the .wav file.
    """
    try:
        temp_dir = tempfile.mkdtemp()
        output_template = os.path.join(temp_dir, "audio.%(ext)s")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": output_template,
            "quiet": True,
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }],
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)

        wav_path = os.path.join(temp_dir, "audio.wav")

        # Verify file exists and is non-empty
        if not os.path.exists(wav_path) or os.path.getsize(wav_path) < 1000:
            print(f"⚠️ Audio file was not created or is too small: {wav_path}")
            return None

        print(f"✅ Audio downloaded to: {wav_path}")
        return wav_path

    except Exception as e:
        print(f"❌ Error downloading audio: {e}")
        return None

def transcribe_audio(audio_path: str) -> str:
    """
    Transcribes a given audio file using Whisper and returns the transcript.
    """
    try:
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 1000:
            print(f"⚠️ Invalid or missing audio file: {audio_path}")
            return None

        result = whisper_model.transcribe(audio_path)
        print("✅ Transcription complete.")
        return result["text"]

    except Exception as e:
        print(f"❌ Error during transcription: {e}")
        return None

def build_rag_chain_from_transcript(transcript: str):
    """
    Build a QA chain using LangChain from the video transcript.
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(transcript)

        embeddings = OpenAIEmbeddings()
        vectordb = Chroma.from_texts(chunks, embedding=embeddings, collection_name="video_chunks")

        retriever = vectordb.as_retriever()

        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are a helpful medical assistant answering questions strictly based on the provided transcript below.
Do NOT answer from general knowledge or guess if the answer is not explicitly in the context.
Only respond based on this video transcript.

Transcript:
{context}

Question:
{question}

Helpful Answer (based only on the transcript):
""".strip()
        )

        chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(temperature=0, model_name="gpt-4o"),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,
            chain_type_kwargs={"prompt": prompt_template}
        )

        print("✅ QA chain built successfully.")
        return chain

    except Exception as e:
        print(f"❌ Error building RAG chain: {e}")
        return None
