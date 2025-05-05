# utils/audio_utils.py
import whisper

def transcribe_audio(audio_path, model_size="small"):
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path)
    return result["text"]
