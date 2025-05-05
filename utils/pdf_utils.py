# utils/pdf_utils.py
import fitz  # PyMuPDF
from openai import OpenAI

client = OpenAI()

def summarize_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Summarize this medical document in simple language."},
            {"role": "user", "content": full_text}
        ]
    )

    return response.choices[0].message.content.strip()
