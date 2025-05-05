# utils/advice_utils.py
from openai import OpenAI

client = OpenAI()

def generate_advice(text: str) -> str:
    prompt = f"""You're a helpful medical assistant. Based on the following information, give simple, friendly advice in 2–3 sentences.
    
    Text: {text}

    Advice:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a compassionate health assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("OpenAI Error:", e)
        return "Sorry, I couldn't generate advice at the moment."
