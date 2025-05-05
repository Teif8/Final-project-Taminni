# utils/image_utils.py

import base64
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load .env variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Extract diagnosis name
def extract_diagnosis_name(gpt_text: str) -> str:
    lines = gpt_text.splitlines()
    for line in lines:
        if "likely" in line.lower() or ":" in line or "**" in line:
            return line.replace("*", "").strip().split(":")[0]
    return "Unclear Condition"

# Main classifier function
def classify_image(image_path: str) -> str:
    try:
        # Convert image to base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        # Request with expanded prompt
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": (
                            "You're an AI medical assistant. Please examine this image and respond with:\n"
                            "A very short observation of what you see.\n"
                            " What the condition might be.\n"
                            " What the user can do at home to ease the condition until they see a doctor.\n"
                            " What type of doctor the user should visit.\n"
                            "Be concise and medically helpful."
                        )},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ],
                }
            ],
            max_tokens=600
        )

        raw = response.choices[0].message.content.strip()
        lines = raw.splitlines()

        formatted = f"""
        <strong>⚠️ Disclaimer:</strong><br>
        This is not a medical diagnosis. It is a preliminary analysis based on the image you provided. Always consult a licensed healthcare provider for an accurate diagnosis and treatment plan.
        <br><br>

        <strong>🔍 Initial Observation:</strong><br>
        {lines[0]}<br><br>

        <strong>🧺 What You Can Do:</strong><br>
        {lines[2] if len(lines) > 2 else "Try to keep the area clean and avoid irritation until you can see a doctor."}<br><br>

        <strong>🧑‍⚕️ Suggested Specialist:</strong><br>
        {lines[3] if len(lines) > 3 else "Please consult a dermatologist for proper evaluation and care."}
        """
        return formatted

    except Exception as e:
        return f"❌ Error classifying image: {str(e)}"
