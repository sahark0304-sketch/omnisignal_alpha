import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
# וודא שהמפתח ב-.env מעודכן
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

print("\n--- Available Models for your API Key ---")
try:
    for model in client.models.list():
        # הדפסה של שם המודל בלבד כדי למנוע שגיאות
        print(f"Model Name: {model.name}")
except Exception as e:
    print(f"Error fetching models: {e}")