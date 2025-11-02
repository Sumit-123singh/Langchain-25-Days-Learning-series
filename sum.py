import google.generativeai as genai
import os

# Configure your key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

try:
    # Simple test: create a model and generate text
    model = genai.GenerativeModel("gemini-1.5-t")
    response = model.generate_content("Say hello in a fun way")
    print("✅ API Key works! Response:", response.text)

except Exception as e:
    print("❌ API Key error:", e)
