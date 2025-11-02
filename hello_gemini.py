from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import promptTemplate
from dotenv import load_dotenv
import os
# Step 1: Initialize Gemini LLM
load_dotenv()
api_key=os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

# # Step 2: Run query
response = llm.invoke("Hello, what is LangChain?")

print(response.content)


