# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec

# ================================
# Load environment variables
# ================================
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east1-gcp")
INDEX_NAME = "notes-rag"

if not GOOGLE_API_KEY or not PINECONE_API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY and PINECONE_API_KEY in your .env file")

# ================================
# Configure Google Gemini
# ================================
genai.configure(api_key=GOOGLE_API_KEY)

# ================================
# Initialize Pinecone (new SDK)
# ================================
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it doesn't exist
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Connect to index
index = pc.Index(INDEX_NAME)

# ================================
# Initialize FastAPI
# ================================
app = FastAPI(title="Notes RAG Chatbot (Gemini + Pinecone)")

# ================================
# Pydantic models
# ================================
class NotesInput(BaseModel):
    notes: str

class QueryInput(BaseModel):
    query: str

# ================================
# Endpoints
# ================================
@app.post("/upload_notes/")
async def upload_notes(data: NotesInput):
    """Embed notes and store in Pinecone"""
    try:
        embed = genai.embed_content(model="models/embedding-001", content=data.notes)
        vector = embed["embedding"]

        index.upsert(
            vectors=[{
                "id": f"note-{hash(data.notes)}",
                "values": vector,
                "metadata": {"text": data.notes}
            }]
        )

        return {"message": "✅ Notes uploaded successfully!"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading notes: {str(e)}")


@app.post("/ask/")
async def ask_question(data: QueryInput):
    """Query Pinecone for relevant notes and answer using Gemini"""
    try:
        query_embed = genai.embed_content(model="models/embedding-001", content=data.query)["embedding"]

        results = index.query(vector=query_embed, top_k=3, include_metadata=True)
        matches = results.get("matches", [])

        if not matches:
            return {"answer": "⚠️ No relevant notes found."}

        context = "\n".join([m["metadata"]["text"] for m in matches])

        model = genai.GenerativeModel("gemini-2.5-pro")
        prompt = f"Context:\n{context}\n\nQuestion:\n{data.query}\n\nAnswer:"
        response = model.generate_content(prompt)

        return {"answer": response.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")


@app.get("/")
async def home():
    return {"message": "✅ Notes RAG Chatbot with Gemini + Pinecone is running successfully!"}
