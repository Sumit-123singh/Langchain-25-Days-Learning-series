import os
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from sentence_transformers import SentenceTransformer

# 1️⃣ Create Pinecone instance
pc = Pinecone(api_key="pcsk_5ENe6X_FSWhpWeRuj8XQADXzJP1hjeM9qjSaJdp2E7aapamW3zLQhoYQYuWfDkEn7nc1V2", environment="us-east1-gcp")

# 2️⃣ Create index if it doesn't exist
index_name = "notes-index"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print(f"✅ Index '{index_name}' created")

# 3️⃣ Connect to the index
index = pc.Index(index_name)

# 4️⃣ Example: add embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
texts = ["AI is the future", "I love pizza", "Neural networks are cool"]
embeddings = model.encode(texts).tolist()

vectors = [(f"id_{i}", emb) for i, emb in enumerate(embeddings)]
index.upsert(vectors=vectors)
print("✅ Embeddings uploaded")

# 5️⃣ Query example
query_text = "Tell me about artificial intelligence"
query_vector = model.encode([query_text]).tolist()[0]
result = index.query(vector=query_vector, top_k=2)
print("\n🔍 Query results:")
for match in result["matches"]:
    print(f"Score: {match['score']:.4f}, Text: {texts[int(match['id'].split('_')[1])]}")
stats = index.describe_index_stats()
print(stats)
