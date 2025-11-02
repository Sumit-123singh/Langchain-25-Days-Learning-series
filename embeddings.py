# Install dependencies (only once)
# pip install sentence-transformers

from sentence_transformers import SentenceTransformer, util

# Load a pretrained embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sentences
sent1 = "I love eating pizza on weekends."
sent2 = "On weekends, I enjoy having pizza."
sent3 = "I am going to the gym."

# Convert sentences to embeddings
emb1 = model.encode(sent1, convert_to_tensor=True)
emb2 = model.encode(sent2, convert_to_tensor=True)
emb3 = model.encode(sent3, convert_to_tensor=True)

# Compute cosine similarity
sim_AB = util.pytorch_cos_sim(emb1, emb2)
sim_AC = util.pytorch_cos_sim(emb1, emb3)

print("Similarity (A,B):", sim_AB.item())
print("Similarity (A,C):", sim_AC.item())
