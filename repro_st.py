from sentence_transformers import SentenceTransformer
print("loading...")
m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("loaded")
print(m.encode(["hello world"], normalize_embeddings=True))
print("done")
