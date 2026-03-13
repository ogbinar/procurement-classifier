import os
import re
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

PERSIST_DIR = "/projects/smc-cpg-classify/data/chroma_unspsc"
COLLECTION = "unspsc_ref"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBED_MODEL_NAME = "BAAI/bge-m3"



def normalize_query(q: str) -> str:
    """
    Light normalization to reduce noise from specs:
    - remove common unit patterns (mm, inch, etc.)
    - remove pure numbers
    - keep core nouns
    This is intentionally simple for debugging.
    """
    q = q.replace("\xa0", " ").strip().lower()

    # remove sizes/units like 1.5mm, 10pcs, 2x, etc.
    q = re.sub(r"\b\d+(\.\d+)?\s*(mm|cm|m|in|inch|inches|pcs|pc|pack|box|g|kg|ml|l)\b", " ", q)
    # remove standalone numbers
    q = re.sub(r"\b\d+(\.\d+)?\b", " ", q)

    # collapse whitespace
    q = re.sub(r"\s+", " ", q).strip()
    return q


def main():
    print("=== UNSPSC RETRIEVE (Chroma) ===")
    print("Persist dir:", PERSIST_DIR)
    print("Collection:", COLLECTION)

    client = chromadb.PersistentClient(
        path=PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    col = client.get_collection(COLLECTION)

    print("\nLoading embedding model:", EMBED_MODEL_NAME)
    model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
    print("✅ Model loaded\n")

    while True:
        q = input("Query (blank to exit): ").strip()
        if not q:
            break

        q_norm = normalize_query(q)
        print("\n--- QUERY ---")
        print("Raw :", q)
        print("Norm:", q_norm)

        q_emb = model.encode([q_norm], normalize_embeddings=True).tolist()
        res = col.query(query_embeddings=q_emb, n_results=5)

        ids = res["ids"][0]
        metas = res["metadatas"][0]
        docs = res["documents"][0]

        print("\n=== TOP 20 CANDIDATES ===")
        for rank, (code, meta, doc) in enumerate(zip(ids, metas, docs), start=1):
            desc = meta.get("description", "")
            seg = meta.get("segment", "")
            fam = meta.get("family", "")
            cls = meta.get("class", "")
            com = meta.get("commodity", "")

            print(f"\n#{rank}  {code}  |  {desc}")
            print(f"    Segment : {seg}")
            print(f"    Family  : {fam}")
            print(f"    Class   : {cls}")
            print(f"    Commodity: {com}")
            # show a short snippet of what was indexed
            snippet = (doc or "")[:220].replace("\n", " ")
            print(f"    Doc: {snippet}...")

        print("\n")


if __name__ == "__main__":
    main()
