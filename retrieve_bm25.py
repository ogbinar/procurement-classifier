import os
import re
import pickle
import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# --- Paths ---
PERSIST_DIR = "/projects/smc-cpg-classify/data/chroma_unspsc"
COLLECTION  = "unspsc_ref"
EMBED_MODEL = "all-MiniLM-L6-v2"

INDEX_DIR   = "/projects/smc-cpg-classify/data/index"
BM25_PKL    = os.path.join(INDEX_DIR, "bm25.pkl")
IDS_PKL     = os.path.join(INDEX_DIR, "bm25_ids.pkl")
META_PARQUET= os.path.join(INDEX_DIR, "bm25_meta.parquet")

# --- Retrieval knobs ---
TOPK_BM25 = 50
TOPK_VEC  = 50
TOPK_FUSED = 5
RRF_K = 60

TOKEN_RE = re.compile(r"[a-z0-9]+")

def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric tokens"""
    return TOKEN_RE.findall(text.lower().replace("\xa0", " "))

def normalize_query(query: str) -> str:
    """Normalize query by removing measurement noise and extra whitespace"""
    normalized = query.replace("\xa0", " ").strip().lower()
    # remove obvious measurement noise
    normalized = re.sub(r"\b\d+(\.\d+)?\s*(mm|cm|m|in|inch|inches|pcs|pc|pack|box|g|kg|ml|l)\b", " ", normalized)
    normalized = re.sub(r"\b\d+(\.\d+)?\b", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized

def rrf_fuse(rank_lists: dict[str, list[str]], k: int = 60) -> list[tuple[str, float, dict]]:
    """
    Perform Reciprocal Rank Fusion on ranked lists from different sources
    
    Args:
        rank_lists: Dictionary mapping source names to ranked IDs
        k: RRF parameter (default 60)
        
    Returns:
        List of tuples (id, fused_score, debug_info)
    """
    scores = {}
    debug = {}

    for source, ids in rank_lists.items():
        for rank, _id in enumerate(ids, start=1):
            scores[_id] = scores.get(_id, 0.0) + 1.0 / (k + rank)
            debug.setdefault(_id, {})[f"{source}_rank"] = rank

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(_id, score, debug.get(_id, {})) for _id, score in fused]

def main():
    print("=== HYBRID RETRIEVE (BM25 + Chroma + RRF) ===")

    # Load BM25 index
    try:
        with open(BM25_PKL, "rb") as f:
            bm25 = pickle.load(f)
        with open(IDS_PKL, "rb") as f:
            bm25_ids = pickle.load(f)
        meta = pd.read_parquet(META_PARQUET).set_index("Code")
    except Exception as e:
        print(f"Error loading index files: {e}")
        return

    # Load Chroma + embedder
    try:
        client = chromadb.PersistentClient(
            path=PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        collection = client.get_collection(COLLECTION)
        model = SentenceTransformer(EMBED_MODEL, device="cpu")
    except Exception as e:
        print(f"Error loading Chroma or model: {e}")
        return

    while True:
        query = input("\nQuery (blank to exit): ").strip()
        if not query:
            break

        normalized_query = normalize_query(query)
        print(f"\n--- QUERY ---")
        print(f"Raw : {query}")
        print(f"Norm: {normalized_query}")

        # 1) BM25 retrieval
        try:
            query_tokens = tokenize(normalized_query)
            bm25_scores = bm25.get_scores(query_tokens)
            # Get top K indices
            top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:TOPK_BM25]
            bm25_top_ids = [bm25_ids[i] for i in top_indices]
        except Exception as e:
            print(f"Error in BM25 retrieval: {e}")
            bm25_top_ids = []

        # 2) Vector retrieval
        try:
            query_embedding = model.encode([normalized_query], normalize_embeddings=True).tolist()
            vector_results = collection.query(query_embeddings=query_embedding, n_results=TOPK_VEC)
            vector_top_ids = vector_results["ids"][0]
        except Exception as e:
            print(f"Error in vector retrieval: {e}")
            vector_top_ids = []

        # 3) Fuse results with RRF
        try:
            fused_results = rrf_fuse({"bm25": bm25_top_ids, "vec": vector_top_ids}, k=RRF_K)[:TOPK_FUSED]
        except Exception as e:
            print(f"Error in RRF fusion: {e}")
            fused_results = []

        print(f"\n=== TOP {TOPK_FUSED} FUSED (RRF) ===")
        for rank, (code, score, debug_info) in enumerate(fused_results, start=1):
            # Get metadata for this code
            try:
                row = meta.loc[code]
                description = row["Description"]
                segment = row["Segment Name"]
                family = row["Family Name"]
                class_name = row["Class Name"]
                commodity = row["Commodity Name"]
            except KeyError:
                # Handle case where code is not found in metadata
                description = segment = family = class_name = commodity = "N/A"
            
            print(f"\n#{rank}  {code} | {description}")
            print(f"    fused_score: {score:.6f} | ranks: {debug_info}")
            print(f"    Segment : {segment}")
            print(f"    Family  : {family}")
            print(f"    Class   : {class_name}")
            print(f"    Commodity: {commodity}")

if __name__ == "__main__":
    main()
