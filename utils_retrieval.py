import os
import re
import pickle
from typing import Dict, List, Tuple

import chromadb
import pandas as pd
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# --- Paths ---
PERSIST_DIR = "data/chroma_unspsc"
COLLECTION = "unspsc_ref"
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_MODEL = "BAAI/bge-m3"
BATCH_SIZE = 256

INDEX_DIR = "data/index"
BM25_PKL = os.path.join(INDEX_DIR, "bm25.pkl")
IDS_PKL = os.path.join(INDEX_DIR, "bm25_ids.pkl")
META_PARQUET = os.path.join(INDEX_DIR, "bm25_meta.parquet")

# --- Retrieval knobs ---
TOPK_BM25 = 50
TOPK_VEC = 50
RRF_K = 60

TOKEN_RE = re.compile(r"[a-z0-9]+")

def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower().replace("\xa0", " "))

def normalize_query(query: str) -> str:
    normalized = query.replace("\xa0", " ").strip().lower()
    normalized = re.sub(r"\b\d+(\.\d+)?\s*(mm|cm|m|in|inch|inches|pcs|pc|pack|box|g|kg|ml|l)\b", " ", normalized)
    normalized = re.sub(r"\b\d+(\.\d+)?\b", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized

def rrf_fuse(rank_lists: Dict[str, List[str]], k: int = 60) -> List[Tuple[str, float, Dict]]:
    scores = {}
    debug = {}

    for source, ids in rank_lists.items():
        for rank, _id in enumerate(ids, start=1):
            scores[_id] = scores.get(_id, 0.0) + 1.0 / (k + rank)
            debug.setdefault(_id, {})[f"{source}_rank"] = rank

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(_id, score, debug.get(_id, {})) for _id, score in fused]

def load_retrieval_assets():
    with open(BM25_PKL, "rb") as f:
        bm25 = pickle.load(f)
    with open(IDS_PKL, "rb") as f:
        bm25_ids = pickle.load(f)

    meta = pd.read_parquet(META_PARQUET).set_index("Code")

    client = chromadb.PersistentClient(
        path=PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection(COLLECTION)
    model = SentenceTransformer(EMBED_MODEL, device="cpu")

    return bm25, bm25_ids, meta, collection, model

def retrieve_candidates(
    query: str,
    bm25,
    bm25_ids,
    meta: pd.DataFrame,
    collection,
    model,
    topk_bm25: int = TOPK_BM25,
    topk_vec: int = TOPK_VEC,
    topk_fused: int = 5,
    rrf_k: int = RRF_K,
):
    normalized_query = normalize_query(query)

    # BM25
    query_tokens = tokenize(normalized_query)
    bm25_scores = bm25.get_scores(query_tokens)
    top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:topk_bm25]
    bm25_top_ids = [bm25_ids[i] for i in top_indices]

    # Vector
    query_embedding = model.encode([normalized_query], normalize_embeddings=True).tolist()
    vector_results = collection.query(query_embeddings=query_embedding, n_results=topk_vec)
    vector_top_ids = vector_results["ids"][0]

    # Fuse
    fused_results = rrf_fuse({"bm25": bm25_top_ids, "vec": vector_top_ids}, k=rrf_k)[:topk_fused]

    rows = []
    for rank, (code, score, debug_info) in enumerate(fused_results, start=1):
        try:
            row = meta.loc[code]
            description = row.get("Description", "")
            segment = row.get("Segment Name", "")
            family = row.get("Family Name", "")
            class_name = row.get("Class Name", "")
            commodity = row.get("Commodity Name", "")
        except KeyError:
            description = segment = family = class_name = commodity = ""

        rows.append({
            "rank": rank,
            "code": code,
            "description": description,
            "fused_score": score,
            "bm25_rank": debug_info.get("bm25_rank"),
            "vec_rank": debug_info.get("vec_rank"),
            "segment": segment,
            "family": family,
            "class_name": class_name,
            "commodity": commodity,
        })

    return normalized_query, pd.DataFrame(rows)