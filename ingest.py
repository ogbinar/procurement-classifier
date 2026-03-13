import os
import time
import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

CSV_PATH = "/projects/smc-cpg-classify/data/reference.csv"

PERSIST_DIR = "/projects/smc-cpg-classify/data/chroma_unspsc"
COLLECTION = "unspsc_ref"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBED_MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 256


def read_csv_robust(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, dtype=str, encoding="utf-8")
    except UnicodeDecodeError:
        print("⚠️ UTF-8 failed. Trying latin1...")
        df = pd.read_csv(path, dtype=str, encoding="latin1")
    return df.fillna("")


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("\xa0", " ", regex=False)
                .str.strip()
            )
    return df


def dedupe_on_code(df: pd.DataFrame) -> pd.DataFrame:
    df["_concat_len"] = df["concat"].astype(str).apply(len)
    df = df.sort_values(["Code", "_concat_len"], ascending=[True, False])
    df = df.drop_duplicates(subset=["Code"], keep="first").drop(columns=["_concat_len"])
    return df


def build_doc(row: pd.Series) -> str:
    desc = row.get("Description", "")
    path = row.get("concat", "")
    if desc and path:
        return f"{desc}. Path: {path}"
    return path or desc


def main():
    t0 = time.time()
    print("=== UNSPSC INGEST (Chroma) ===")
    print("CSV:", CSV_PATH)
    print("Persist dir:", PERSIST_DIR)
    print("Collection:", COLLECTION)

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Missing CSV: {CSV_PATH}")

    print("\n[1/6] Loading CSV...")
    df = read_csv_robust(CSV_PATH)
    df = clean_df(df)

    required_cols = [
        "Code", "Description", "Segment Name", "Family Name",
        "Class Name", "Commodity Name", "concat"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    before = len(df)
    df = dedupe_on_code(df)
    after = len(df)
    if after != before:
        print(f"ℹ️ Deduped on Code: {before} → {after} (removed {before - after})")
    print(f"Rows to ingest: {after}")

    print("\n[2/6] Initializing Chroma...")
    os.makedirs(PERSIST_DIR, exist_ok=True)
    client = chromadb.PersistentClient(
        path=PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )

    # POC: recreate collection
    try:
        client.delete_collection(COLLECTION)
        print(f"🧹 Deleted existing collection '{COLLECTION}'")
    except Exception:
        print("ℹ️ No existing collection to delete (or delete not supported).")

    col = client.get_or_create_collection(name=COLLECTION)
    print("✅ Chroma collection ready.")

    print("\n[3/6] Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    print("✅ Model loaded:", EMBED_MODEL_NAME)

    print("\n[4/6] Building docs + metadata...")
    ids, docs, metas = [], [], []
    for _, row in df.iterrows():
        code = str(row["Code"]).strip()
        if not code:
            continue

        ids.append(code)
        docs.append(build_doc(row))
        metas.append({
            "code": code,
            "description": row.get("Description", ""),
            "segment": row.get("Segment Name", ""),
            "family": row.get("Family Name", ""),
            "class": row.get("Class Name", ""),
            "commodity": row.get("Commodity Name", ""),
        })

    print(f"✅ Prepared {len(ids)} docs.")

    print("\n[5/6] Embedding + upserting to Chroma...")
    embeddings = model.encode(
        docs,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).tolist()

    # Upsert in chunks to avoid huge payloads
    chunk = 2000
    for i in range(0, len(ids), chunk):
        col.upsert(
            ids=ids[i:i+chunk],
            documents=docs[i:i+chunk],
            metadatas=metas[i:i+chunk],
            embeddings=embeddings[i:i+chunk],
        )
        print(f"  - Upserted {min(i+chunk, len(ids))}/{len(ids)}")

    print("✅ Upsert complete.")

    print("\n[6/6] Smoke test query...")
    q = "permanent marker artline waterproof quick drying"
    q_emb = model.encode([q], normalize_embeddings=True).tolist()
    res = col.query(query_embeddings=q_emb, n_results=5)

    print("Top 5 IDs:", res["ids"][0])
    print("Top 1 meta:", res["metadatas"][0][0])

    dt = time.time() - t0
    print(f"\n🎉 DONE. Ingested {len(ids)} rows in {dt:.1f}s")
    print(f"📍 Persisted at: {PERSIST_DIR}")


if __name__ == "__main__":
    main()
