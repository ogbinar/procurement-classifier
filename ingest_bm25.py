import os
import re
import pickle
import pandas as pd
from rank_bm25 import BM25Okapi

CSV_PATH = "/projects/smc-cpg-classify/data/reference.csv"
OUT_DIR  = "/projects/smc-cpg-classify/data/index"
BM25_PKL = os.path.join(OUT_DIR, "bm25.pkl")
IDS_PKL  = os.path.join(OUT_DIR, "bm25_ids.pkl")
META_PARQUET = os.path.join(OUT_DIR, "bm25_meta.parquet")

TOKEN_RE = re.compile(r"[a-z0-9]+")

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
                df[col].astype(str)
                .str.replace("\xa0", " ", regex=False)
                .str.strip()
            )
    return df

def dedupe_on_code(df: pd.DataFrame) -> pd.DataFrame:
    df["_concat_len"] = df["concat"].astype(str).apply(len)
    df = df.sort_values(["Code", "_concat_len"], ascending=[True, False])
    df = df.drop_duplicates(subset=["Code"], keep="first").drop(columns=["_concat_len"])
    return df

def make_doc(row: pd.Series) -> str:
    # BM25 likes raw text; keep both
    desc = row.get("Description", "")
    path = row.get("concat", "")
    if desc and path:
        return f"{desc} {path}"
    return path or desc

def tokenize(text: str) -> list[str]:
    text = text.lower()
    return TOKEN_RE.findall(text)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = read_csv_robust(CSV_PATH)
    df = clean_df(df)
    df = dedupe_on_code(df)

    ids = df["Code"].astype(str).tolist()
    docs = [make_doc(r) for _, r in df.iterrows()]
    tokenized = [tokenize(d) for d in docs]

    print(f"Building BM25 on {len(ids)} docs...")
    bm25 = BM25Okapi(tokenized)

    with open(BM25_PKL, "wb") as f:
        pickle.dump(bm25, f)

    with open(IDS_PKL, "wb") as f:
        pickle.dump(ids, f)

    # handy for debugging and printing descriptions quickly
    df[["Code","Description","Segment Name","Family Name","Class Name","Commodity Name","concat"]].to_parquet(
        META_PARQUET, index=False
    )

    print("✅ Saved:")
    print(" -", BM25_PKL)
    print(" -", IDS_PKL)
    print(" -", META_PARQUET)

if __name__ == "__main__":
    main()

