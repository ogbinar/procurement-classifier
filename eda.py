import pandas as pd

CSV_PATH = "/projects/smc-cpg-classify/data/reference.csv"

try:
    df = pd.read_csv(CSV_PATH, dtype=str, encoding="utf-8")
except UnicodeDecodeError:
    print("⚠️ UTF-8 failed. Trying latin1...")
    df = pd.read_csv(CSV_PATH, dtype=str, encoding="latin1")

df = df.fillna("")


print("=== BASIC INFO ===")
print(f"Shape: {df.shape}")
print("\nColumns:")
print(df.columns.tolist())

print("\n=== HEAD ===")
print(df.head())

print("\n=== DATA TYPES ===")
print(df.dtypes)

print("\n=== NULL COUNTS ===")
print(df.isna().sum())

print("\n=== DUPLICATE ROWS ===")
print("Total duplicates:", df.duplicated().sum())

# ---- Check code uniqueness (very important for ingestion) ----
if "code" in df.columns:
    print("\n=== CODE UNIQUENESS ===")
    print("Unique codes:", df["code"].nunique())
    print("Total rows:", len(df))
    print("Duplicate codes:", df["code"].duplicated().sum())

# ---- Cardinality check per column ----
print("\n=== CARDINALITY (Unique values per column) ===")
for col in df.columns:
    print(f"{col}: {df[col].nunique()}")

# ---- Text length analysis (important for embedding quality) ----
print("\n=== TEXT LENGTH STATS ===")
for col in df.columns:
    if df[col].dtype == "object":
        lengths = df[col].fillna("").astype(str).apply(len)
        print(f"\n{col}")
        print("  Mean length:", round(lengths.mean(), 2))
        print("  Max length:", lengths.max())
        print("  Min length:", lengths.min())

# ---- Sample random rows ----
print("\n=== RANDOM SAMPLE ===")
print(df.sample(min(5, len(df))))
