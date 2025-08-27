# build_index.py
import os
import json
import pandas as pd
import numpy as np
import faiss
import openai
from tqdm import tqdm

"""
Usage:
    export OPENAI_API_KEY="sk-..."
    python build_index.py
"""

OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in env before running.")
openai.api_key = OPENAI_KEY

INPUT_CSV = "/mnt/data/Shrimad-bhagvad-gita-hindi.csv"
INDEX_PATH = "gita_index.faiss"
META_PATH = "gita_meta.json"
EMB_MODEL = "text-embedding-3-large"
BATCH = 32

# Load CSV
df = pd.read_csv(INPUT_CSV)

# Expected columns: chapter, verse, text_hindi (adjust names if different)
# If your CSV headers differ, print df.head() to inspect and adjust
texts = []
metas = []

for i, row in df.iterrows():
    chap = row.get("chapter") or row.get("adhyay") or row.get("Chapter")
    verse = row.get("verse") or row.get("shloka") or row.get("Verse")
    text = row.get("text_hindi") or row.get("shloka") or row.get("Text")
    if pd.isna(text):
        continue
    chunk = f"Chapter {chap} Verse {verse}: {text}"
    texts.append(chunk)
    metas.append({
        "chapter": chap,
        "verse": verse,
        "text_hindi": text
    })

print(f"Loaded {len(texts)} shlokas.")

def embed_batch(batch):
    resp = openai.Embedding.create(model=EMB_MODEL, input=batch)
    return [r["embedding"] for r in resp["data"]]

all_embs = []
for i in tqdm(range(0, len(texts), BATCH)):
    batch = texts[i:i+BATCH]
    embs = embed_batch(batch)
    all_embs.extend(embs)

emb_matrix = np.array(all_embs, dtype="float32")
dim = emb_matrix.shape[1]

index = faiss.IndexFlatL2(dim)
index.add(emb_matrix)
faiss.write_index(index, INDEX_PATH)
with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(metas, f, ensure_ascii=False, indent=2)

print(f"Saved index to {INDEX_PATH}, metadata to {META_PATH}")
