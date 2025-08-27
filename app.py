# app.py
import os
import json
import numpy as np
import faiss
import openai
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in env before running.")
openai.api_key = OPENAI_KEY

INDEX_PATH = "gita_index.faiss"
META_PATH = "gita_meta.json"
EMB_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-5-mini"  # change if needed

# Load index and meta
index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

app = FastAPI(title="Krishna Chatbot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

class Query(BaseModel):
    question: str
    top_k: int = 4

def embed(text: str):
    resp = openai.Embedding.create(model=EMB_MODEL, input=text)
    return np.array(resp["data"][0]["embedding"], dtype="float32")

def build_prompt(question, verses):
    block = "\n\n".join(
        [f"Chapter {v['chapter']} Verse {v['verse']}:\n{v['text_hindi']}" for v in verses]
    )
    system = (
        "You are Krishna, the divine teacher from the Bhagavad Gita. "
        "Speak as Krishna, addressing the user as 'my dear Arjuna'. "
        "Quote shlokas (chapter:verse) and explain them in simple English with practical life advice."
    )
    user = f"Question: {question}\n\nRelevant shlokas:\n{block}\n\nAnswer as Krishna."
    return system, user

@app.post("/ask")
def ask(q: Query):
    q_emb = embed(q.question).reshape(1, -1)
    D, I = index.search(q_emb, q.top_k)
    verses = [meta[int(idx)] for idx in I[0]]
    system, user = build_prompt(q.question, verses)

    resp = openai.ChatCompletion.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        max_tokens=512,
        temperature=0.7
    )
    answer = resp["choices"][0]["message"]["content"].strip()
    return {"answer": answer, "verses": verses}
