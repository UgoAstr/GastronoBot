"""
rag_backend.py — Module backend RAG (Recherche Hybride + Génération via Groq)
Utilisé par app.py (Streamlit).
"""

# ─────────────────────────── Imports ───────────────────────────
from api_groq import API_KEYS

import pandas as pd
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import util, SentenceTransformer
from groq import Groq

# ─────────────────────────── Init Device ───────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────── Chargement des données ───────────────────────────
df_intern = pd.read_csv("intern_recipe_df.csv")
df_intern["embedding"] = df_intern["embedding"].apply(
    lambda x: np.fromstring(x.strip("[]"), sep=" ")
)

embeddings = torch.tensor(
    np.stack(df_intern["embedding"].tolist(), axis=0), dtype=torch.float32
).to(device)

# ─────────────────────────── BM25 ───────────────────────────
corpus = df_intern["text"].tolist()
tokenized_corpus = [doc.lower().split() for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

# ─────────────────────────── Embedding model ───────────────────────────
embeddings_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# ─────────────────────────── Groq client ───────────────────────────
client = Groq(api_key=API_KEYS)


# ═══════════════════════════ RETRIEVAL ═══════════════════════════
def hybrid_search(req: str, top_k: int = 3) -> list[dict]:
    """Recherche hybride : cosine similarity + BM25."""
    # Score sémantique
    req_embed = embeddings_model.encode(req, convert_to_tensor=True).to(device)
    vector_scores = util.cos_sim(req_embed, embeddings).flatten().cpu().numpy()

    # Score BM25
    tokenized_query = req.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)

    # Normalisation min-max
    def normalize(arr):
        diff = np.max(arr) - np.min(arr)
        return arr if diff == 0 else (arr - np.min(arr)) / diff

    v_norm = normalize(vector_scores)
    b_norm = normalize(bm25_scores)
    final_scores = (v_norm + b_norm) / 2

    # Top-K
    topk_indices = np.argsort(final_scores)[::-1][:top_k]

    return [
        {"page": df_intern.iloc[idx]["page_number"], "text": df_intern.iloc[idx]["text"]}
        for idx in topk_indices
    ]


# ═══════════════════════════ AUGMENTATION ═══════════════════════════
def augmentation_groq(req: str, top: int) -> list[dict]:
    """Construit le prompt système + utilisateur avec les extraits RAG."""
    top_results = hybrid_search(req, top_k=top)

    system_msg = (
        "You are an assistant specializing in culinary recipes from around the world, "
        "created by university researchers.\n"
        "Based on these excerpts, please provide a detailed answer to the question "
        "and use only the information provided and give the page numbers.\n\n"
        "Do not try to give more information than what is provided.\n\n"
        "If you cannot find an answer to the question, do not try to invent one.\n\n"
    )

    user_content = f"This is the question: {req}\n\n"
    user_content += "Here are some excerpts to help you answer the question:\n\n"
    for i, e in enumerate(top_results, 1):
        user_content += f"Extract {i}\nPages: {e['page']}\nTexte: {e['text']}\n\n"

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content},
    ]


# ═══════════════════════════ GENERATION ═══════════════════════════
def generation_gorq(req: str, top: int = 3) -> str:
    """Génère une réponse en anglais via Groq."""
    messages = augmentation_groq(req, top)

    completion = client.chat.completions.create(
        model="qwen/qwen3-32b",
        messages=messages,
        temperature=0.4,
        max_completion_tokens=1000,
        top_p=1,
        reasoning_effort="none",
        stream=True,
        stop=None,
    )

    return "".join(chunk.choices[0].delta.content or "" for chunk in completion)


# ═══════════════════════════ TRADUCTION ═══════════════════════════
def _translate(text: str, direction: str) -> str:
    """Traduit FR→EN ou EN→FR via Llama 3.1."""
    if direction == "fr2en":
        prompt = "the only task is to translate the following text from French to English. I just want the translation:\n\n" + text
    else:
        prompt = "the only task is to translate the following text from English to French. I just want the translation:\n\n" + text

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_completion_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )
    return "".join(chunk.choices[0].delta.content or "" for chunk in completion)


def fr_to_eng(req: str) -> str:
    return _translate(req, "fr2en")


def eng_to_fr(req: str) -> str:
    return _translate(req, "en2fr")


def generation_gorq_fr(req: str, top: int = 3) -> str:
    """Pipeline complet : FR → EN (requête) → Génération → FR (réponse)."""
    req_eng = fr_to_eng(req)
    gen_eng = generation_gorq(req_eng, top)
    gen_fr = eng_to_fr(gen_eng)
    return gen_fr
