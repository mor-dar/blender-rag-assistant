#!/usr/bin/env python3
# scripts/evaluate_demo.py
"""
Evaluate retrieval quality + latency for Blender Bot's demo DB.

Metrics:
- Recall@1, Recall@3
- Mean Reciprocal Rank (MRR)
- Median latency (s) over queries

Outputs:
- outputs/eval_demo.json
- outputs/eval_demo.md (human-readable table)

Usage examples:
  # Evaluate an existing DB
  python scripts/evaluate_demo.py --persist_dir ./data/vector_db --collection blender_docs --k 5

  # Build a tiny demo DB (5 Blender Manual pages) and then evaluate
  python scripts/evaluate_demo.py --autobuild-demo --persist_dir ./data/vector_db --collection blender_docs

You may override the embedding model or device:
  --embedding_model multi-qa-MiniLM-L6-cos-v1
  --device cpu|cuda
"""
import argparse
import json
import os
import time
import statistics
from pathlib import Path
from typing import List, Dict

# Vector store + embeddings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Optional tiny demo builder
import requests
from bs4 import BeautifulSoup
import chromadb
from chromadb.utils import embedding_functions

# ---------- Defaults (align with your README) ----------
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "multi-qa-MiniLM-L6-cos-v1")
DEFAULT_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
DEFAULT_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/vector_db")
DEFAULT_COLLECTION = os.getenv("CHROMA_COLLECTION_NAME", "blender_docs")

# Five authoritative Blender Manual pages for the tiny demo DB.
DEMO_URLS = [
    "https://docs.blender.org/manual/en/4.5/render/shader_nodes/shader/principled.html",
    "https://docs.blender.org/manual/en/4.5/modeling/modifiers/generate/subdivision_surface.html",
    "https://docs.blender.org/manual/en/4.5/modeling/meshes/editing/uv.html",
    "https://docs.blender.org/manual/en/4.5/modeling/meshes/editing/mesh/merge.html",
    "https://docs.blender.org/manual/en/4.5/render/eevee/introduction.html",
]

# Hand-written queries mapped to gold URLs (2 per page; tune freely)
DEFAULT_EVAL_SET: List[Dict] = [
    # Buttons
    {
        "question": "What are Blender UI buttons and how are they generally interacted with?",
        "gold_urls": ["https://docs.blender.org/manual/en/4.5/interface/controls/buttons/buttons.html"],
    },
    {
        "question": "Where are buttons used across Blender and what kinds of actions do they provide?",
        "gold_urls": ["https://docs.blender.org/manual/en/4.5/interface/controls/buttons/buttons.html"],
    },
    # Decorators
    {
        "question": "What do the small decorators next to a property do (e.g., add keyframe, reset, drivers)?",
        "gold_urls": ["https://docs.blender.org/manual/en/4.5/interface/controls/buttons/decorators.html"],
    },
    {
        "question": "How can I tell if a property has keyframes or a driver from the UI and act on it?",
        "gold_urls": ["https://docs.blender.org/manual/en/4.5/interface/controls/buttons/decorators.html"],
    },
    # Eyedropper
    {
        "question": "How do I use the Eyedropper to sample a color from the screen?",
        "gold_urls": ["https://docs.blender.org/manual/en/4.5/interface/controls/buttons/eyedropper.html"],
    },
    {
        "question": "Can the Eyedropper pick data like objects or materials, and how is it used?",
        "gold_urls": ["https://docs.blender.org/manual/en/4.5/interface/controls/buttons/eyedropper.html"],
    },
    # Fields
    {
        "question": "How do numeric fields work and what shortcuts help with precise input or stepping?",
        "gold_urls": ["https://docs.blender.org/manual/en/4.5/interface/controls/buttons/fields.html"],
    },
    {
        "question": "Which field types are available (e.g., toggles, sliders, text) and how do you enter values?",
        "gold_urls": ["https://docs.blender.org/manual/en/4.5/interface/controls/buttons/fields.html"],
    },
    # Menus
    {
        "question": "How do menus work in Blender's UI, including context menus and popovers?",
        "gold_urls": ["https://docs.blender.org/manual/en/4.5/interface/controls/buttons/menus.html"],
    },
    {
        "question": "How do I open a menu for a button or property to access additional actions?",
        "gold_urls": ["https://docs.blender.org/manual/en/4.5/interface/controls/buttons/menus.html"],
    },
]

# ---------- Helpers ----------
def norm_url(u: str) -> str:
    return u.rstrip("/").split("#")[0].lower()

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    text = " ".join(text.split())
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i : i + chunk_size])
        i += max(1, chunk_size - overlap)
    return chunks

def fetch_page(url: str) -> str:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    # Strip nav/aside/footer where possible
    for tag in soup(["nav", "aside", "footer", "script", "style"]):
        tag.decompose()
    return soup.get_text(separator=" ")

def autobuild_demo_db(persist_dir: str, collection_name: str, embedding_model: str, device: str) -> None:
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model, device=device)
    try:
        collection = client.get_or_create_collection(name=collection_name, embedding_function=embed_fn)
    except TypeError:
        # Older Chroma versions may not accept embedding_function in get_or_create_collection
        collection = client.get_or_create_collection(name=collection_name)

    docs, metas, ids = [], [], []
    for idx, url in enumerate(DEMO_URLS):
        text = fetch_page(url)
        for j, chunk in enumerate(chunk_text(text)):
            docs.append(chunk)
            metas.append({"source": url})
            ids.append(f"demo-{idx}-{j}")
    if hasattr(collection, "add"):
        collection.add(documents=docs, metadatas=metas, ids=ids)
    else:
        # Fallback: recreate via vectorstore if needed (rare)
        pass

def load_retriever(persist_dir: str, collection_name: str, embedding_model: str, device: str, k: int = 5):
    embedder = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectordb = Chroma(
        collection_name=collection_name,
        persist_directory=persist_dir,
        embedding_function=embedder,
    )
    return vectordb.as_retriever(search_type="similarity", search_kwargs={"k": k})

# ---------- Evaluation ----------
def evaluate(retriever, eval_set: List[Dict], k: int = 5) -> Dict:
    per_query = []
    latencies = []
    hits_at_1 = 0
    hits_at_3 = 0
    mrr_vals = []

    for item in eval_set:
        q = item["question"]
        golds = [norm_url(u) for u in item.get("gold_urls", [])]
        t0 = time.perf_counter()
        docs = retriever.get_relevant_documents(q)
        dt = time.perf_counter() - t0

        latencies.append(dt)
        retrieved_urls = []
        for d in docs:
            src = d.metadata.get("source") or d.metadata.get("url") or ""
            retrieved_urls.append(norm_url(str(src)))

        # Compute metrics for this query
        rr = 0.0
        first_match_rank = None
        for rank, ru in enumerate(retrieved_urls, start=1):
            if any(ru == g or (g in ru) or (ru in g) for g in golds):
                if rank == 1:
                    hits_at_1 += 1
                if rank <= 3:
                    hits_at_3 += 1
                rr = 1.0 / rank
                first_match_rank = rank
                break
        if rr > 0.0:
            mrr_vals.append(rr)

        per_query.append({
            "question": q,
            "gold_urls": item.get("gold_urls", []),
            "retrieved_urls": retrieved_urls,
            "first_hit_rank": first_match_rank,
            "latency_sec": round(dt, 3),
        })

    n = len(eval_set)
    metrics = {
        "queries": n,
        "recall_at_1": round(hits_at_1 / n, 3),
        "recall_at_3": round(hits_at_3 / n, 3),
        "mrr": round((sum(mrr_vals) / n) if n > 0 else 0.0, 3),
        "median_latency_sec": round(statistics.median(latencies), 3) if latencies else None,
        "k": k,
    }
    return {"metrics": metrics, "per_query": per_query}

def write_reports(results: Dict, out_dir: str = "outputs"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    json_path = Path(out_dir) / "eval_demo.json"
    md_path = Path(out_dir) / "eval_demo.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Markdown table
    m = results["metrics"]
    lines = []
    lines.append("# Demo Retrieval Evaluation\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Queries | {m['queries']} |")
    lines.append(f"| k | {m['k']} |")
    lines.append(f"| Recall@1 | {m['recall_at_1']} |")
    lines.append(f"| Recall@3 | {m['recall_at_3']} |")
    lines.append(f"| MRR | {m['mrr']} |")
    lines.append(f"| Median latency (s) | {m['median_latency_sec']} |")
    lines.append("\n## Per‑query details\n")
    lines.append("| Question | First Hit Rank | Latency (s) | Top URLs |")
    lines.append("|---|---:|---:|---|")
    for row in results["per_query"]:
        urls = "<br>".join(row["retrieved_urls"][:5])
        lines.append(f"| {row['question']} | {row['first_hit_rank'] or ''} | {row['latency_sec']} | {urls} |")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote {json_path} and {md_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--persist_dir", default=DEFAULT_PERSIST_DIR)
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)
    ap.add_argument("--embedding_model", default=DEFAULT_EMBEDDING_MODEL)
    ap.add_argument("--device", default=DEFAULT_DEVICE, choices=["cpu", "cuda"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--autobuild-demo", action="store_true", help="Build a tiny demo DB from 5 Blender Manual pages")
    args = ap.parse_args()

    if args.autobuild_demo:
        print(f"Building tiny demo DB into {args.persist_dir!r} (collection={args.collection!r}) ...")
        autobuild_demo_db(args.persist_dir, args.collection, args.embedding_model, args.device)

    print("Loading retriever ...")
    retriever = load_retriever(args.persist_dir, args.collection, args.embedding_model, args.device, k=args.k)

    print("Running evaluation ...")
    results = evaluate(retriever, DEFAULT_EVAL_SET, k=args.k)
    write_reports(results)

if __name__ == "__main__":
    print("Running evaluate_demo.py ...")
    main()
