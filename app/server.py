from fastapi import FastAPI, UploadFile, File, Body, HTTPException
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from datetime import datetime, timezone

import os
import io
import time
import json
import re
import fitz  # pymupdf
import httpx

from bs4 import BeautifulSoup
from urllib.parse import urlparse
from typing import List, Dict, Optional

from pathlib import Path
from rag_pipeline import process_url_for_rag, FAISSVectorStore, get_backend

# import pymupdf4llm

# --- Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
DATA_DIR = os.path.join(BASE_DIR, "data")
STATIC_DIR = os.path.join(BASE_DIR, "static")
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "vector_stores")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

LIGATURE_MAP = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
}

# Global registry for loaded vector stores
vector_stores: Dict[str, FAISSVectorStore] = {}


def hlog(logtxt: str):
    print(logtxt)


app = FastAPI(title="PDF → JSON Extractor + RAG Search")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
try:
    app.mount("/static", StaticFiles(directory=STATIC_DIR, html=False), name="static")
    app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
    app.mount(
        "/vector_stores", StaticFiles(directory=VECTOR_STORE_DIR), name="vector_stores"
    )
except Exception:
    pass


# -----------------------------
# Helpers
# -----------------------------
def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines()]
    text = "\n".join(ln for ln in lines if ln)
    return text


def _build_txt_filename_from_url(url: str) -> str:
    p = urlparse(url)
    host = p.netloc or "unknown"
    path = p.path.strip("/").replace("/", "_")
    base = sanitize_basename(f"{host}_{path or 'index'}")
    return f"{base}.txt"


def _normalize_text(s: str) -> str:
    for k, v in LIGATURE_MAP.items():
        s = s.replace(k, v)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"(?<![\.!?:;])\n(?!\n)", " ", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()


def sanitize_basename(name: str) -> str:
    base = os.path.splitext(os.path.basename(name))[0]
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    return base or "pdf"


def _split_paragraphs(block_text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", block_text.strip())
    paras = []
    for p in parts:
        p = _normalize_text(p)
        if p:
            paras.append(p)
    return paras


def extract_paragraphs_pymupdf_with_pages(pdf_bytes: bytes) -> list[dict]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_count = doc.page_count
    out = []
    pid = 1
    for page_index, page in enumerate(doc, start=1):
        blocks = page.get_text("blocks")
        blocks.sort(key=lambda b: (round(b[1], 1), round(b[0], 1)))
        for b in blocks:
            text = b[4]
            if not text or not text.strip():
                continue
            for para in _split_paragraphs(text):
                out.append(
                    {
                        "paragraph_id": pid,
                        "page_num": page_index,
                        "paragraph_text": para,
                    }
                )
                pid += 1
    doc.close()
    return out, page_count


# -----------------------------
# API Endpoints
# -----------------------------


@app.post("/api/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a .pdf file")

    pdf_bytes = await file.read()
    safe_name_pdf = sanitize_basename(file.filename) + ".pdf"
    upload_path = os.path.join(UPLOAD_DIR, safe_name_pdf)

    with open(upload_path, "wb") as out:
        out.write(pdf_bytes)

    try:
        paragraphs, page_count = extract_paragraphs_pymupdf_with_pages(pdf_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")

    result = {
        "created_at": utc_timestamp(),
        "source_file": f"{safe_name_pdf}",
        "page_count": page_count,
        "chunk_count": len(paragraphs),
        "paragraphs": paragraphs,
    }

    json_filename = sanitize_basename(file.filename) + ".json"
    json_path = os.path.join(OUTPUT_DIR, json_filename)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return JSONResponse(
        content={
            **result,
            "json_url": f"/outputs/{json_filename}",
            "json_filename": json_filename,
        },
        status_code=200,
    )


@app.get("/api/download/{basename}")
def download_json(basename: str):
    safe = sanitize_basename(basename)
    path = os.path.join(OUTPUT_DIR, f"{safe}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="application/json", filename=f"{safe}.json")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root_index():
    return HTMLResponse(
        open(os.path.join(STATIC_DIR, "index.html"), encoding="utf-8").read()
    )


@app.get("/index.html")
def index_alias():
    return HTMLResponse(
        open(os.path.join(STATIC_DIR, "index.html"), encoding="utf-8").read()
    )


@app.post("/api/fetch_url")
async def api_fetch_url(payload: dict = Body(...)):
    """
    Process URL and create vector store for semantic search

    Body:
    {
      "url": "https://example.com/page",
      "max_tokens_per_chunk": 512,
      "embed_backend": "google|openai|sbert|ollama",
      "collection_name": "my_collection"  # optional, will be auto-generated if not provided
    }
    """
    url = (payload.get("url") or "").strip()
    if not url or not url.lower().startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Provide a valid http(s) URL")

    max_tokens = int(payload.get("max_tokens_per_chunk") or 512)
    embed_backend = payload.get("embed_backend")
    collection_name = payload.get("collection_name")

    try:
        pkg = process_url_for_rag(
            url,
            max_tokens_per_chunk=max_tokens,
            embed_backend=embed_backend,
            create_vector_store=True,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Pipeline failed: {e!s}")

    # Generate collection name if not provided
    if not collection_name:
        base = (
            re.sub(r"[^a-zA-Z0-9_-]", "_", (pkg["title"] or "document"))[:60]
            or "document"
        )
        collection_name = f"{base}_{int(time.time())}"

    collection_name = sanitize_basename(collection_name)

    # Prepare output files
    stem = collection_name
    records_slim = []
    embeddings = []
    all_markdown = []

    for r in pkg["records"]:
        slim = {k: v for k, v in r.items() if k != "embedding"}
        h = r.get("heading") or ""
        body = r.get("markdown", "")
        if h:
            all_markdown.append(f"# {h}\n{body}\n")
        else:
            all_markdown.append(f"{body}\n")

        records_slim.append(slim)
        embeddings.append({"id": r["id"], "embedding": r["embedding"]})

    all_md = "\n".join(all_markdown).strip()

    meta = {
        "source_url": pkg["source_url"],
        "title": pkg["title"],
        "collection_name": collection_name,
        "created_at": utc_timestamp(),
        "record_count": len(pkg["records"]),
        "max_tokens_per_chunk": max_tokens,
        "embed_backend": embed_backend or os.getenv("EMBED_BACKEND") or "google",
    }

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    meta_path = Path(OUTPUT_DIR) / f"{stem}.json"
    emb_path = Path(OUTPUT_DIR) / f"{stem}.embeddings.jsonl"
    txt_path = Path(OUTPUT_DIR) / f"{stem}.txt"

    # Save files
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {"meta": meta, "records": records_slim}, f, ensure_ascii=False, indent=2
        )

    with open(emb_path, "w", encoding="utf-8") as f:
        for row in embeddings:
            f.write(json.dumps(row) + "\n")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(all_md)

    # Save vector store to disk and load into memory
    if "vector_store" in pkg:
        vector_store_path = Path(VECTOR_STORE_DIR) / collection_name
        pkg["vector_store"].save(str(vector_store_path))
        vector_stores[collection_name] = pkg["vector_store"]
        hlog(
            f"✓ Vector store '{collection_name}' created and loaded with {len(pkg['records'])} vectors"
        )

    return JSONResponse(
        content={
            "message": "OK",
            "source_url": pkg["source_url"],
            "title": pkg["title"],
            "collection_name": collection_name,
            "record_count": meta["record_count"],
            "outputs": {
                "records_json": f"/outputs/{meta_path.name}",
                "embeddings_jsonl": f"/outputs/{emb_path.name}",
                "markdown_txt": f"/outputs/{txt_path.name}",
            },
            "vector_store": {
                "available": True,
                "stats": (
                    pkg["vector_store"].get_stats() if "vector_store" in pkg else None
                ),
            },
            "params": {
                "max_tokens_per_chunk": max_tokens,
                "embed_backend": meta["embed_backend"],
            },
        }
    )


@app.post("/api/search")
async def api_search(payload: dict = Body(...)):
    """
    Semantic search in a vector store collection

    Body:
    {
      "query": "what are you looking for?",
      "collection": "collection_name",
      "k": 5,
      "embed_backend": "google|openai|sbert|ollama"  # optional, must match collection's backend
    }
    """
    query = payload.get("query", "").strip()
    collection = payload.get("collection", "").strip()
    k = int(payload.get("k", 5))
    embed_backend = payload.get("embed_backend")

    if not query:
        raise HTTPException(status_code=400, detail="Missing 'query' parameter")

    if not collection:
        raise HTTPException(status_code=400, detail="Missing 'collection' parameter")

    # Load vector store if not in memory
    if collection not in vector_stores:
        vector_store_path = Path(VECTOR_STORE_DIR) / collection
        if not (vector_store_path.with_suffix(".faiss").exists()):
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection}' not found. Available: {list(vector_stores.keys())}",
            )

        # Load from disk
        try:
            store = FAISSVectorStore()
            store.load(str(vector_store_path))
            vector_stores[collection] = store
            hlog(f"✓ Loaded vector store '{collection}' from disk")
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to load collection: {e}"
            )

    store = vector_stores[collection]

    try:
        backend = get_backend(embed_backend)
        results = store.search_with_text(query, k=k, backend=backend)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

    # Format results for frontend
    formatted_results = []
    for res in results:
        rec = res["record"]
        formatted_results.append(
            {
                "rank": res["rank"],
                "score": round(res["score"], 4),
                "id": rec["id"],
                "title": rec.get("title"),
                "heading": rec.get("heading"),
                "url": rec.get("url"),
                "anchor": rec.get("anchor"),
                "breadcrumbs": rec.get("breadcrumbs", []),
                "markdown": (
                    rec["markdown"][:300] + "..."
                    if len(rec["markdown"]) > 300
                    else rec["markdown"]
                ),
                "full_markdown": rec["markdown"],
                "tokens_est": rec.get("tokens_est"),
            }
        )

    return JSONResponse(
        content={
            "query": query,
            "collection": collection,
            "k": k,
            "results_count": len(formatted_results),
            "results": formatted_results,
            "store_stats": store.get_stats(),
        }
    )


@app.get("/api/collections")
def list_collections():
    """List all available vector store collections"""
    collections = []

    # Check disk
    for item in Path(VECTOR_STORE_DIR).glob("*.faiss"):
        name = item.stem
        is_loaded = name in vector_stores

        stats = None
        if is_loaded:
            stats = vector_stores[name].get_stats()

        collections.append({"name": name, "loaded": is_loaded, "stats": stats})

    return JSONResponse(
        content={
            "collections": collections,
            "loaded_count": len(vector_stores),
            "total_count": len(collections),
        }
    )


@app.delete("/api/collection/{collection_name}")
def delete_collection(collection_name: str):
    """Delete a vector store collection"""
    collection_name = sanitize_basename(collection_name)

    # Remove from memory
    if collection_name in vector_stores:
        del vector_stores[collection_name]

    # Remove from disk
    vector_store_path = Path(VECTOR_STORE_DIR) / collection_name
    deleted_files = []

    for ext in [".faiss", ".pkl"]:
        file_path = vector_store_path.with_suffix(ext)
        if file_path.exists():
            file_path.unlink()
            deleted_files.append(str(file_path.name))

    if not deleted_files:
        raise HTTPException(
            status_code=404, detail=f"Collection '{collection_name}' not found"
        )

    return JSONResponse(
        content={
            "message": f"Collection '{collection_name}' deleted",
            "deleted_files": deleted_files,
        }
    )
