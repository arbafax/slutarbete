###################################################
#
#   file: server.py
#
#   Holds the API endpoints.
#   Read and write towards the file system
#   Uses rag_pipeline.py
#
###################################################

from fastapi import FastAPI, UploadFile, File, Body, HTTPException
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from datetime import datetime, timezone
from dotenv import load_dotenv

import os
import time
import json
import re
import requests

from typing import Dict, Optional
from pathlib import Path
from rag_pipeline import (
    process_url_for_rag,
    process_pdf_for_rag,
    FAISSVectorStore,
    get_backend,
)
from helpers import (
    utc_timestamp,
    hlog,
    sanitize_basename,
    getApiKey,
    STATIC_DIR,
    UPLOAD_DIR,
    OUTPUT_DIR,
    VECTOR_STORE_DIR,
)

from google import genai
from google.genai import types

genai_client = genai.Client(api_key=getApiKey("GOOGLE_API_KEY"))

# Global registry for loaded vector stores
vector_stores: Dict[str, FAISSVectorStore] = {}


# -----------------------------
# LLM Backends
# -----------------------------


class LLMBackend:
    """
    Basklass för LLM backends.
    Definierar interface för olika språkmodeller.
    """

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generera text baserat på system- och user-prompt.

        Args:
            system_prompt (str): Systeminstruktioner för modellen
            user_prompt (str): Användarens fråga/prompt

        Returns:
            str: Genererad text från modellen
        """
        raise NotImplementedError


class GoogleLLMBackend(LLMBackend):
    """
    Google Gemini LLM backend.
    Använder Google Gemini API för textgenerering.
    """

    def __init__(self, model: str = "gemini-2.0-flash"):
        """
        Initiera Google LLM backend.

        Args:
            model (str): Modellnamn (default: "gemini-2.0-flash")
        """
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generera text med Google Gemini.

        Args:
            system_prompt (str): Systeminstruktioner
            user_prompt (str): Användarfråga

        Returns:
            str: Genererad text
        """
        response = genai_client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.3,
                max_output_tokens=2048,
            ),
        )
        return response.text


class OpenAILLMBackend(LLMBackend):
    """
    OpenAI GPT backend.
    Använder OpenAI API för textgenerering.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initiera OpenAI backend.

        Args:
            model (str): Modellnamn (default: "gpt-4o-mini")
        """
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generera text med OpenAI GPT.

        Args:
            system_prompt (str): Systeminstruktioner
            user_prompt (str): Användarfråga

        Returns:
            str: Genererad text
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=2048,
        )
        return response.choices[0].message.content


class OllamaLLMBackend(LLMBackend):
    """
    Ollama lokal LLM backend.
    Använder lokalt installerad Ollama för textgenerering.
    """

    def __init__(self, model: str = "llama3.2", host: str = "http://localhost:11434"):
        """
        Initiera Ollama backend.

        Args:
            model (str): Modellnamn (default: "llama3.2")
            host (str): Ollama server URL (default: "http://localhost:11434")
        """
        self.model = model
        self.host = host.rstrip("/")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generera text med Ollama.

        Args:
            system_prompt (str): Systeminstruktioner
            user_prompt (str): Användarfråga

        Returns:
            str: Genererad text
        """
        r = requests.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "prompt": user_prompt,
                "system": system_prompt,
                "stream": False,
            },
        )
        r.raise_for_status()
        return r.json()["response"]


def get_llm_backend(kind: Optional[str] = None) -> LLMBackend:
    """
    Hämta rätt LLM backend baserat på namn.

    Args:
        kind (Optional[str]): Backend-typ ("google", "openai", "ollama")

    Returns:
        LLMBackend: Instans av vald backend
    """
    backend = (kind or "google").lower()
    if backend == "openai":
        return OpenAILLMBackend()
    if backend == "ollama":
        return OllamaLLMBackend()
    return GoogleLLMBackend()


DEFAULT_SYSTEM_PROMPT = """Du är en hjälpsam assistent som svarar på frågor baserat på den kontext som ges.

VIKTIGA REGLER:
- Svara ENDAST baserat på informationen i den bifogade kontexten
- Om kontexten inte innehåller tillräcklig information för att svara, säg "Det finns inte tillräckligt med information i dokumentet för att svara på den frågan."
- Gissa inte eller hitta på information
- Formulera dig tydligt och dela upp svaret i läsbara stycken
- Var koncis men informativ
- Om du citerar från kontexten, var tydlig med det"""


# -----------------------------
# FastAPI App
# -----------------------------

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
# API Endpoints
# -----------------------------


@app.post("/api/upload_pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    embed_backend: str = "google",
    max_tokens_per_chunk: int = 512,
    collection_name: Optional[str] = None,  # NEW: Optional collection name
):
    """
    API endpoint för att ladda upp och processa PDF-filer.

    Args:
        file (UploadFile): PDF-fil att ladda upp
        embed_backend (str): Embedding backend ("google", "openai", "ollama")
        max_tokens_per_chunk (int): Max tokens per chunk
        collection_name (Optional[str]): Anpassat namn för samlingen

    Returns:
        JSONResponse: Status och information om processad PDF
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a .pdf file")

    pdf_bytes = await file.read()
    safe_name_pdf = sanitize_basename(file.filename) + ".pdf"
    upload_path = os.path.join(UPLOAD_DIR, safe_name_pdf)

    with open(upload_path, "wb") as out:
        out.write(pdf_bytes)

    try:
        pkg = process_pdf_for_rag(
            pdf_bytes,
            filename=file.filename,
            max_tokens_per_chunk=max_tokens_per_chunk,
            embed_backend=embed_backend,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")

    # NEW: Use custom collection name if provided, otherwise use filename
    if collection_name and collection_name.strip():
        collection_name = sanitize_basename(collection_name.strip())
    else:
        collection_name = sanitize_basename(file.filename)

    records_slim = []
    embeddings = []
    all_text = []

    for r in pkg["records"]:
        slim = {k: v for k, v in r.items() if k != "embedding"}
        records_slim.append(slim)
        embeddings.append({"id": r["id"], "embedding": r["embedding"]})
        h = r.get("heading") or ""
        body = r.get("markdown", "")
        if h:
            all_text.append(f"# {h}\n{body}\n")
        else:
            all_text.append(f"{body}\n")

    all_text_str = "\n".join(all_text).strip()

    meta = {
        "source_file": safe_name_pdf,
        "title": pkg["title"],
        "collection_name": collection_name,
        "created_at": utc_timestamp(),
        "record_count": len(pkg["records"]),
        "max_tokens_per_chunk": max_tokens_per_chunk,
        "embed_backend": embed_backend,
        "type": "pdf",
    }

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    stem = collection_name

    meta_path = Path(OUTPUT_DIR) / f"{stem}.json"
    emb_path = Path(OUTPUT_DIR) / f"{stem}.embeddings.jsonl"
    txt_path = Path(OUTPUT_DIR) / f"{stem}.txt"

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {"meta": meta, "records": records_slim}, f, ensure_ascii=False, indent=2
        )

    with open(emb_path, "w", encoding="utf-8") as f:
        for row in embeddings:
            f.write(json.dumps(row) + "\n")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(all_text_str)

    vector_store_path = Path(VECTOR_STORE_DIR) / collection_name
    pkg["vector_store"].save(str(vector_store_path))
    vector_stores[collection_name] = pkg["vector_store"]

    hlog(
        f"✓ PDF '{file.filename}' processed: {len(pkg['records'])} chunks as '{collection_name}'"
    )
    hlog(f"/outputs/{meta_path.name}")

    return JSONResponse(
        content={
            "message": "OK",
            "source_file": safe_name_pdf,
            "title": pkg["title"],
            "collection_name": collection_name,
            "record_count": meta["record_count"],
            "outputs": {
                "records_json": f"/outputs/{meta_path.name}",
                "embeddings_jsonl": f"/outputs/{emb_path.name}",
                "text_file": f"/outputs/{txt_path.name}",
            },
            "vector_store": {
                "available": True,
                "stats": pkg["vector_store"].get_stats(),
            },
            "params": {
                "max_tokens_per_chunk": max_tokens_per_chunk,
                "embed_backend": embed_backend,
            },
        },
        status_code=200,
    )


@app.get("/api/download/{basename}")
def download_json(basename: str):
    """
    Ladda ner JSON-fil från outputs.

    Args:
        basename (str): Filnamn utan extension

    Returns:
        FileResponse: JSON-fil
    """
    safe = sanitize_basename(basename)
    path = os.path.join(OUTPUT_DIR, f"{safe}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="application/json", filename=f"{safe}.json")


@app.get("/api/health")
def health():
    """
    Health check endpoint.

    Returns:
        dict: Status OK
    """
    return {"status": "ok"}


@app.get("/")
def root_index():
    """
    Serve index.html på root path.

    Returns:
        HTMLResponse: index.html
    """
    return HTMLResponse(
        open(os.path.join(STATIC_DIR, "index.html"), encoding="utf-8").read()
    )


@app.get("/index.html")
def index_alias():
    """
    Serve index.html på /index.html path.

    Returns:
        HTMLResponse: index.html
    """
    return HTMLResponse(
        open(os.path.join(STATIC_DIR, "index.html"), encoding="utf-8").read()
    )


@app.post("/api/fetch_url")
async def api_fetch_url(payload: dict = Body(...)):
    """
    Hämta och processa innehåll från URL för RAG.

    Args:
        payload (dict): Request body med url, max_tokens_per_chunk, embed_backend, collection_name

    Returns:
        JSONResponse: Status och information om processat innehåll
    """
    url = (payload.get("url") or "").strip()
    if not url or not url.lower().startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Provide a valid http(s) URL")

    max_tokens = int(payload.get("max_tokens_per_chunk") or 512)
    embed_backend = payload.get("embed_backend")
    collection_name = payload.get("collection_name")  # NEW: Get custom name

    try:
        pkg = process_url_for_rag(
            url,
            max_tokens_per_chunk=max_tokens,
            embed_backend=embed_backend,
            create_vector_store=True,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Pipeline failed: {e!s}")

    # NEW: Use custom collection name if provided
    if collection_name and collection_name.strip():
        collection_name = sanitize_basename(collection_name.strip())
    else:
        # Generate automatic name from title
        base = (
            re.sub(r"[^a-zA-Z0-9_-]", "_", (pkg["title"] or "document"))[:60]
            or "document"
        )
        collection_name = f"{base}_{int(time.time())}"

    collection_name = sanitize_basename(collection_name)

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

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {"meta": meta, "records": records_slim}, f, ensure_ascii=False, indent=2
        )

    with open(emb_path, "w", encoding="utf-8") as f:
        for row in embeddings:
            f.write(json.dumps(row) + "\n")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(all_md)

    if "vector_store" in pkg:
        vector_store_path = Path(VECTOR_STORE_DIR) / collection_name
        pkg["vector_store"].save(str(vector_store_path))
        vector_stores[collection_name] = pkg["vector_store"]
        hlog(
            f"✓ Vector store '{collection_name}' created with {len(pkg['records'])} vectors"
        )

    hlog(f"URL:  -->   /outputs/{meta_path.name}")

    return JSONResponse(
        content={
            "message": "OK",
            "source_url": pkg["source_url"],
            "title": pkg["title"],
            "collection_name": collection_name,  # FIXED: Return actual collection name
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
    Sök i vector store med semantisk sökning.

    Args:
        payload (dict): Request body med query, collection, k, embed_backend

    Returns:
        JSONResponse: Sökresultat med rankade träffar
    """
    query = payload.get("query", "").strip()
    collection = payload.get("collection", "").strip()
    k = int(payload.get("k", 5))
    embed_backend = payload.get("embed_backend")

    if not query:
        raise HTTPException(status_code=400, detail="Missing 'query' parameter")
    if not collection:
        raise HTTPException(status_code=400, detail="Missing 'collection' parameter")

    if collection not in vector_stores:
        vector_store_path = Path(VECTOR_STORE_DIR) / collection
        if not vector_store_path.with_suffix(".faiss").exists():
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection}' not found. Available: {list(vector_stores.keys())}",
            )
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


@app.post("/api/ask")
async def api_ask(payload: dict = Body(...)):
    """
    RAG-baserad fråga-svar med LLM
    """
    query = payload.get("query", "").strip()
    collection = payload.get("collection", "").strip()
    k = int(payload.get("k", 5))
    llm_backend_name = payload.get("llm_backend", "google")
    embed_backend_name = payload.get("embed_backend")
    custom_system_prompt = payload.get("system_prompt")

    if not query:
        raise HTTPException(status_code=400, detail="Missing 'query' parameter")
    if not collection:
        raise HTTPException(status_code=400, detail="Missing 'collection' parameter")

    # 1. Ladda vector store
    if collection not in vector_stores:
        vector_store_path = Path(VECTOR_STORE_DIR) / collection
        if not vector_store_path.with_suffix(".faiss").exists():
            raise HTTPException(
                status_code=404, detail=f"Collection '{collection}' not found"
            )
        try:
            store = FAISSVectorStore()
            store.load(str(vector_store_path))
            vector_stores[collection] = store
            hlog(f"✓ Loaded vector store '{collection}' from disk")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load: {e}")

    store = vector_stores[collection]

    # 2. Semantisk sökning
    try:
        embed_backend = get_backend(embed_backend_name)
        search_results = store.search_with_text(query, k=k, backend=embed_backend)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

    if not search_results:
        return JSONResponse(
            content={
                "query": query,
                "answer": "Inga relevanta dokument hittades för att svara på frågan.",
                "sources": [],
                "context_used": "",
            }
        )

    # 3. Bygg kontext
    context_parts = []
    sources = []

    for i, res in enumerate(search_results, 1):
        rec = res["record"]
        heading = rec.get("heading", "")
        text = rec.get("markdown", "")
        score = res["score"]

        if heading:
            context_parts.append(f"[Källa {i}: {heading}]\n{text}")
        else:
            context_parts.append(f"[Källa {i}]\n{text}")

        sources.append(
            {
                "rank": i,
                "score": round(score, 4),
                "heading": heading,
                "preview": text[:200] + "..." if len(text) > 200 else text,
                "url": rec.get("url"),
                "anchor": rec.get("anchor"),
            }
        )

    context = "\n\n---\n\n".join(context_parts)

    # 4. Bygg user prompt
    user_prompt = f"""KONTEXT:
{context}

---

FRÅGA: {query}

Svara på frågan baserat på kontexten ovan."""

    # 5. System prompt
    system_prompt = custom_system_prompt or DEFAULT_SYSTEM_PROMPT

    # 6. Anropa LLM
    try:
        llm = get_llm_backend(llm_backend_name)
        hlog(f"✓ Sending to LLM backend USER_PROMPT: {user_prompt}")
        answer = llm.generate(system_prompt, user_prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {e}")

    hlog(f"✓ RAG answer generated for '{query[:50]}...' using {llm_backend_name}")

    return JSONResponse(
        content={
            "query": query,
            "answer": answer,
            "sources": sources,
            "context_used": context,
            "params": {
                "collection": collection,
                "k": k,
                "llm_backend": llm_backend_name,
                "embed_backend": embed_backend_name or "google",
            },
        }
    )


@app.get("/api/collections")
def list_collections():
    collections = []
    for item in Path(VECTOR_STORE_DIR).glob("*.faiss"):
        name = item.stem
        is_loaded = name in vector_stores
        stats = vector_stores[name].get_stats() if is_loaded else None
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
    collection_name = sanitize_basename(collection_name)

    if collection_name in vector_stores:
        del vector_stores[collection_name]

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
