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

import os
import io
import time
import json
import re
import fitz  # pymupdf
import httpx
import requests

from bs4 import BeautifulSoup
from urllib.parse import urlparse
from typing import List, Dict, Optional

from pathlib import Path
from rag_pipeline import process_url_for_rag, FAISSVectorStore, get_backend

from google import genai
from google.genai import types

# --- API Keys ---
GOOGLE_API_KEY = "AIzaSyBxHq5cPvm-0GGo8fHtPLDNDihB9X_oUlM"
genai_client = genai.Client(api_key=GOOGLE_API_KEY)

# --- Paths ---
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
    """
    Hjälpfunktion för loggning till konsol.
    
    Args:
        logtxt (str): Texten som ska loggas
    
    Returns:
        None
    """
    print(logtxt)


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
- Svara ENDAST baserat på informationen i kontexten nedan
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
# Helpers
# -----------------------------


def utc_timestamp() -> str:
    """
    Generera UTC timestamp i ISO-format.
    
    Returns:
        str: UTC timestamp (t.ex. "2025-01-15T10:30:00Z")
    """
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _html_to_text(html: str) -> str:
    """
    Konvertera HTML till ren text.
    
    Args:
        html (str): HTML-sträng
    
    Returns:
        str: Rengjord text utan HTML-taggar
    """
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines()]
    text = "\n".join(ln for ln in lines if ln)
    return text


def _build_txt_filename_from_url(url: str) -> str:
    """
    Skapa filnamn baserat på URL.
    
    Args:
        url (str): URL att konvertera
    
    Returns:
        str: Säkert filnamn med .txt extension
    """
    p = urlparse(url)
    host = p.netloc or "unknown"
    path = p.path.strip("/").replace("/", "_")
    base = sanitize_basename(f"{host}_{path or 'index'}")
    return f"{base}.txt"


def _normalize_text(s: str) -> str:
    """
    Normalisera text genom att ersätta ligaturer och fixa radbrytningar.
    
    Args:
        s (str): Text att normalisera
    
    Returns:
        str: Normaliserad text
    """
    for k, v in LIGATURE_MAP.items():
        s = s.replace(k, v)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"(?<![\.!?:;])\n(?!\n)", " ", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()


def sanitize_basename(name: str) -> str:
    """
    Sanera filnamn genom att ta bort ogiltiga tecken.
    
    Args:
        name (str): Filnamn att sanera
    
    Returns:
        str: Sanerat filnamn (endast A-Z, a-z, 0-9, ., _, -)
    """
    base = os.path.splitext(os.path.basename(name))[0]
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    return base or "pdf"


def _split_paragraphs(block_text: str) -> list[str]:
    """
    Dela upp text i paragrafer.
    
    Args:
        block_text (str): Text att dela upp
    
    Returns:
        list[str]: Lista med paragrafer
    """
    parts = re.split(r"\n\s*\n", block_text.strip())
    paras = []
    for p in parts:
        p = _normalize_text(p)
        if p:
            paras.append(p)
    return paras


def extract_paragraphs_pymupdf_with_pages(pdf_bytes: bytes) -> list[dict]:
    """
    Extrahera paragrafer från PDF med sidnummer.
    
    Args:
        pdf_bytes (bytes): PDF-fil som bytes
    
    Returns:
        tuple: (list[dict], int) - Lista med paragrafer och antal sidor
               Varje dict innehåller: paragraph_id, page_num, paragraph_text
    """
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


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extrahera all text från PDF.
    
    Args:
        pdf_bytes (bytes): PDF-fil som bytes
    
    Returns:
        str: All text från PDF:en
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            full_text.append(text)
    doc.close()
    return "\n\n".join(full_text)


def process_pdf_for_rag(
    pdf_bytes: bytes,
    filename: str,
    max_tokens_per_chunk: int = 512,
    embed_backend: Optional[str] = None,
) -> Dict:
    """
    Processa PDF för RAG (Retrieval Augmented Generation).
    Extraherar text, skapar chunks, genererar embeddings och bygger vector store.
    
    Args:
        pdf_bytes (bytes): PDF-fil som bytes
        filename (str): Filnamn
        max_tokens_per_chunk (int): Max antal tokens per chunk (default: 512)
        embed_backend (Optional[str]): Embedding backend att använda
    
    Returns:
        Dict: Dictionary med source_file, title, records, vector_store
    """
    from rag_pipeline import (
        Block,
        split_markdown_into_blocks,
        build_records,
        embed_records,
        get_backend,
        FAISSVectorStore,
    )

    full_text = extract_text_from_pdf(pdf_bytes)
    full_text = _normalize_text(full_text)
    paragraphs = re.split(r"\n\s*\n", full_text)

    blocks = []
    for i, para in enumerate(paragraphs):
        if para.strip():
            lines = para.split("\n", 1)
            first_line = lines[0].strip()
            is_heading = len(first_line) < 80 and (
                first_line.isupper() or re.match(r"^\d+[\.\)]\s+", first_line)
            )
            if is_heading and len(lines) > 1:
                heading = first_line
                content = lines[1]
            else:
                heading = f"Section {i+1}"
                content = para
            blocks.append(Block(level=1, heading=heading, content=content))

    if not blocks:
        blocks = [Block(level=1, heading="Document Content", content=full_text)]

    title = os.path.splitext(filename)[0]
    records = build_records(
        url=f"pdf://{filename}",
        title=title,
        blocks=blocks,
        max_tokens_per_chunk=max_tokens_per_chunk,
    )

    backend = get_backend(embed_backend)
    embed_records(records, backend=backend)

    dimension = backend.get_dimension()
    vector_store = FAISSVectorStore(dimension=dimension)
    vector_store.add_records(records)

    return {
        "source_file": filename,
        "title": title,
        "records": records,
        "vector_store": vector_store,
    }


# -----------------------------
# API Endpoints
# -----------------------------


@app.post("/api/upload_pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    embed_backend: str = "google",
    max_tokens_per_chunk: int = 512,
):
    """
    API endpoint för att ladda upp och processa PDF-filer.
    
    Args:
        file (UploadFile): PDF-fil att ladda upp
        embed_backend (str): Embedding backend ("google", "openai", "ollama")
        max_tokens_per_chunk (int): Max tokens per chunk
    
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

    hlog(f"✓ PDF '{file.filename}' processed: {len(pkg['records'])} chunks")

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

    if not collection_name:
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
    RAG-baserad fråga-svar med LLM.
    Söker först relevanta dokument, sedan genererar svar med LLM.
    
    Args:
        payload (dict): Request body med query, collection, k, llm_backend, 
                       embed_backend, system_prompt
    
    Returns:
        JSONResponse: LLM-genererat svar med källor och kontext
    """
    query = payload.get("query", "").strip()
    collection = payload.get("collection", "").strip()
    k =