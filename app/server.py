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

from fastapi import Body
from pathlib import Path
from rag_pipeline import process_url_for_rag

# import pdfplumber
# from pypdf import PdfReader

import pymupdf4llm

# --- Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
DATA_DIR = os.path.join(BASE_DIR, "data")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

LIGATURE_MAP = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
}


def hlog(logtxt: str):
    print(logtxt)


from fastapi.staticfiles import StaticFiles

app = FastAPI(title="PDF → JSON Extractor")

# CORS (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend (mounted at /static to avoid shadowing the API)
try:
    app.mount("/static", StaticFiles(directory=STATIC_DIR, html=False), name="static")
    app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
except Exception:
    pass


# -----------------------------
# Helpers
# -----------------------------
def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _html_to_text(html: str) -> str:
    # Robust HTML→text: ta bort script/style och komprimera whitespace
    soup = BeautifulSoup(html, "lxml")  # fallbackar till html.parser om lxml saknas
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # normalisera
    lines = [ln.strip() for ln in text.splitlines()]
    text = "\n".join(ln for ln in lines if ln)  # slopa tomrader
    return text


def _build_txt_filename_from_url(url: str) -> str:
    p = urlparse(url)
    host = p.netloc or "unknown"
    path = p.path.strip("/").replace("/", "_")
    base = sanitize_basename(f"{host}_{path or 'index'}")
    return f"{base}.txt"


def _normalize_text(s: str) -> str:
    # ersätt ligaturer och normalisera whitespace utan att ta bort legitima blanksteg
    for k, v in LIGATURE_MAP.items():
        s = s.replace(k, v)

    # ta bort hårda radbrytningar i mitten av meningar men behåll blankrader som styckesdelare
    s = s.replace("\r\n", "\n").replace("\r", "\n")

    # slå ihop radbrytningar där raden inte ser ut att avsluta en mening
    s = re.sub(r"(?<![\.!?:;])\n(?!\n)", " ", s)  # rad->mellanrum om inte meningsslut

    # komprimera mer än två spaces, men lämna ett space
    s = re.sub(r"[ \t]{2,}", " ", s)

    return s.strip()


def sanitize_basename(name: str) -> str:
    base = os.path.splitext(os.path.basename(name))[0]
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    return base or "pdf"


def _split_paragraphs(block_text: str) -> list[str]:
    # dela på tomma rader/indrag/avstånd – enkel men oftast träffsäker
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
        # sortera top-to-bottom, left-to-right
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
                        # lägg till fler metadata här om du vill: "x0": b[0], "y0": b[1], ...
                    }
                )
                pid += 1
    doc.close()
    return out, page_count


# -----------------------------
# API
# -----------------------------

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse


@app.post("/api/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    # 1) validera filtyp
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a .pdf file")

    pdf_bytes = await file.read()

    # 2) Spara originalet från upload
    safe_name_pdf = sanitize_basename(file.filename) + ".pdf"

    upload_path = os.path.join(UPLOAD_DIR, safe_name_pdf)
    with open(upload_path, "wb") as out:
        out.write(pdf_bytes)

    # 3) extrahera
    try:
        paragraphs, page_count = extract_paragraphs_pymupdf_with_pages(pdf_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")

    # 4) resultatstruktur
    result = {
        "created_at": utc_timestamp(),
        "source_file": f"{safe_name_pdf}",  # (montera om du vill exponera uploads)
        # "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "page_count": page_count,
        "chunk_count": len(paragraphs),
        "paragraphs": paragraphs,
    }

    # 5) skriv JSON till disk
    json_filename = sanitize_basename(file.filename) + ".json"
    json_path = os.path.join(OUTPUT_DIR, json_filename)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 6) returnera resultat + länk till JSON-filen
    #    (filen är åtkomlig via /outputs/<json_filename> tack vare StaticFiles-mounten)
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
    hlog(path)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="application/json", filename=f"{safe}.json")


# Simple health endpoint
@app.get("/api/health")
def health():
    return {"status": "ok"}


# Serve the SPA index
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


from fastapi import Body


@app.post("/api/fetch_url_old")
async def fetch_url_old(payload: dict = Body(...)):
    url = (payload.get("url") or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="Missing 'url'")
    if not (url.startswith("http://") or url.startswith("https://")):
        raise HTTPException(
            status_code=400, detail="URL must start with http:// or https://"
        )

    # Hämta sidan
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; PDFJSONBot/1.0; +local-dev)"}
        timeout = httpx.Timeout(15.0, connect=10.0)
        async with httpx.AsyncClient(
            headers=headers, timeout=timeout, follow_redirects=True
        ) as client:
            resp = await client.get(url)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Fetch failed: {e!s}")

    # Bestäm innehållstyp
    ctype = resp.headers.get("content-type", "").lower()

    if "text/html" in ctype or ctype.startswith("text/") or resp.text:
        text = _html_to_text(resp.text)
    else:
        # Enkel fallback: om det inte är HTML, skriv råbytes som text om möjligt
        try:
            text = resp.text  # httpx försöker dekoda med rätt encoding
        except Exception:
            raise HTTPException(
                status_code=415, detail=f"Unsupported content-type: {ctype}"
            )

    # Spara till /outputs
    txt_name = _build_txt_filename_from_url(url)
    txt_path = os.path.join(OUTPUT_DIR, txt_name)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)

    # Svara frontend
    return JSONResponse(
        content={
            "message": "OK",
            "source_url": url,
            "txt_filename": txt_name,
            "txt_url": f"/outputs/{txt_name}",
            "chars": len(text),
            "words": len(text.split()),
            "created_at": utc_timestamp(),
        },
        status_code=200,
    )


@app.post("/api/fetch_url")
async def api_fetch_url(payload: dict = Body(...)):
    """
    Body:
    {
      "url": "https://exempel.se/sida",
      "max_tokens_per_chunk": 512,            # valfritt
      "embed_backend": "openai|sbert|ollama"  # valfritt, annars ENV EMBED_BACKEND/openai
    }
    Returnerar paths till sparade filer och lite statistik.
    """
    url = (payload.get("url") or "").strip()
    if not url or not url.lower().startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Provide a valid http(s) URL")

    max_tokens = int(payload.get("max_tokens_per_chunk") or 512)
    embed_backend = payload.get("embed_backend")  # None => env/standard används

    try:
        pkg = process_url_for_rag(
            url, max_tokens_per_chunk=max_tokens, embed_backend=embed_backend
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Pipeline failed: {e!s}")

    # skriv ut som två filer: metadata+md och embeddings
    ts = ""  ## int(time.time())
    base = (
        re.sub(r"[^a-zA-Z0-9_-]", "_", (pkg["title"] or "document"))[:60] or "document"
    )
    stem = f"rag_{base}_{ts}"

    # 1) records utan den stora vektorn (för läsbarhet)
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
    chars = len(all_md)
    words = len(all_md.split())

    meta = {
        "source_url": pkg["source_url"],
        "title": pkg["title"],
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "record_count": len(pkg["records"]),
        "max_tokens_per_chunk": max_tokens,
        "embed_backend": embed_backend or os.getenv("EMBED_BACKEND") or "openai",
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

    return JSONResponse(
        content={
            "message": "OK",
            "source_url": pkg["source_url"],
            "title": pkg["title"],
            "record_count": meta["record_count"],
            "outputs": {
                "records_json": f"/outputs/{meta_path.name}",
                "embeddings_jsonl": f"/outputs/{emb_path.name}",
            },
            "params": {
                "max_tokens_per_chunk": max_tokens,
                "embed_backend": meta["embed_backend"],
            },
        }
    )
