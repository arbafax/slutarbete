from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import os
import io
import json
import re
from typing import List, Dict, Optional

import pdfplumber
from pypdf import PdfReader

# --- Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
DATA_DIR = os.path.join(BASE_DIR, "data")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

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
app.mount("/static", StaticFiles(directory=STATIC_DIR, html=False), name="static")

# -----------------------------
# Helpers
# -----------------------------


def sanitize_basename(name: str) -> str:
    base = os.path.splitext(os.path.basename(name))[0]
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    return base or "pdf"


def split_paragraphs(text: str) -> List[str]:
    """
    Split page text into paragraphs. Heuristics:
    - Prefer double newlines as paragraph boundaries
    - Also split on single newlines preceded by end punctuation for long lines
    - Remove tiny fragments
    """
    if not text:
        return []

    # Normalize line endings
    t = text.replace("\r\n", "\n").replace("\r", "\n")

    # First pass: split on blank lines
    parts = re.split(r"\n\s*\n", t)

    paras: List[str] = []
    for part in parts:
        # Secondary split: lines that likely end a sentence
        chunks = re.split(r"(?<=\.|:|;|\?|!|\))\n(?=\S)", part)
        for ch in chunks:
            para = re.sub(r"\s+", " ", ch).strip()  ## 1.0
            # para = re.sub(r"(?<=\S)\s+", " ", ch).rstrip()  ## 1.1
            # para = ch  ## låt alla whitespace vara kvar 1.2
            if len(para) >= 25:  # ignore tiny bits
                paras.append(para)

    return paras


def build_chapter_map_from_outlines(reader: PdfReader) -> Dict[int, str]:
    """Return a mapping of 1-based page number → chapter title using PDF outlines if present."""
    chapter_map: Dict[int, str] = {}

    try:
        outlines = reader.outlines  # may raise or be empty
    except Exception:
        outlines = []

    def flatten(out, current_title_prefix: Optional[str] = None):
        for item in out:
            if isinstance(item, list):
                flatten(item, current_title_prefix)
            else:
                try:
                    # pypdf Destination or Bookmark-like
                    page_obj = item.page
                    page_num = reader.get_page_number(page_obj) + 1  # 1-based
                    title = str(getattr(item, "title", "")).strip()
                    if current_title_prefix:
                        title = (
                            f"{current_title_prefix} — {title}"
                            if title
                            else current_title_prefix
                        )
                    if title:
                        chapter_map[page_num] = title
                except Exception:
                    continue

    if outlines:
        flatten(outlines)

    # Forward-fill chapters across pages
    if chapter_map:
        last_title = None
        for p in range(1, len(reader.pages) + 1):
            if p in chapter_map:
                last_title = chapter_map[p]
            elif last_title:
                chapter_map[p] = last_title

    return chapter_map


def infer_heading_from_page(page: pdfplumber.page.Page) -> Optional[str]:
    """
    Very light-weight heading inference using text lines:
    - Look at first ~10 lines
    - Prefer short lines (< 80 chars) that are ALL CAPS or Title Case
    - Return the best candidate if any
    """
    try:
        raw = page.extract_text(layout=True) or ""
        lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
        candidates: List[str] = []
        for ln in lines[:10]:
            if len(ln) <= 80:
                if ln.isupper():
                    candidates.append((0, ln))  # all caps best
                elif re.match(r"^[A-ZÅÄÖ][A-Za-zÅÄÖåäö0-9 ,.'\-:;()\/]+$", ln):
                    candidates.append((1, ln))  # title case-ish
        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1]
    except Exception:
        pass
    return None


def extract_paragraphs_with_metadata(pdf_bytes: bytes, filename: str) -> Dict:
    """
    Extract paragraphs from a PDF with metadata.
    Returns a dict with keys: source_pdf, total_pages, paragraphs (list), saved_json_path
    """
    basename = sanitize_basename(filename)

    # Try pypdf for outlines/chapters
    reader = PdfReader(io.BytesIO(pdf_bytes))
    chapter_map = build_chapter_map_from_outlines(reader)

    paragraphs: List[Dict] = []
    para_id = 1

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        total_pages = len(pdf.pages)
        for idx, page in enumerate(pdf.pages, start=1):
            try:
                text = page.extract_text(layout=True)
            except Exception:
                text = page.extract_text()  # fallback

            page_paras = split_paragraphs(text or "")

            # Heading inference for this page (used as fallback and as per-page heading)
            heading = infer_heading_from_page(page)

            # Chapter from outline if present; else fallback to heading
            chapter = chapter_map.get(idx, heading)

            for ptext in page_paras:
                paragraphs.append(
                    {
                        "paragraph_id": para_id,
                        "page_num": idx,
                        "paragraph_text": ptext,
                        "heading": heading,
                        "chapter": chapter,
                    }
                )
                para_id += 1

    # Save JSON
    output_path = os.path.join(DATA_DIR, f"{basename}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_pdf": filename,
                "total_pages": len(reader.pages),
                "paragraphs": paragraphs,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "source_pdf": filename,
        "total_pages": len(reader.pages),
        "paragraph_count": len(paragraphs),
        "saved_json": output_path,
    }


# -----------------------------
# API
# -----------------------------


@app.post("/api/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a .pdf file")

    pdf_bytes = await file.read()

    # Save original upload (optional)
    safe_name = sanitize_basename(file.filename) + ".pdf"
    upload_path = os.path.join(UPLOAD_DIR, safe_name)
    with open(upload_path, "wb") as out:
        out.write(pdf_bytes)

    try:
        result = extract_paragraphs_with_metadata(pdf_bytes, file.filename)
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")


@app.get("/api/download/{basename}")
def download_json(basename: str):
    safe = sanitize_basename(basename)
    path = os.path.join(DATA_DIR, f"{safe}.json")
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


# # Root serves index.html via StaticFiles
# @app.get("/index.html")
# def root_index():
#     return HTMLResponse(
#         open(os.path.join(STATIC_DIR, "index.html"), encoding="utf-8").read()
#     )
