from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from datetime import datetime, timezone

import os
import io
import json
import re
import fitz  # pymupdf

from typing import List, Dict, Optional

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
app.mount("/static", StaticFiles(directory=STATIC_DIR, html=False), name="static")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


# -----------------------------
# Helpers
# -----------------------------
def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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


# # Root serves index.html via StaticFiles
# @app.get("/index.html")
# def root_index():
#     return HTMLResponse(
#         open(os.path.join(STATIC_DIR, "index.html"), encoding="utf-8").read()
#     )
