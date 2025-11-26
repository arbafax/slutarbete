###################################################
#
#   file: helpers.py
#
#   Serves with helper functionalities
#
###################################################

import os
import re
import fitz

from datetime import datetime, timezone

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


def utc_timestamp() -> str:
    """
    Generera UTC timestamp i ISO-format.

    Returns:
        str: UTC timestamp (t.ex. "2025-01-15T10:30:00Z")
    """
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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
# Filnamn / text-normalisering
# -----------------------------

LIGATURE_MAP = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
}


def normalize_text(s: str) -> str:
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


# def split_paragraphs(block_text: str) -> list[str]:
#     """
#     Dela upp text i paragrafer.

#     Args:
#         block_text (str): Text att dela upp

#     Returns:
#         list[str]: Lista med paragrafer
#     """
#     parts = re.split(r"\n\s*\n", block_text.strip())
#     paras = []
#     for p in parts:
#         p = normalize_text(p)
#         if p:
#             paras.append(p)
#     return paras


# def _html_to_text(html: str) -> str:
#     """
#     Konvertera HTML till ren text.

#     Args:
#         html (str): HTML-sträng

#     Returns:
#         str: Rengjord text utan HTML-taggar
#     """
#     soup = BeautifulSoup(html, "lxml")
#     for tag in soup(["script", "style", "noscript"]):
#         tag.decompose()
#     text = soup.get_text(separator="\n")
#     lines = [ln.strip() for ln in text.splitlines()]
#     text = "\n".join(ln for ln in lines if ln)
#     return text

# def extract_paragraphs_pymupdf_with_pages(pdf_bytes: bytes) -> list[dict]:
#     """
#     Extrahera paragrafer från PDF med sidnummer.

#     Args:
#         pdf_bytes (bytes): PDF-fil som bytes

#     Returns:
#         tuple: (list[dict], int) - Lista med paragrafer och antal sidor
#                Varje dict innehåller: paragraph_id, page_num, paragraph_text
#     """
#     doc = fitz.open(stream=pdf_bytes, filetype="pdf")
#     page_count = doc.page_count
#     out = []
#     pid = 1
#     for page_index, page in enumerate(doc, start=1):
#         blocks = page.get_text("blocks")
#         blocks.sort(key=lambda b: (round(b[1], 1), round(b[0], 1)))
#         for b in blocks:
#             text = b[4]
#             if not text or not text.strip():
#                 continue
#             for para in split_paragraphs(text):
#                 out.append(
#                     {
#                         "paragraph_id": pid,
#                         "page_num": page_index,
#                         "paragraph_text": para,
#                     }
#                 )
#                 pid += 1
#     doc.close()
#     return out, page_count


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
