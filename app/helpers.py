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
from dotenv import load_dotenv

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

load_dotenv()


def getDefaultSystemPrompt() -> str:
    return """Du är en hjälpsam assistent som svarar på frågor baserat på den kontext som ges.

VIKTIGA REGLER:
- Svara ENDAST baserat på informationen i den bifogade kontexten
- Om kontexten inte innehåller tillräcklig information för att svara, säg "Det finns inte tillräckligt med information i dokumentet för att svara på den frågan."
- Gissa inte eller hitta på information
- Formulera dig tydligt och dela upp svaret i läsbara stycken
- Var koncis men informativ
- Om du citerar från kontexten, var tydlig med det"""


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
# API-keys management
# -----------------------------
def getApiKey(apiKey: str) -> str:
    return os.getenv("apiKey")


# -----------------------------
# File namne and text normalizing
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
