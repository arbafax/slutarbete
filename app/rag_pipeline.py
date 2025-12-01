###################################################
#
#   file: rag_pipeline.py
#
#   Helper classes for RAG as well as a pipeline for a RAG Process
#   Now with FAISS vector store integration
#
###################################################


import numpy as np
import os, re, math, json, html, time, pickle
from dataclasses import dataclass
from typing import List, Dict, Optional
from urllib.parse import urljoin, urlparse
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from google import genai
from google.genai import types

from helpers import extract_text_from_pdf, normalize_text, getApiKey

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("WARNING: faiss not installed. Run: pip install faiss-cpu")


client = genai.Client(api_key=getApiKey("GOOGLE_API_KEY"))

# ---------------- Embeddings-backends ----------------


class EmbeddingBackend:
    """
    Basklass för embedding backends.
    Definierar interface för olika embedding-modeller.
    """

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Skapa embeddings för lista av texter.

        Args:
            texts (List[str]): Lista med textsträngar

        Returns:
            List[List[float]]: Lista med embedding-vektorer
        """
        raise NotImplementedError

    def get_dimension(self) -> int:
        """
        Hämta dimensionen på embedding-vektorer.

        Returns:
            int: Antal dimensioner i embedding-vektorn
        """
        raise NotImplementedError


class OpenAIBackend(EmbeddingBackend):
    """
    OpenAI embedding backend.
    Använder OpenAI API för att skapa text embeddings.
    """

    def __init__(self, model: str = "text-embedding-3-large"):
        """
        Initiera OpenAI backend.

        Args:
            model (str): OpenAI modellnamn (default: "text-embedding-3-large")
        """
        from openai import OpenAI

        self.client = OpenAI(api_key=getApiKey("OPENAI_API_KEY"))
        self.model = model
        self._dimension = 3072 if "large" in model else 1536

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Skapa embeddings med OpenAI.

        Args:
            texts (List[str]): Texter att embedda

        Returns:
            List[List[float]]: Embedding-vektorer
        """
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]

    def get_dimension(self) -> int:
        """
        Hämta embedding dimension.

        Returns:
            int: Dimension (1536 eller 3072)
        """
        return self._dimension


class SBERTBackend(EmbeddingBackend):
    """
    Sentence-BERT embedding backend.
    Använder sentence-transformers för lokal embedding.
    """

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initiera SBERT backend.

        Args:
            model (str): Sentence-transformers modellnamn
        """
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model)
        self._dimension = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Skapa embeddings med SBERT.

        Args:
            texts (List[str]): Texter att embedda

        Returns:
            List[List[float]]: Normaliserade embedding-vektorer
        """
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def get_dimension(self) -> int:
        """
        Hämta embedding dimension.

        Returns:
            int: Dimension från modellen
        """
        return self._dimension


class OllamaBackend(EmbeddingBackend):
    """
    Ollama lokal embedding backend.
    Använder lokalt installerad Ollama för embeddings.
    """

    def __init__(
        self, model: str = "nomic-embed-text", host: str = "http://localhost:11434"
    ):
        """
        Initiera Ollama backend.

        Args:
            model (str): Ollama modellnamn (default: "nomic-embed-text")
            host (str): Ollama server URL (default: "http://localhost:11434")
        """
        self.model = model
        self.host = host.rstrip("/")
        self._dimension = 768  # default for nomic-embed-text

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Skapa embeddings med Ollama.

        Args:
            texts (List[str]): Texter att embedda

        Returns:
            List[List[float]]: Embedding-vektorer
        """
        out = []
        for t in texts:
            r = requests.post(
                f"{self.host}/api/embeddings", json={"model": self.model, "prompt": t}
            )
            r.raise_for_status()
            out.append(r.json()["embedding"])
        return out

    def get_dimension(self) -> int:
        """
        Hämta embedding dimension.

        Returns:
            int: Dimension (768 för nomic-embed-text)
        """
        return self._dimension


class GoogleBackend(EmbeddingBackend):
    """
    Google Gemini embedding backend.
    Använder Google Gemini API för embeddings.
    """

    def __init__(self, model: str = "text-embedding-004"):
        """
        Initiera Google backend.

        Args:
            model (str): Google modellnamn (default: "text-embedding-004")
        """
        self.model = model
        self._dimension = 768

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Skapa embeddings med Google Gemini.

        Args:
            texts (List[str]): Texter att embedda

        Returns:
            List[List[float]]: Embedding-vektorer
        """
        results = []
        for text in texts:
            resp = client.models.embed_content(
                model=self.model,
                contents=text,
                config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
            )
            results.append(resp.embeddings[0].values)
        return results

    def get_dimension(self) -> int:
        """
        Hämta embedding dimension.

        Returns:
            int: Dimension (768)
        """
        return self._dimension


# ============= NYA FAS 1 BACKENDS =============


class CohereBackend(EmbeddingBackend):
    """
    Cohere embedding backend - Mycket robust mot brusig data.
    Utmärkt för PDF och HTML med formatfel.
    Stöder flera språk och stora kontextlängder (512 tokens).
    """

    def __init__(self, model: str = "embed-multilingual-v3.0"):
        """
        Initiera Cohere backend.

        Args:
            model (str): Cohere modellnamn (default: "embed-multilingual-v3.0")
                Alternativ: "embed-english-v3.0", "embed-multilingual-light-v3.0"
        """
        try:
            import cohere
        except ImportError:
            raise ImportError("Cohere package not installed. Run: pip install cohere")

        api_key = getApiKey("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY not found in environment")

        self.client = cohere.Client(api_key)
        self.model = model
        self._dimension = 1024  # Cohere v3 models use 1024 dimensions

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Skapa embeddings med Cohere.

        Args:
            texts (List[str]): Texter att embedda

        Returns:
            List[List[float]]: Embedding-vektorer
        """
        response = self.client.embed(
            texts=texts, model=self.model, input_type="search_document"
        )
        return response.embeddings

    def get_dimension(self) -> int:
        """
        Hämta embedding dimension.

        Returns:
            int: Dimension (1024)
        """
        return self._dimension


class BGEBackend(EmbeddingBackend):
    """
    BGE-M3 embedding backend - State-of-the-art öppen källkod.
    Multilingual, utmärkt för RAG, hanterar långa texter.
    Tränad specifikt för informationssökning.
    """

    def __init__(self, model: str = "BAAI/bge-m3"):
        """
        Initiera BGE backend.

        Args:
            model (str): BGE modellnamn (default: "BAAI/bge-m3")
                Alternativ: "BAAI/bge-large-en-v1.5", "BAAI/bge-base-en-v1.5"
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )

        print(f"Loading BGE model {model}... (this may take a moment first time)")
        self.model = SentenceTransformer(model)
        self._dimension = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Skapa embeddings med BGE.

        Args:
            texts (List[str]): Texter att embedda

        Returns:
            List[List[float]]: Normaliserade embedding-vektorer
        """
        # BGE models benefit from instruction prefix for queries
        # For documents, we don't need prefix
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def get_dimension(self) -> int:
        """
        Hämta embedding dimension.

        Returns:
            int: Dimension från modellen (typically 1024)
        """
        return self._dimension


class E5Backend(EmbeddingBackend):
    """
    E5 embedding backend - Modern öppen källkod från Microsoft.
    Balanserad prestanda/kostnad, bra för svenska texter.
    Robust mot stavfel och ostrukturerad text.
    """

    def __init__(self, model: str = "intfloat/multilingual-e5-large"):
        """
        Initiera E5 backend.

        Args:
            model (str): E5 modellnamn (default: "intfloat/multilingual-e5-large")
                Alternativ: "intfloat/e5-large-v2", "intfloat/multilingual-e5-base"
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )

        print(f"Loading E5 model {model}... (this may take a moment first time)")
        self.model = SentenceTransformer(model)
        self._dimension = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Skapa embeddings med E5.

        E5 models förväntar sig prefix "passage: " för dokument
        och "query: " för sökfrågor. Vi använder passage för alla chunks.

        Args:
            texts (List[str]): Texter att embedda

        Returns:
            List[List[float]]: Normaliserade embedding-vektorer
        """
        # Add E5 prefix for better performance
        prefixed_texts = [f"passage: {text}" for text in texts]
        return self.model.encode(prefixed_texts, normalize_embeddings=True).tolist()

    def get_dimension(self) -> int:
        """
        Hämta embedding dimension.

        Returns:
            int: Dimension från modellen (1024 för large, 768 för base)
        """
        return self._dimension


def get_backend(kind: Optional[str] = None) -> EmbeddingBackend:
    """
    Hämta rätt embedding backend baserat på namn.

    Args:
        kind (Optional[str]): Backend-typ ("google", "openai", "sbert", "ollama",
                              "cohere", "bge", "e5")

    Returns:
        EmbeddingBackend: Instans av vald backend
    """
    backend = (kind or os.getenv("EMBED_BACKEND") or "google").lower()

    # Original backends
    if backend == "openai":
        return OpenAIBackend()
    if backend == "sbert":
        return SBERTBackend()
    if backend == "ollama":
        return OllamaBackend()
    if backend == "google":
        return GoogleBackend()

    # FAS 1: Nya moderna backends
    if backend == "cohere":
        return CohereBackend()
    if backend in ["bge", "bge-m3"]:
        return BGEBackend()
    if backend == "e5":
        return E5Backend()

    # fallback
    return GoogleBackend()


# ---------------- FAISS Vector Store ----------------


class FAISSVectorStore:
    """
    FAISS-baserad vector store för semantisk sökning.
    Lagrar embeddings och associerade records för snabb similarity search.
    """

    def __init__(self, dimension: int = 768, index_type: str = "flat"):
        """
        Initiera FAISS vector store.

        Args:
            dimension (int): Embedding-vektorns dimension (default: 768)
            index_type (str): Index-typ - 'flat' för exakt sökning,
                            'ivf' för approximativ (snabbare med många vektorer)
        """
        if not FAISS_AVAILABLE:
            raise ImportError("faiss not installed. Run: pip install faiss-cpu")

        self.dimension = dimension
        self.index_type = index_type

        # Create index
        if index_type == "flat":
            # Exact search using inner product (for normalized vectors = cosine similarity)
            self.index = faiss.IndexFlatIP(dimension)
        elif index_type == "ivf":
            # Approximate search (faster for large datasets)
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)  # 100 clusters
        else:
            raise ValueError(f"Unknown index_type: {index_type}")

        self.id_to_record = {}  # Store full records
        self.record_ids = []  # Keep track of record IDs in order
        self.is_trained = False

    def add_records(self, records: List[Dict]):
        """
        Lägg till records med embeddings till store.

        Args:
            records (List[Dict]): Lista med records som innehåller 'embedding' och andra metadata

        Returns:
            None
        """
        if not records:
            return

        vectors = []
        for r in records:
            # Normalize vector for cosine similarity
            vec = np.array(r["embedding"], dtype="float32")
            vec = vec / np.linalg.norm(vec)  # normalize
            vectors.append(vec)

            # Store record
            idx = len(self.record_ids)
            self.id_to_record[idx] = r
            self.record_ids.append(r["id"])

        vectors_array = np.array(vectors, dtype="float32")

        # Train index if needed (for IVF)
        if self.index_type == "ivf" and not self.is_trained:
            self.index.train(vectors_array)
            self.is_trained = True

        self.index.add(vectors_array)

    def search(self, query_embedding: List[float], k: int = 5) -> List[Dict]:
        """
        Sök efter k närmaste grannarna i vector store.

        Args:
            query_embedding (List[float]): Query-vektor att söka efter
            k (int): Antal resultat att returnera (default: 5)

        Returns:
            List[Dict]: Lista med dicts innehållande 'record', 'score' (cosine similarity), 'rank'
        """
        if self.index.ntotal == 0:
            return []

        # Normalize query vector
        query_vec = np.array([query_embedding], dtype="float32")
        query_vec = query_vec / np.linalg.norm(query_vec)

        # Search
        k = min(k, self.index.ntotal)  # Can't search for more than we have
        distances, indices = self.index.search(query_vec, k)

        results = []
        for rank, (score, idx) in enumerate(zip(distances[0], indices[0]), 1):
            if idx == -1:  # FAISS returns -1 for unfilled results
                continue
            if idx in self.id_to_record:
                results.append(
                    {
                        "record": self.id_to_record[idx],
                        "score": float(score),  # This is cosine similarity (0-1)
                        "rank": rank,
                    }
                )

        return results

    def search_with_text(
        self, query_text: str, k: int = 5, backend: Optional[EmbeddingBackend] = None
    ) -> List[Dict]:
        """
        Sök med text-query (embeddar automatiskt).

        Args:
            query_text (str): Textsträng att söka efter
            k (int): Antal resultat (default: 5)
            backend (Optional[EmbeddingBackend]): Embedding backend att använda

        Returns:
            List[Dict]: Sökresultat med records och scores
        """
        backend = backend or get_backend()
        query_embedding = backend.embed([query_text])[0]
        return self.search(query_embedding, k)

    def save(self, base_path: str):
        """
        Spara index och metadata till disk.

        Args:
            base_path (str): Bas-sökväg för filer (skapar .faiss och .pkl)

        Returns:
            None
        """
        base_path = Path(base_path)
        base_path.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(base_path) + ".faiss")

        # Save metadata
        metadata = {
            "id_to_record": self.id_to_record,
            "record_ids": self.record_ids,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "is_trained": self.is_trained,
        }
        with open(str(base_path) + ".pkl", "wb") as f:
            pickle.dump(metadata, f)

        print(f"✓ Saved vector store to {base_path}.faiss + .pkl")

    def load(self, base_path: str):
        """
        Ladda index och metadata från disk.

        Args:
            base_path (str): Bas-sökväg till sparade filer

        Returns:
            None
        """
        base_path = Path(base_path)

        # Load FAISS index
        self.index = faiss.read_index(str(base_path) + ".faiss")

        # Load metadata
        with open(str(base_path) + ".pkl", "rb") as f:
            metadata = pickle.load(f)

        self.id_to_record = metadata["id_to_record"]
        self.record_ids = metadata["record_ids"]
        self.dimension = metadata["dimension"]
        self.index_type = metadata["index_type"]
        self.is_trained = metadata["is_trained"]

        print(f"✓ Loaded vector store from {base_path} ({self.index.ntotal} vectors)")

    def get_stats(self) -> Dict:
        """
        Hämta statistik om vector store.

        Returns:
            Dict: Statistik med total_vectors, dimension, index_type, etc.
        """
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "is_trained": self.is_trained,
            "records_count": len(self.id_to_record),
        }


# ---------------- Helpers for scraping ----------------


def approx_token_count(text: str) -> int:
    """
    Uppskatta antal tokens i text.

    Args:
        text (str): Text att räkna tokens för

    Returns:
        int: Uppskattat antal tokens (~4 tecken per token)
    """
    return max(1, math.ceil(len(text) / 4))


def slugify(s: str) -> str:
    """
    Konvertera sträng till URL-säker slug.

    Args:
        s (str): Sträng att konvertera

    Returns:
        str: URL-säker slug (max 80 tecken)
    """
    s = re.sub(r"[^\w\s-]", "", s).strip().lower()
    s = re.sub(r"[\s_-]+", "-", s)
    return s[:80] or "chunk"


# ---------------- Steps ----------------


@dataclass
class ScrapeResult:
    """
    Dataclass för resultat från web scraping.

    Attributes:
        url (str): Ursprunglig URL
        base_url (str): Bas-URL (schema + domain)
        html (str): Hämtad HTML-kod
    """

    url: str
    base_url: str
    html: str


def scrape_url(url: str, timeout: int = 20) -> ScrapeResult:
    """
    Hämta HTML-innehåll från URL.

    Args:
        url (str): URL att hämta
        timeout (int): Timeout i sekunder (default: 20)

    Returns:
        ScrapeResult: Objekt med url, base_url och html
    """
    headers = {"User-Agent": "Mozilla/5.0 (compatible; RAG-Scraper/1.0; +local-dev)"}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    r.encoding = r.apparent_encoding or r.encoding
    base = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
    return ScrapeResult(url=url, base_url=base, html=r.text)


from bs4 import BeautifulSoup, Comment
import html


def extract_main_content(raw_html: str) -> str:
    """
    FAS 1 FÖRBÄTTRAD: Aggressiv extrahering av huvudinnehåll från HTML.
    Fokuserar på att ta bort så mycket brus som möjligt.

    Args:
        raw_html (str): Rå HTML-kod

    Returns:
        str: Rengjord HTML med huvudinnehåll
    """
    if not raw_html or not raw_html.strip():
        return "<html><body><p>Tomt innehåll</p></body></html>"

    try:
        soup = BeautifulSoup(raw_html, "html.parser")
    except Exception as e:
        raise Exception(f"Kunde inte parsa HTML: {str(e)}")

    # 1. Ta bort alla script, style, iframe, form, button, etc
    for tag in soup(
        [
            "script",
            "style",
            "noscript",
            "template",
            "svg",
            "iframe",
            "form",
            "button",
            "input",
            "select",
            "textarea",
        ]
    ):
        try:
            tag.decompose()
        except Exception:
            continue

    # 2. Ta bort HTML-kommentarer
    try:
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()
    except Exception:
        pass

    # 3. Försök hitta huvudinnehållet med vanliga selektorer
    main_content = None

    # Prioritetsordning av selektorer för att hitta huvudinnehåll
    content_selectors = [
        "main",
        "article",
        '[role="main"]',
        ".main-content",
        ".content",
        "#main-content",
        "#content",
        ".post-content",
        ".entry-content",
        ".article-content",
        ".article-body",
    ]

    for selector in content_selectors:
        try:
            main_content = soup.select_one(selector)
            if main_content and len(main_content.get_text(strip=True)) > 200:
                # Hitta något med substans
                break
        except Exception:
            continue

    # Om vi inte hittar huvudinnehåll, använd body men filtrera brus
    if not main_content:
        main_content = soup.find("body")
        if not main_content:
            main_content = soup

        # Ta bort explicita nav/footer/header/aside tags
        try:
            for tag in main_content.find_all(["nav", "footer", "header", "aside"]):
                tag.decompose()
        except Exception:
            pass

    # 4. AGGRESSIV brusfiltrering - ta bort element med brus-relaterade klasser/IDs
    noise_patterns = [
        "cookie",
        "gdpr",
        "consent",
        "advertisement",
        "ad-",
        "ads-",
        "sidebar",
        "side-bar",
        "widget",
        "navigation",
        "nav-",
        "menu",
        "footer",
        "header",
        "banner",
        "popup",
        "modal",
        "overlay",
        "social",
        "share",
        "sharing",
        "comment",
        "comments",
        "related",
        "recommend",
        "breadcrumb",
        "pagination",
        "pager",
        "search-box",
        "searchbox",
        "login",
        "signup",
        "subscribe",
        "newsletter",
        "promo",
        "sponsored",
        "tracking",
    ]

    try:
        for tag in list(main_content.find_all(True)):
            try:
                classes = " ".join(tag.get("class", [])).lower()
                id_ = (tag.get("id") or "").lower()

                # Kolla om något brus-mönster matchar
                if any(
                    pattern in classes or pattern in id_ for pattern in noise_patterns
                ):
                    tag.decompose()
                    continue

                # 5. Ta bort element med väldigt kort text (ofta brus)
                # Undantag: rubriker och list-items kan vara korta
                text = tag.get_text(strip=True)
                if (
                    len(text) < 20
                    and tag.name
                    not in ["h1", "h2", "h3", "h4", "h5", "h6", "li", "td", "th"]
                    and tag.name
                    not in ["p", "div", "span"]  # Behåll strukturella element
                ):
                    tag.decompose()
            except Exception:
                # Skippa problem-element
                continue
    except Exception:
        pass

    try:
        return html.unescape(str(main_content))
    except Exception as e:
        raise Exception(f"Kunde inte konvertera till sträng: {str(e)}")


def clean_html(raw_html: str) -> str:
    """
    Förbättrad version av clean_html med aggressiv brusrensning.

    Args:
        raw_html (str): Rå HTML-kod

    Returns:
        str: Rengjord HTML
    """
    return extract_main_content(raw_html)


def normalize_html(cleaned_html: str) -> BeautifulSoup:
    """
    Normalisera HTML genom att ta bort extra whitespace.

    Args:
        cleaned_html (str): Rengjord HTML

    Returns:
        BeautifulSoup: Normaliserad soup-objekt
    """
    if not cleaned_html or not cleaned_html.strip():
        # Returnera tom soup om input är tom
        return BeautifulSoup("<html><body></body></html>", "html.parser")

    soup = BeautifulSoup(cleaned_html, "html.parser")

    # Säker whitespace-normalisering
    for text_node in soup.find_all(string=True):
        try:
            if text_node.parent and text_node.parent.name in ("pre", "code"):
                continue
            new = re.sub(r"\s+", " ", text_node)
            text_node.replace_with(new)
        except Exception:
            # Skippa problem-noder
            continue

    # Ta bort tomma element (säkert)
    for tag in list(soup.find_all()):
        try:
            if tag.name not in ("img", "br") and not tag.get_text(strip=True):
                tag.decompose()
        except Exception:
            # Skippa problem-element
            continue

    return soup


def resolve_links(soup: BeautifulSoup, base_url: str) -> BeautifulSoup:
    """
    Gör relativa länkar till absoluta.

    Args:
        soup (BeautifulSoup): Soup-objekt med HTML
        base_url (str): Bas-URL för att lösa relativa länkar

    Returns:
        BeautifulSoup: Soup med absoluta länkar
    """
    for tag in soup.find_all(["a", "img", "script", "link", "source"]):
        attr = "href" if tag.name in ("a", "link") else "src"
        if tag.has_attr(attr):
            tag[attr] = urljoin(base_url, tag[attr])
    return soup


def html_to_markdown(soup: BeautifulSoup) -> str:
    """
    Konvertera HTML till Markdown.

    Args:
        soup (BeautifulSoup): HTML soup-objekt

    Returns:
        str: Markdown-formaterad text
    """
    return md(
        str(soup),
        heading_style="ATX",
        convert=[
            "br",
            "p",
            "img",
            "a",
            "table",
            "pre",
            "code",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
        ],
    )


@dataclass
class Block:
    """
    Dataclass för textblock med rubrik.

    Attributes:
        level (int): Rubriknivå (1-6)
        heading (str): Rubriktext
        content (str): Innehåll under rubriken
    """

    level: int
    heading: str
    content: str


def split_markdown_into_blocks(
    markdown_text: str, min_level: int = 1, max_level: int = 6
) -> List[Block]:
    """
    Dela upp Markdown i block baserat på rubriker.

    Args:
        markdown_text (str): Markdown-text att dela upp
        min_level (int): Minsta rubriknivå att behandla (default: 1)
        max_level (int): Högsta rubriknivå att behandla (default: 6)

    Returns:
        List[Block]: Lista med Block-objekt
    """
    lines = markdown_text.splitlines()
    pattern = re.compile(r"^(#{1,6})\s+(.*)$")
    blocks, curr_heading, curr_level, buff = [], None, None, []

    def flush():
        nonlocal blocks, curr_heading, curr_level, buff
        if curr_heading is not None:
            blocks.append(
                Block(
                    level=curr_level,
                    heading=curr_heading,
                    content="\n".join(buff).strip(),
                )
            )
        curr_heading, curr_level, buff = None, None, []

    for line in lines:
        m = pattern.match(line)
        if m:
            level = len(m.group(1))
            heading = m.group(2).strip()
            if min_level <= level <= max_level:
                flush()
                curr_heading, curr_level = heading, level
                buff = []
                continue
        buff.append(line)
    flush()
    return blocks


# ============= FAS 1: OVERLAPPING CHUNKS =============


def chunk_text_with_overlap(
    text: str, max_tokens: int = 512, overlap_tokens: int = 50
) -> List[str]:
    """
    FAS 1 FÖRBÄTTRAD: Dela upp text i chunks med överlapp för att bevara kontext.

    Overlapping chunks förhindrar att viktiga information förloras vid chunk-gränser.
    Detta är särskilt viktigt för RAG där kontext kan spänna över flera meningar.

    Args:
        text (str): Text att dela upp
        max_tokens (int): Max tokens per chunk (default: 512)
        overlap_tokens (int): Antal tokens överlapp mellan chunks (default: 50, ~10%)

    Returns:
        List[str]: Lista med överlappande text-chunks
    """
    if approx_token_count(text) <= max_tokens:
        return [text.strip()]

    chunks = []
    paragraphs = re.split(r"\n{2,}", text)

    current_chunk = []
    current_tokens = 0
    previous_overlap = []  # Spara text för överlapp

    for para in paragraphs:
        if not para.strip():
            continue

        para_tokens = approx_token_count(para)

        # Om ett enskilt stycke är större än max, dela upp det på meningar
        if para_tokens > max_tokens:
            sentences = re.split(r"(?<=[.!?])\s+", para)

            for sentence in sentences:
                if not sentence.strip():
                    continue

                sent_tokens = approx_token_count(sentence)

                # Om vi skulle överskrida gränsen, spara nuvarande chunk
                if current_tokens + sent_tokens > max_tokens and len(current_chunk) > 0:
                    # Spara chunk
                    chunk_text = "\n\n".join(current_chunk).strip()
                    chunks.append(chunk_text)

                    # Förbered nästa chunk med överlapp från föregående
                    current_chunk = previous_overlap.copy()
                    current_tokens = sum(
                        approx_token_count(s) for s in previous_overlap
                    )

                    # Lägg till mening
                    current_chunk.append(sentence)
                    current_tokens += sent_tokens
                else:
                    current_chunk.append(sentence)
                    current_tokens += sent_tokens

                # Uppdatera överlapp-buffer (behåll sista ~overlap_tokens)
                while (
                    len(previous_overlap) > 0
                    and sum(approx_token_count(s) for s in previous_overlap)
                    > overlap_tokens
                ):
                    previous_overlap.pop(0)
                previous_overlap.append(sentence)
                if len(previous_overlap) > 5:  # Max 5 meningar i överlapp
                    previous_overlap.pop(0)
        else:
            # Normalt stycke - behandla som förut men med överlapp
            if current_tokens + para_tokens > max_tokens and len(current_chunk) > 0:
                # Spara chunk
                chunk_text = "\n\n".join(current_chunk).strip()
                chunks.append(chunk_text)

                # Förbered nästa chunk med överlapp
                current_chunk = previous_overlap.copy()
                current_tokens = sum(approx_token_count(p) for p in previous_overlap)

            current_chunk.append(para)
            current_tokens += para_tokens

            # Uppdatera överlapp-buffer
            while (
                len(previous_overlap) > 0
                and sum(approx_token_count(p) for p in previous_overlap)
                > overlap_tokens
            ):
                previous_overlap.pop(0)
            previous_overlap.append(para)
            if len(previous_overlap) > 3:  # Max 3 stycken i överlapp
                previous_overlap.pop(0)

    # Lägg till sista chunken
    if current_chunk:
        chunk_text = "\n\n".join(current_chunk).strip()
        chunks.append(chunk_text)

    return chunks


def chunk_text(text: str, max_tokens: int = 512, hard_limit: int = 2048) -> List[str]:
    """
    ÄLDRE VERSION: Dela upp text i chunks baserat på token-gräns.
    Behålls för bakåtkompatibilitet.

    För nya implementationer, använd chunk_text_with_overlap() istället.

    Args:
        text (str): Text att dela upp
        max_tokens (int): Max tokens per chunk (default: 512)
        hard_limit (int): Hård gräns för enskild mening (default: 2048)

    Returns:
        List[str]: Lista med text-chunks
    """
    if approx_token_count(text) <= max_tokens:
        return [text.strip()]
    paragraphs = re.split(r"\n{2,}", text)
    chunks, cur = [], []

    def cur_tokens():
        return approx_token_count("\n\n".join(cur))

    for p in paragraphs:
        if not p.strip():
            continue
        if approx_token_count(p) > hard_limit:
            sentences = re.split(r"(?<=[.!?])\s+", p)
            for s in sentences:
                if not s.strip():
                    continue
                if cur and cur_tokens() + approx_token_count(s) > max_tokens:
                    chunks.append("\n\n".join(cur).strip())
                    cur = []
                cur.append(s)
        else:
            if cur and cur_tokens() + approx_token_count(p) > max_tokens:
                chunks.append("\n\n".join(cur).strip())
                cur = []
            cur.append(p)
    if cur:
        chunks.append("\n\n".join(cur).strip())
    return chunks


def _breadcrumbs(blocks: List[Block], idx: int) -> List[str]:
    """
    Bygg breadcrumb-trail för block baserat på rubriknivåer.

    Args:
        blocks (List[Block]): Lista med alla block
        idx (int): Index för aktuellt block

    Returns:
        List[str]: Lista med överordnade rubriker
    """
    me = blocks[idx]
    trail, cur_level = [], me.level
    for k in range(idx - 1, -1, -1):
        b = blocks[k]
        if b.level < cur_level:
            trail.append(b.heading)
            cur_level = b.level
    return list(reversed(trail))


def build_records(
    url: str,
    title: Optional[str],
    blocks: List[Block],
    max_tokens_per_chunk: int = 512,
    use_overlap: bool = True,
) -> List[Dict]:
    """
    Bygg records från blocks för RAG-system.

    FAS 1: Nu med stöd för overlapping chunks.

    Args:
        url (str): Käll-URL
        title (Optional[str]): Dokumenttitel
        blocks (List[Block]): Lista med textblock
        max_tokens_per_chunk (int): Max tokens per chunk (default: 512)
        use_overlap (bool): Använd overlapping chunks (default: True)

    Returns:
        List[Dict]: Lista med records innehållande metadata och text-chunks
    """
    records = []
    doc_id = slugify(title or urlparse(url).path or "document")

    for i, b in enumerate(blocks):
        block_id = f"{doc_id}--{slugify(b.heading) or f'block-{i}'}"

        # FAS 1: Använd overlapping chunks om aktiverat
        if use_overlap:
            chunks = chunk_text_with_overlap(
                b.content or "", max_tokens=max_tokens_per_chunk, overlap_tokens=50
            )
        else:
            chunks = chunk_text(b.content or "", max_tokens=max_tokens_per_chunk)

        for j, ch in enumerate(chunks):
            records.append(
                {
                    "id": f"{block_id}--{j}",
                    "url": url,
                    "title": title,
                    "heading": b.heading,
                    "level": b.level,
                    "markdown": ch,
                    "chunk_index": j,
                    "chunk_count": None,
                    "block_index": i,
                    "tokens_est": approx_token_count(ch),
                    "anchor": f"#{slugify(b.heading)}" if b.heading else None,
                    "breadcrumbs": _breadcrumbs(blocks, i),
                }
            )

    # Set chunk_count per block
    by_block = {}
    for r in records:
        by_block.setdefault(r["block_index"], []).append(r)
    for _, arr in by_block.items():
        n = len(arr)
        for r in arr:
            r["chunk_count"] = n
    return records


def embed_records(
    records: List[Dict],
    backend: Optional[EmbeddingBackend] = None,
    batch_size: int = 64,
) -> None:
    """
    Lägg till embeddings till records.

    Args:
        records (List[Dict]): Lista med records att embedda
        backend (Optional[EmbeddingBackend]): Embedding backend att använda
        batch_size (int): Antal records per batch (default: 64)

    Returns:
        None (modifierar records in-place)
    """
    backend = backend or get_backend()
    texts = [r["markdown"] for r in records]
    for start in range(0, len(texts), batch_size):
        vecs = backend.embed(texts[start : start + batch_size])
        for i, v in enumerate(vecs):
            records[start + i]["embedding"] = v


def process_url_for_rag(
    url: str,
    max_tokens_per_chunk: int = 512,
    embed_backend: Optional[str] = None,
    create_vector_store: bool = True,
    use_overlap: bool = True,
) -> Dict:
    """
    Komplett pipeline för att processa URL för RAG.
    Scrapar, rensar, chunkar, embeddar och skapar vector store.

    FAS 1: Med förbättrad HTML-rensning och overlapping chunks.

    Args:
        url (str): URL att processa
        max_tokens_per_chunk (int): Max tokens per chunk (default: 512)
        embed_backend (Optional[str]): Embedding backend att använda
        create_vector_store (bool): Om vector store ska skapas (default: True)
        use_overlap (bool): Använd overlapping chunks (default: True)

    Returns:
        Dict: Dictionary med source_url, title, records, och optionellt vector_store
    """
    try:
        scraped = scrape_url(url)
    except Exception as e:
        raise Exception(f"Kunde inte hämta URL: {str(e)}")

    try:
        cleaned = clean_html(scraped.html)
        soup = normalize_html(cleaned)
        soup = resolve_links(soup, scraped.base_url)
    except Exception as e:
        raise Exception(f"Kunde inte rensa HTML: {str(e)}")

    # Säker title-extraktion
    title = None
    try:
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        elif soup.find("h1"):
            # Fallback: använd första h1 som title
            title = soup.find("h1").get_text(strip=True)
    except Exception:
        pass  # title förblir None

    # Om title fortfarande är None, använd URL som fallback
    if not title:
        from urllib.parse import urlparse

        parsed = urlparse(url)
        title = parsed.netloc or "Untitled"

    try:
        markdown = html_to_markdown(soup)
    except Exception as e:
        raise Exception(f"Kunde inte konvertera till Markdown: {str(e)}")

    try:
        blocks = split_markdown_into_blocks(markdown, min_level=1, max_level=6)
        if not blocks:
            blocks = [Block(level=1, heading="Innehåll", content=markdown)]
    except Exception as e:
        raise Exception(f"Kunde inte dela upp i blocks: {str(e)}")

    try:
        records = build_records(
            scraped.url,
            title,
            blocks,
            max_tokens_per_chunk=max_tokens_per_chunk,
            use_overlap=use_overlap,
        )
    except Exception as e:
        raise Exception(f"Kunde inte bygga records: {str(e)}")

    try:
        backend = get_backend(embed_backend)
        embed_records(records, backend=backend)
    except Exception as e:
        raise Exception(f"Kunde inte skapa embeddings ({embed_backend}): {str(e)}")

    result = {"source_url": scraped.url, "title": title, "records": records}

    # Create vector store if requested
    if create_vector_store and FAISS_AVAILABLE:
        dimension = backend.get_dimension()
        vector_store = FAISSVectorStore(dimension=dimension)
        vector_store.add_records(records)
        result["vector_store"] = vector_store

    return result


def process_pdf_for_rag(
    pdf_bytes: bytes,
    filename: str,
    max_tokens_per_chunk: int = 512,
    embed_backend: Optional[str] = None,
    use_overlap: bool = True,
) -> Dict:
    """
    Processa PDF för RAG (Retrieval Augmented Generation).
    Extraherar text, skapar chunks, genererar embeddings och bygger vector store.

    FAS 1: Med overlapping chunks.

    Args:
        pdf_bytes (bytes): PDF-fil som bytes
        filename (str): Filnamn
        max_tokens_per_chunk (int): Max antal tokens per chunk (default: 512)
        embed_backend (Optional[str]): Embedding backend att använda
        use_overlap (bool): Använd overlapping chunks (default: True)

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
    full_text = normalize_text(full_text)
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
        use_overlap=use_overlap,
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


# Keep old functions for backwards compatibility
def cosine_similarity(vec1, vec2):
    """
    Beräkna cosine similarity mellan två vektorer.

    Args:
        vec1: Första vektorn (list eller numpy array)
        vec2: Andra vektorn (list eller numpy array)

    Returns:
        float: Cosine similarity (0-1)
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def create_embeddings(
    text, model="text-embedding-004", task_type="SEMANTIC_SIMILARITY"
):
    """
    Skapa embeddings med Google API (legacy funktion).

    Args:
        text (str): Text att embedda
        model (str): Modellnamn (default: "text-embedding-004")
        task_type (str): Task type (default: "SEMANTIC_SIMILARITY")

    Returns:
        Response-objekt från Google API
    """
    return client.models.embed_content(
        model=model, contents=text, config=types.EmbedContentConfig(task_type=task_type)
    )
