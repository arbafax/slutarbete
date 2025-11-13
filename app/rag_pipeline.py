###################################################
#
#   file: rag_pipeline.py
#
#   Helper classes for RAG as well as a pipeline for a RAG Process
#   File partly created by ChatGPT 5.0
#
###################################################


import numpy as np

import os, re, math, json, html, time
from dataclasses import dataclass
from typing import List, Dict, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from google import genai
from google.genai import types


## create an openAI client
my_api_key = "AIzaSyBxHq5cPvm-0GGo8fHtPLDNDihB9X_oUlM"
client = genai.Client(api_key=my_api_key)


# ---------------- Embeddings-backends ----------------


class EmbeddingBackend:
    def embed(self, texts: List[str]) -> List[List[float]]:
        print("##### EmbeddingBackend.embed()")
        raise NotImplementedError


class OpenAIBackend(EmbeddingBackend):
    def __init__(self, model: str = "text-embedding-3-large"):
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model

    def embed(self, texts: List[str]) -> List[List[float]]:
        print("##### OpenAIBackend.embed()")
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]


class SBERTBackend(EmbeddingBackend):
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model)

    def embed(self, texts: List[str]) -> List[List[float]]:
        print("##### SBERTBackend.embed()")
        return self.model.encode(texts, normalize_embeddings=True).tolist()


class OllamaBackend(EmbeddingBackend):
    def __init__(
        self, model: str = "nomic-embed-text", host: str = "http://localhost:11434"
    ):
        self.model = model
        self.host = host.rstrip("/")

    def embed(self, texts: List[str]) -> List[List[float]]:
        print("##### OllamaBackend.embed()")
        out = []
        for t in texts:
            r = requests.post(
                f"{self.host}/api/embeddings", json={"model": self.model, "prompt": t}
            )
            r.raise_for_status()
            out.append(r.json()["embedding"])
        return out


def get_backend(kind: Optional[str] = None) -> EmbeddingBackend:
    print("### get_backend()")
    backend = (kind or os.getenv("EMBED_BACKEND") or "openai").lower()
    if backend == "openai":
        return OpenAIBackend()
    if backend == "sbert":
        return SBERTBackend()
    if backend == "ollama":
        return OllamaBackend()

    ## fulkodat
    return SBERTBackend()
    # raise ValueError("Unknown EMBED_BACKEND; use openai | sbert | ollama")


# ---------------- Helpers for scraping ----------------


def approx_token_count(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))  # ~4 chars/token


def slugify(s: str) -> str:
    s = re.sub(r"[^\w\s-]", "", s).strip().lower()
    s = re.sub(r"[\s_-]+", "-", s)
    return s[:80] or "chunk"


def is_probably_nav_or_footer(tag) -> bool:
    classes = " ".join(tag.get("class", [])).lower()
    id_ = (tag.get("id") or "").lower()
    bad = [
        "cookie",
        "gdpr",
        "consent",
        "banner",
        "nav",
        "navbar",
        "menu",
        "footer",
        "subscribe",
        "newsletter",
        "share",
        "social",
    ]
    return any(w in classes or w in id_ for w in bad)


# ---------------- Steps ----------------


@dataclass
class ScrapeResult:
    url: str
    base_url: str
    html: str


def scrape_url(url: str, timeout: int = 20) -> ScrapeResult:
    print("### scrape_url()")
    headers = {"User-Agent": "Mozilla/5.0 (compatible; RAG-Scraper/1.0; +local-dev)"}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    r.encoding = r.apparent_encoding or r.encoding
    base = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
    return ScrapeResult(url=url, base_url=base, html=r.text)


def clean_html(raw_html: str) -> str:
    print("### clean_html()")
    soup = BeautifulSoup(raw_html, "html.parser")
    for tag in soup(["script", "style", "noscript", "template", "svg"]):
        tag.decompose()
    for tag in list(soup.find_all(True)):
        try:
            if is_probably_nav_or_footer(tag):
                tag.decompose()
        except Exception:
            pass
    return html.unescape(str(soup))


def normalize_html(cleaned_html: str) -> BeautifulSoup:
    print("### normalize_html()")
    soup = BeautifulSoup(cleaned_html, "html.parser")
    for text_node in soup.find_all(string=True):
        if text_node.parent and text_node.parent.name in ("pre", "code"):
            continue
        new = re.sub(r"\s+", " ", text_node)
        text_node.replace_with(new)
    for tag in list(soup.find_all()):
        if tag.name not in ("img", "br") and not tag.get_text(strip=True):
            tag.decompose()
    return soup


def resolve_links(soup: BeautifulSoup, base_url: str) -> BeautifulSoup:
    print("### resolve_links()")
    for tag in soup.find_all(["a", "img", "script", "link", "source"]):
        attr = "href" if tag.name in ("a", "link") else "src"
        if tag.has_attr(attr):
            tag[attr] = urljoin(base_url, tag[attr])
    return soup


def html_to_markdown(soup: BeautifulSoup) -> str:
    print("### html_to_markdown()")
    return md(
        str(soup),
        heading_style="ATX",
        # strip=["script", "style"],
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
    level: int
    heading: str
    content: str


def split_markdown_into_blocks(
    markdown_text: str, min_level: int = 1, max_level: int = 6
):
    print("### split_markdown_into_blocks()")
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


def chunk_text(text: str, max_tokens: int = 512, hard_limit: int = 2048) -> List[str]:
    print("### chunk_text()")
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
    print("### _breadcrumbs()")
    me = blocks[idx]
    trail, cur_level = [], me.level
    for k in range(idx - 1, -1, -1):
        b = blocks[k]
        if b.level < cur_level:
            trail.append(b.heading)
            cur_level = b.level
    return list(reversed(trail))


def build_records(
    url: str, title: Optional[str], blocks: List[Block], max_tokens_per_chunk: int = 512
) -> List[Dict]:
    print("### build_records()")
    records = []
    doc_id = slugify(title or urlparse(url).path or "document")
    for i, b in enumerate(blocks):
        block_id = f"{doc_id}--{slugify(b.heading) or f'block-{i}'}"
        for j, ch in enumerate(
            chunk_text(b.content or "", max_tokens=max_tokens_per_chunk)
        ):
            records.append(
                {
                    "id": f"{block_id}--{j}",
                    "url": url,
                    "title": title,
                    "heading": b.heading,
                    "level": b.level,
                    "markdown": ch,
                    "chunk_index": j,
                    "chunk_count": None,  # sätts ev senare
                    "block_index": i,
                    "tokens_est": approx_token_count(ch),
                    "anchor": f"#{slugify(b.heading)}" if b.heading else None,
                    "breadcrumbs": _breadcrumbs(blocks, i),
                }
            )
    # sätt chunk_count per block
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
    print("### embed_records()")
    backend = backend or get_backend()
    texts = [r["markdown"] for r in records]
    for start in range(0, len(texts), batch_size):
        vecs = backend.embed(texts[start : start + batch_size])
        for i, v in enumerate(vecs):
            records[start + i]["embedding"] = v


def process_url_for_rag(
    url: str, max_tokens_per_chunk: int = 512, embed_backend: Optional[str] = None
) -> Dict:
    print("### process_url_for_rag()")
    scraped = scrape_url(url)
    cleaned = clean_html(scraped.html)
    soup = normalize_html(cleaned)
    soup = resolve_links(soup, scraped.base_url)
    title = soup.title.string.strip() if (soup.title and soup.title.string) else None
    markdown = html_to_markdown(soup)
    blocks = split_markdown_into_blocks(markdown, min_level=1, max_level=6)
    if not blocks:
        blocks = [Block(level=1, heading="Innehåll", content=markdown)]
    records = build_records(
        scraped.url, title, blocks, max_tokens_per_chunk=max_tokens_per_chunk
    )
    backend = get_backend(embed_backend)
    embed_records(records, backend=backend)
    return {"source_url": scraped.url, "title": title, "records": records}


def cosine_similarity(vec1, vec2):
    """
    Calculates the cosine similarity between two vectors.

    Args:
    vec1 (np.ndarray): The first vector.
    vec2 (np.ndarray): The second vector.

    Returns:
    float: The cosine similarity between the two vectors.
    """
    print("### cosine_similarity()")
    # Compute the dot product of the two vectors and divide by the product of their norms
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    ### end cosine_similarity


def semantic_search(query, text_chunks, embeddings, k=5):
    """
    Performs semantic search on the text chunks using the given query and embeddings.

    Args:
    query (str): The query for the semantic search.
    text_chunks (List[str]): A list of text chunks to search through.
    embeddings (List[dict]): A list of embeddings for the text chunks.
    k (int): The number of top relevant text chunks to return. Default is 5.

    Returns:
    List[str]: A list of the top k most relevant text chunks based on the query.
    """
    # Create an embedding for the query
    query_embedding = create_embeddings(query).data[0].embedding
    similarity_scores = []  # Initialize a list to store similarity scores

    # Calculate similarity scores between the query embedding and each text chunk embedding
    for i, chunk_embedding in enumerate(embeddings):
        similarity_score = cosine_similarity(
            np.array(query_embedding), np.array(chunk_embedding.embedding)
        )
        similarity_scores.append(
            (i, similarity_score)
        )  # Append the index and similarity score

    # Sort the similarity scores in descending order
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    # Get the indices of the top k most similar text chunks
    top_indices = [index for index, _ in similarity_scores[:k]]
    # Return the top k most relevant text chunks
    return [text_chunks[index] for index in top_indices]
    ### end semantic_search


def create_embeddings(
    text,
    model="text-embedding-004",
    task_type="SEMANTIC_SIMILARITY",  ## observera att vi använder en annan modell när vi embeddar än när vi chattar
):
    return client.models.embed_content(
        model=model, contents=text, config=types.EmbedContentConfig(task_type=task_type)
    )
    ### end create_embeddings
