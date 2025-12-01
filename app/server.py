###################################################
#
#   file: server.py (FÖRBÄTTRAD MED FELHANTERING)
#
#   Holds the API endpoints with multi-URL support
#   Read and write towards the file system
#   Uses rag_pipeline.py
#
#   FÖRBÄTTRING: Alla exceptions fångas och returneras som
#   strukturerade felmeddelanden till frontend
#
###################################################

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Body,
    HTTPException,
)
from fastapi.responses import (
    JSONResponse,
    FileResponse,
    HTMLResponse,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import os
import time
import json
import re
import requests
import traceback

from typing import Dict, Optional, List
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
    getDefaultSystemPrompt,
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

# Track URLs per collection
collection_metadata: Dict[str, Dict] = {}


# -----------------------------
# Felhantering Helpers
# -----------------------------


def create_error_response(error: Exception, context: str = "") -> Dict:
    """
    Skapa strukturerat felmeddelande för frontend.

    Args:
        error: Exception-objektet
        context: Kontext där felet uppstod

    Returns:
        Dict med error-information
    """
    error_type = type(error).__name__
    error_msg = str(error)

    # Bygg användarvänligt meddelande
    user_message = f"Fel vid {context}: {error_msg}" if context else error_msg

    # Hantera specifika feltyper
    if "API key" in error_msg or "authentication" in error_msg.lower():
        user_message = f"API-nyckel saknas eller är ogiltig. Kontrollera din .env-fil.\n\nDetaljer: {error_msg}"
    elif "not installed" in error_msg or "No module named" in error_msg:
        missing_package = error_msg.split("'")[1] if "'" in error_msg else "paketet"
        user_message = f"Saknat paket: {missing_package}\n\nInstallera med: pip install {missing_package} --break-system-packages"
    elif "Connection" in error_msg or "timeout" in error_msg.lower():
        user_message = f"Nätverksfel: Kunde inte ansluta.\n\nDetaljer: {error_msg}"
    elif "Permission" in error_msg or "Access denied" in error_msg:
        user_message = (
            f"Åtkomstfel: Kontrollera filrättigheter.\n\nDetaljer: {error_msg}"
        )

    return {
        "success": False,
        "error": {
            "type": error_type,
            "message": user_message,
            "details": error_msg,
            "context": context,
            "traceback": traceback.format_exc() if os.getenv("DEBUG") else None,
        },
    }


# -----------------------------
# LLM Backends
# -----------------------------


class LLMBackend:
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


class GoogleLLMBackend(LLMBackend):
    def __init__(self, model: str = "gemini-2.0-flash"):
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
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
        except Exception as e:
            raise Exception(f"Google LLM fel: {str(e)}")


class OpenAILLMBackend(LLMBackend):
    def __init__(self, model: str = "gpt-4o-mini"):
        try:
            from openai import OpenAI

            self.client = OpenAI()
            self.model = model
        except Exception as e:
            raise Exception(f"OpenAI initialization fel: {str(e)}")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
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
        except Exception as e:
            raise Exception(f"OpenAI LLM fel: {str(e)}")


class OllamaLLMBackend(LLMBackend):
    def __init__(self, model: str = "llama3.2", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host.rstrip("/")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            r = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": user_prompt,
                    "system": system_prompt,
                    "stream": False,
                },
                timeout=120,
            )
            r.raise_for_status()
            return r.json()["response"]
        except requests.exceptions.ConnectionError:
            raise Exception(f"Ollama server är inte igång. Starta med: ollama serve")
        except Exception as e:
            raise Exception(f"Ollama LLM fel: {str(e)}")


def get_llm_backend(kind: Optional[str] = None) -> LLMBackend:
    try:
        backend = (kind or "google").lower()
        if backend == "openai":
            return OpenAILLMBackend()
        if backend == "ollama":
            return OllamaLLMBackend()
        return GoogleLLMBackend()
    except Exception as e:
        raise Exception(f"Kunde inte ladda LLM backend '{kind}': {str(e)}")


# -----------------------------
# FastAPI App
# -----------------------------

app = FastAPI(title="PDF → JSON Extractor + RAG Search")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    app.mount("/static", StaticFiles(directory=STATIC_DIR, html=False), name="static")
    app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
    app.mount(
        "/vector_stores", StaticFiles(directory=VECTOR_STORE_DIR), name="vector_stores"
    )
except Exception:
    pass


# -----------------------------
# Helper Functions
# -----------------------------


def load_collection_metadata(collection_name: str) -> Dict:
    """Load metadata for a collection"""
    meta_path = Path(OUTPUT_DIR) / f"{collection_name}.meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"indexed_urls": [], "total_records": 0}


def save_collection_metadata(collection_name: str, metadata: Dict):
    """Save metadata for a collection"""
    meta_path = Path(OUTPUT_DIR) / f"{collection_name}.meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


# -----------------------------
# API Endpoints
# -----------------------------


@app.post("/api/upload_pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    embed_backend: str = "google",
    max_tokens_per_chunk: int = 512,
    collection_name: Optional[str] = None,
):
    """
    Upload and process PDF for RAG. Supports adding to existing collections.
    """
    try:
        # Validera fil
        if not file.filename.lower().endswith(".pdf"):
            return JSONResponse(
                status_code=400,
                content=create_error_response(
                    Exception("Endast .pdf-filer tillåtna"), "filvalidering"
                ),
            )

        pdf_bytes = await file.read()
        safe_name_pdf = sanitize_basename(file.filename) + ".pdf"
        upload_path = os.path.join(UPLOAD_DIR, safe_name_pdf)

        # Spara uppladdad fil
        with open(upload_path, "wb") as out:
            out.write(pdf_bytes)

        # Processa PDF
        try:
            pkg = process_pdf_for_rag(
                pdf_bytes,
                filename=file.filename,
                max_tokens_per_chunk=max_tokens_per_chunk,
                embed_backend=embed_backend,
            )
        except Exception as e:
            return JSONResponse(
                status_code=500, content=create_error_response(e, "PDF-processering")
            )

        # Bestäm collection name
        is_new_collection = False
        if collection_name and collection_name.strip():
            collection_name = sanitize_basename(collection_name.strip())
        else:
            collection_name = f"Samling_{int(time.time())}"
            is_new_collection = True

        # Ladda eller skapa vector store
        try:
            vector_store_path = Path(VECTOR_STORE_DIR) / collection_name
            backend_obj = get_backend(embed_backend)
            dimension = backend_obj.get_dimension()

            if (
                vector_store_path.with_suffix(".faiss").exists()
                and not is_new_collection
            ):
                # Ladda befintlig
                if collection_name not in vector_stores:
                    store = FAISSVectorStore(dimension=dimension)
                    store.load(str(vector_store_path))
                    vector_stores[collection_name] = store
                    hlog(f"✓ Loaded existing vector store '{collection_name}'")
                else:
                    store = vector_stores[collection_name]

                store.add_records(pkg["records"])
                store.save(str(vector_store_path))
                hlog(
                    f"✓ Added {len(pkg['records'])} new records to '{collection_name}'"
                )
            else:
                # Skapa ny
                store = FAISSVectorStore(dimension=dimension)
                store.add_records(pkg["records"])
                store.save(str(vector_store_path))
                vector_stores[collection_name] = store
                hlog(
                    f"✓ Created new vector store '{collection_name}' with {len(pkg['records'])} vectors"
                )
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content=create_error_response(e, "vector store-skapande"),
            )

        # Uppdatera metadata
        metadata = load_collection_metadata(collection_name)
        if safe_name_pdf not in metadata.get("indexed_pdfs", []):
            metadata.setdefault("indexed_pdfs", []).append(safe_name_pdf)
        metadata["total_records"] = store.get_stats()["total_vectors"]
        metadata["last_updated"] = utc_timestamp()
        metadata["embed_backend"] = embed_backend or "google"
        save_collection_metadata(collection_name, metadata)

        # Spara individuell PDF-data
        stem = f"{collection_name}_{int(time.time())}"
        records_slim = []
        all_text = []

        for r in pkg["records"]:
            slim = {k: v for k, v in r.items() if k != "embedding"}
            records_slim.append(slim)
            h = r.get("heading") or ""
            body = r.get("markdown", "")
            if h:
                all_text.append(f"# {h}\n{body}\n")
            else:
                all_text.append(f"{body}\n")

        all_text_str = "\n".join(all_text).strip()

        meta_path = Path(OUTPUT_DIR) / f"{stem}.json"
        txt_path = Path(OUTPUT_DIR) / f"{stem}.txt"

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "source_file": safe_name_pdf,
                    "title": pkg["title"],
                    "records": records_slim,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(all_text_str)

        hlog(
            f"✓ PDF '{file.filename}' processed: {len(pkg['records'])} chunks as '{collection_name}'"
        )

        return JSONResponse(
            content={
                "success": True,
                "message": "OK",
                "source_file": safe_name_pdf,
                "title": pkg["title"],
                "collection_name": collection_name,
                "record_count": len(pkg["records"]),
                "total_records": metadata["total_records"],
                "total_vectors": store.get_stats()["total_vectors"],
                "indexed_pdfs": metadata["indexed_pdfs"],
                "outputs": {
                    "records_json": f"/outputs/{meta_path.name}",
                    "text_file": f"/outputs/{txt_path.name}",
                },
                "vector_store": {
                    "available": True,
                    "stats": store.get_stats(),
                },
                "params": {
                    "max_tokens_per_chunk": max_tokens_per_chunk,
                    "embed_backend": embed_backend,
                },
            },
            status_code=200,
        )

    except Exception as e:
        # Catch-all för oväntade fel
        hlog(f"❌ Unexpected error in upload_pdf: {str(e)}")
        return JSONResponse(
            status_code=500,
            content=create_error_response(e, "PDF-uppladdning (oväntat fel)"),
        )


@app.post("/api/fetch_url")
async def api_fetch_url(payload: dict = Body(...)):
    """
    Fetch and process URL for RAG. Supports adding to existing collections.
    """
    try:
        url = (payload.get("url") or "").strip()
        if not url or not url.lower().startswith(("http://", "https://")):
            return JSONResponse(
                status_code=400,
                content=create_error_response(
                    Exception("Ogiltig URL. Måste börja med http:// eller https://"),
                    "URL-validering",
                ),
            )

        max_tokens = int(payload.get("max_tokens_per_chunk") or 512)
        embed_backend = payload.get("embed_backend")
        collection_name = payload.get("collection_name")

        # Processa URL
        try:
            pkg = process_url_for_rag(
                url,
                max_tokens_per_chunk=max_tokens,
                embed_backend=embed_backend,
                create_vector_store=False,
            )
        except Exception as e:
            return JSONResponse(
                status_code=502,
                content=create_error_response(e, f"URL-hämtning ({url})"),
            )

        # Bestäm collection name
        is_new_collection = False
        if collection_name and collection_name.strip():
            collection_name = sanitize_basename(collection_name.strip())
        else:
            collection_name = f"Samling_{int(time.time())}"
            is_new_collection = True

        collection_name = sanitize_basename(collection_name)

        # Ladda eller skapa vector store
        try:
            vector_store_path = Path(VECTOR_STORE_DIR) / collection_name
            backend_obj = get_backend(embed_backend)
            dimension = backend_obj.get_dimension()

            if (
                vector_store_path.with_suffix(".faiss").exists()
                and not is_new_collection
            ):
                if collection_name not in vector_stores:
                    store = FAISSVectorStore(dimension=dimension)
                    store.load(str(vector_store_path))
                    vector_stores[collection_name] = store
                    hlog(f"✓ Loaded existing vector store '{collection_name}'")
                else:
                    store = vector_stores[collection_name]

                store.add_records(pkg["records"])
                store.save(str(vector_store_path))
                hlog(
                    f"✓ Added {len(pkg['records'])} new records to '{collection_name}'"
                )
            else:
                store = FAISSVectorStore(dimension=dimension)
                store.add_records(pkg["records"])
                store.save(str(vector_store_path))
                vector_stores[collection_name] = store
                hlog(
                    f"✓ Created new vector store '{collection_name}' with {len(pkg['records'])} vectors"
                )
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content=create_error_response(e, "vector store-skapande"),
            )

        # Uppdatera metadata
        metadata = load_collection_metadata(collection_name)
        if url not in metadata.get("indexed_urls", []):
            metadata.setdefault("indexed_urls", []).append(url)
        metadata["total_records"] = store.get_stats()["total_vectors"]
        metadata["last_updated"] = utc_timestamp()
        metadata["embed_backend"] = embed_backend or "google"
        save_collection_metadata(collection_name, metadata)

        # Spara individuell URL-data
        stem = f"{collection_name}_{int(time.time())}"
        records_slim = []
        for r in pkg["records"]:
            slim = {k: v for k, v in r.items() if k != "embedding"}
            records_slim.append(slim)

        meta_path = Path(OUTPUT_DIR) / f"{stem}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "url": pkg["source_url"],
                    "title": pkg["title"],
                    "records": records_slim,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        return JSONResponse(
            content={
                "success": True,
                "message": "OK",
                "source_url": pkg["source_url"],
                "title": pkg["title"],
                "collection_name": collection_name,
                "record_count": len(pkg["records"]),
                "total_records": metadata["total_records"],
                "total_vectors": store.get_stats()["total_vectors"],
                "indexed_urls": metadata["indexed_urls"],
                "vector_store": {
                    "available": True,
                    "stats": store.get_stats(),
                },
                "params": {
                    "max_tokens_per_chunk": max_tokens,
                    "embed_backend": embed_backend or "google",
                },
            }
        )

    except Exception as e:
        hlog(f"❌ Unexpected error in fetch_url: {str(e)}")
        return JSONResponse(
            status_code=500,
            content=create_error_response(e, "URL-hämtning (oväntat fel)"),
        )


@app.get("/api/collection_info/{collection_name}")
def get_collection_info(collection_name: str):
    """Get information about a collection including all indexed URLs"""
    try:
        collection_name = sanitize_basename(collection_name)

        vector_store_path = Path(VECTOR_STORE_DIR) / collection_name
        if not vector_store_path.with_suffix(".faiss").exists():
            return JSONResponse(
                status_code=404,
                content=create_error_response(
                    Exception(f"Samlingen '{collection_name}' finns inte"),
                    "collection lookup",
                ),
            )

        metadata = load_collection_metadata(collection_name)

        if collection_name not in vector_stores:
            store = FAISSVectorStore()
            store.load(str(vector_store_path))
            vector_stores[collection_name] = store

        stats = vector_stores[collection_name].get_stats()

        return JSONResponse(
            content={
                "success": True,
                "collection_name": collection_name,
                "indexed_urls": metadata.get("indexed_urls", []),
                "indexed_pdfs": metadata.get("indexed_pdfs", []),
                "total_records": metadata.get("total_records", stats["total_vectors"]),
                "total_vectors": stats["total_vectors"],
                "last_updated": metadata.get("last_updated"),
                "embed_backend": metadata.get("embed_backend", "unknown"),
                "vector_store": {
                    "available": True,
                    "stats": stats,
                },
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=create_error_response(e, "collection info-hämtning"),
        )


@app.post("/api/search")
async def api_search(payload: dict = Body(...)):
    try:
        query = payload.get("query", "").strip()
        collection = payload.get("collection", "").strip()
        k = int(payload.get("k", 5))
        embed_backend = payload.get("embed_backend")

        if not query:
            return JSONResponse(
                status_code=400,
                content=create_error_response(Exception("Sökfråga saknas"), "sökning"),
            )
        if not collection:
            return JSONResponse(
                status_code=400,
                content=create_error_response(Exception("Samling saknas"), "sökning"),
            )

        # Ladda collection
        if collection not in vector_stores:
            vector_store_path = Path(VECTOR_STORE_DIR) / collection
            if not vector_store_path.with_suffix(".faiss").exists():
                return JSONResponse(
                    status_code=404,
                    content=create_error_response(
                        Exception(f"Samlingen '{collection}' finns inte"), "sökning"
                    ),
                )
            try:
                store = FAISSVectorStore()
                store.load(str(vector_store_path))
                vector_stores[collection] = store
                hlog(f"✓ Loaded vector store '{collection}' from disk")
            except Exception as e:
                return JSONResponse(
                    status_code=500,
                    content=create_error_response(e, "vector store-laddning"),
                )

        store = vector_stores[collection]

        # Sök
        try:
            backend = get_backend(embed_backend)
            results = store.search_with_text(query, k=k, backend=backend)
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content=create_error_response(e, "sökning i vector store"),
            )

        # Formatera resultat
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
                "success": True,
                "query": query,
                "collection": collection,
                "k": k,
                "results_count": len(formatted_results),
                "results": formatted_results,
                "store_stats": store.get_stats(),
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500, content=create_error_response(e, "sökning (oväntat fel)")
        )


@app.post("/api/ask")
async def api_ask(payload: dict = Body(...)):
    try:
        query = payload.get("query", "").strip()
        collection = payload.get("collection", "").strip()
        k = int(payload.get("k", 5))
        llm_backend_name = payload.get("llm_backend", "google")
        embed_backend_name = payload.get("embed_backend")
        custom_system_prompt = payload.get("system_prompt")

        if not query:
            return JSONResponse(
                status_code=400,
                content=create_error_response(Exception("Fråga saknas"), "AI-fråga"),
            )
        if not collection:
            return JSONResponse(
                status_code=400,
                content=create_error_response(Exception("Samling saknas"), "AI-fråga"),
            )

        # Ladda collection
        if collection not in vector_stores:
            vector_store_path = Path(VECTOR_STORE_DIR) / collection
            if not vector_store_path.with_suffix(".faiss").exists():
                return JSONResponse(
                    status_code=404,
                    content=create_error_response(
                        Exception(f"Samlingen '{collection}' finns inte"), "AI-fråga"
                    ),
                )
            try:
                store = FAISSVectorStore()
                store.load(str(vector_store_path))
                vector_stores[collection] = store
            except Exception as e:
                return JSONResponse(
                    status_code=500,
                    content=create_error_response(e, "vector store-laddning"),
                )

        store = vector_stores[collection]

        # Sök relevanta dokument
        try:
            embed_backend = get_backend(embed_backend_name)
            search_results = store.search_with_text(query, k=k, backend=embed_backend)
        except Exception as e:
            return JSONResponse(
                status_code=500, content=create_error_response(e, "dokumentsökning")
            )

        if not search_results:
            return JSONResponse(
                content={
                    "success": True,
                    "query": query,
                    "answer": "Inga relevanta dokument hittades.",
                    "sources": [],
                    "context_used": "",
                }
            )

        # Bygg kontext
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

        user_prompt = f"""KONTEXT:
{context}

---

FRÅGA: {query}

Svara på frågan baserat på kontexten ovan."""

        system_prompt = custom_system_prompt or getDefaultSystemPrompt()

        # Generera svar med LLM
        try:
            llm = get_llm_backend(llm_backend_name)
            answer = llm.generate(system_prompt, user_prompt)
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content=create_error_response(
                    e, f"LLM-generering ({llm_backend_name})"
                ),
            )

        return JSONResponse(
            content={
                "success": True,
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
    except Exception as e:
        return JSONResponse(
            status_code=500, content=create_error_response(e, "AI-fråga (oväntat fel)")
        )


@app.get("/api/collections")
def list_collections():
    try:
        collections = []
        for item in Path(VECTOR_STORE_DIR).glob("*.faiss"):
            name = item.stem
            is_loaded = name in vector_stores
            stats = vector_stores[name].get_stats() if is_loaded else None

            metadata = load_collection_metadata(name)
            url_count = len(metadata.get("indexed_urls", []))
            pdf_count = len(metadata.get("indexed_pdfs", []))

            collections.append(
                {
                    "name": name,
                    "loaded": is_loaded,
                    "stats": stats,
                    "url_count": url_count,
                    "pdf_count": pdf_count,
                }
            )

        return JSONResponse(
            content={
                "success": True,
                "collections": collections,
                "loaded_count": len(vector_stores),
                "total_count": len(collections),
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500, content=create_error_response(e, "collection-listning")
        )


@app.delete("/api/collection/{collection_name}")
def delete_collection(collection_name: str):
    try:
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

        meta_path = Path(OUTPUT_DIR) / f"{collection_name}.meta.json"
        if meta_path.exists():
            meta_path.unlink()
            deleted_files.append(str(meta_path.name))

        if not deleted_files:
            return JSONResponse(
                status_code=404,
                content=create_error_response(
                    Exception(f"Samlingen '{collection_name}' finns inte"),
                    "collection-radering",
                ),
            )

        return JSONResponse(
            content={
                "success": True,
                "message": f"Collection '{collection_name}' deleted",
                "deleted_files": deleted_files,
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500, content=create_error_response(e, "collection-radering")
        )


@app.get("/api/download/{basename}")
def download_json(basename: str):
    try:
        safe = sanitize_basename(basename)
        path = os.path.join(OUTPUT_DIR, f"{safe}.json")
        if not os.path.exists(path):
            return JSONResponse(
                status_code=404,
                content=create_error_response(
                    Exception(f"Filen '{safe}.json' finns inte"), "fil-nedladdning"
                ),
            )
        return FileResponse(
            path, media_type="application/json", filename=f"{safe}.json"
        )
    except Exception as e:
        return JSONResponse(
            status_code=500, content=create_error_response(e, "fil-nedladdning")
        )


@app.get("/api/health")
def health():
    return {"status": "ok", "success": True}


@app.get("/")
def root_index():
    try:
        return HTMLResponse(
            open(os.path.join(STATIC_DIR, "index.html"), encoding="utf-8").read()
        )
    except Exception as e:
        return HTMLResponse(
            f"<h1>Fel: {str(e)}</h1><p>Kunde inte ladda index.html</p>", status_code=500
        )


@app.get("/index.html")
def index_alias():
    try:
        return HTMLResponse(
            open(os.path.join(STATIC_DIR, "index.html"), encoding="utf-8").read()
        )
    except Exception as e:
        return HTMLResponse(
            f"<h1>Fel: {str(e)}</h1><p>Kunde inte ladda index.html</p>", status_code=500
        )
