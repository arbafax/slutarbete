**VIKTIGT** Denna dokumentation är automatgenererad och _**inte**_ verifierad 2025-12-01 15:01

# RAG Search System - Teknisk dokumentation

Teknisk dokumentation för utvecklare och avancerade användare.

## Innehållsförteckning

- [Systemarkitektur](#systemarkitektur)
- [Teknisk stack](#teknisk-stack)
- [Kodstruktur](#kodstruktur)
- [API-dokumentation](#api-dokumentation)
- [Embedding-backends](#embedding-backends)
- [LLM-backends](#llm-backends)
- [Vector Store](#vector-store)
- [RAG Pipeline](#rag-pipeline)
- [Utvecklingsguide](#utvecklingsguide)
- [Konfiguration](#konfiguration)
- [Prestanda och optimering](#prestanda-och-optimering)

---

## Systemarkitektur

### Övergripande arkitektur

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (HTML/JS)                    │
│              - Material Design Interface                 │
│              - Drag & Drop Upload                        │
│              - Real-time Progress Tracking               │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP/REST
                         ▼
┌─────────────────────────────────────────────────────────┐
│                  FastAPI Server (server.py)              │
│              - RESTful API Endpoints                     │
│              - File Upload Handling                      │
│              - Error Management                          │
│              - CORS Support                              │
└────────────┬──────────────────────────┬─────────────────┘
             │                          │
             ▼                          ▼
┌────────────────────────┐   ┌──────────────────────────┐
│   RAG Pipeline         │   │   Helper Functions       │
│   (rag_pipeline.py)    │   │   (helpers.py)           │
│   - Web Scraping       │   │   - Text Normalization   │
│   - PDF Processing     │   │   - File Management      │
│   - Text Chunking      │   │   - API Key Management   │
│   - Embeddings         │   │   - Logging              │
│   - Vector Store       │   └──────────────────────────┘
└────────────┬───────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│              External Services                           │
│   - Google Gemini API (Embeddings & LLM)                │
│   - OpenAI API (Optional)                               │
│   - Cohere API (Optional)                               │
│   - Ollama (Local, Optional)                            │
└─────────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│              Local Storage                               │
│   - uploads/      (Temporary file storage)              │
│   - outputs/      (Generated JSON files)                │
│   - vector_stores/ (FAISS indices)                      │
│   - static/       (Frontend assets)                     │
└─────────────────────────────────────────────────────────┘
```

### Dataflöde

#### PDF-uppladdning och indexering:
```
User → Upload PDF → server.py → rag_pipeline.py
                                      ↓
                         extract_text_from_pdf()
                                      ↓
                         normalize_text()
                                      ↓
                         split_into_blocks()
                                      ↓
                         chunk_text_with_overlap()
                                      ↓
                         embed_records()
                                      ↓
                         FAISSVectorStore.add_records()
                                      ↓
                         save to vector_stores/
```

#### URL-scraping och indexering:
```
User → Submit URL → server.py → rag_pipeline.py
                                      ↓
                         scrape_url()
                                      ↓
                         clean_html()
                                      ↓
                         html_to_markdown()
                                      ↓
                         split_markdown_into_blocks()
                                      ↓
                         build_records()
                                      ↓
                         embed_records()
                                      ↓
                         FAISSVectorStore.add_records()
```

#### Semantisk sökning:
```
User → Search Query → server.py → FAISSVectorStore.search_with_text()
                                      ↓
                         embed query with backend
                                      ↓
                         FAISS similarity search
                                      ↓
                         Return top-k results
                                      ↓
                         Display to user
```

#### AI-fråga:
```
User → Question → server.py → FAISSVectorStore.search_with_text()
                                      ↓
                         Retrieve relevant chunks
                                      ↓
                         Build context from chunks
                                      ↓
                         LLMBackend.generate()
                                      ↓
                         Return answer + sources
```

---

## Teknisk stack

### Backend
- **Python 3.9+**
- **FastAPI** - Modern web framework
- **Uvicorn** - ASGI server
- **Pydantic** - Data validation

### Document Processing
- **PyMuPDF (fitz)** - PDF text extraction
- **BeautifulSoup4** - HTML parsing
- **markdownify** - HTML to Markdown conversion
- **requests** - HTTP client för web scraping

### AI & Machine Learning
- **FAISS** - Vector similarity search (Facebook AI)
- **NumPy** - Numerical computing
- **google-genai** - Google Gemini API client
- **openai** (optional) - OpenAI API client
- **sentence-transformers** (optional) - Local embeddings
- **cohere** (optional) - Cohere API client

### Frontend
- **Vanilla JavaScript** - No frameworks
- **Material Design** - UI components
- **Fetch API** - HTTP requests

---

## Kodstruktur

### Huvudfiler

#### `server.py`
FastAPI-server med alla API-endpoints.

**Klasser:**
- `LLMBackend` - Abstrakt basklass för LLM-backends
- `GoogleLLMBackend` - Google Gemini LLM
- `OpenAILLMBackend` - OpenAI GPT
- `OllamaLLMBackend` - Ollama lokal LLM

**Huvudfunktioner:**
- `create_error_response()` - Strukturerad felhantering
- `get_llm_backend()` - Factory för LLM-backends
- `load_collection_metadata()` - Ladda samlingens metadata
- `save_collection_metadata()` - Spara samlingens metadata

**Endpoints:**
- `POST /api/upload_pdf` - Ladda upp PDF
- `POST /api/process_url` - Processa URL
- `POST /api/search` - Semantisk sökning
- `POST /api/ask` - AI-fråga
- `GET /api/collections` - Lista samlingar
- `DELETE /api/collection/{name}` - Ta bort samling
- `GET /api/download/{basename}` - Ladda ner JSON
- `GET /api/health` - Hälsokontroll

#### `rag_pipeline.py`
RAG-pipeline med all dokument- och embeddingsbearbetning.

**Klasser:**

**Embedding Backends:**
- `EmbeddingBackend` - Abstrakt basklass
- `OpenAIBackend` - OpenAI embeddings
- `SBERTBackend` - Sentence-BERT embeddings
- `OllamaBackend` - Ollama embeddings
- `GoogleBackend` - Google Gemini embeddings
- `CohereBackend` - Cohere v3 embeddings
- `BGEM3Backend` - BGE-M3 embeddings
- `E5Backend` - E5 multilingual embeddings

**Data-strukturer:**
- `ScrapedContent` - Dataclass för scrapad content
- `Block` - Dataclass för text-block med heading
- `FAISSVectorStore` - FAISS vector store wrapper

**Huvudfunktioner:**
- `get_backend()` - Factory för embedding backends
- `scrape_url()` - Scrapa webbsida
- `clean_html()` - Rensa HTML från skräp
- `normalize_html()` - Normalisera HTML-struktur
- `html_to_markdown()` - Konvertera HTML till Markdown
- `split_markdown_into_blocks()` - Dela upp Markdown i blocks
- `chunk_text()` - Enkel chunking
- `chunk_text_with_overlap()` - Chunking med överlapp
- `build_records()` - Bygg records från blocks
- `embed_records()` - Generera embeddings för records
- `process_url_for_rag()` - Komplett URL-pipeline
- `process_pdf_for_rag()` - Komplett PDF-pipeline

#### `helpers.py`
Hjälpfunktioner och konfiguration.

**Konstanter:**
- `BASE_DIR`, `UPLOAD_DIR`, `OUTPUT_DIR`, etc. - Mappsökvägar
- `LIGATURE_MAP` - Mapping för text-ligaturer

**Funktioner:**
- `getDefaultSystemPrompt()` - Hämta default system prompt
- `utc_timestamp()` - UTC timestamp i ISO-format
- `hlog()` - Loggningshjälpfunktion
- `getApiKey()` - Hämta API-nyckel från miljövariabler
- `normalize_text()` - Normalisera text (ligaturer, radbrytningar)
- `sanitize_basename()` - Sanera filnamn
- `extract_text_from_pdf()` - Extrahera text från PDF-bytes

#### `index.html`
Frontend med Material Design.

**Sektioner:**
- PDF-uppladdning (ny samling / lägg till)
- URL-extrahering (ny samling / lägg till)
- Semantisk sökning
- AI-frågor
- Samlingshantering

---

## API-dokumentation

### POST /api/upload_pdf

Ladda upp och processa PDF-filer.

**Request:**
```
Content-Type: multipart/form-data

Parameters:
- file: UploadFile (required, multiple allowed)
- collection_name: str (optional för ny samling)
- existing_collection: str (optional för att lägga till)
- embed_backend: str (optional, default: "google")
- max_tokens: int (optional, default: 512)
- use_overlap: bool (optional, default: true)
```

**Response:**
```json
{
  "success": true,
  "collection_name": "min_samling",
  "total_files": 3,
  "total_chunks": 145,
  "details": [
    {
      "filename": "dokument.pdf",
      "chunks": 48,
      "title": "Dokument"
    }
  ],
  "download_url": "/api/download/min_samling"
}
```

**Error Response:**
```json
{
  "success": false,
  "error": {
    "type": "ValueError",
    "message": "Användarvänligt felmeddelande",
    "details": "Tekniska detaljer",
    "context": "PDF-uppladdning",
    "traceback": "..." // Only if DEBUG=true
  }
}
```

---

### POST /api/process_url

Processa en eller flera URLs.

**Request:**
```json
{
  "urls": ["https://example.com/page1", "https://example.com/page2"],
  "collection_name": "min_samling",
  "existing_collection": null,
  "embed_backend": "google",
  "max_tokens": 512,
  "use_overlap": true
}
```

**Response:**
```json
{
  "success": true,
  "collection_name": "min_samling",
  "total_urls": 2,
  "total_chunks": 87,
  "results": [
    {
      "url": "https://example.com/page1",
      "title": "Example Page",
      "chunks": 45,
      "status": "success"
    }
  ]
}
```

---

### POST /api/search

Semantisk sökning i samling.

**Request:**
```json
{
  "collection": "min_samling",
  "query": "information om produkt",
  "k": 5,
  "embed_backend": "google"
}
```

**Response:**
```json
{
  "success": true,
  "query": "information om produkt",
  "results_count": 5,
  "results": [
    {
      "score": 0.8523,
      "heading": "Produktinformation",
      "markdown": "Text chunk här...",
      "url": "https://example.com/page1",
      "anchor": "#produktinformation",
      "chunk_index": 2
    }
  ]
}
```

---

### POST /api/ask

Ställ fråga till AI baserat på samling.

**Request:**
```json
{
  "collection": "min_samling",
  "query": "Vad är huvudfunktionerna?",
  "k": 5,
  "llm_backend": "google",
  "embed_backend": "google",
  "system_prompt": null
}
```

**Response:**
```json
{
  "success": true,
  "query": "Vad är huvudfunktionerna?",
  "answer": "Baserat på dokumenten så är huvudfunktionerna...",
  "sources": [
    {
      "rank": 1,
      "score": 0.8523,
      "heading": "Funktioner",
      "preview": "Text förhandsvisning...",
      "url": "https://example.com/page1",
      "anchor": "#funktioner"
    }
  ],
  "context_used": "Hela kontexten som användes...",
  "params": {
    "collection": "min_samling",
    "k": 5,
    "llm_backend": "google",
    "embed_backend": "google"
  }
}
```

---

### GET /api/collections

Lista alla tillgängliga samlingar.

**Response:**
```json
{
  "success": true,
  "collections": [
    {
      "name": "min_samling",
      "loaded": true,
      "stats": {
        "total_records": 145,
        "dimension": 768
      },
      "url_count": 2,
      "pdf_count": 3
    }
  ],
  "loaded_count": 1,
  "total_count": 5
}
```

---

### DELETE /api/collection/{collection_name}

Ta bort en samling.

**Response:**
```json
{
  "success": true,
  "message": "Collection 'min_samling' deleted",
  "deleted_files": [
    "min_samling.faiss",
    "min_samling.pkl",
    "min_samling.meta.json"
  ]
}
```

---

### GET /api/download/{basename}

Ladda ner samling som JSON.

**Response:**
- Content-Type: application/json
- File download med alla records och metadata

---

### GET /api/health

Hälsokontroll.

**Response:**
```json
{
  "status": "ok",
  "success": true
}
```

---

## Embedding-backends

### Översikt

Systemet stöder flera embedding-backends för olika användningsfall:

| Backend | Dimension | Lokal/API | Bäst för | API-nyckel |
|---------|-----------|-----------|----------|------------|
| Google Gemini | 768 | API | Allround, multilingual | GOOGLE_API_KEY |
| OpenAI | 1536/3072 | API | Hög kvalitet | OPENAI_API_KEY |
| Cohere v3 | 1024 | API | Robust mot brus | COHERE_API_KEY |
| BGE-M3 | 1024 | Lokal | State-of-the-art | - |
| E5 | 768 | Lokal | Multilingual | - |
| Sentence-BERT | 384 | Lokal | Enkel, snabb | - |
| Ollama | 768 | Lokal | Privacy-first | - |

### Implementation

Alla backends implementerar `EmbeddingBackend`-interfacet:

```python
class EmbeddingBackend:
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts"""
        raise NotImplementedError
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        raise NotImplementedError
```

### Exempel: Lägga till ny backend

```python
class CustomBackend(EmbeddingBackend):
    def __init__(self, model: str = "custom-model"):
        # Initialize your model
        self.model = load_model(model)
        self._dimension = 512
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        # Generate embeddings
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
    
    def get_dimension(self) -> int:
        return self._dimension

# Registrera i get_backend()
def get_backend(backend_name: Optional[str] = None) -> EmbeddingBackend:
    backend = (backend_name or "google").lower()
    if backend == "custom":
        return CustomBackend()
    # ... existing backends
```

---

## LLM-backends

### Översikt

Systemet stöder flera LLM-backends för AI-frågor:

| Backend | Modell | Lokal/API | Kostnad | API-nyckel |
|---------|--------|-----------|---------|------------|
| Google Gemini | gemini-2.0-flash | API | Gratis tier | GOOGLE_API_KEY |
| OpenAI | gpt-4o-mini | API | Betald | OPENAI_API_KEY |
| Ollama | llama3.2 | Lokal | Gratis | - |

### Implementation

Alla LLM-backends implementerar `LLMBackend`-interfacet:

```python
class LLMBackend:
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response from LLM"""
        raise NotImplementedError
```

### Exempel: Google LLM Backend

```python
class GoogleLLMBackend(LLMBackend):
    def __init__(self, model: str = "gemini-2.0-flash"):
        self.model = model
    
    def generate(self, system_prompt: str, user_prompt: str) -> str:
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
```

---

## Vector Store

### FAISS Implementation

`FAISSVectorStore` är en wrapper runt Facebook's FAISS-bibliotek.

**Huvudfunktioner:**

```python
class FAISSVectorStore:
    def __init__(self, dimension: int = 768):
        """Initialize vector store with embedding dimension"""
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product
        self.records = []
    
    def add_records(self, records: List[Dict]):
        """Add records with embeddings to index"""
        # Extract embeddings and normalize
        vectors = np.array([r["embedding"] for r in records])
        faiss.normalize_L2(vectors)
        
        # Add to FAISS index
        self.index.add(vectors)
        self.records.extend(records)
    
    def search(self, query_vector: List[float], k: int = 5) -> List[Dict]:
        """Search for similar vectors"""
        query = np.array([query_vector], dtype=np.float32)
        faiss.normalize_L2(query)
        
        scores, indices = self.index.search(query, k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0:
                results.append({
                    "record": self.records[idx],
                    "score": float(score)
                })
        return results
    
    def save(self, path: str):
        """Save index and records to disk"""
        faiss.write_index(self.index, f"{path}.faiss")
        with open(f"{path}.pkl", "wb") as f:
            pickle.dump(self.records, f)
    
    def load(self, path: str):
        """Load index and records from disk"""
        self.index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}.pkl", "rb") as f:
            self.records = pickle.load(f)
```

### Similarity Metrics

FAISS använder **Inner Product (IP)** för similarity search:

```python
# IndexFlatIP använder inner product
index = faiss.IndexFlatIP(dimension)

# Normalisera vektorer för cosine similarity
faiss.normalize_L2(vectors)

# Efter normalisering: IP = cosine similarity
similarity = np.dot(v1, v2) / (||v1|| * ||v2||)
```

**Alternativa index-typer:**

```python
# L2 distance (Euclidean)
index = faiss.IndexFlatL2(dimension)

# IVF för snabbare sökning (approximate)
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist=100)

# HNSW för hög prestanda
index = faiss.IndexHNSWFlat(dimension, 32)
```

---

## RAG Pipeline

### URL Processing Pipeline

```python
def process_url_for_rag(url, max_tokens_per_chunk=512, 
                        embed_backend=None, use_overlap=True):
    # 1. Scrape URL
    scraped = scrape_url(url)
    
    # 2. Clean HTML
    cleaned = clean_html(scraped.html)
    soup = normalize_html(cleaned)
    soup = resolve_links(soup, scraped.base_url)
    
    # 3. Extract title
    title = soup.title.string if soup.title else "Untitled"
    
    # 4. Convert to Markdown
    markdown = html_to_markdown(soup)
    
    # 5. Split into blocks by headings
    blocks = split_markdown_into_blocks(markdown)
    
    # 6. Chunk blocks
    records = build_records(url, title, blocks, 
                           max_tokens_per_chunk, use_overlap)
    
    # 7. Generate embeddings
    backend = get_backend(embed_backend)
    embed_records(records, backend)
    
    # 8. Create vector store
    vector_store = FAISSVectorStore(backend.get_dimension())
    vector_store.add_records(records)
    
    return {
        "source_url": url,
        "title": title,
        "records": records,
        "vector_store": vector_store
    }
```

### PDF Processing Pipeline

```python
def process_pdf_for_rag(pdf_bytes, filename, max_tokens_per_chunk=512,
                        embed_backend=None, use_overlap=True):
    # 1. Extract text from PDF
    full_text = extract_text_from_pdf(pdf_bytes)
    full_text = normalize_text(full_text)
    
    # 2. Split into paragraphs
    paragraphs = re.split(r"\n\s*\n", full_text)
    
    # 3. Create blocks with heuristic heading detection
    blocks = []
    for para in paragraphs:
        lines = para.split("\n", 1)
        first_line = lines[0].strip()
        
        # Heuristic: short, uppercase, or numbered line = heading
        is_heading = (len(first_line) < 80 and 
                     (first_line.isupper() or 
                      re.match(r"^\d+[\.\)]\s+", first_line)))
        
        if is_heading and len(lines) > 1:
            heading = first_line
            content = lines[1]
        else:
            heading = f"Section {i+1}"
            content = para
        
        blocks.append(Block(level=1, heading=heading, content=content))
    
    # 4-8: Same as URL processing
    records = build_records(...)
    embed_records(...)
    vector_store = FAISSVectorStore(...)
    
    return {
        "source_file": filename,
        "title": title,
        "records": records,
        "vector_store": vector_store
    }
```

### Text Chunking

**Enkel chunking:**
```python
def chunk_text(text, max_tokens=512):
    words = text.split()
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for word in words:
        word_tokens = approx_token_count(word)
        if current_tokens + word_tokens > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_tokens = word_tokens
        else:
            current_chunk.append(word)
            current_tokens += word_tokens
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks
```

**Chunking med överlapp:**
```python
def chunk_text_with_overlap(text, max_tokens=512, overlap_tokens=50):
    words = text.split()
    chunks = []
    start_idx = 0
    
    while start_idx < len(words):
        # Calculate chunk end
        end_idx = start_idx
        current_tokens = 0
        
        while end_idx < len(words) and current_tokens < max_tokens:
            current_tokens += approx_token_count(words[end_idx])
            end_idx += 1
        
        # Create chunk
        chunk = " ".join(words[start_idx:end_idx])
        chunks.append(chunk)
        
        # Move start with overlap
        overlap_words = min(overlap_tokens // 2, end_idx - start_idx)
        start_idx = end_idx - overlap_words
        
        if start_idx >= len(words):
            break
    
    return chunks
```

---

## Utvecklingsguide

### Sätta upp utvecklingsmiljö

```bash
# Klona/ladda ner projektet
cd rag-search

# Skapa virtuell miljö
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# eller
venv\Scripts\activate  # Windows

# Installera dependencies i utvecklingsläge
pip install -r requirements.txt
pip install --upgrade black flake8 pytest

# Sätt DEBUG-läge
echo "DEBUG=true" >> .env
```

### Kodstandarder

**Formattering:**
```bash
# Formatera kod med black
black server.py rag_pipeline.py helpers.py

# Lint med flake8
flake8 --max-line-length=100 *.py
```

**Docstrings:**
Använd Google-stil docstrings:

```python
def function_name(param1: str, param2: int) -> bool:
    """
    Kort beskrivning av funktionen.
    
    Längre beskrivning om nödvändigt. Förklara vad funktionen gör,
    varför den finns, och eventuella viktiga detaljer.
    
    Args:
        param1 (str): Beskrivning av parameter 1
        param2 (int): Beskrivning av parameter 2
    
    Returns:
        bool: Beskrivning av returvärdet
    
    Raises:
        ValueError: När detta exception kastas
        
    Example:
        >>> function_name("test", 42)
        True
    """
    pass
```

### Testing

Skapa `tests/test_pipeline.py`:

```python
import pytest
from rag_pipeline import chunk_text, normalize_text

def test_chunk_text():
    text = "This is a test. " * 100
    chunks = chunk_text(text, max_tokens=50)
    assert len(chunks) > 1
    assert all(approx_token_count(c) <= 50 for c in chunks)

def test_normalize_text():
    text = "Hello\r\nWorld\r\n\r\nTest"
    normalized = normalize_text(text)
    assert "\r" not in normalized
    assert "\n\n" in normalized
```

Kör tester:
```bash
pytest tests/
```

### Lägga till nya endpoints

1. Definiera endpoint i `server.py`:

```python
@app.post("/api/my_endpoint")
async def my_endpoint(payload: dict = Body(...)):
    try:
        # Validera input
        if not payload.get("required_field"):
            return JSONResponse(
                status_code=400,
                content=create_error_response(
                    Exception("required_field saknas"),
                    "my_endpoint"
                )
            )
        
        # Bearbeta
        result = do_something(payload)
        
        # Returnera resultat
        return JSONResponse(content={
            "success": True,
            "result": result
        })
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=create_error_response(e, "my_endpoint")
        )
```

2. Uppdatera frontend i `index.html`:

```javascript
async function callMyEndpoint(data) {
    try {
        const response = await fetch('/api/my_endpoint', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            await handleFetchError(response);
            return;
        }
        
        const result = await response.json();
        // Hantera resultat
    } catch (error) {
        showError(error.message, 'my_endpoint');
    }
}
```

---

## Konfiguration

### Miljövariabler (.env)

```env
# OBLIGATORISKA
GOOGLE_API_KEY=your_key_here

# VALFRIA
OPENAI_API_KEY=your_key_here
COHERE_API_KEY=your_key_here

# INSTÄLLNINGAR
DEBUG=false
MAX_UPLOAD_SIZE=100000000  # 100MB
DEFAULT_CHUNK_SIZE=512
DEFAULT_EMBEDDING_BACKEND=google
DEFAULT_LLM_BACKEND=google

# OLLAMA (om du använder lokal Ollama)
OLLAMA_HOST=http://localhost:11434
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_LLM_MODEL=llama3.2
```

### Anpassa standardvärden

I `helpers.py`:

```python
def getDefaultSystemPrompt() -> str:
    return """Din anpassade system-prompt här"""

# Eller läs från fil:
def getDefaultSystemPrompt() -> str:
    prompt_file = os.path.join(BASE_DIR, "system_prompt.txt")
    if os.path.exists(prompt_file):
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read()
    return "Default prompt..."
```

### Server-inställningar

I `server.py` eller vid start:

```python
# CORS-inställningar
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ändra för produktion
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Filstorlekar
app.add_middleware(
    MaxSizeMiddleware,
    max_size=100 * 1024 * 1024  # 100MB
)
```

Starta med custom port:
```bash
uvicorn server:app --host 0.0.0.0 --port 8080
```

---

## Prestanda och optimering

### Vector Store Optimering

**Använd IVF-index för stora dataset:**

```python
class FAISSVectorStore:
    def __init__(self, dimension: int = 768, use_ivf: bool = False):
        if use_ivf:
            # IVF = Inverted File Index
            quantizer = faiss.IndexFlatIP(dimension)
            nlist = 100  # Antal clusters
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            self.index.nprobe = 10  # Sökparameter
        else:
            self.index = faiss.IndexFlatIP(dimension)
```

**GPU-acceleration (om tillgängligt):**

```python
import faiss

# Kolla GPU-tillgänglighet
if faiss.get_num_gpus() > 0:
    res = faiss.StandardGpuResources()
    index_cpu = faiss.IndexFlatIP(dimension)
    index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
```

### Batch-bearbetning

**För stora PDF-uppsättningar:**

```python
async def batch_process_pdfs(files: List[UploadFile], batch_size: int = 5):
    results = []
    for i in range(0, len(files), batch_size):
        batch = files[i:i+batch_size]
        
        # Bearbeta batch parallellt
        tasks = [process_single_pdf(f) for f in batch]
        batch_results = await asyncio.gather(*tasks)
        
        results.extend(batch_results)
    
    return results
```

### Caching

**Cache embeddings:**

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_embed(text: str, backend: str = "google") -> tuple:
    backend_obj = get_backend(backend)
    embedding = backend_obj.embed([text])[0]
    return tuple(embedding)  # tuple för hashability
```

### Minneshantering

**Frigör resurser:**

```python
import gc

def cleanup_resources():
    # Töm globala vector stores som inte används
    for name in list(vector_stores.keys()):
        if not recently_used(name):
            del vector_stores[name]
    
    # Kör garbage collector
    gc.collect()

# Kör regelbundet
import schedule
schedule.every(30).minutes.do(cleanup_resources)
```

### Monitering

**Lägg till logging:**

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_system.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

@app.post("/api/search")
async def search(...):
    logger.info(f"Search request: collection={collection}, query={query}")
    start_time = time.time()
    
    # ... processing ...
    
    duration = time.time() - start_time
    logger.info(f"Search completed in {duration:.2f}s")
```

---

## Säkerhet

### API-nycklar

- **Lagra aldrig nycklar i kod**
- Använd `.env`-filen
- Lägg till `.env` i `.gitignore`
- Använd olika nycklar för dev/prod

### Input-validering

```python
from pydantic import BaseModel, validator

class SearchRequest(BaseModel):
    collection: str
    query: str
    k: int = 5
    
    @validator('k')
    def validate_k(cls, v):
        if not 1 <= v <= 50:
            raise ValueError('k must be between 1 and 50')
        return v
    
    @validator('query')
    def validate_query(cls, v):
        if len(v) > 1000:
            raise ValueError('Query too long')
        return v
```

### Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/api/search")
@limiter.limit("10/minute")
async def search(request: Request, ...):
    # ...
```

---

## Felsökning

### Debug-läge

Aktivera i `.env`:
```env
DEBUG=true
```

Detta ger:
- Detaljerade tracebacks i felmeddelanden
- Utökad logging
- Stack traces i API-svar

### Vanliga fel

**"FAISS not available":**
```bash
pip install --upgrade faiss-cpu
```

**"API key invalid":**
- Kontrollera `.env`-filen
- Testa nyckeln på respektive plattform
- Starta om servern efter ändring

**"Out of memory":**
- Minska chunk-storlek
- Använd batch-bearbetning
- Minska antal resultat (k)
- Överväg GPU-acceleration

**"Slow performance":**
- Använd IVF-index för stora dataset
- Cache embeddings
- Använd lättare embedding-modeller
- Optimera chunk-storlek

---

## Licens och bidrag

### Licens
[Ange licens här]

### Bidra

1. Forka projektet
2. Skapa feature branch (`git checkout -b feature/AmazingFeature`)
3. Committa ändringar (`git commit -m 'Add AmazingFeature'`)
4. Pusha till branch (`git push origin feature/AmazingFeature`)
5. Öppna Pull Request

**Riktlinjer:**
- Följ kodstandarder (black, flake8)
- Lägg till tester för ny funktionalitet
- Uppdatera dokumentation
- Skriv tydliga commit-meddelanden

---

## Support

För teknisk support:
- Öppna ett issue på GitHub
- Kontrollera loggar (`rag_system.log`)
- Sätt `DEBUG=true` för mer information
- Inkludera systeminfo och fel-meddelanden

---

**Lycka till med utvecklingen!** 🚀
