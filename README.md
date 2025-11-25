# Slutarbete
Projekt som ska visa att jag lärt mig något på AI-kursen från YH

Projektet bygger på ett backend med FastAPI och en frontend med HTMX och Jinja-templates

## Python-bibliotek som behövs

För att köra applikationen behöver följande Python-bibliotek installeras:

### Kärnbibliotek
- **FastAPI** - Web framework för API
- **uvicorn** - ASGI server för att köra FastAPI
- **python-multipart** - För att hantera filuppladdningar

### AI & Machine Learning
- **google-genai** - Google Gemini API för LLM och embeddings
- **openai** - OpenAI API (valfritt, för GPT-modeller)
- **sentence-transformers** - För lokala SBERT embeddings (valfritt)
- **faiss-cpu** - FAISS vector store för semantisk sökning
- **numpy** - Numeriska beräkningar för vektorer

### Web scraping & HTML-parsing
- **requests** - HTTP-förfrågningar
- **httpx** - Asynkron HTTP-klient
- **beautifulsoup4** - HTML-parsing
- **lxml** - HTML-parser (används av BeautifulSoup)
- **markdownify** - Konvertera HTML till Markdown

### PDF-hantering
- **PyMuPDF** (fitz) - PDF-extraktion och parsing

### Hjälpbibliotek
- **python-dateutil** - Datumhantering

### Installation

Skapa och aktivera virtuell miljö:

```bash
# Skapa virtuell miljö
python3 -m venv .venv

# Aktivera (macOS/Linux)
source .venv/bin/activate

# Aktivera (Windows)
.venv\Scripts\activate
```

Installera alla bibliotek:

```bash
pip install fastapi uvicorn[standard] python-multipart
pip install google-genai openai
pip install sentence-transformers
pip install faiss-cpu numpy
pip install requests httpx beautifulsoup4 lxml markdownify
pip install PyMuPDF
pip install python-dateutil
```

Eller använd en requirements.txt:

```bash
pip install -r requirements.txt
```

### requirements.txt innehåll:
```
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
python-multipart>=0.0.6
google-genai>=0.2.0
openai>=1.3.0
sentence-transformers>=2.2.2
faiss-cpu>=1.7.4
numpy>=1.24.0
requests>=2.31.0
httpx>=0.25.0
beautifulsoup4>=4.12.0
lxml>=4.9.3
markdownify>=0.11.6
PyMuPDF>=1.23.0
python-dateutil>=2.8.2
```

## Snabbstart 

### Kör servern

```bash
# Stå i projektets root-mapp (där ligger mappen _app_)
source .venv/bin/activate

# Navigera till app-mappen
cd app

# Starta servern
uvicorn server:app --reload --host 0.0.0.0 --port 8000

# Öppna http://127.0.0.1:8000 med webläsare
```

## Lokal LLM med Ollama (Valfritt)

Vill man köra lokal LLM istället för Google/OpenAI:

### Kolla att Ollama finns

```bash
ollama --version
```

Om "command not found":

```bash
brew install ollama
```

### Starta Ollama-servern i ett separat fönster

```bash
ollama serve
```

Låt den processen ligga kvar igång.

### Hämta en modell (om du inte redan gjort det)

I ytterligare ett terminalfönster (eller samma där du kör serve innan du startar servern):

```bash
ollama pull llama3.1
```

### Snabbtest att API:et verkligen svarar

```bash
curl http://localhost:11434/api/tags
# ska ge JSON med listade modeller

curl http://localhost:11434/api/generate -d '{"model":"llama3.1","prompt":"Hej! Säg något kort."}'
# ska ge ett JSON-svar med text
```

## Så här fungerar webbappen:

### 1. Ladda upp PDF eller URL

Systemet:
* Extraherar all text från PDF:en eller webbsidan
* Delar upp i meningsfulla chunks (~512 tokens)
* Skapar embeddings med AI
* Bygger sökbart FAISS-index

### 2. Sök i dokumentet

Skriv naturlig språk som:
* "Vad säger dokumentet om AI?"
* "Sammanfatta huvudpunkterna"
* "Hitta information om säkerhet"

### 3. Få rankade resultat

De mest relevanta styckena baserat på semantisk likhet!

### 4. Ställ frågor med /api/ask

Systemet:
* Söker relevanta dokument
* Skickar kontext till LLM
* Genererar svar baserat på dokumenten

## API Endpoints

- **GET /** - Frontend
- **POST /api/upload_pdf** - Ladda upp PDF
- **POST /api/fetch_url** - Processa URL
- **POST /api/search** - Semantisk sökning
- **POST /api/ask** - RAG-baserad fråga-svar
- **GET /api/collections** - Lista alla collections
- **DELETE /api/collection/{name}** - Radera collection
- **GET /api/health** - Health check

## Konfiguration

### API-nycklar

Lägg till dina API-nycklar i .env filen:

```bash
GOOGLE_API_KEY=din-google-api-nyckel
OPENAI_API_KEY=din-openai-api-nyckel
```

### Embedding backends

Välj mellan:
- `google` - Google Gemini (default, 768 dimensioner)
- `openai` - OpenAI (1536 eller 3072 dimensioner)
- `sbert` - Lokal sentence-transformers (384 dimensioner)
- `ollama` - Lokal Ollama (768 dimensioner)

### LLM backends

Välj mellan:
- `google` - Google Gemini (default)
- `openai` - OpenAI GPT
- `ollama` - Lokal Ollama

## Projektstruktur

```
slutarbete/
├── app/
│   ├── server.py          # FastAPI endpoints
│   ├── rag_pipeline.py    # RAG-logik och embeddings
│   ├── static/            # Frontend filer
│   ├── uploads/           # Uppladdade PDF:er
│   ├── outputs/           # Genererade JSON/text
│   └── vector_stores/     # FAISS index
├── .venv/                 # Virtuell miljö
└── README.md
```

## Anteckningar för utvecklingen

### Kommande förbättringar
