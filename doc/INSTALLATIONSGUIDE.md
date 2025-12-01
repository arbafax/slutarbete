**VIKTIGT** Denna dokumentation är automatgenererad och _**inte**_ verifierad 2025-12-01 15:01

# RAG Search System - Installationsguide

En komplett guide för att installera och köra RAG Search System lokalt på din PC eller Mac.

## Innehållsförteckning

- [Systemkrav](#systemkrav)
- [Vad är RAG Search System?](#vad-är-rag-search-system)
- [Installation steg för steg](#installation-steg-för-steg)
  - [1. Installera Python](#1-installera-python)
  - [2. Ladda ner projektet](#2-ladda-ner-projektet)
  - [3. Skapa virtuell miljö (rekommenderat)](#3-skapa-virtuell-miljö-rekommenderat)
  - [4. Installera beroenden](#4-installera-beroenden)
  - [5. Konfigurera API-nycklar](#5-konfigurera-api-nycklar)
  - [6. Skapa mappstruktur](#6-skapa-mappstruktur)
- [Starta systemet](#starta-systemet)
- [Testa att det fungerar](#testa-att-det-fungerar)
- [Vanliga problem och lösningar](#vanliga-problem-och-lösningar)

---

## Systemkrav

### Minimikrav
- **Operativsystem:** Windows 10/11, macOS 10.14+, eller Linux
- **Python:** Version 3.9 eller senare
- **RAM:** Minst 4 GB (8 GB rekommenderas)
- **Diskutrymme:** 2 GB ledigt utrymme
- **Internetanslutning:** Krävs för installation och API-anrop

### Rekommenderat
- **RAM:** 8 GB eller mer för bättre prestanda
- **Processor:** Multi-core processor för snabbare bearbetning
- **SSD:** För snabbare läs/skriv-operationer

---

## Vad är RAG Search System?

RAG Search System är ett avancerat verktyg som låter dig:

1. **Ladda upp PDF-filer** och extrahera text från dem
2. **Scrapa webbsidor** och hämta innehåll från URL:er
3. **Skapa semantiska sökbara samlingar** med AI-drivna embeddings
4. **Söka intelligent** i dina dokument med naturligt språk
5. **Ställa frågor till AI** som svarar baserat på dina dokument

Systemet använder moderna AI-tekniker som RAG (Retrieval Augmented Generation) och vektorbaserad sökning för att ge exakta svar på dina frågor.

---

## Installation steg för steg

### 1. Installera Python

#### Windows:
1. Gå till [python.org/downloads](https://www.python.org/downloads/)
2. Ladda ner senaste Python 3.x (3.9 eller senare)
3. Kör installationsfilen
4. **VIKTIGT:** Markera "Add Python to PATH" under installationen
5. Klicka "Install Now"
6. Verifiera installationen genom att öppna Command Prompt (CMD) och skriva:
   ```bash
   python --version
   ```
   Du ska se något liknande: `Python 3.11.x`

#### macOS:
Python 3 är ofta förinstallerat på moderna Mac-datorer. Kontrollera version:
```bash
python3 --version
```

Om du behöver installera eller uppgradera:

**Alternativ 1: Via Homebrew (rekommenderat)**
```bash
# Installera Homebrew om du inte har det
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Installera Python
brew install python@3.11
```

**Alternativ 2: Från python.org**
1. Gå till [python.org/downloads](https://www.python.org/downloads/)
2. Ladda ner senaste Python 3.x för macOS
3. Öppna .pkg-filen och följ instruktionerna

---

### 2. Ladda ner projektet

Skapa en mapp för projektet och placera alla filer där:

#### Windows:
```bash
# Öppna Command Prompt och navigera till önskad plats
cd C:\Users\DittNamn\Documents
mkdir rag-search
cd rag-search

# Kopiera alla projektfiler till denna mapp
```

#### macOS/Linux:
```bash
# Öppna Terminal
cd ~/Documents
mkdir rag-search
cd rag-search

# Kopiera alla projektfiler till denna mapp
```

Du bör nu ha följande filer i mappen:
- `server.py`
- `rag_pipeline.py`
- `helpers.py`
- `index.html`

---

### 3. Skapa virtuell miljö (rekommenderat)

En virtuell miljö isolerar projektets beroenden från systemet.

#### Windows:
```bash
# I projektmappen
python -m venv venv

# Aktivera miljön
venv\Scripts\activate

# Du ska se (venv) framför din kommandoprompt
```

#### macOS/Linux:
```bash
# I projektmappen
python3 -m venv venv

# Aktivera miljön
source venv/bin/activate

# Du ska se (venv) framför din prompt
```

**Tips:** För att avaktivera miljön senare, skriv bara `deactivate`.

---

### 4. Installera beroenden

Skapa en fil som heter `requirements.txt` i projektmappen med följande innehåll:

```txt
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
python-multipart>=0.0.6
requests>=2.31.0
beautifulsoup4>=4.12.0
markdownify>=0.11.6
python-dotenv>=1.0.0
PyMuPDF>=1.23.0
numpy>=1.24.0
faiss-cpu>=1.7.4
google-genai>=1.0.0
```

Installera sedan alla paket:

#### Windows:
```bash
pip install -r requirements.txt
```

#### macOS/Linux:
```bash
pip3 install -r requirements.txt
```

**OBS:** Installationen kan ta några minuter beroende på din internetanslutning.

#### Valfria beroenden

För ytterligare funktionalitet kan du installera:

**För OpenAI-stöd:**
```bash
pip install openai
```

**För Ollama lokal AI:**
```bash
pip install ollama
# Och installera Ollama från: https://ollama.ai
```

**För Sentence-BERT embeddings:**
```bash
pip install sentence-transformers torch
```

**För Cohere embeddings:**
```bash
pip install cohere
```

---

### 5. Konfigurera API-nycklar

Skapa en fil som heter `.env` i projektmappen:

#### Windows (Command Prompt):
```bash
echo. > .env
notepad .env
```

#### macOS/Linux (Terminal):
```bash
touch .env
nano .env
```

Lägg till följande innehåll i `.env`-filen:

```env
# Google Gemini API (OBLIGATORISK för grundfunktionalitet)
GOOGLE_API_KEY=din_google_api_nyckel_här

# OpenAI API (valfri - endast om du vill använda OpenAI)
OPENAI_API_KEY=din_openai_api_nyckel_här

# Cohere API (valfri - endast om du vill använda Cohere embeddings)
COHERE_API_KEY=din_cohere_api_nyckel_här

# Debug-läge (sätt till true för mer detaljerad felsökning)
DEBUG=false
```

#### Skaffa Google API-nyckel (OBLIGATORISKT):

1. Gå till [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Logga in med ditt Google-konto
3. Klicka på "Get API Key" eller "Create API Key"
4. Kopiera nyckeln och klistra in den i `.env`-filen efter `GOOGLE_API_KEY=`

**OBS:** Håll din API-nyckel hemlig! Dela aldrig din `.env`-fil med andra.

---

### 6. Skapa mappstruktur

Skapa nödvändiga mappar för projektet:

#### Windows:
```bash
mkdir static
mkdir uploads
mkdir outputs
mkdir data
mkdir vector_stores
```

#### macOS/Linux:
```bash
mkdir static uploads outputs data vector_stores
```

Flytta `index.html` till `static`-mappen:

#### Windows:
```bash
move index.html static\
```

#### macOS/Linux:
```bash
mv index.html static/
```

Din projektstruktur bör nu se ut så här:
```
rag-search/
├── .env
├── server.py
├── rag_pipeline.py
├── helpers.py
├── requirements.txt
├── static/
│   └── index.html
├── uploads/
├── outputs/
├── data/
├── vector_stores/
└── venv/
```

---

## Starta systemet

### 1. Aktivera virtuell miljö (om du använde en)

#### Windows:
```bash
venv\Scripts\activate
```

#### macOS/Linux:
```bash
source venv/bin/activate
```

### 2. Starta servern

#### Windows:
```bash
python -m uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

#### macOS/Linux:
```bash
python3 -m uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

Du ska se något liknande:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### 3. Öppna webbgränssnittet

Öppna din webbläsare och navigera till:
```
http://localhost:8000
```

Du ska nu se RAG Search System-gränssnittet!

---

## Testa att det fungerar

### Snabbtest

1. **Testa URL-scraping:**
   - I webbgränssnittet, gå till "Extrahera från URL"
   - Ange en URL (t.ex. `https://sv.wikipedia.org/wiki/Sverige`)
   - Ge samlingen ett namn (t.ex. "test")
   - Klicka "Extrahera"
   - Vänta tills processen är klar

2. **Testa sökning:**
   - Gå till "Sök i samling"
   - Välj din nyskapade samling
   - Skriv en sökfråga (t.ex. "befolkning")
   - Klicka "Sök"
   - Du ska se relevanta resultat

3. **Testa AI-frågor:**
   - Gå till "Fråga AI om samling"
   - Välj din samling
   - Ställ en fråga (t.ex. "Vad är Sveriges befolkning?")
   - Klicka "Fråga AI"
   - Vänta på svaret från AI:n

### API-test (avancerat)

Testa att API:t fungerar genom att öppna:
```
http://localhost:8000/api/health
```

Du ska se:
```json
{"status": "ok", "success": true}
```

---

## Vanliga problem och lösningar

### Problem: "Python is not recognized" eller "command not found"

**Lösning Windows:**
- Python är inte tillagt i PATH
- Ominstallera Python och markera "Add Python to PATH"
- Eller lägg till manuellt via Systeminställningar → Miljövariabler

**Lösning macOS/Linux:**
- Använd `python3` istället för `python`
- Installera via Homebrew: `brew install python@3.11`

---

### Problem: "No module named 'faiss'" eller liknande

**Lösning:**
```bash
pip install faiss-cpu --break-system-packages
```

På vissa system behöver man `--break-system-packages`:
```bash
pip install --break-system-packages -r requirements.txt
```

---

### Problem: "API key saknas eller är ogiltig"

**Lösning:**
1. Kontrollera att `.env`-filen finns i projektmappen
2. Öppna `.env` och verifiera att `GOOGLE_API_KEY` är korrekt ifylld
3. Starta om servern efter att ha ändrat `.env`
4. Testa din API-nyckel på [Google AI Studio](https://makersuite.google.com/)

---

### Problem: "Address already in use" eller port 8000 upptagen

**Lösning:**
Använd en annan port:
```bash
python -m uvicorn server:app --reload --host 0.0.0.0 --port 8080
```

Öppna då istället:
```
http://localhost:8080
```

---

### Problem: Servern startar men webbsidan visar "Cannot GET /"

**Lösning:**
Kontrollera att `index.html` ligger i `static/`-mappen:
```bash
# Windows
dir static

# macOS/Linux
ls -la static/
```

Du ska se `index.html` i listan.

---

### Problem: Import-fel eller ModuleNotFoundError

**Lösning:**
1. Kontrollera att virtuell miljö är aktiverad (du ska se `(venv)` i prompten)
2. Installera om beroenden:
   ```bash
   pip install --upgrade -r requirements.txt
   ```
3. Om problemet kvarstår, radera `venv`-mappen och skapa en ny:
   ```bash
   # Windows
   rmdir /s venv
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt

   # macOS/Linux
   rm -rf venv
   python3 -m venv venv
   source venv/bin/activate
   pip3 install -r requirements.txt
   ```

---

### Problem: "Permission denied" vid installation

**Lösning Windows:**
- Kör Command Prompt som administratör

**Lösning macOS/Linux:**
- Använd `--user` flaggan:
  ```bash
  pip3 install --user -r requirements.txt
  ```
- Eller använd `sudo` (ej rekommenderat med virtuell miljö):
  ```bash
  sudo pip3 install -r requirements.txt
  ```

---

### Problem: Långsam prestanda eller hänger sig

**Lösning:**
1. Kontrollera att du har tillräckligt med RAM (minst 4 GB)
2. Använd mindre chunks vid PDF/URL-bearbetning (256 tokens istället för 512)
3. Minska antal resultat (k-värdet) vid sökning
4. Överväg att använda lättare embedding-modeller (t.ex. Sentence-BERT istället för OpenAI)

---

### Problem: PDF-uppladdning misslyckas

**Lösning:**
1. Kontrollera att PDF:en inte är skadad
2. Verifiera att `PyMuPDF` är korrekt installerat:
   ```bash
   pip install --upgrade PyMuPDF
   ```
3. Testa med en mindre PDF först
4. Kontrollera att `uploads/`-mappen existerar och har skrivbehörighet

---

### Problem: Ollama fungerar inte

**Lösning:**
1. Installera Ollama från [ollama.ai](https://ollama.ai)
2. Starta Ollama-servern:
   ```bash
   ollama serve
   ```
3. Ladda ner önskad modell:
   ```bash
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```
4. Kontrollera att Ollama körs på rätt port (standard: 11434)

---

## Nästa steg

När installationen är klar, se [ANVÄNDARGUIDE.md](ANVÄNDARGUIDE.md) för detaljerade instruktioner om hur du använder systemet.

För teknisk information och API-dokumentation, se [TEKNISK_DOKUMENTATION.md](TEKNISK_DOKUMENTATION.md).

---

## Support och hjälp

Om du stöter på problem som inte täcks i denna guide:

1. Kontrollera loggen i terminalen där servern körs
2. Sätt `DEBUG=true` i `.env`-filen för mer detaljerad loggning
3. Verifiera att alla filer finns på rätt plats
4. Starta om servern efter konfigurationsändringar

---

**Lycka till med din RAG Search System-installation!** 🚀
