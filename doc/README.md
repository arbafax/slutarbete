**VIKTIGT** Denna dokumentation är automatgenererad och _**inte**_ verifierad 2025-12-01 15:01

# RAG Search System - Dokumentation

Välkommen till dokumentationen för RAG Search System! Detta är ett kraftfullt verktyg för att skapa sökbara samlingar från PDF-filer och webbsidor, med AI-assisterad sökning och frågefunktionalitet.

## Dokumentationsöversikt

Dokumentationen är uppdelad i tre huvuddokument:

### 1. [INSTALLATIONSGUIDE.md](INSTALLATIONSGUIDE.md)
**För alla användare - Börja här!**

Denna guide hjälper dig att:
- Installera Python och nödvändiga verktyg
- Sätta upp projektet på din PC eller Mac
- Konfigurera API-nycklar
- Starta systemet första gången
- Lösa vanliga installationsproblem

**Passar för:** Alla som vill komma igång, oavsett teknisk bakgrund.

---

### 2. [ANVÄNDARGUIDE.md](ANVÄNDARGUIDE.md)
**För daglig användning**

Denna guide visar dig hur du:
- 📄 Laddar upp och bearbetar PDF-filer
- 🌐 Extraherar innehåll från webbsidor
- 🔍 Söker semantiskt i dina dokument
- 🤖 Ställer frågor till AI baserat på dina dokument
- 📊 Hanterar och organiserar samlingar
- 💡 Använder systemet effektivt

**Passar för:** Alla användare som vill lära sig använda systemets funktioner optimalt.

---

### 3. [TEKNISK_DOKUMENTATION.md](TEKNISK_DOKUMENTATION.md)
**För utvecklare och avancerade användare**

Denna guide innehåller:
- 🏗️ Systemarkitektur och design
- 🔧 API-dokumentation
- 💻 Kodstruktur och implementation
- ⚙️ Konfiguration och anpassning
- 🚀 Prestanda och optimering
- 🛠️ Utvecklingsguide

**Passar för:** Utvecklare som vill förstå systemet på djupet eller bidra med kod.

---

## 🚀 Snabbstart

### Steg 1: Installation
Följ [INSTALLATIONSGUIDE.md](INSTALLATIONSGUIDE.md) för detaljerade instruktioner.

**Snabbversion:**
```bash
# 1. Installera Python 3.9+
# 2. Skapa projektmapp och lägg till filer
# 3. Skapa virtuell miljö
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# eller: venv\Scripts\activate  # Windows

# 4. Installera dependencies
pip install -r requirements.txt

# 5. Skapa .env-fil med API-nyckel
echo "GOOGLE_API_KEY=din_nyckel" > .env

# 6. Starta servern
python3 -m uvicorn server:app --reload
```

### Steg 2: Öppna webbgränssnittet
Gå till `http://localhost:8000` i din webbläsare.

### Steg 3: Testa systemet
1. Ladda upp en PDF eller ange en URL
2. Vänta på att bearbetningen blir klar
3. Sök i samlingen eller ställ frågor till AI

Se [ANVÄNDARGUIDE.md](ANVÄNDARGUIDE.md) för detaljerad användningsinformation.

---

## 🎯 Vad kan systemet göra?

### Dokumentbearbetning
- **PDF-extrahering:** Extrahera och indexera text från PDF-filer
- **Webbscraping:** Hämta innehåll från webbsidor
- **Intelligent chunking:** Dela upp dokument i meningsfulla segment
- **Multiformat:** Stöd för olika dokumenttyper

### AI-funktioner
- **Semantisk sökning:** Hitta relevant information baserat på betydelse
- **Embeddings:** Flera backends (Google, OpenAI, Cohere, lokala modeller)
- **AI-assisterad Q&A:** Ställ frågor och få exakta svar från dina dokument
- **Flera LLM-modeller:** Google Gemini, OpenAI, Ollama (lokal)

### Samlingshantering
- **Skapa samlingar:** Organisera dokument i tematiska samlingar
- **Utöka samlingar:** Lägg till nya dokument i befintliga samlingar
- **Exportera:** Ladda ner samlingar som JSON
- **Radera:** Ta bort oanvända samlingar

---

## 🛠️ Systemkrav

### Minimikrav
- Python 3.9 eller senare
- 4 GB RAM
- 2 GB diskutrymme
- Internetanslutning

### Rekommenderat
- Python 3.11+
- 8 GB RAM eller mer
- SSD-disk
- Stabil internetanslutning för API-anrop

---

## 🔑 API-nycklar

Systemet kräver minst en API-nyckel för att fungera:

### Obligatorisk
- **Google API-nyckel** - Gratis från [Google AI Studio](https://makersuite.google.com/app/apikey)
  - Används för embeddings och LLM
  - Gratis tier tillgänglig

### Valfria
- **OpenAI API-nyckel** - För OpenAI embeddings/GPT
- **Cohere API-nyckel** - För Cohere embeddings
- **Ollama** - Lokal installation, ingen API-nyckel behövs

Se [INSTALLATIONSGUIDE.md](INSTALLATIONSGUIDE.md) för detaljer om hur du skaffar API-nycklar.

---

## 📖 Vanliga användningsfall

### Forskare
- Analysera forskningsartiklar
- Hitta relevanta citat och referenser
- Sammanfatta flera studier

### Företagsanvändare
- Analysera årsredovisningar
- Söka i policydokument
- Jämföra konkurrentinformation

### Studenter
- Sammanfatta kursmaterial
- Hitta information inför tentor
- Organisera kurslitteratur

### Produktteam
- Skapa intern dokumentationsportal
- AI-assisterad support
- Kunskapsbas för teamet

Se [ANVÄNDARGUIDE.md](ANVÄNDARGUIDE.md) för fler exempel och best practices.

---

## 🤝 Support och hjälp

### Problem med installation?
Se "Vanliga problem och lösningar" i [INSTALLATIONSGUIDE.md](INSTALLATIONSGUIDE.md).

### Frågor om användning?
Kolla [ANVÄNDARGUIDE.md](ANVÄNDARGUIDE.md) för detaljerad information.

### Tekniska frågor?
Se [TEKNISK_DOKUMENTATION.md](TEKNISK_DOKUMENTATION.md) eller API-dokumentationen.

### Fortfarande fast?
- Sätt `DEBUG=true` i `.env`-filen
- Kontrollera loggar i terminalen
- Verifiera API-nycklar
- Starta om servern

---

## 📋 Projektstruktur

```
rag-search/
├── .env                    # API-nycklar och konfiguration
├── server.py               # FastAPI-server med endpoints
├── rag_pipeline.py         # RAG-pipeline och embeddings
├── helpers.py              # Hjälpfunktioner
├── requirements.txt        # Python-dependencies
│
├── static/
│   └── index.html         # Webbgränssnitt
│
├── uploads/               # Temporära uppladdade filer
├── outputs/               # Genererade JSON-filer
├── vector_stores/         # FAISS-index och metadata
└── venv/                  # Virtuell Python-miljö
```

---

## 🔧 Teknisk stack

### Backend
- **Python 3.9+**
- **FastAPI** - Modern web framework
- **FAISS** - Vector similarity search
- **BeautifulSoup4** - HTML parsing
- **PyMuPDF** - PDF processing

### AI & ML
- **Google Gemini** - Embeddings och LLM
- **OpenAI** (valfri) - GPT-modeller
- **Cohere** (valfri) - Robust embeddings
- **Sentence-transformers** (valfri) - Lokala embeddings
- **Ollama** (valfri) - Lokal LLM

### Frontend
- **Vanilla JavaScript** - Ingen framework
- **Material Design** - UI-komponenter

---

## 📝 Licens

[Ange licens här]

---

## 🎉 Kom igång nu!

1. **[Installera systemet →](INSTALLATIONSGUIDE.md)**
2. **[Lär dig använda det →](ANVÄNDARGUIDE.md)**
3. **[Utforska tekniken →](TEKNISK_DOKUMENTATION.md)**

---

**Lycka till med ditt RAG Search System!** 🚀

*Uppdaterad: December 2024*
