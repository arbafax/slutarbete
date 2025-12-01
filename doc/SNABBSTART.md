**VIKTIGT** Denna dokumentation är automatgenererad och _**inte**_ verifierad 2025-12-01 15:01

# RAG Search System - Snabbstart Checklista

En kortfattad checklista för att komma igång snabbt. För detaljerad information, se [INSTALLATIONSGUIDE.md](INSTALLATIONSGUIDE.md).

## ✅ Före installation

- [ ] Python 3.9 eller senare installerat
- [ ] Minst 4 GB RAM tillgängligt
- [ ] 2 GB ledigt diskutrymme
- [ ] Internetanslutning

## ✅ Installationssteg

### 1. Förbered projekt
```bash
# Skapa projektmapp
mkdir rag-search
cd rag-search

# Lägg alla projektfiler här:
# - server.py
# - rag_pipeline.py
# - helpers.py
# - requirements.txt
# - index.html
```

### 2. Skapa mappstruktur
```bash
# Windows
mkdir static uploads outputs data vector_stores
move index.html static\

# macOS/Linux
mkdir static uploads outputs data vector_stores
mv index.html static/
```

### 3. Virtuell miljö
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Installera paket
```bash
# Windows
pip install -r requirements.txt

# macOS/Linux
pip3 install -r requirements.txt
```

### 5. Konfigurera API-nyckel
```bash
# Skapa .env-fil
# Windows: echo. > .env
# macOS/Linux: touch .env

# Öppna och lägg till:
# GOOGLE_API_KEY=din_google_api_nyckel_här
```

**Skaffa Google API-nyckel:**
1. Gå till https://makersuite.google.com/app/apikey
2. Logga in med Google-konto
3. Klicka "Get API Key" eller "Create API Key"
4. Kopiera nyckeln till .env-filen

### 6. Starta servern
```bash
# Windows
python -m uvicorn server:app --reload --host 0.0.0.0 --port 8000

# macOS/Linux
python3 -m uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

### 7. Öppna webbgränssnitt
Öppna webbläsare och gå till: **http://localhost:8000**

---

## ✅ Snabbtest

### Test 1: URL-extrahering
- [ ] Gå till "Extrahera från URL"
- [ ] Ange: `https://sv.wikipedia.org/wiki/Sverige`
- [ ] Samlingens namn: "test"
- [ ] Klicka "Extrahera"
- [ ] Vänta på resultat

### Test 2: Sökning
- [ ] Gå till "Sök i samling"
- [ ] Välj samling: "test"
- [ ] Sökfråga: "befolkning"
- [ ] Klicka "Sök"
- [ ] Kontrollera resultat

### Test 3: AI-fråga
- [ ] Gå till "Fråga AI om samling"
- [ ] Välj samling: "test"
- [ ] Fråga: "Vad är Sveriges befolkning?"
- [ ] Klicka "Fråga AI"
- [ ] Läs svaret

**Om alla tester fungerar: GRATTIS! Systemet är korrekt installerat! 🎉**

---

## 🆘 Snabb felsökning

### Python hittas inte
**Windows:**
- Ominstallera Python, markera "Add Python to PATH"

**macOS/Linux:**
- Använd `python3` istället för `python`

### Paket kan inte installeras
```bash
# Prova:
pip install --upgrade pip
pip install -r requirements.txt --break-system-packages
```

### API-nyckel fungerar inte
- [ ] Kontrollera att `.env` finns i projektmappen
- [ ] Öppna `.env` och verifiera att nyckeln är korrekt
- [ ] Starta om servern efter ändringar i `.env`
- [ ] Testa nyckeln på https://makersuite.google.com/

### Port 8000 upptagen
```bash
# Använd annan port, t.ex. 8080:
python -m uvicorn server:app --reload --host 0.0.0.0 --port 8080

# Öppna då: http://localhost:8080
```

### Servern startar men ingen sida visas
- [ ] Kontrollera att `index.html` ligger i `static/`-mappen
- [ ] Testa: http://localhost:8000/index.html

---

## 📚 Nästa steg

När installationen fungerar:

1. **Läs användarguiden** - [ANVÄNDARGUIDE.md](ANVÄNDARGUIDE.md)
   - Lär dig alla funktioner
   - Tips och bästa praxis
   - Vanliga användningsfall

2. **Utforska embedding-modeller**
   - Prova olika backends (Google, Cohere, BGE-M3)
   - Optimera för dina dokument
   - Jämför kvalitet och hastighet

3. **Bygg dina samlingar**
   - Ladda upp dina egna PDFs
   - Scrapa relevanta webbsidor
   - Organisera i tematiska samlingar

4. **Avancerad användning**
   - Läs teknisk dokumentation - [TEKNISK_DOKUMENTATION.md](TEKNISK_DOKUMENTATION.md)
   - Anpassa system-prompts
   - Optimera prestanda

---

## 💡 Tips

### Produktivitet
- Skapa alias för att starta servern:
  ```bash
  # I ~/.bashrc eller ~/.zshrc:
  alias rag-start='cd ~/rag-search && source venv/bin/activate && python3 -m uvicorn server:app --reload'
  ```

### Backup
- Säkerhetskopiera `vector_stores/`-mappen regelbundet
- Exportera viktiga samlingar som JSON
- Spara `.env`-filen säkert (men dela aldrig den!)

### Uppdateringar
```bash
# Uppdatera alla paket
pip install --upgrade -r requirements.txt

# Uppdatera specifikt paket
pip install --upgrade fastapi
```

---

## 📞 Behöver mer hjälp?

- **Installation:** Se [INSTALLATIONSGUIDE.md](INSTALLATIONSGUIDE.md)
- **Användning:** Se [ANVÄNDARGUIDE.md](ANVÄNDARGUIDE.md)
- **Teknisk info:** Se [TEKNISK_DOKUMENTATION.md](TEKNISK_DOKUMENTATION.md)
- **Översikt:** Se [README.md](README.md)

---

**Lycka till!** 🚀
