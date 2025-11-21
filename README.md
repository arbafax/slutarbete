# Slutarbete
Projekt som ska visa att jag lärt mig något på AI-kursen från YH

Projektet bygger på ett backend med FastAPI och en frontend med HTMX och Jinja-templates

#### Snabbstart 


### Kör
        # Stå i projektets root-mapp (där ligger mappen _app_

        source .venv/bin/activate

        uvicorn server:app --reload --host 0.0.0.0 --port 8000

        # Öppna http://127.0.0.1:8000 med webläsare

Vill man köra lokal LLM 

#### Kolla att Ollama finns

        ollama --version

Om “command not found”:

        brew install ollama

#### Starta Ollama-servern i ett separat fönster

        ollama serve

Låt den processen ligga kvar igång.

#### Hämta en modell (om du inte redan gjort det)

I ytterligare ett terminalfönster (eller samma där du kör serve innan du startar servern):

        ollama pull llama3.1

#### Snabbtest att API:et verkligen svarar

        curl http://localhost:11434/api/tags
        # ska ge JSON med listade modeller
        
        curl http://localhost:11434/api/generate -d '{"model":"llama3.1","prompt":"Hej! Säg något kort."}'
        # ska ge ett JSON-svar med text


### Server.py

För att köra server.py i en teminal navigera till mappen app /slutarbete och kör

        source .venv/bin/activate

navigera till mappen /app och kör

        uvicorn server:app --reload --host 0.0.0.0 --port 8000

## Så här fungerar det:

1. Ladda upp PDF → Systemet:

* Extraherar all text från PDF:en
* Delar upp i meningsfulla chunks (~512 tokens)
* Skapar embeddings med AI
* Bygger sökbart FAISS-index


2. Sök i dokumentet → Skriv naturlig språk som:

* "Vad säger dokumentet om AI?"
* "Sammanfatta huvudpunkterna"
* "Hitta information om säkerhet"


Få rankade resultat → De mest relevanta styckena baserat på semantisk likhet!

## Anteckningar av/för utvecklingen

### saker att kolla upp

kolla upp beautiful soup 

bs_obj = bs4(html.text, "html5lib")

### kommande steg i utvecklingen
* Skicka ett antal embeddings tillsammans med frågan till en LLM
* Sätta system-promt
* Sätta fråge-prompt
* Styla responsen
* Dokumentera koden
* Skriva användarinstruktioner
 