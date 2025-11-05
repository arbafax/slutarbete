# Slutarbete
Projekt som ska visa att jag lärt mig något på AI-kursen från YH

Projektet bygger på ett backend med FastAPI och en frontend med HTMX och Jinja-templates

#### Snabbstart – miljö & skelett
##### Förbered maskinen
        # Homebrew (Apple Silicon)
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

        # Lägg till brew i din shell (zsh)
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
        eval "$(/opt/homebrew/bin/brew shellenv)"

        # Python-hantering: uv (snabbt venv + dep)
        curl -LsSf https://astral.sh/uv/install.sh | sh

##### Skapa projekt
        mkdir ai-slutarbete && cd ai-slutarbete
        uv venv
        source .venv/bin/activate

        uv pip install fastapi uvicorn jinja2 python-multipart sqlmodel sqlalchemy \
        pydantic-settings httpx
        # AI-klienter – välj det du vill prova:
        uv pip install openai
        # (valfritt) för Ollama-klient:
        uv pip install ollama

##### Installera Ollama lokalt
        brew install ollama
        ollama serve
        ollama pull llama3.1

##### app (FastAPI + HTMX)
        ai-slutarbete/
        app/
            main.py
            templates/
            index.html
        .env         # fylls vid behov (API-nycklar etc.)

### Kör
        # Stå i projektets root-mapp (där ligger mappen _app_

        source .venv/bin/activate

        uvicorn app.main:app --reload

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

