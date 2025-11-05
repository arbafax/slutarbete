import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates
from pydantic_settings import BaseSettings, SettingsConfigDict

import os
import httpx
import traceback


class Settings(BaseSettings):
    OPENAI_API_KEY: str | None = None
    OLLAMA_MODEL: str = "llama3.2:3b"  # snabb och bra för demo
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request, "answer": None}
    )


async def call_local_llm(prompt: str) -> str:
    import httpx

    OLLAMA_URL = "http://127.0.0.1:11434"
    async with httpx.AsyncClient(timeout=120) as client:  # längre timeout
        # Health check
        t = await client.get(f"{OLLAMA_URL}/api/tags")
        t.raise_for_status()

        # Kör modellen (justera namnet om du bytte)
        r = await client.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": "llama3.1:8b", "prompt": prompt, "stream": False},
        )
        r.raise_for_status()
        return r.json().get("response", "(ingen respons)")


##### async def call_openai(prompt: str) -> str:
#####     # Minimal OpenAI (Text responses med Responses API)
#####     headers = {"Authorization": f"Bearer {settings.OPENAI_API_KEY}"}
#####     payload = {"model": "gpt-4o-mini", "input": prompt}
#####     async with httpx.AsyncClient() as client:
#####         r = await client.post(
#####             "https://api.openai.com/v1/responses", json=payload, headers=headers
#####         )
#####     r.raise_for_status()
#####     data = r.json()
#####     # hämta text på ett enkelt sätt
#####     out = data.get("output", [])
#####     if out and isinstance(out, list):
#####         # concat eventuella text-chunks
#####         return "".join(
#####             [c.get("content", [{}])[0].get("text", "") for c in out if "content" in c]
#####         )
#####     return "(ingen respons)"

import os, httpx, asyncio
from openai import AsyncOpenAI

# client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
print(
    "OPENAI_API_KEY: " + client.api_key[:8] + "..." + client.api_key[-8:]
)  # visar första 8 tecknen


async def call_openai(prompt: str) -> str:
    # Separata tidsgränser för connect/read hjälper när nätet är segt
    # (SDK använder httpx under huven)
    try:
        logger.info("*****")
        resp = await client.responses.create(
            model="gpt-4o-mini",  # eller valfri modell
            input=prompt,
            timeout=60,  # total timeout i sekunder
        )
    except Exception as e:
        # Låt /ask hantera fel och maskera detaljer
        raise
    # Plocka ut texten på ett robust sätt
    # Nya SDK:ns .output_text sammanfogar innehåll
    return getattr(resp, "output_text", "").strip() or "(tomt svar)"


from fastapi import status


@app.post("/ask", response_class=HTMLResponse)
async def ask(
    request: Request, prompt: str = Form(...), provider: str = Form("ollama")
):
    try:
        if provider == "openai":
            if not settings.OPENAI_API_KEY:
                raise RuntimeError("Saknar OPENAI_API_KEY i .env")
            answer = await call_openai(prompt)
        else:
            answer = await call_local_llm(prompt)
    except httpx.ReadTimeout:
        answer = "⚠️ Tidsgräns mot OpenAI överskreds. Försök igen strax."
    except httpx.ConnectError:
        answer = f"⚠️ Kunde inte ansluta till {provider}. Kolla nätverket."
    except Exception:
        logger.error("ASK FAILED: %s")
        logger.debug("TRACE:\n%s", "".join(traceback.format_exc()))
        answer = "⚠️ Ett oväntat fel inträffade. Försök igen."

    if request.headers.get("HX-Request") == "true":
        return templates.TemplateResponse(
            "partials/answer.html", {"request": request, "answer": answer}
        )
    return templates.TemplateResponse(
        "index.html", {"request": request, "answer": answer, "last_prompt": prompt}
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/diag")
async def diag():
    # OBS! Läcker inget hemligt
    return {
        "openai_key_present": bool(settings.OPENAI_API_KEY),
        "ollama_model": settings.OLLAMA_MODEL,
    }
