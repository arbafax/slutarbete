from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates
from pydantic_settings import BaseSettings

import os
import httpx


class Settings(BaseSettings):
    OPENAI_API_KEY: str | None = None
    OLLAMA_MODEL: str = "llama3.1"


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


async def call_openai(prompt: str) -> str:
    # Minimal OpenAI (Text responses med Responses API)
    headers = {"Authorization": f"Bearer {settings.OPENAI_API_KEY}"}
    payload = {"model": "gpt-4o-mini", "input": prompt}
    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://api.openai.com/v1/responses", json=payload, headers=headers
        )
    r.raise_for_status()
    data = r.json()
    # hämta text på ett enkelt sätt
    out = data.get("output", [])
    if out and isinstance(out, list):
        # concat eventuella text-chunks
        return "".join(
            [c.get("content", [{}])[0].get("text", "") for c in out if "content" in c]
        )
    return "(ingen respons)"


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
    except Exception as e:
        answer = f"⚠️ Ett fel inträffade: {type(e).__name__}: {e} <br>***{settings.OPENAI_API_KEY}"

    # Om HTMX-anrop → returnera enbart partialen som ska in i #result
    if request.headers.get("HX-Request") == "true":
        return templates.TemplateResponse(
            "partials/answer.html", {"request": request, "answer": answer}
        )

    # Direktladdning (om någon gör vanlig POST) → hela sidan
    return templates.TemplateResponse(
        "index.html", {"request": request, "answer": answer, "last_prompt": prompt}
    )
