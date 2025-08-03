"""
FastAPI Telegram webhook bot:
- Answers (Hebrew) about a single PDF via Gemini.
- Uses Telegram webhooks (HTTPS) — ideal for Cloud Run.
- Validates Telegram's secret token header.

ENV:
  TELEGRAM_BOT_TOKEN=...
  TELEGRAM_WEBHOOK_SECRET=some-long-random-string
  GEMINI_API_KEY=...
  PDF_PATH=./doc.pdf
  GEMINI_MODEL=gemini-2.5-flash   (or gemini-1.5-pro)
"""

from __future__ import annotations
import os
import time
from typing import Any, Dict, Optional, Tuple
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request, status
from google import genai
from google.genai import types

load_dotenv()

BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
WEBHOOK_SECRET: str = os.getenv("TELEGRAM_WEBHOOK_SECRET", "")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
PDF_PATH: str = os.getenv("PDF_PATH", "./doc.pdf")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

if not BOT_TOKEN or not GEMINI_API_KEY:
    raise SystemExit("Missing TELEGRAM_BOT_TOKEN or GEMINI_API_KEY")

TELEGRAM_API_BASE: str = f"https://api.telegram.org/bot{BOT_TOKEN}"

SYSTEM_HE: str = (
    "אתה מסייע בעברית בלבד. ענה בקצרה ובדיוק אך ורק על סמך המסמך המצורף (PDF). "
    "אם המידע אינו במסמך – אמור שאין מספיק מידע. הוסף ציטוט קצר במרכאות כשאפשר."
)

app = FastAPI(title="Hebrew PDF Q&A (Telegram Webhook)")

# HTTP client used for Telegram replies
http_client = httpx.AsyncClient(timeout=15.0)

# Gemini client + ephemeral file cache (48h retention → refresh ~40h)
genai_client = genai.Client(api_key=GEMINI_API_KEY)
_uploaded_file: Dict[str, Any] = {"uri": None, "mime": None, "ts": 0.0}


def _ensure_file_uploaded() -> Tuple[str, str]:
    """
    Upload local PDF to Gemini Files API if needed.
    Returns:
        (file_uri, mime_type)
    Raises:
        FileNotFoundError: if PDF_PATH does not exist.
        RuntimeError: on failed upload.
    """
    global _uploaded_file
    if _uploaded_file["uri"] and (time.time() - _uploaded_file["ts"] < 40 * 3600):
        return _uploaded_file["uri"], _uploaded_file["mime"]

    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF not found at {PDF_PATH}")

    uploaded = genai_client.files.upload(path=PDF_PATH)
    uri: Optional[str] = getattr(uploaded, "uri", None)
    mime: Optional[str] = getattr(uploaded, "mime_type", None)
    if not uri or not mime:
        raise RuntimeError("Failed to upload PDF to Gemini Files API.")

    _uploaded_file = {"uri": uri, "mime": mime, "ts": time.time()}
    return uri, mime


async def answer_question(q: str) -> str:
    """
    Generate a Hebrew answer grounded in the uploaded PDF.
    Args:
        q: user question text
    Returns:
        Hebrew answer string (or a short fallback).
    """
    file_uri, mime = _ensure_file_uploaded()
    resp = genai_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_text(SYSTEM_HE),
            types.Part.from_uri(file_uri=file_uri, mime_type=mime),
            types.Part.from_text(f"שאלה: {q}\nענה בעברית פשוטה, עם ציטוט קצר אם ניתן."),
        ],
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=800,
            response_mime_type="text/plain",
        ),
    )
    return (resp.text or "").strip() or "לא הצלחתי להפיק תשובה מהמסמך."


async def send_message(chat_id: int, text: str) -> None:
    """Send a text message via Telegram Bot API."""
    payload = {"chat_id": chat_id, "text": text}
    r = await http_client.post(f"{TELEGRAM_API_BASE}/sendMessage", json=payload)
    r.raise_for_status()


@app.get("/healthz")
async def healthz() -> Dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.post("/webhook")
async def telegram_webhook(
    request: Request,
    x_telegram_bot_api_secret_token: Optional[str] = Header(default=None),
) -> Dict[str, str]:
    """
    Telegram webhook endpoint.
    Validates the secret header (if configured), extracts message text, answers, and replies.
    """
    if WEBHOOK_SECRET:
        if x_telegram_bot_api_secret_token != WEBHOOK_SECRET:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Bad token")

    update: Dict[str, Any] = await request.json()

    # Only handle text messages
    msg: Optional[Dict[str, Any]] = update.get("message")
    if not msg:
        return {"status": "ignored"}

    chat = msg.get("chat", {})
    chat_id: Optional[int] = chat.get("id")

    text: str = (msg.get("text") or "").strip()
    if not chat_id or not text:
        return {"status": "ignored"}

    try:
        ans = await answer_question(text)
        await send_message(chat_id, ans)
        return {"status": "ok"}
    except Exception as e:
        try:
            await send_message(chat_id, f"שגיאה: {e}")
        except Exception:
            pass
        return {"status": "error"}
