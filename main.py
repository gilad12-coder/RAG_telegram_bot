from __future__ import annotations
import os
import time
import json
import uuid
import asyncio
import logging
import contextlib
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, Tuple, List
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request, status
from google import genai
from google.genai import types

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

load_dotenv()

BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
WEBHOOK_SECRET: str = os.getenv("TELEGRAM_WEBHOOK_SECRET", "")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8080"))
RELOAD: bool = os.getenv("RELOAD", "false").lower() in ("true", "1", "yes")
PDF_PATH: str = "./guide_HR.pdf"

FILE_CACHE_HOURS: float = 24.0
HTTP_TIMEOUT: float = 15.0
MAX_OUTPUT_TOKENS: int = 8192
TEMPERATURE: float = 0.1
MAX_TG_LEN: int = 4096

if not BOT_TOKEN or not GEMINI_API_KEY:
    logger.critical("Missing TELEGRAM_BOT_TOKEN or GEMINI_API_KEY")
    raise SystemExit("שגיאה קריטית: יש להגדיר את TELEGRAM_BOT_TOKEN ו-GEMINI_API_KEY")

TELEGRAM_API_BASE: str = f"https://api.telegram.org/bot{BOT_TOKEN}"

SYSTEM_PROMPT: str = (
    "אתה 'המסייע המשאני', בוט מידע של חיל משאבי האנוש בצה\"ל. "
    "תפקידך הוא לענות על שאלות אך ורק על סמך המידע מספר 'ניהול משרד המשא\"ן בשגרה' המצורף. "
    "התשובות שלך צריכות להיות מדויקות, תמציתיות, וכתובות בעברית תקנית. "
    "אם המידע אינו נמצא במסמך, עליך לציין זאת במפורש ולומר: 'המידע המבוקש אינו מופיע בספר'. "
    "במידת האפשר, שלב בתשובתך ציטוט קצר ורלוונטי מהמסמך במרכאות."
)

genai_client: genai.Client = genai.Client(api_key=GEMINI_API_KEY)
_uploaded_file: Dict[str, Any] = {"uri": None, "mime": None, "ts": 0.0}
http_client: Optional[httpx.AsyncClient] = None


def _extract_text_from_response(resp: Any) -> str:
    """Extracts best-effort plain text from a GenerateContent response."""
    text: str = (getattr(resp, "text", "") or "").strip()
    if text:
        return text
    parts: List[str] = []
    for cand in (getattr(resp, "candidates", None) or []):
        content = getattr(cand, "content", None)
        for part in (getattr(content, "parts", None) or []):
            t = getattr(part, "text", None)
            if t:
                parts.append(str(t))
    return "\n".join(p for p in parts if p).strip()


def _to_jsonable(obj: Any) -> Any:
    """Converts an arbitrary object to a JSON-serializable structure for logging."""
    try:
        return json.loads(json.dumps(obj, default=lambda o: getattr(o, "__dict__", str(o))))
    except Exception:
        return str(obj)


def _chunk(text: str, n: int = 4000) -> List[str]:
    """Splits text into chunks of at most n characters, preserving order."""
    if not text:
        return []
    return [text[i:i + n] for i in range(0, len(text), n)]


def _ensure_file_uploaded() -> Tuple[str, str]:
    """Ensures the PDF is uploaded to Gemini Files API and returns (uri, mime)."""
    global _uploaded_file
    cache_age_hours: float = (time.time() - _uploaded_file["ts"]) / 3600.0
    if _uploaded_file["uri"] and cache_age_hours < FILE_CACHE_HOURS:
        logger.info(f"Using cached file (URI: {_uploaded_file['uri']}). Cache age: {cache_age_hours:.2f} hours.")
        return _uploaded_file["uri"], _uploaded_file["mime"]
    if not os.path.exists(PDF_PATH):
        logger.error(f"PDF not found at configured path: {PDF_PATH}")
        raise FileNotFoundError(f"קובץ ה-PDF לא נמצא בנתיב: {PDF_PATH}")
    try:
        logger.info(f"Uploading file '{PDF_PATH}' to Gemini Files API...")
        try:
            uploaded: Any = genai_client.files.upload(file=PDF_PATH)
        except TypeError:
            uploaded = genai_client.files.upload(path=PDF_PATH)
        uri: Optional[str] = getattr(uploaded, "uri", None)
        mime: Optional[str] = getattr(uploaded, "mime_type", None)
        if not uri or not mime:
            logger.error("File API upload succeeded but returned invalid metadata.")
            raise RuntimeError("ההעלאה הצליחה אך לא החזירה מטא-דאטה תקין.")
        _uploaded_file = {"uri": uri, "mime": mime, "ts": time.time()}
        logger.info(f"File uploaded successfully. URI: {uri}")
        return uri, mime
    except Exception as e:
        logger.error(f"Failed to upload PDF to Gemini Files API: {e}", exc_info=True)
        raise RuntimeError(f"כשל בהעלאת קובץ ה-PDF: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes shared clients and preloads the PDF; cleans up on shutdown."""
    global http_client
    logger.info("Application lifespan starting up...")
    http_client = httpx.AsyncClient(timeout=HTTP_TIMEOUT)
    try:
        _ensure_file_uploaded()
    except Exception as e:
        logger.error(f"File warmup failed: {e}")
    yield
    if http_client:
        logger.info("Closing HTTP client...")
        await http_client.aclose()
    logger.info("Application lifespan finished.")


app: FastAPI = FastAPI(
    title="המסייע המשאני (Telegram Webhook)",
    description="בוט טלגרם למענה על שאלות מתוך ספר המשא\"ן באמצעות Gemini.",
    lifespan=lifespan,
)


async def answer_question(question: str) -> str:
    """Generates a Hebrew answer to the question using the uploaded PDF as context."""
    req_id: str = uuid.uuid4().hex[:8]
    t0: float = time.monotonic()
    logger.info(f"[{req_id}] Q received: {question!r}")
    try:
        file_uri, mime_type = _ensure_file_uploaded()
        try:
            response = await asyncio.wait_for(
                genai_client.aio.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[
                        types.Part.from_text(text=SYSTEM_PROMPT),
                        types.Part.from_uri(file_uri=file_uri, mime_type=mime_type),
                        types.Part.from_text(text=f"שאלה: {question}"),
                    ],
                    config=types.GenerateContentConfig(
                        temperature=TEMPERATURE,
                        max_output_tokens=MAX_OUTPUT_TOKENS,
                        response_mime_type="text/plain",
                    ),
                ),
                timeout=12.0,
            )
        except asyncio.TimeoutError:
            dt = (time.monotonic() - t0) * 1000.0
            logger.warning(f"[{req_id}] Generation timed out after {dt:.0f} ms")
            return "המענה מתעכב לרגע. אנא נסה שוב בעוד מספר שניות."
        answer: str = _extract_text_from_response(response) or "לא הצלחתי למצוא תשובה לשאלתך בספר. ייתכן שהמידע אינו קיים או שהשאלה אינה ברורה מספיק."
        dt_ms: float = (time.monotonic() - t0) * 1000.0
        usage = getattr(response, "usage_metadata", None)
        usage_dict: Dict[str, Any] = {}
        if usage:
            for k in ("input_tokens", "output_tokens", "total_tokens", "prompt_token_count", "candidates_token_count", "total_token_count"):
                if hasattr(usage, k):
                    usage_dict[k] = getattr(usage, k)
        cand_count = len(getattr(response, "candidates", []) or [])
        finish_reason = None
        safety = None
        if cand_count:
            c0 = response.candidates[0]
            finish_reason = getattr(c0, "finish_reason", None)
            safety = getattr(c0, "safety_ratings", None)
        preview = (answer.replace("\n", " ")[:200] + ("…" if len(answer) > 200 else ""))
        logger.info(f"[{req_id}] OK model={GEMINI_MODEL} latency_ms={dt_ms:.0f} tokens={usage_dict or '-'} candidates={cand_count} finish={finish_reason} preview={preview!r}")
        logger.debug(f"[{req_id}] FULL_ANSWER:\n{answer}")
        logger.debug(f"[{req_id}] RAW_RESPONSE_META: { _to_jsonable({'usage': usage_dict, 'safety': _to_jsonable(safety)}) }")
        return answer
    except FileNotFoundError:
        logger.error(f"[{req_id}] PDF missing at {PDF_PATH}", exc_info=True)
        return "אני מתנצל, נראה שקובץ המידע המרכזי (ספר המשא\"ן) אינו זמין כרגע. אנא נסה שוב מאוחר יותר."
    except Exception as e:
        dt_ms: float = (time.monotonic() - t0) * 1000.0
        logger.error(f"[{req_id}] Generation error after {dt_ms:.0f} ms: {e}", exc_info=True)
        return "אני מתנצל, אירעה שגיאה טכנית בעת עיבוד בקשתך."


async def send_message(chat_id: int, text: str) -> int:
    """Sends a Telegram message and returns the resulting message_id."""
    if not http_client:
        logger.error("HTTP client not initialized when trying to send message.")
        raise RuntimeError("HTTP client not initialized")
    payload: Dict[str, Any] = {"chat_id": chat_id, "text": text}
    response: httpx.Response = await http_client.post(f"{TELEGRAM_API_BASE}/sendMessage", json=payload)
    response.raise_for_status()
    data: Dict[str, Any] = response.json()
    msg_id: Optional[int] = (data.get("result") or {}).get("message_id")
    if msg_id is None:
        logger.warning(f"sendMessage returned no message_id: {data}")
        msg_id = -1
    logger.info(f"Message sent to chat_id={chat_id} message_id={msg_id}.")
    return msg_id


async def send_chat_action(chat_id: int, action: str = "typing") -> None:
    """Sends a chat action (e.g., typing) to indicate the bot is working."""
    if not http_client:
        raise RuntimeError("HTTP client not initialized")
    resp = await http_client.post(f"{TELEGRAM_API_BASE}/sendChatAction", json={"chat_id": chat_id, "action": action})
    resp.raise_for_status()


async def edit_message_text(chat_id: int, message_id: int, text: str) -> None:
    """Edits an existing Telegram message text by message_id."""
    if not http_client:
        raise RuntimeError("HTTP client not initialized")
    resp = await http_client.post(
        f"{TELEGRAM_API_BASE}/editMessageText",
        json={"chat_id": chat_id, "message_id": message_id, "text": text},
    )
    resp.raise_for_status()


async def _typing_pinger(chat_id: int, stop: asyncio.Event) -> None:
    """Continuously sends typing actions until stop is set."""
    while not stop.is_set():
        with contextlib.suppress(Exception):
            await send_chat_action(chat_id, "typing")
        await asyncio.sleep(4.0)


async def handle_update(chat_id: int, text: str) -> None:
    """Handles a user update: acknowledges, shows progress, generates, and delivers the final answer."""
    req: str = os.urandom(3).hex()
    logger.info(f"[{req}] handle_update start chat_id={chat_id} text={text!r}")
    placeholder_id: int = await send_message(chat_id, "קיבלתי! עובד על זה…")
    stop = asyncio.Event()
    pinger = asyncio.create_task(_typing_pinger(chat_id, stop))
    try:
        answer: str = await answer_question(text)
        chunks: List[str] = _chunk(answer, MAX_TG_LEN - 50) or ["לא הצלחתי להפיק תשובה."]
        await edit_message_text(chat_id, placeholder_id, chunks[0])
        for extra in chunks[1:]:
            await send_message(chat_id, extra)
        logger.info(f"[{req}] handle_update done; sent {len(chunks)} message(s)")
    except Exception as e:
        logger.error(f"[{req}] handle_update failed: {e}", exc_info=True)
        with contextlib.suppress(Exception):
            await edit_message_text(chat_id, placeholder_id, "אירעה שגיאה בעיבוד הבקשה. נסו שוב מאוחר יותר.")
    finally:
        stop.set()
        with contextlib.suppress(Exception):
            await pinger


@app.get("/healthz", include_in_schema=False)
async def healthz() -> Dict[str, str]:
    """Returns a simple health status."""
    return {"status": "ok"}


@app.post("/webhook")
async def telegram_webhook(
    request: Request,
    x_telegram_bot_api_secret_token: Optional[str] = Header(default=None),
) -> Dict[str, str]:
    """Receives Telegram updates, validates, and schedules asynchronous handling."""
    logger.info("Webhook endpoint triggered.")
    if WEBHOOK_SECRET:
        if x_telegram_bot_api_secret_token != WEBHOOK_SECRET:
            logger.warning("Invalid webhook secret token received.")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="טוקן אימות לא תקין")
        logger.info("Webhook secret token validated successfully.")
    else:
        logger.info("No webhook secret configured, skipping validation.")
    try:
        update: Dict[str, Any] = await request.json()
    except Exception:
        logger.error("Failed to parse incoming webhook JSON.", exc_info=True)
        return {"status": "error", "reason": "invalid_json"}
    message: Optional[Dict[str, Any]] = update.get("message")
    if not message:
        logger.info("Update received is not a message, skipping.")
        return {"status": "ok", "reason": "not_a_message"}
    chat: Dict[str, Any] = message.get("chat", {})
    chat_id: Optional[int] = chat.get("id")
    text: str = (message.get("text") or "").strip()
    if not chat_id or not text:
        logger.warning(f"Invalid message payload: chat_id={chat_id}, text='{text}'")
        return {"status": "ok", "reason": "invalid_message_payload"}
    asyncio.create_task(handle_update(chat_id, text))
    return {"status": "accepted"}


if __name__ == "__main__":
    try:
        import uvicorn
    except ImportError:
        logger.critical("uvicorn is required to run the server. Install it with: pip install uvicorn")
        raise SystemExit(1)
    logger.info(f"Starting server on {HOST}:{PORT} (reload={RELOAD})")
    logger.info(f"Health check available at: http://{HOST}:{PORT}/healthz")
    logger.info(f"Webhook endpoint available at: http://{HOST}:{PORT}/webhook")
    if not os.path.exists(PDF_PATH):
        logger.warning(f"PDF file not found at {PDF_PATH}. The bot will fail until the file is available.")
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=RELOAD,
        log_level="info",
    )