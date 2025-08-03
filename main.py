from __future__ import annotations
import os
import time
import json
import uuid
import asyncio
import logging
import contextlib
from collections import deque
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, Tuple, List, Deque
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
MAX_TURNS: int = 16
MAX_HISTORY_CHARS: int = 16000
DISPLAY_BUDGET: int = MAX_TG_LEN - 50
STREAM_EDIT_INTERVAL_SEC: float = 1.0

if not BOT_TOKEN or not GEMINI_API_KEY:
    logger.critical("Missing TELEGRAM_BOT_TOKEN or GEMINI_API_KEY")
    raise SystemExit("שגיאה קריטית: יש להגדיר את TELEGRAM_BOT_TOKEN ו-GEMINI_API_KEY")

TELEGRAM_API_BASE: str = f"https://api.telegram.org/bot{BOT_TOKEN}"

SYSTEM_PROMPT = """
אתה 'המסייע המש״אני' - מערכת סיוע למש"קים/ות משאבי אנוש בצה"ל.

## מקור הידע היחיד:
הספר 'ניהול משרד המשא"ן בשגרה' המצורף - זהו המקור היחיד שממנו תענה.

## כללי תשובה מחייבים:

### מקור המידע:
ענה אך ורק ממה שכתוב בספר המצורף
אם המידע לא מופיע בספר: אמור "המידע אינו מופיע בספר המצורף"
אסור להוסיף מידע כללי או הנחות - רק מה שכתוב מפורשות
צטט תמיד את מיקום המידע: [פרק X, עמ' Y]

### מבנה התשובה:
הליכים: הצג בשלבים ממוספרים (1, 2, 3...)
רשימות: השתמש בתבליטים
ציטוט: העתק ניסוח מדויק מהספר כשרלוונטי

### סגנון:
תמציתי וברור - ישר לעניין
עברית תקנית ומקצועית
פנייה ישירה (גוף שני)

### כשחסר מידע:
אם השאלה על נושא שלא בספר: "נושא זה אינו מכוסה בספר"
אם חסרים פרטים לתשובה מלאה: "בספר מופיע רק: [המידע החלקי שיש]"
אם צריך הבהרה מהשואל: "לתשובה מדויקת, אנא ציין: [מה חסר]"

### דוגמת תשובה:
שאלה: "איך מעדכנים חופשה?"
תשובה: "לפי הספר [פרק 3, עמ' 22]:
1. פתח את מערכת X
2. בחר 'עדכון חופשות'
3. הזן מספר אישי
4. מלא את הפרטים הנדרשים
5. שמור

הערה: לחופשות מיוחדות יש נוהל נוסף [עמ' 23]"

## חשוב:
אין לענות מידע כללי על צה"ל או משא"ן שלא מהספר
אין להמציא או להניח נהלים
חובה לציין כשמידע לא מופיע בספר

התפקיד: לספק מידע מדויק ואמין רק מהספר המצורף לסיוע בעבודה היומיומית.
"""

genai_client: genai.Client = genai.Client(api_key=GEMINI_API_KEY)
_uploaded_file: Dict[str, Any] = {"uri": None, "mime": None, "ts": 0.0}
http_client: Optional[httpx.AsyncClient] = None
chat_histories: Dict[int, Deque[Dict[str, str]]] = {}


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


def _get_history(chat_id: int) -> Deque[Dict[str, str]]:
    """Returns a bounded deque for a chat's turn-by-turn history."""
    if chat_id not in chat_histories:
        chat_histories[chat_id] = deque(maxlen=MAX_TURNS * 2)
    return chat_histories[chat_id]


def _prune_history_by_chars(hist: Deque[Dict[str, str]]) -> None:
    """Prunes history to keep total characters under the configured cap."""
    total = sum(len(h["text"]) for h in hist)
    while total > MAX_HISTORY_CHARS and hist:
        dropped = hist.popleft()
        total -= len(dropped["text"])


def _build_contents_for_chat(chat_id: int, question: str, file_uri: str, mime_type: str) -> List[types.Content]:
    """Builds a contents list that includes the PDF context and recent chat turns."""
    contents: List[types.Content] = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="המסמך המצורף הוא מקור הידע הראשי לתשובותיך."),
                types.Part.from_uri(file_uri=file_uri, mime_type=mime_type),
            ],
        )
    ]
    hist = _get_history(chat_id)
    _prune_history_by_chars(hist)
    for turn in hist:
        role = "user" if turn["role"] == "user" else "model"
        contents.append(
            types.Content(role=role, parts=[types.Part.from_text(text=turn["text"])])
        )
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=question)]))
    return contents


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


async def answer_question(chat_id: int, question: str) -> str:
    """Generates an answer using conversation history and the uploaded PDF as context."""
    req_id: str = uuid.uuid4().hex[:8]
    t0: float = time.monotonic()
    logger.info(f"[{req_id}] Q chat_id={chat_id} received: {question!r}")
    try:
        file_uri, mime_type = _ensure_file_uploaded()
        contents = _build_contents_for_chat(chat_id, question, file_uri, mime_type)
        response = await genai_client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=TEMPERATURE,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                response_mime_type="text/plain",
            ),
        )
        answer: str = _extract_text_from_response(response) or "לא הצלחתי למצוא תשובה לשאלתך בספר. ייתכן שהמידע אינו קיים או שהשאלה אינה ברורה מספיק."
        hist = _get_history(chat_id)
        hist.append({"role": "user", "text": question})
        hist.append({"role": "assistant", "text": answer})
        dt_ms: float = (time.monotonic() - t0) * 1000.0
        preview = (answer.replace("\n", " ")[:200] + ("…" if len(answer) > 200 else ""))
        logger.info(f"[{req_id}] OK latency_ms={dt_ms:.0f} preview={preview!r}")
        logger.debug(f"[{req_id}] FULL_ANSWER:\n{answer}")
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


async def stream_answer(chat_id: int, question: str, placeholder_id: int) -> str:
    """Streams model output, periodically editing the placeholder message; returns the final answer."""
    req_id: str = uuid.uuid4().hex[:8]
    t0: float = time.monotonic()
    logger.info(f"[{req_id}] stream start chat_id={chat_id} q={question!r}")
    file_uri, mime_type = _ensure_file_uploaded()
    contents = _build_contents_for_chat(chat_id, question, file_uri, mime_type)
    stream = await genai_client.aio.models.generate_content_stream(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=TEMPERATURE,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            response_mime_type="text/plain",
        ),
    )
    buf: List[str] = []
    last_edit: float = 0.0
    async for chunk in stream:
        piece: str = (getattr(chunk, "text", "") or "")
        if not piece:
            continue
        buf.append(piece)
        now = time.monotonic()
        if now - last_edit >= STREAM_EDIT_INTERVAL_SEC:
            current: str = "".join(buf)
            await edit_message_text(chat_id, placeholder_id, current[:DISPLAY_BUDGET])
            last_edit = now
    final_answer: str = "".join(buf).strip()
    if not final_answer:
        final_answer = "לא הצלחתי למצוא תשובה לשאלתך בספר. ייתכן שהמידע אינו קיים או שהשאלה אינה ברורה מספיק."
    dt_ms: float = (time.monotonic() - t0) * 1000.0
    preview = (final_answer.replace("\n", " ")[:200] + ("…" if len(final_answer) > 200 else ""))
    logger.info(f"[{req_id}] stream done latency_ms={dt_ms:.0f} preview={preview!r}")
    return final_answer


async def handle_update(chat_id: int, text: str) -> None:
    """Handles a user update with acknowledgment, progress feedback, conversational answer, and delivery."""
    req: str = os.urandom(3).hex()
    logger.info(f"[{req}] handle_update start chat_id={chat_id} text={text!r}")
    placeholder_id: int = await send_message(chat_id, "קיבלתי! עובד על זה…")
    stop = asyncio.Event()
    pinger = asyncio.create_task(_typing_pinger(chat_id, stop))
    try:
        answer: str = await stream_answer(chat_id, text, placeholder_id)
        hist = _get_history(chat_id)
        hist.append({"role": "user", "text": text})
        hist.append({"role": "assistant", "text": answer})
        await edit_message_text(chat_id, placeholder_id, answer[:DISPLAY_BUDGET])
        if len(answer) > DISPLAY_BUDGET:
            for extra in _chunk(answer[DISPLAY_BUDGET:], MAX_TG_LEN - 50):
                await send_message(chat_id, extra)
        logger.info(f"[{req}] handle_update done; total_len={len(answer)}")
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
    """Receives Telegram updates, validates, and schedules asynchronous handling with conversation context."""
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