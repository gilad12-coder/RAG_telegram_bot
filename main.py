from __future__ import annotations
import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, Tuple
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request, status
from google import genai
from google.genai import types

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
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

FILE_CACHE_HOURS = 24
HTTP_TIMEOUT = 15.0
MAX_OUTPUT_TOKENS = 8192
TEMPERATURE = 0.1

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

genai_client = genai.Client(api_key=GEMINI_API_KEY)
_uploaded_file: Dict[str, Any] = {"uri": None, "mime": None, "ts": 0.0}
http_client: Optional[httpx.AsyncClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """ניהול מחזור החיים של היישום וניקוי משאבים."""
    global http_client
    logger.info("Application lifespan starting up...")
    http_client = httpx.AsyncClient(timeout=HTTP_TIMEOUT)
    yield
    if http_client:
        logger.info("Closing HTTP client...")
        await http_client.aclose()
    logger.info("Application lifespan finished.")


app = FastAPI(
    title="המסייע המשאני (Telegram Webhook)",
    description="בוט טלגרם למענה על שאלות מתוך ספר המשא\"ן באמצעות Gemini.",
    lifespan=lifespan
)


def _ensure_file_uploaded() -> Tuple[str, str]:
    """
    פונקציה זו מוודאת שהקובץ המקומי הועלה ל-Gemini Files API.
    היא משתמשת במטמון כדי להימנע מהעלאות חוזרות ונשנות.

    Returns:
        Tuple of (file_uri, mime_type) - מטא-דאטה של הקובץ שהועלה.

    Raises:
        FileNotFoundError: אם קובץ ה-PDF לא נמצא בנתיב שהוגדר.
        RuntimeError: במקרה של כישלון בהעלאת הקובץ.
    """
    global _uploaded_file

    cache_age_hours = (time.time() - _uploaded_file["ts"]) / 3600
    if _uploaded_file["uri"] and cache_age_hours < FILE_CACHE_HOURS:
        logger.info(f"Using cached file (URI: {_uploaded_file['uri']}). Cache age: {cache_age_hours:.2f} hours.")
        return _uploaded_file["uri"], _uploaded_file["mime"]

    if not os.path.exists(PDF_PATH):
        logger.error(f"PDF not found at configured path: {PDF_PATH}")
        raise FileNotFoundError(f"קובץ ה-PDF לא נמצא בנתיב: {PDF_PATH}")

    try:
        logger.info(f"Uploading file '{PDF_PATH}' to Gemini Files API...")
        uploaded = genai_client.files.upload(path=PDF_PATH)
        uri = getattr(uploaded, "uri", None)
        mime = getattr(uploaded, "mime_type", None)

        if not uri or not mime:
            logger.error("File API upload succeeded but returned invalid metadata.")
            raise RuntimeError("ההעלאה הצליחה אך לא החזירה מטא-דאטה תקין.")

        _uploaded_file = {"uri": uri, "mime": mime, "ts": time.time()}
        logger.info(f"File uploaded successfully. URI: {uri}")
        return uri, mime

    except Exception as e:
        logger.error(f"Failed to upload PDF to Gemini Files API: {e}", exc_info=True)
        raise RuntimeError(f"כשל בהעלאת קובץ ה-PDF: {e}")


async def answer_question(question: str) -> str:
    """
    מייצר תשובה בעברית לשאלה, בהתבסס על תוכן קובץ ה-PDF.

    Args:
        question: שאלת המשתמש.

    Returns:
        מחרוזת תשובה בעברית.
    """
    logger.info(f"Generating answer for question: '{question}'")
    try:
        file_uri, mime_type = _ensure_file_uploaded()

        generation_config = types.GenerateContentConfig(
            temperature=TEMPERATURE,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            response_mime_type="text/plain",
        )

        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            generation_config=generation_config,
            system_instruction=SYSTEM_PROMPT
        )

        response = await model.generate_content_async(
            [
                types.Part.from_uri(file_uri=file_uri, mime_type=mime_type),
                types.Part.from_text(f"שאלה: {question}"),
            ]
        )

        answer = (
                             response.text or "").strip() or "לא הצלחתי למצוא תשובה לשאלתך בספר. ייתכן שהמידע אינו קיים או שהשאלה אינה ברורה מספיק."
        logger.info(f"Generated answer: '{answer[:100]}...'")
        return answer

    except FileNotFoundError as e:
        logger.error(f"File not found during answer generation: {e}", exc_info=True)
        return "אני מתנצל, נראה שקובץ המידע המרכזי (ספר המשא\"ן) אינו זמין כרגע. אנא נסה שוב מאוחר יותר."
    except Exception as e:
        logger.error(f"Error in answer_question: {e}", exc_info=True)
        return f"אני מתנצל, אירעה שגיאה טכנית בעת עיבוד בקשתך."


async def send_message(chat_id: int, text: str) -> None:
    """
    שולח הודעת טקסט למשתמש דרך ה-API של טלגרם.

    Args:
        chat_id: מזהה הצ'אט בטלגרם.
        text: תוכן ההודעה לשליחה.

    Raises:
        httpx.HTTPStatusError: במקרה של שגיאת API.
    """
    if not http_client:
        logger.error("HTTP client not initialized when trying to send message.")
        raise RuntimeError("HTTP client not initialized")

    logger.info(f"Sending message to chat_id {chat_id}: '{text[:100]}...'")
    payload = {"chat_id": chat_id, "text": text}
    try:
        response = await http_client.post(f"{TELEGRAM_API_BASE}/sendMessage", json=payload)
        response.raise_for_status()
        logger.info(f"Message sent successfully to chat_id {chat_id}.")
    except httpx.HTTPStatusError as e:
        logger.error(
            f"Failed to send message to chat_id {chat_id}. Status: {e.response.status_code}, Response: {e.response.text}",
            exc_info=True)
        raise


@app.get("/healthz", include_in_schema=False)
async def healthz() -> Dict[str, str]:
    """נקודת קצה לבדיקת חיות המערכת."""
    return {"status": "ok"}


@app.post("/webhook")
async def telegram_webhook(
        request: Request,
        x_telegram_bot_api_secret_token: Optional[str] = Header(default=None),
) -> Dict[str, str]:
    """
    נקודת הקצה הראשית המקבלת עדכונים מטלגרם (Webhook).
    הפונקציה מאמתת את הבקשה, מחלצת את תוכן ההודעה, קוראת לפונקציה
    שמייצרת תשובה, ושולחת את התשובה בחזרה למשתמש.

    Args:
        request: אובייקט הבקשה מ-FastAPI.
        x_telegram_bot_api_secret_token: טוקן סודי לאימות (אופציונלי).

    Returns:
        מילון המציין את סטטוס העיבוד.

    Raises:
        HTTPException: במקרה של כשל באימות.
    """
    logger.info("Webhook endpoint triggered.")
    if WEBHOOK_SECRET:
        if x_telegram_bot_api_secret_token != WEBHOOK_SECRET:
            logger.warning("Invalid webhook secret token received.")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="טוקן אימות לא תקין"
            )
        logger.info("Webhook secret token validated successfully.")
    else:
        logger.info("No webhook secret configured, skipping validation.")

    try:
        update: Dict[str, Any] = await request.json()
    except Exception:
        logger.error("Failed to parse incoming webhook JSON.", exc_info=True)
        return {"status": "error", "reason": "invalid_json"}

    message = update.get("message")
    if not message:
        logger.info("Update received is not a message, skipping.")
        return {"status": "ok", "reason": "not_a_message"}

    chat = message.get("chat", {})
    chat_id = chat.get("id")
    text = (message.get("text") or "").strip()

    if not chat_id or not text:
        logger.warning(f"Received invalid message payload: chat_id={chat_id}, text='{text}'")
        return {"status": "ok", "reason": "invalid_message_payload"}

    logger.info(f"Processing message from chat_id {chat_id} with text: '{text}'")
    try:
        answer = await answer_question(text)
        await send_message(chat_id, answer)
        logger.info(f"Successfully processed and answered message from chat_id {chat_id}.")
        return {"status": "ok", "action": "answered"}

    except Exception as e:
        logger.error(f"Unhandled exception in telegram_webhook for chat_id {chat_id}: {e}", exc_info=True)
        error_message = f"אני מתנצל, אירעה שגיאה כללית במערכת. אנא נסה שוב מאוחר יותר."
        try:
            await send_message(chat_id, error_message)
        except Exception as send_err:
            logger.error(f"Failed to send final error message to user {chat_id}: {send_err}", exc_info=True)
            pass
        return {"status": "error", "details": str(e)}


if __name__ == "__main__":
    """
    Entry point for running the server directly.
    """
    try:
        import uvicorn
    except ImportError:
        logger.critical("uvicorn is required to run the server. Install it with: pip install uvicorn")

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
        log_level="info"
    )