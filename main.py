from __future__ import annotations
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, Tuple
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request, status
from google import genai
from google.genai import types

load_dotenv()

BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN")
WEBHOOK_SECRET: str = os.getenv("TELEGRAM_WEBHOOK_SECRET")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
PDF_PATH: str = os.getenv("PDF_PATH")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL")

FILE_CACHE_HOURS = 24
HTTP_TIMEOUT = 15.0
MAX_OUTPUT_TOKENS = 8192
TEMPERATURE = 0.1

if not BOT_TOKEN or not GEMINI_API_KEY:
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
    http_client = httpx.AsyncClient(timeout=HTTP_TIMEOUT)
    yield
    if http_client:
        await http_client.aclose()


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
        return _uploaded_file["uri"], _uploaded_file["mime"]

    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"קובץ ה-PDF לא נמצא בנתיב: {PDF_PATH}")

    try:
        print(f"מעלה את הקובץ '{PDF_PATH}' ל-Gemini Files API...")
        uploaded = genai_client.files.upload(path=PDF_PATH)
        uri = getattr(uploaded, "uri", None)
        mime = getattr(uploaded, "mime_type", None)

        if not uri or not mime:
            raise RuntimeError("ההעלאה הצליחה אך לא החזירה מטא-דאטה תקין.")

        _uploaded_file = {"uri": uri, "mime": mime, "ts": time.time()}
        print("הקובץ הועלה בהצלחה.")
        return uri, mime

    except Exception as e:
        raise RuntimeError(f"כשל בהעלאת קובץ ה-PDF: {e}")


async def answer_question(question: str) -> str:
    """
    מייצר תשובה בעברית לשאלה, בהתבסס על תוכן קובץ ה-PDF.

    Args:
        question: שאלת המשתמש.

    Returns:
        מחרוזת תשובה בעברית.
    """
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

        return (
                    response.text or "").strip() or "לא הצלחתי למצוא תשובה לשאלתך בספר. ייתכן שהמידע אינו קיים או שהשאלה אינה ברורה מספיק."

    except FileNotFoundError:
        return "אני מתנצל, נראה שקובץ המידע המרכזי (ספר המשא\"ן) אינו זמין כרגע. אנא נסה שוב מאוחר יותר."
    except Exception as e:
        print(f"Error in answer_question: {e}")
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
        raise RuntimeError("HTTP client not initialized")

    payload = {"chat_id": chat_id, "text": text}
    response = await http_client.post(f"{TELEGRAM_API_BASE}/sendMessage", json=payload)
    response.raise_for_status()


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
    if WEBHOOK_SECRET and x_telegram_bot_api_secret_token != WEBHOOK_SECRET:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="טוקן אימות לא תקין"
        )

    try:
        update: Dict[str, Any] = await request.json()
    except Exception:
        return {"status": "error", "reason": "invalid_json"}

    message = update.get("message")
    if not message:
        return {"status": "ok", "reason": "not_a_message"}

    chat = message.get("chat", {})
    chat_id = chat.get("id")
    text = (message.get("text") or "").strip()

    if not chat_id or not text:
        return {"status": "ok", "reason": "invalid_message_payload"}

    try:
        answer = await answer_question(text)
        await send_message(chat_id, answer)
        return {"status": "ok", "action": "answered"}

    except Exception as e:
        print(f"Error in telegram_webhook: {e}")
        error_message = f"אני מתנצל, אירעה שגיאה כללית במערכת. אנא נסה שוב מאוחר יותר."
        try:
            await send_message(chat_id, error_message)
        except Exception as send_err:
            print(f"Failed to send error message to user: {send_err}")
            pass
        return {"status": "error", "details": str(e)}