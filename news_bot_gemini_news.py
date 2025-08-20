import asyncio
import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
from urllib.parse import quote_plus, urlparse

from dotenv import load_dotenv
import feedparser
from google import genai
from google.genai import types
from telegram import Bot
import telegram

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="newsbot.log",
)

BOT_TOKEN = os.getenv("BOT_TOKEN", "")
CHANNEL_ID = os.getenv("CHANNEL_ID", "")

NEWS_HL = os.getenv("NEWS_HL", "ru")
NEWS_GL = os.getenv("NEWS_GL", "RU")
NEWS_CEID = os.getenv("NEWS_CEID", "RU:ru")

NEWS_TAGS_RAW = os.getenv(
    "NEWS_TAGS",
    """[
  "Каталония OR Catalonia",
  "Страна Басков OR Basque Country",
  "Галисия (Испания) OR Galicia Spain",
  "Фландрия OR Flanders",
  "Валлония OR Wallonia",
  "Бретань OR Brittany",
  "Корсика OR Corsica",
  "Шотландия OR Scotland",
  "Уэльс OR Wales",
  "Ирландия Северная OR Northern Ireland",
  "Силезия OR Silesia",
  "Трансильвания OR Transylvania",
  "Галичина OR Halychyna OR Galicia Ukraine",
  "Косово OR Kosovo",
  "Воеводина OR Vojvodina",
  "Абхазия OR Abkhazia",
  "Южная Осетия OR South Ossetia",
  "Приднестровье OR Transnistria",
  "Крым OR Crimea",
  "Донбасс OR Donbas",
  "Нагорный Карабах OR Artsakh OR Nagorno-Karabakh",
  "Палестина OR West Bank OR Gaza",
  "Голанские высоты OR Golan Heights",
  "Курдистан OR Kurdistan",
  "Тибет OR Tibet",
  "Синьцзян OR Xinjiang OR East Turkestan",
  "Кашмир OR Kashmir",
  "Белуджистан OR Balochistan",
  "Тайвань OR Taiwan",
  "Ачех OR Aceh",
  "Западное Папуа OR West Papua",
  "Кабинда OR Cabinda",
  "Западная Сахара OR Western Sahara",
  "Сомалиленд OR Somaliland",
  "Дарфур OR Darfur",
  "Новая Каледония OR Kanaky",
  "Бугенвиль OR Bougainville",
  "Квебек OR Quebec"
]""",
)
MAX_PER_RUN = int(os.getenv("MAX_PER_RUN", "5"))
MAX_PER_TAG = int(os.getenv("MAX_PER_TAG", "2"))
MAX_AGE_HOURS = int(os.getenv("MAX_AGE_HOURS", "24"))

ENABLE_SUMMARY = os.getenv("ENABLE_SUMMARY", "true").lower() == "true"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

STATE_FILE = Path(__file__).parent / "news_state.json"

bot: Optional[Bot] = Bot(token=BOT_TOKEN) if BOT_TOKEN else None
gemini_client: Optional[genai.Client] = genai.Client() if ENABLE_SUMMARY else None

def parse_tags(raw: str) -> List[str]:
    raw = raw.strip()
    if not raw:
        return []
    if raw.startswith("["):
        try:
            arr = json.loads(raw)
            return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            pass
    return [x.strip() for x in raw.split(",") if x.strip()]

def google_news_rss_url(query: str) -> str:
    q = quote_plus(query)
    return f"https://news.google.com/rss/search?q={q}&hl={NEWS_HL}&gl={NEWS_GL}&ceid={NEWS_CEID}"

def load_state() -> Dict[str, Any]:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"seen": {}, "last_run": None}

def save_state(state: Dict[str, Any]) -> None:
    try:
        tmp = STATE_FILE.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        os.replace(tmp, STATE_FILE)
    except Exception as e:
        logging.error(f"Failed to save state: {e}")

def hash_id(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()

def extract_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().lstrip("www.")
    except Exception:
        return "source"

def html_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def summarize(title: str, snippet: str, url: str) -> Optional[str]:
    if not (ENABLE_SUMMARY and gemini_client):
        return None
    try:
        prompt = (
            "Суммируй новость на русском в 1–2 предложениях. "
            "Не добавляй фактов, которых нет в источниках. "
            "Верни только краткий текст без ссылок и без Markdown.\n\n"
            f"Заголовок: {title}\n"
            f"Описание/сниппет: {snippet}\n"
            f"URL: {url}\n"
        )
        resp = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.5,
                max_output_tokens=160,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        text = (resp.text or "").strip()
        if not text:
            return None
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return " ".join(sentences[:2]).strip()
    except Exception as e:
        logging.warning(f"Gemini summarize failed: {e}")
        return None

async def post_message(text: str) -> None:
    if not bot:
        logging.critical("Bot is not initialized.")
        return
    try:
        await bot.send_message(chat_id=CHANNEL_ID, text=text, parse_mode="HTML", disable_web_page_preview=False)
    except telegram.error.TelegramError as e:
        logging.error(f"Telegram send_message error: {e}")

def should_post(published_ts: float) -> bool:
    if MAX_AGE_HOURS <= 0:
        return True
    age_hours = (time.time() - published_ts) / 3600.0
    return age_hours <= MAX_AGE_HOURS

def build_message(title: str, summary: Optional[str], link: str, domain: str, tag: str, published: Optional[datetime]) -> str:
    parts = [f"<b>{html_escape(title)}</b>"]
    if summary:
        parts.append(html_escape(summary))
    parts.append(f"Источник: <i>{html_escape(domain)}</i>")
    if published:
        dt = published.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        parts.append(f"Время: <i>{dt}</i>")
    parts.append(html_escape(link))
    # хештег из тега
    tag_hash = "#" + "".join(w.capitalize() for w in re.split(r"[\\s_/.,:;!()\\[\\]{}+-]+", tag) if w)[:50]
    parts.append(tag_hash)
    return "\n\n".join(parts)

async def run_once():
    if not BOT_TOKEN or not CHANNEL_ID:
        logging.critical("BOT_TOKEN/CHANNEL_ID missing.")
        return

    state = load_state()
    seen: Dict[str, Dict[str, str]] = state.get("seen", {})

    tags = parse_tags(NEWS_TAGS_RAW)
    if not tags:
        logging.warning("No NEWS_TAGS provided — nothing to do.")
        return

    total_posted = 0
    for tag in tags:
        if total_posted >= MAX_PER_RUN:
            break
        url = google_news_rss_url(tag)
        logging.info(f"Fetch RSS for tag '{tag}': {url}")
        feed = feedparser.parse(url)
        count_for_tag = 0

        for entry in feed.entries:
            if total_posted >= MAX_PER_RUN or count_for_tag >= MAX_PER_TAG:
                break

            title = entry.get("title", "").strip()
            link = entry.get("link", "").strip()
            summary_html = entry.get("summary", "") or entry.get("description", "") or ""
            published_parsed = entry.get("published_parsed") or entry.get("updated_parsed")
            if published_parsed:
                published_ts = time.mktime(published_parsed)
                published_dt = datetime.fromtimestamp(published_ts, tz=timezone.utc)
            else:
                published_ts = time.time()
                published_dt = None

            uid = hash_id(link or title)
            if uid in seen:
                continue
            if not should_post(published_ts):
                continue

            plain_snippet = re.sub("<[^>]+>", " ", summary_html)
            plain_snippet = re.sub(r"\s+", " ", plain_snippet).strip()
            short = summarize(title, plain_snippet, link) if ENABLE_SUMMARY else None

            domain = extract_domain(link)
            msg = build_message(title, short or plain_snippet, link, domain, tag, published_dt)
            await post_message(msg)

            seen[uid] = {"tag": tag, "url": link}
            total_posted += 1
            count_for_tag += 1

    state["seen"] = seen
    state["last_run"] = datetime.now(timezone.utc).isoformat()
    try:
        tmp = STATE_FILE.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        os.replace(tmp, STATE_FILE)
    except Exception as e:
        logging.error(f"Failed to save state: {e}")

if __name__ == "__main__":
    logging.info("News bot run started.")
    try:
        asyncio.run(run_once())
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
    logging.info("News bot run finished.")
