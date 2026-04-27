import os
import time
import random
import hashlib
import json
import ssl
import smtplib
import re
from datetime import datetime
from email.message import EmailMessage
from email.utils import formataddr
from typing import Dict

import requests
import pandas as pd
from bs4 import BeautifulSoup
import google.generativeai as genai
from google.api_core import exceptions


# ==========================================================
# CONFIG
# ==========================================================

MODEL_NAME = "gemini-2.5-flash"
MAX_RETRIES = 3
REQUEST_DELAY = 2
SAVE_INTERVAL = 10
MAX_TEXT_LENGTH = 8000
MIN_PARAGRAPH_LENGTH = 50

# 🔴 TEST MODE
MAX_TEST_EMAILS = 4  # set None for unlimited

# Email
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
SENDER_EMAIL = "synapsebiopharma@gmail.com"
SENDER_NAME = "Synapse News Bot"
RECIPIENTS = ["krishna@synapsebiopharma.com"]

# Explicit (testing)
EMAIL_PASSWORD = "nwtungnsbcndsxne"


# ==========================================================
# GEMINI SETUP
# ==========================================================

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not set")

genai.configure(api_key=api_key)
model = genai.GenerativeModel(MODEL_NAME)


# ==========================================================
# HELPERS
# ==========================================================

def get_url_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()


def load_cache(file="summary_cache.json") -> Dict[str, str]:
    if os.path.exists(file):
        with open(file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: Dict[str, str], file="summary_cache.json"):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def markdown_to_plain_text(md: str) -> str:
    if not isinstance(md, str):
        return ""

    text = md
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

    text = re.sub(
        r"^##\s*(.*)",
        lambda m: f"\n{m.group(1).upper()}\n" + "-" * 60,
        text,
        flags=re.MULTILINE,
    )

    text = re.sub(r"^\s*[-*]\s+", "• ", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"`(.*?)`", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def is_valid_summary(summary: str) -> bool:
    if not isinstance(summary, str):
        return False

    s = summary.lower().strip()

    if (
        s.startswith("⚠️")
        or s.startswith("error")
        or "summarization failed" in s
        or len(s) < 150
    ):
        return False

    required = ["key highlights", "historical context", "importance"]
    return any(h in s for h in required)


# ==========================================================
# FETCH & SUMMARIZE
# ==========================================================

def fetch_article_text(url: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()

        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        article = soup.find("article") or soup
        paragraphs = [
            p.get_text(strip=True)
            for p in article.find_all("p")
            if len(p.get_text(strip=True)) > MIN_PARAGRAPH_LENGTH
        ]

        return " ".join(paragraphs)[:MAX_TEXT_LENGTH]

    except Exception:
        return ""


def summarize_article(text: str) -> str:
    if not text:
        return "⚠️ No content extracted"

    prompt = f"""
You are a pharma/biotech news analyst.

STRICT RULES:
- Do NOT include source URLs
- Do NOT include separators
- Do NOT repeat headings
- Do NOT add greetings or sign-offs

FORMAT:

## Key Highlights
- 3–5 bullets

## Historical Context
- 3–5 bullets

## Importance
- 3–5 bullets

OPTIONAL (only if relevant):
- Financial Impact
- Market Implications

ARTICLE TEXT:
{text}
"""

    for i in range(MAX_RETRIES):
        try:
            return model.generate_content(prompt).text.strip()
        except exceptions.ResourceExhausted:
            time.sleep(5 * (2 ** i))
        except Exception:
            pass

    return "⚠️ Summarization failed"


# ==========================================================
# EMAIL (HTML + TEXT)
# ==========================================================

def send_news_alert_email(headline, summary, source_url, company=None) -> bool:
    msg = EmailMessage()
    msg["From"] = formataddr((SENDER_NAME, SENDER_EMAIL))
    msg["To"] = ", ".join(RECIPIENTS)

    subject = headline if not company else f"{company} – {headline}"
    msg["Subject"] = subject[:150]

    # Plain text fallback
    text_body = f"""
FYIP

{headline}

{summary}

Source:
{source_url}

Regards,
Synapse News Bot
""".strip()

    msg.set_content(text_body)

    # HTML version
    html_body = f"""
<html>
<body style="font-family: Aptos, sans-serif; font-size:14px;">
    <p>FYIP</p>

   <h1 style="font-weight: bold; color: orange;">
        {headline}
    </h1>

    <pre style="white-space: pre-wrap; font-family: Aptos, sans-serif;">
{summary}
    </pre>

    <p>
        <strong>Source:</strong>
        <a href="{source_url}" target="_blank">View full article</a>
    </p>

    <p>
        Regards,<br>
        <strong>Synapse News Bot</strong>
    </p>
</body>
</html>
"""

    msg.add_alternative(html_body, subtype="html")

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=context) as server:
            server.login(SENDER_EMAIL, EMAIL_PASSWORD)
            server.send_message(msg)
        return True
    except Exception as e:
        print(f"❌ Email failed: {e}")
        return False


# ==========================================================
# MAIN PROCESS
# ==========================================================

def process_articles(
    input_file="news_results.xlsx",
    output_file="news_summarized.xlsx",
):
    df = pd.read_excel(input_file)
    df.columns = df.columns.str.lower().str.strip()

    # 🚫 Exclude Financial and Other tags
    df = df[~df["tag"].isin(["Financial", "Other"])]

    # print(df)
    cache = load_cache()
    emails_sent = 0

    for idx, row in df.iterrows():
        url = row.get("link")
        headline = row.get("title", "No Title")
        company = row.get("company")

        if not isinstance(url, str) or not url.startswith("http"):
            continue

        url_hash = get_url_hash(url)

        if url_hash in cache:
            clean_summary = cache[url_hash]
        else:
            text = fetch_article_text(url)
            raw = summarize_article(text)
            clean_summary = markdown_to_plain_text(raw)
            cache[url_hash] = clean_summary

        df.at[idx, "summary"] = clean_summary

        if (
            is_valid_summary(clean_summary)
            and emails_sent < MAX_TEST_EMAILS
        ):
            if send_news_alert_email(
                headline=headline,
                summary=clean_summary,
                source_url=url,
                company=company,
            ):
                emails_sent += 1
                print(f"📧 Email sent ({emails_sent}/{MAX_TEST_EMAILS})")

        time.sleep(REQUEST_DELAY)

    df.to_excel(output_file, index=False)
    save_cache(cache)
    print("✅ Processing complete")


# ==========================================================
# ENTRY
# ==========================================================

if __name__ == "__main__":
    process_articles()
