import time
import re
import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime, timezone
import os
import logging
import smtplib
import ssl
from urllib.parse import quote_plus
from email.message import EmailMessage
from email.utils import formataddr, parsedate_to_datetime
from zoneinfo import ZoneInfo
import requests
from dotenv import load_dotenv

# --- Import AI filter (from ai_agent.py) ---
from agent.ai_agent import enrich_with_ai as enrich_news_data
from agent.ai_agent import keyword_based_filter


load_dotenv()

# ─── Logging setup ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("scraper_debug.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ─── Why RSS instead of HTML scraping? ───────────────────────────────────────
# Google's HTML search now serves a JS-challenge/bot-detection page to plain
# requests (no real browser). The Google News RSS feed is an official, stable,
# no-auth endpoint that returns clean XML — no CAPTCHA, no obfuscated
# selectors, no JS required.
#
# RSS feed format:
#  https://news.google.com/rss/search?q=QUERY+when:24h&hl=en-IN&gl=IN&ceid=IN:en
#
# "when:24h" limits results to the last 24 hours (same as tbs=qdr:d).
# ─────────────────────────────────────────────────────────────────────────────

companies = [
    "Chime Biologics",
    "Sandoz",
    "Stada",
    "Formycon",
    "Gedeon Richter",
    "Boehringer Ingelheim",
    "Hikma",
    "Amgen",
    "Amneal Kashiv Biosciences",
    "Organon",
    "Apotex",
    "Plant Form",
    "Coherus Biosciences",
    "Viatris-Mylan",
    "Pfizer",
    "Biogen",
    "Libbs Farmaceutica",
    "Blau Farmaceutica",
    "Cristália",
    "Farmacore",
    "Bionovis",
    "Biosidus",
    "Dr. Reddy's",
    "Intas",
    "Biocon",
    "Aurobindo Pharma",
    "Zydus Biotech",
    "Hetero Biopharma",
    "Shilpa Biologicals",
    "Stelis",
    "Samsung",
    "Celltrion",
    "Prestige Biopharma",
    "Daiichi Sankyo",
    "Dong A ST - Meiji Seika",
    "Kyowa Kirin",
    "JCR Pharma",
    "Kidswell Bio",
    "Nichi-Iko",
    "Mochida Pharmaceutical",
    "Polpharma",
    "Fresenius Kabi",
    "MS Pharma",
    "Alvotech",
    "Advanz Pharma",
    "Fuji Pharma",
    "Teva",
    "Jamp Pharma",
    "Samsung Bioepis"
]

# 🔹 Block unwanted keywords
# FIX #10 (minor): Use word-boundary matching to avoid false positives like
# "MSN Labs" being blocked because "msn" appears as a substring.
blocked_keywords = ["advertisement", "sponsored", "promo", "offer", "discount"]
blocked_whole_words = ["msn"]  # Matched as whole words only


def is_relevant(title: str, source: str = "") -> bool:
    text = (title + " " + source).lower()
    if any(word in text for word in blocked_keywords):
        return False
    # Whole-word matching for "msn" to avoid blocking "MSN Labs"
    if any(re.search(rf"\b{re.escape(word)}\b", text) for word in blocked_whole_words):
        return False
    return True


def decode_google_news_url(rss_link: str) -> str:
    """
    Google News RSS items wrap the real article URL in a Google redirect.
    This extracts the real URL where possible, falling back to the redirect URL.
    """
    if not rss_link.startswith("https://news.google.com"):
        return rss_link

    try:
        match = re.search(r"[?&]url=([^&]+)", rss_link)
        if match:
            return match.group(1)
    except Exception:
        pass

    return rss_link


def parse_rss_date(pub_date_str: str) -> str:
    """
    Convert RSS pubDate string (RFC 2822) to readable format.
    e.g. 'Mon, 24 Feb 2026 10:30:00 GMT' → '24-02-2026 10:30 UTC'
    """
    if not pub_date_str:
        return ""
    try:
        dt = parsedate_to_datetime(pub_date_str)
        dt_utc = dt.astimezone(timezone.utc)
        return dt_utc.strftime("%d-%m-%Y %H:%M UTC")
    except Exception:
        return pub_date_str


def fetch_google_news(company_name: str, query: str, limit: int = 15,
                      lang: str = "en-IN", country: str = "IN") -> list:
    """
    Fetch news from Google News RSS feed.
    Uses 'when:24h' to restrict to last 24 hours.

    Args:
        company_name: Clean company name stored in results (e.g. "Cipla")
        query: Search query string, may differ from company_name
    """
    encoded_query = quote_plus(f"{query} when:24h")
    url = (
        f"https://news.google.com/rss/search"
        f"?q={encoded_query}"
        f"&hl={lang}"
        f"&gl={country}"
        f"&ceid={country}:{lang.split('-')[0]}"
    )

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/rss+xml, application/xml, text/xml, */*",
    }

    logger.info(f"🔍 Fetching RSS for '{company_name}': {url}")

    try:
        response = requests.get(url, headers=headers, timeout=15)
    except requests.RequestException as e:
        logger.error(f"❌ Network error for '{company_name}': {e}")
        return []

    if response.status_code != 200:
        logger.error(f"❌ HTTP {response.status_code} for query: '{company_name}'")
        return []

    logger.debug(f"✅ HTTP 200 — RSS size: {len(response.text):,} chars")

    # ── Detect bot-detection / JS challenge page ───────────────────────────
    content_type = response.headers.get("Content-Type", "")
    if "html" in content_type and "xml" not in content_type:
        logger.error(
            "🚫 Received HTML instead of XML — Google may be rate-limiting. "
            "Wait a few minutes and try again."
        )
        with open(f"debug_bot_response_{company_name[:20]}.html", "w", encoding="utf-8") as f:
            f.write(response.text)
        logger.error(f"   Saved response to debug_bot_response_{company_name[:20]}.html")
        return []

    # ── Parse XML ─────────────────────────────────────────────────────────
    try:
        root = ET.fromstring(response.content)
    except ET.ParseError as e:
        logger.error(f"❌ XML parse error for '{company_name}': {e}")
        logger.debug(f"   Raw response snippet: {response.text[:500]}")
        with open(f"debug_xml_error_{company_name[:20]}.txt", "w", encoding="utf-8") as f:
            f.write(response.text)
        return []

    channel = root.find("channel")
    if channel is None:
        logger.warning(f"⚠️  No <channel> element found in RSS for '{company_name}'")
        return []

    items = channel.findall("item")
    logger.info(f"📰 Found {len(items)} RSS items for '{company_name}'")

    if not items:
        logger.warning(
            f"⚠️  0 articles in RSS for '{company_name}'. "
            "This may mean no news in last 24h, or the query returned nothing."
        )
        return []

    results = []
    seen_links = set()

    for idx, item in enumerate(items[:limit]):
        item_num = idx + 1
        try:
            # ── Title ──────────────────────────────────────────────────────
            title_el = item.find("title")
            if title_el is not None and title_el.text:
                raw_title = title_el.text.strip()
                title_parts = raw_title.rsplit(" - ", 1)
                title = title_parts[0].strip()
                logger.debug(f"  [{item_num}] ✅ Title  : {title[:80]}")
            else:
                title = ""
                logger.warning(f"  [{item_num}] ⚠️  Title NOT FOUND in <title> tag")

            # ── Source ─────────────────────────────────────────────────────
            source_el = item.find("source")
            if source_el is not None and source_el.text:
                source = source_el.text.strip()
                logger.debug(f"  [{item_num}] ✅ Source : {source}")
            elif title_el is not None and title_el.text and " - " in title_el.text:
                source = title_el.text.rsplit(" - ", 1)[-1].strip()
                logger.debug(f"  [{item_num}] ✅ Source : {source} (from title fallback)")
            else:
                source = ""
                logger.warning(
                    f"  [{item_num}] ⚠️  Source NOT FOUND — "
                    "tried <source> tag and title suffix"
                )

            # ── Link ───────────────────────────────────────────────────────
            # FIX #1: In RSS XML, <link> text is often stored as the tag's
            # .tail (text after the closing tag), not .text (text inside the
            # tag). We check both .text and .tail, then fall back to <guid>.
            link = ""
            link_el = item.find("link")
            if link_el is not None:
                # Prefer .text if non-empty, otherwise try .tail
                candidate = (link_el.text or "").strip() or (link_el.tail or "").strip()
                if candidate:
                    link = candidate
                    logger.debug(f"  [{item_num}] ✅ Link   : {link[:80]}")

            # Fallback: <guid> usually contains the same URL
            if not link:
                guid_el = item.find("guid")
                if guid_el is not None and guid_el.text:
                    link = guid_el.text.strip()
                    logger.debug(
                        f"  [{item_num}] ✅ Link   : {link[:80]} (from <guid> fallback)"
                    )

            if not link:
                logger.warning(
                    f"  [{item_num}] ⚠️  Link NOT FOUND — "
                    "tried <link>.text, <link>.tail, and <guid>"
                )

            # ── Deduplicate ────────────────────────────────────────────────
            if link and link in seen_links:
                logger.debug(f"  [{item_num}] ⏩ Duplicate link, skipped")
                continue
            if link:
                seen_links.add(link)

            # ── Date ───────────────────────────────────────────────────────
            pub_date_el = item.find("pubDate")
            if pub_date_el is not None and pub_date_el.text:
                date = parse_rss_date(pub_date_el.text.strip())
                logger.debug(f"  [{item_num}] ✅ Date   : {date}")
            else:
                date = ""
                logger.warning(f"  [{item_num}] ⚠️  Date NOT FOUND in <pubDate> tag")

            # ── Relevance filter ───────────────────────────────────────────
            if title and is_relevant(title, source):
                results.append({
                    "Company": company_name,  # FIX: store clean company name, not raw query
                    "Title": title,
                    "Source": source,
                    "Link": link,
                    "Date": date,
                })
                logger.debug(f"  [{item_num}] ✅ Added to results")
            else:
                if not title:
                    logger.debug(f"  [{item_num}] ⏩ Skipped (empty title)")
                else:
                    logger.debug(f"  [{item_num}] ⏩ Skipped (irrelevant): {title[:60]}")

        except Exception as e:
            logger.exception(f"  [{item_num}] ❌ Unexpected error parsing item: {e}")

    logger.info(f"✅ Collected {len(results)} valid articles for '{company_name}'")
    return results


# ---------------- COLLECT RESULTS ---------------- #
all_results = []
for company in companies:
    logger.info(f"─── Fetching news for: {company} ───")
    # Pass company name separately from query so stored Company is always clean
    results = fetch_google_news(company_name=company, query=company, limit=15)
    all_results.extend(results)
    time.sleep(2)

logger.info(f"📊 Total articles collected across all companies: {len(all_results)}")

# ---------------- SAVE RAW RESULTS ---------------- #
df = pd.DataFrame(all_results)
raw_filename = "news_results_raw.xlsx"
df.to_excel(raw_filename, index=False, engine="openpyxl")
logger.info(f"✅ Saved raw results: {raw_filename} ({len(df)} rows)")

# ---------- ENRICH WITH AI (with keyword-based backup) ---------- #
logger.info("🤖 Enriching results with AI (with keyword-based backup)...")
enriched_results = []

try:
    enriched_results = enrich_news_data(all_results)

    # FIX #6: Check for non-empty dicts, not just list truthiness.
    # enrich_news_data returning [{}] or [{}, {}] should be treated as failure.
    valid_enriched = [r for r in enriched_results if r and isinstance(r, dict) and any(r.values())]
    if valid_enriched:
        enriched_results = valid_enriched
        logger.info(f"✅ AI enrichment successful! Processed {len(enriched_results)} items")
    else:
        logger.warning("⚠️  AI returned empty/blank results. Falling back to keyword-based filtering...")
        enriched_results = keyword_based_filter(all_results)

except Exception as e:
    logger.exception(f"❌ AI enrichment failed: {e}")
    logger.info("⚠️  Using keyword-based backup filtering instead...")
    try:
        enriched_results = keyword_based_filter(all_results)
        logger.info(f"✅ Backup filtering completed! Processed {len(enriched_results)} items")
    except Exception as backup_error:
        logger.error(f"❌ Backup filtering also failed: {backup_error}")
        logger.warning("⚠️  Using raw results without filtering...")
        enriched_results = all_results

# Final safety net
if not enriched_results:
    logger.warning("⚠️  No enriched results available. Using raw results as fallback...")
    enriched_results = all_results

# Convert enriched results into DataFrame
df_enriched = pd.DataFrame(enriched_results)

# Add generation timestamp as a separate sheet, not a data row.
# FIX #7 (partially addressed): Timestamp now goes into a dedicated metadata
# sheet so it doesn't corrupt the main data table. The main sheet stays clean
# and machine-readable.
timestamp = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%d-%m-%Y %H:%M:%S")

filename = "news_results.xlsx"
with pd.ExcelWriter(filename, engine="openpyxl") as writer:
    df_enriched.to_excel(writer, index=False, sheet_name="News")
    meta_df = pd.DataFrame([{"Generated On (IST)": timestamp}])
    meta_df.to_excel(writer, index=False, sheet_name="Metadata")

logger.info(f"✅ Saved enriched results with metadata sheet: {filename}")

# ---------------- EMAIL SENDING VIA SMTP ---------------- #
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
SENDER_EMAIL = "synapsebiopharma@gmail.com"
SENDER_NAME = "Vedolizumab Bot"

# FIX #8: Fail loudly if EMAIL_PASSWORD is missing instead of passing None
# to smtplib and getting a cryptic TypeError at login time.
PASSWORD = os.getenv("EMAIL_PASSWORD")
if not PASSWORD:
    raise EnvironmentError(
        "❌ EMAIL_PASSWORD environment variable is not set. "
        "Add it to your .env file or export it before running this script."
    )

recipients = [
    "krishna@synapsebiopharma.com",
    # "utkarsh@synapsebiopharma.com",
    # "diksha@synapsebiopharma.com",
    "dipanshi@synapsebiopharma.com",
    # "navita@synapsebiopharma.com",
    # "satish@synapsebiopharma.com"
]

msg = EmailMessage()
msg["From"] = formataddr((SENDER_NAME, SENDER_EMAIL))
msg["To"] = ", ".join(recipients)
msg["Subject"] = f"Daily Pharma News Report - {timestamp}"
msg.set_content("Attached are the pharma news reports:\n\n- Raw Results\n- AI Enriched Results")

# Attach both Excel files — guard against missing files explicitly
for file in [raw_filename, filename]:
    if not os.path.exists(file):
        logger.error(f"❌ Attachment file not found, skipping: {file}")
        continue
    with open(file, "rb") as f:
        file_data = f.read()
    msg.add_attachment(
        file_data,
        maintype="application",
        subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=os.path.basename(file)
    )

try:
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=context) as server:
        server.login(SENDER_EMAIL, PASSWORD)
        server.send_message(msg)
    logger.info("✅ Email sent successfully with raw + enriched reports!")
except Exception as e:
    logger.exception(f"❌ Error sending email via SMTP: {e}")
