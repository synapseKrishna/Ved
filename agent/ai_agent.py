import os
import time
import json
import pandas as pd
import google.generativeai as genai
from google.api_core import exceptions
from dotenv import load_dotenv
import re

# 1. Load the environment variables from your .env file
load_dotenv(override=True)

# 2. Setup API Key securely
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("🔴 GEMINI_API_KEY not set in environment or .env file")

genai.configure(api_key=api_key)

# 3. Optimized Configuration for Gemini 2.5 Flash Free Tier
MODEL_NAME = "gemini-2.5-flash"
BATCH_SIZE = 25          # 12 requests total for 300 rows (stays under 15 RPM)
COOL_DOWN_SECONDS = 5    # Prevents burst-limit throttling
MAX_RETRIES = 3

# ─── KEYWORD-BASED BACKUP FILTERING DICTIONARIES ─────────────────────────────
# Used when AI fails or gets rate-limited
BLOCKED_KEYWORDS = [
    "advertisement", "sponsored", "promo", "offer", "discount", "msn",
    "stock price", "market", "investor", "earnings", "trading", "crypto",
    "sports","entertainment","stock",
    "dividend", "bull market", "bear market", "nasdaq",
    "s&p 500", "dow jones", "pe ratio", "yield", "bonds", "hedge fund",
    "put option", "call option", "short selling", "volatility", "revenue", "profit margin",
    "stock split", "buyback", "dilution", "shareholders"
]

TAG_KEYWORDS = {
    "Approval": [
        "fda approval", "approved", "drug approval", "approved by fda",
        "ema approval", "regulatory approval", "gets approved", "receives approval"
    ],
    "Clinical Trial": [
        "clinical trial", "phase", "trial", "clinical study", "trial results",
        "trial data", "enrolled", "recruitment", "phase i", "phase ii", "phase iii"
    ],
    "Pipeline": [
        "pipeline", "development", "developing", "novel", "investigational",
        "lead candidate", "candidate molecule", "preclinical", "ind"
    ],
    "Collaboration": [
        "partnership", "collaboration", "partnered", "collaborated", "joint venture",
        "license", "deal", "agreement", "signed agreement", "announced partnership"
    ],
    "Acquisition": [
        "acquisition", "acquired", "acquisition of", "buyout", "takeover",
        "merger", "merged", "merge", "acquires", "purchased"
    ],
    "Inspection": [
        "inspection", "fda inspection", "audit", "warning letter",
        "compliance", "enforcement", "investigation", "FDA finds"
    ],
    "Financial": [
        "funding", "funding round", "investment", "investor", "capital raise",
        "series a", "series b", "series c", "ipo", "valuation", "raised"
    ],
}

# Initialize the model with System Instructions
model = genai.GenerativeModel(
    model_name=MODEL_NAME,
    system_instruction=(
        """You are a pharmaceutical competitive intelligence analyst working for a company monitoring vedolizumab and the biologics / biosimilar market.

Your job is to evaluate whether a piece of news is RELEVANT for daily monitoring. Apply the criteria below — vedolizumab is the primary signal, but biologics, biosimilars, FDA actions, manufacturing, partnerships, and market events may also qualify.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALWAYS RELEVANT — include if the item matches ANY of these:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. VEDOLIZUMAB / ENTYVIO / IBD PRIMARY SIGNAL
   - Any mention of: vedolizumab, Entyvio, Takeda Entyvio, vedolizumab biosimilar, anti-integrin therapy, α4β7 integrin, gut-selective biologic
   - Any news about vedolizumab formulation, lifecycle management, IV or subcutaneous versions, label expansion, patents, exclusivity, litigation, interchangeability, or biosimilar entry
   - Any regulatory, clinical, commercial, patent, or manufacturing news directly involving vedolizumab or Entyvio
   - Clinical trial results or pipeline news for ulcerative colitis, Crohn’s disease, inflammatory bowel disease, pouchitis, or other IBD indications when the product is biologic, biosimilar, or directly competes with vedolizumab

2. BIOSIMILARS & BIOLOGICS 
   - FDA approval, launch, CRL, acceptance, review, interchangeability decision, or litigation involving biosimilars 
   - Biosimilars or biologics in gastroenterology, immunology, autoimmune disease, inflammation, IBD, rheumatology, dermatology, or related specialty markets
   - Biosimilar launches, pricing, market access, payer coverage, formulary changes, substitution/interchangeability, or PBM decisions
   - Biologic lifecycle management, formulation change, new route of administration, device change, or new strength that affects competition
   - Patent litigation, settlements, PTAB/IPR, exclusivity disputes, or launch-at-risk events involving biologics or biosimilars

3. FDA / REGULATORY ACTIONS
   - FDA approvals, complete response letters, rejections, tentative approvals, BLA/sBLA/aBLA filings, NDA/ANDA only if relevant to biologics, biosimilars, IBD, immunology, or monitored companies
   - FDA inspections, Form 483 observations, warning letters, EIRs, import alerts, consent decrees, cGMP violations, or enforcement actions involving biologic or biosimilar manufacturing sites
   - FDA guidance, policy, or rule changes affecting biosimilars, biologics, interchangeability, BLA submissions, biologic manufacturing, labeling, pharmacovigilance, or post-marketing requirements
   - Advisory committee meetings, PDUFA dates, FDA decisions, or regulatory milestones for biologics, biosimilars, IBD, immunology, or autoimmune disease products

4. MARKET LAUNCHES, APPROVALS & COMMERCIAL EVENTS
   - launch or approval of biologics, biosimilars, specialty injectables, immunology drugs, IBD therapies, or autoimmune therapies
   - Market entry, commercialization, distribution, payer access, formulary placement, or major pricing actions for biologics or biosimilars 
   - Any market activity involving vedolizumab competitors or IBD-relevant drugs, including but not limited to:
     Humira / adalimumab biosimilars, Stelara / ustekinumab biosimilars, Skyrizi, Rinvoq, Zeposia, Omvoh, Tremfya, Simponi, Remicade / infliximab biosimilars, Cimzia, Entyvio, and other IBD biologics or advanced therapies

5. MANUFACTURING, SUPPLY CHAIN & CAPACITY
   - Biologic or biosimilar manufacturing expansions, new facilities, fill-finish capacity, cell culture capacity, sterile injectable capacity, CDMO deals, or supply agreements
   - Manufacturing events that affect supply of biologics or biosimilars
   - FDA inspections, warning letters, import alerts, or cGMP issues at plants producing or supplying biologics, biosimilars, sterile injectables, or monoclonal antibodies
   - Technology transfers, scale-up, process validation, batch failures, supply disruptions, recalls, shortages, or quality issues involving biologics or biosimilars

6. PARTNERSHIPS, DEALS, LICENSING & ACQUISITION
   - Mergers, acquisitions, licensing agreements, commercialization deals, co-promotion deals, distribution agreements, or asset purchases involving biologics, biosimilars, IBD drugs, immunology products
   - Out-licensing or in-licensing of rights for biologics, biosimilars, immunology, autoimmune, or IBD therapies
   - CDMO, manufacturing, supply, or development partnerships for biologics or biosimilars
   - Deals involving companies active in biologics, biosimilars, immunology, IBD, or specialty injectable markets

7. CLINICAL TRIALS & PIPELINE NEWS — ONLY IF RELEVANT
   - Clinical data, trial starts, trial failures, trial completions, label expansion studies, or regulatory trial updates for IBD, ulcerative colitis, Crohn’s disease, pouchitis, autoimmune disease, immunology, or biosimilar development
   - Comparative trials, switching studies, interchangeability studies, immunogenicity studies, pharmacokinetic studies, or real-world evidence involving biosimilars or biologics
   - Pipeline news involving monoclonal antibodies, biologics, biosimilars, or advanced therapies that could affect the IBD/immunology market

8. MONITORED COMPANIES
   Include relevant biologics, biosimilar, IBD, FDA, manufacturing, deal, or market news involving:
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
    "Polpharma Biologics"
    "Fresenius Kabi"
    "Samsung Biologics"
   - Any other company developing, manufacturing, launching, or litigating biologics, biosimilars, IBD drugs, or immunology therapies

9. EARNINGS & FINANCIAL REPORTS — ONLY WHEN BUSINESS-RELEVANT
   - Quarterly earnings, annual results, investor presentations, or guidance updates from monitored companies ONLY IF they mention:
     biologics revenue, biosimilar revenue, Entyvio, vedolizumab, IBD portfolio, immunology portfolio, specialty medicines, biologic manufacturing, FDA approvals,launches, pipeline milestones, or biosimilar strategy
   - Exclude earnings items that only discuss share price, margins, generic business, broad revenue, or analyst commentary without biologics / biosimilar / IBD relevance

10. INDUSTRY MEETINGS & CONFERENCES
   - Major scientific, regulatory, or industry meetings where biologics, biosimilars, IBD, gastroenterology, immunology, FDA policy, or market access are discussed
   - Relevant conferences include: DDW, ACG, ECCO, UEG Week, AGA, Crohn’s & Colitis Congress, BIO, DIA, RAPS, ASHP, AAM, GRx+Biosims, Festival of Biologics, World Biosimilar Congress
   - Investor days or R&D days from monitored companies when they cover biologics, biosimilars, IBD, immunology, or pipeline strategy

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DISEASE / THERAPY AREA FILTER — VERY IMPORTANT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

News about diseases such as lung cancer, Alzheimer’s disease, Parkinson’s disease, diabetes, obesity, cardiovascular disease, rare genetic diseases, vaccines, infectious disease, or oncology is NOT relevant by default.

Only mark such items relevant if they also contain one of the following:
   - A direct mention of vedolizumab, Entyvio, IBD, ulcerative colitis, Crohn’s disease, pouchitis, gastroenterology, biologics, biosimilars, monoclonal antibodies, FDA biosimilar policy, biologic manufacturing, or a monitored company’s biologics/biosimilar business
   - A FDA action that affects biologics or biosimilars
   - A manufacturing, acquisition, licensing, or supply-chain event involving biologics, biosimilars, monoclonal antibodies, or sterile injectable biologic products

Do NOT mark an article relevant just because it mentions the word “biosimilar” casually in a broad market context unless there is a concrete company, product, regulatory, launch, manufacturing, deal, or clinical event.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NEVER RELEVANT — exclude these:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- Pure stock/equity/trading news:
  stock price movements, share gains/declines, analyst ratings, buy/sell/hold recommendations, target price updates, brokerage reports, open interest, options activity, stock charts, “stocks to watch,” market-cap commentary, generic corporate financial profiles
- General business news with no specific biologic, biosimilar, IBD, FDA, manufacturing, launch, approval, acquisition, partnership, or regulatory event
- General healthcare policy commentary unless it specifically affects biologics, biosimilars, FDA approval pathways, interchangeability, market access, or IBD/immunology drugs
- Oncology, Alzheimer’s, lung cancer, diabetes, obesity, vaccines, infectious disease, or unrelated therapeutic-area news unless it meets the Disease / Therapy Area Filter above
- Clinical trial news for small molecules, oncology drugs, Alzheimer’s drugs, vaccines, or unrelated therapies unless directly tied to biologics/biosimilars or a monitored company’s relevant biologics business
- International-only product launches, approvals, or deals unless they involve the market, FDA, rights, supply,  manufacturing, or a globally important biosimilar/biologic competitor
- Generic drug ANDA news unless the article also involves biologics, biosimilars, IBD, immunology, FDA enforcement at biologic facilities, or a monitored company’s biologics strategy
- Corporate appointments, leadership changes, awards, ESG, CSR, hiring, office openings, or routine company announcements unless directly tied to biologics, biosimilars, IBD, FDA, manufacturing, or commercialization

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEDUPLICATION — apply before marking relevant:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You will receive all items for a category in one batch. Before scoring relevance, identify groups of items that cover the same underlying news event — even if they come from different sources, are worded differently, or have slightly different details.

For each duplicate group:
   - Mark only the single most informative item as relevant
   - Prefer the original company/FDA/regulatory source over secondary reports
   - Prefer the item with the most specific product, company, regulatory, approval, launch, manufacturing, or deal details
   - Mark all others in the group as not relevant with reason "duplicate of item N"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Respond with ONLY a JSON array, no markdown and no extra text.

Each object must follow this format:
[
  {"id": 1, "relevant": true, "reason": "brief one-line reason citing which category matched"},
  {"id": 2, "relevant": false, "reason": "duplicate of item 1"}
]

Reasons must be short, specific, and must cite the matched category, for example:
- "Vedolizumab / Entyvio primary signal"
- "biosimilar FDA approval"
- "Biologic manufacturing FDA warning letter"
- "IBD clinical pipeline news"
- "biologics licensing deal"
- "excluded: unrelated oncology news"
- "excluded: pure stock/analyst news"
- "duplicate of item 3"
"""
    )
)

# ─── BACKUP FILTERING FUNCTIONS ─────────────────────────────────────────────────

def is_relevant_keyword(title: str, source: str = "") -> bool:
    """Check if an article is relevant based on blocked keywords."""
    text = (title + " " + source).lower()
    return not any(keyword in text for keyword in BLOCKED_KEYWORDS)


def assign_tag_by_keywords(title: str, source: str = "") -> str:
    """Assign a tag based on keyword matching in title and source."""
    text = (title + " " + source).lower()
    
    # Check each tag category
    for tag, keywords in TAG_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in text:
                return tag
    
    return "Other"  # Default tag if no keywords match


def deduplicate_items(items: list) -> list:
    """Remove duplicate items based on title (case-insensitive)."""
    seen_titles = set()
    deduplicated = []
    
    for item in items:
        title = item.get("Title", "").lower().strip()
        if title and title not in seen_titles:
            seen_titles.add(title)
            deduplicated.append(item)
    
    return deduplicated


def keyword_based_filter(news_items_batch: list, use_ai: bool = True) -> list:
    """
    Keyword-based backup filtering when AI fails or gets rate-limited.
    - Removes irrelevant items
    - Assigns tags based on keywords
    - Deduplicates
    """
    print("   🔄 Applying keyword-based backup filtering...")
    
    filtered_items = []
    
    for item in news_items_batch:
        title = item.get("Title", "")
        source = item.get("Source", "")
        
        # Check relevance
        if not title:
            continue
            
        if not is_relevant_keyword(title, source):
            continue
        
        # Create filtered item with tag
        filtered_item = item.copy()
        if "Tag" not in filtered_item:
            filtered_item["Tag"] = assign_tag_by_keywords(title, source)
        
        filtered_items.append(filtered_item)
    
    # Deduplicate
    final_items = deduplicate_items(filtered_items)
    
    print(f"   ✅ Keyword filter: Kept {len(final_items)} items from {len(news_items_batch)} (removed {len(news_items_batch) - len(final_items)} irrelevant/duplicates)")
    
    return final_items

def is_stock_or_finance(title: str, source: str = "") -> bool:
    text = (title + " " + source).lower()
    
    # 1. Match stock ticker symbols or ISIN codes in parentheses, e.g. (NYSE:TEVA), (NASDAQ:AMGN), (PFE), (US7170811035), (PFE.F)
    if re.search(r'\([A-Z]{2,5}\)', title):
        return True
    if re.search(r'\([A-Z0-9.]{5,15}\)', title):
        return True
    if re.search(r'\b(NYSE|NASDAQ|TSE|LSE|OTCMKTS|LON)\s*:\s*[A-Z]{1,5}\b', title):
        return True
    
    # 2. Block stock/finance sites
    blocked_sources = [
        "simply wall.st", "simplywall.st", "seeking alpha", "seekingalpha", "motley fool", 
        "investorplace", "marketbeat", "tradingview", "investing.com", "zacks", 
        "insider monkey", "stocknews", "wall street zen", "benzinga", "investor news", "investornews"
    ]
    source_lower = source.lower()
    if any(src in source_lower for src in blocked_sources):
        return True
        
    # 3. Block strong financial/stock keywords
    blocked_finance_keywords = [
        "stock price", "share price", "valuation check", "valuation analysis", 
        "price target", "target price", "buy or sell", "stocks to watch", 
        "stocks to buy", "equity research", "options activity", "options trade", 
        "options chain", "market cap", "growth story", "insider buy", "insider sell", 
        "insider transaction", "quarterly earnings", "dividend", "yield", "bullish", 
        "bearish", "nasdaq", "nyse", "s&p 500", "dow jones", "brokerage report", 
        "consensus estimate", "analyst consensus", "broker consensus", "stock split", 
        "short selling", "should you buy", "should you sell", "should you hold", 
        "is it a buy", "is it a sell", "time to buy", "worth buying", "worth selling", 
        "worth holding", "outperform", "underperform", "market perform", "price forecast", 
        "stock forecast", "earnings forecast", "revenue forecast", "is a winner", 
        "winner in the", "growth story", "growth outlook", "long-term outlook"
    ]
    
    if any(keyword in text for keyword in blocked_finance_keywords):
        return True
        
    # 4. Whole-word matching for pure stock/investor terms to avoid false positives on words like "shares" as a verb
    if re.search(r'\b(stock|stocks|investor|investors|trading|valuation|ticker|tickers|dividend|dividends|equity|equities|brokerage|brokerages)\b', text):
        return True
        
    return False


def enrich_with_ai(news_items_batch):
    """
    Cleans, deduplicates, and tags a batch of news using Gemini.
    Falls back to keyword-based filtering if AI fails or gets rate-limited.
    """
    if not news_items_batch:
        return []

    # 0. Pre-filter the items in Python to immediately discard stock/finance noise
    filtered_input_batch = []
    for item in news_items_batch:
        title = item.get("Title", item.get("title", ""))
        source = item.get("Source", item.get("source", ""))
        if not is_stock_or_finance(title, source):
            filtered_input_batch.append(item)
            
    if not filtered_input_batch:
        print("   ⏩ All items in this batch filtered out as stock/finance news.")
        return []

    # 1. Assign ID to each item in the batch for robust correlation
    # Create copies of items to avoid modifying the input list in-place unexpectedly
    batch_with_ids = []
    for idx, item in enumerate(filtered_input_batch):
        item_copy = item.copy()
        item_copy["id"] = idx + 1
        batch_with_ids.append(item_copy)

    prompt = f"Process this batch of articles:\n{json.dumps(batch_with_ids, indent=2, ensure_ascii=False)}"

    for attempt in range(MAX_RETRIES):
        try:
            # Force valid JSON output
            response = model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            
            if response.text:
                result = json.loads(response.text)
                print(f"   ✅ AI enrichment successful for this batch")
                
                # Create a dictionary of AI assessments indexed by id
                ai_assessments = {}
                if isinstance(result, list):
                    for r in result:
                        if isinstance(r, dict) and "id" in r:
                            try:
                                ai_assessments[int(r["id"])] = r
                            except (ValueError, TypeError):
                                pass

                enriched_batch = []
                for idx, item in enumerate(filtered_input_batch):
                    item_id = idx + 1
                    assessment = ai_assessments.get(item_id)
                    
                    is_rel = False
                    reason = "Excluded by AI"
                    
                    if assessment:
                        is_rel = assessment.get("relevant")
                        if isinstance(is_rel, str):
                            is_rel = is_rel.lower() == "true"
                        reason = assessment.get("reason", "AI enriched")
                    
                    if is_rel:
                        enriched_item = item.copy()
                        enriched_item["Reason"] = reason
                        enriched_item["Relevant"] = True
                        
                        # Preserve case-insensitive title and source for tag assignment
                        title = item.get("Title", item.get("title", ""))
                        source = item.get("Source", item.get("source", ""))
                        
                        # Add tag if not present
                        if "Tag" not in enriched_item and "tag" not in enriched_item:
                            enriched_item["Tag"] = assign_tag_by_keywords(title, source)
                        
                        enriched_batch.append(enriched_item)
                
                return enriched_batch
                
            return keyword_based_filter(filtered_input_batch)  # Fallback if empty response

        except exceptions.ResourceExhausted as e:
            print(f"   ⏳ AI rate limit hit (Attempt {attempt+1}/{MAX_RETRIES})")
            if attempt < MAX_RETRIES - 1:
                time.sleep(15)
            else:
                print(f"   ⚠️  Max retries exceeded. Switching to keyword-based backup filtering...")
                return keyword_based_filter(filtered_input_batch)
                
        except exceptions.PermissionDenied as e:
            print(f"   🔴 AI permission denied (API key issue): {e}")
            print(f"   ⚠️  Switching to keyword-based backup filtering...")
            return keyword_based_filter(filtered_input_batch)
            
        except json.JSONDecodeError as e:
            print(f"   🔴 AI returned invalid JSON: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(5)
            else:
                print(f"   ⚠️  Could not parse AI response. Switching to keyword-based backup filtering...")
                return keyword_based_filter(filtered_input_batch)
            
        except Exception as e:
            print(f"   🔴 AI error on attempt {attempt+1}: {str(e)[:100]}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(5)
            else:
                print(f"   ⚠️  AI failed after {MAX_RETRIES} attempts. Switching to keyword-based backup filtering...")
                return keyword_based_filter(filtered_input_batch)
            
    # Final fallback (should not reach here, but just in case)
    return keyword_based_filter(news_items_batch)

def process_full_excel(input_filepath, output_filepath):
    """Reads raw Excel, processes in rate-limit-safe batches, and saves to new Excel."""
    print(f"📂 Loading data from {input_filepath}...")
    
    # Read the Excel file into a list of dictionaries
    df = pd.read_excel(input_filepath)
    # Ensure dates/NaNs don't break the JSON parser
    df = df.fillna("") 
    news_items = df.to_dict(orient='records')
    total_rows = len(news_items)
    
    print(f"📊 Found {total_rows} rows. Processing in batches of {BATCH_SIZE}...")
    
    final_results = []
    
    for i in range(0, total_rows, BATCH_SIZE):
        chunk = news_items[i : i + BATCH_SIZE]
        print(f"📦 Processing rows {i+1} to {min(i + BATCH_SIZE, total_rows)} of {total_rows}...")
        
        try:
            cleaned_chunk = enrich_with_ai(chunk)
            final_results.extend(cleaned_chunk)
            
            # Mandatory pause to stay completely under the radar of the rate limits
            if i + BATCH_SIZE < total_rows:
                time.sleep(COOL_DOWN_SECONDS)
                
        except Exception as e:
            print(f"⚠️ Fatal error on batch starting at row {i+1}: {e}")
            break
            
    # Save the cleaned JSON data directly back into a new Excel file
    if final_results:
        output_df = pd.DataFrame(final_results)
        
        # Reorder columns to put 'Tag' at the front for easy reading
        cols = output_df.columns.tolist()
        if 'Tag' in cols:
            cols.insert(0, cols.pop(cols.index('Tag')))
            output_df = output_df[cols]
            
        output_df.to_excel(output_filepath, index=False)
        print(f"✅ Success! Cleaned, tagged, and deduplicated data saved to {output_filepath}")
    else:
        print("❌ No data was successfully processed.")

if __name__ == "__main__":
    # Point these to your actual file names
    INPUT_EXCEL = "news_results_raw.xlsx"
    OUTPUT_EXCEL = "competitive_intelligence_cleaned.xlsx"
    
    process_full_excel(INPUT_EXCEL, OUTPUT_EXCEL)