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
    "sports","entertainment",
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
        "You are an expert AI assistant specializing in pharma/biotech news. "
        "Clean and deduplicate the provided news items. Remove irrelevant stories "
        "(stock, finance, promos, sports, cybertrucks, etc.). For each valid item, add a 'Tag' "
        "field with one of: 'Approval', 'Clinical Trial', 'Pipeline', 'Collaboration', "
        "'Financial', 'Other', 'Acquisition', 'Inspection'. Always return a valid JSON array."
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

def enrich_with_ai(news_items_batch):
    """
    Cleans, deduplicates, and tags a batch of news using Gemini.
    Falls back to keyword-based filtering if AI fails or gets rate-limited.
    """
    prompt = f"Process this batch of articles:\n{json.dumps(news_items_batch, indent=2, ensure_ascii=False)}"

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
                return result
            return keyword_based_filter(news_items_batch)  # Fallback if empty response

        except exceptions.ResourceExhausted as e:
            print(f"   ⏳ AI rate limit hit (Attempt {attempt+1}/{MAX_RETRIES})")
            if attempt < MAX_RETRIES - 1:
                time.sleep(15)
            else:
                print(f"   ⚠️  Max retries exceeded. Switching to keyword-based backup filtering...")
                return keyword_based_filter(news_items_batch)
                
        except exceptions.PermissionDenied as e:
            print(f"   🔴 AI permission denied (API key issue): {e}")
            print(f"   ⚠️  Switching to keyword-based backup filtering...")
            return keyword_based_filter(news_items_batch)
            
        except json.JSONDecodeError as e:
            print(f"   🔴 AI returned invalid JSON: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(5)
            else:
                print(f"   ⚠️  Could not parse AI response. Switching to keyword-based backup filtering...")
                return keyword_based_filter(news_items_batch)
            
        except Exception as e:
            print(f"   🔴 AI error on attempt {attempt+1}: {str(e)[:100]}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(5)
            else:
                print(f"   ⚠️  AI failed after {MAX_RETRIES} attempts. Switching to keyword-based backup filtering...")
                return keyword_based_filter(news_items_batch)
            
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