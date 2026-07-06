"""
Patent Thicket Analysis & Circumvention Strategy (505(b)(2)) — CLAUDE VERSION

Identical logic to ipd3bq__6_.py but uses Claude Sonnet 4.6 (via AnthropicVertex)
instead of Gemini 2.5 Flash for all LLM calls.

Key differences from ipd3bq__6_.py:
  - LLM backend: Claude Sonnet 4.6 via AnthropicVertex (replaces Gemini)
  - No Google Search grounding; online fallback uses Claude's knowledge directly
  - Output BQ tables: Circumvention_Table_Claude, Patent_Thicket_Score_Table_Claude
  - Separate GCS checkpoint: ipd3_checkpoints_claude/
  - Adds Model_Used column to every BQ row
  - Always writes results to local JSON files (for orchestrator eval)
  - Supports --no-bq to skip BQ entirely (for local testing)
  - alloydb_client / cog imports are optional (stubs used if missing)
"""

import os
import re
import json
import time
import asyncio
import argparse
import random
from datetime import datetime
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
from anthropic import AnthropicVertex, RateLimitError

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

# ── Optional imports: AlloyDB client & cog ────────────────────────────────────
# These are optional for testing. If not found, stubs are used and the pipeline
# will skip AlloyDB chunk fetching (online fallback only).

import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

_HAS_ALLOYDB = False
_HAS_COG = False

_alloydb_candidates = [
    os.path.join(SCRIPT_DIR, "cog"),
    os.path.join(SCRIPT_DIR, os.pardir, "cog"),
    SCRIPT_DIR,
]
_alloydb_dir = next(
    (d for d in _alloydb_candidates
     if os.path.exists(os.path.join(d, "alloydb_client.py"))),
    None,
)

if _alloydb_dir is not None:
    _alloydb_dir = os.path.abspath(_alloydb_dir)
    if _alloydb_dir not in sys.path:
        sys.path.insert(0, _alloydb_dir)
    try:
        from alloydb_client import AlloyDBClient
        _HAS_ALLOYDB = True
        print(f"[IPD3BQ-CLAUDE] alloydb_client located at: {_alloydb_dir}/alloydb_client.py")
    except ImportError:
        print("[IPD3BQ-CLAUDE] alloydb_client.py found but failed to import — using stubs")
else:
    print("[IPD3BQ-CLAUDE] alloydb_client.py not found — AlloyDB chunk fetching disabled (online fallback only)")

try:
    from cog import drug_filter as glp1_universe
    _HAS_COG = True
except ImportError:
    glp1_universe = None
    print("[IPD3BQ-CLAUDE] cog.drug_filter not found — GLP-1 drug filter disabled")


# ── AlloyDB Stubs (when alloydb_client is unavailable) ────────────────────────

class _AlloyDBStub:
    """Minimal stub so the pipeline runs without AlloyDB (returns no chunks)."""
    def get_collection(self, name):
        raise RuntimeError(f"AlloyDB stub: collection '{name}' not available")

# ── BigQuery Config ───────────────────────────────────────────────────────────
BQ_PROJECT_ID  = os.getenv("BQ_PROJECT_ID", "cognito-prod-394707")
BQ_DATASET_ID  = os.getenv("BQ_DATASET_ID", "cognito_prod_datamart")
BQ_TABLE_ID    = os.getenv("BQ_LOE_TABLE_NAME", "Master_LOE")
CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
BQ_LOCATION    = os.getenv("BQ_LOCATION", "asia-south1")

# ── LLM Config ────────────────────────────────────────────────────────────────
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
GCP_REGION   = os.getenv("GCP_REGION", "us-east5")
PROJECT_ID   = os.getenv("BQ_PROJECT_ID", "cognito-prod-394707")

# ── Output directory for local JSON results ───────────────────────────────────
OUTPUT_DIR = os.getenv("IPD3_OUTPUT_DIR", os.path.join(SCRIPT_DIR, "ipd3_output"))


def load_data_from_bigquery() -> pd.DataFrame:
    table_ref = f"{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{BQ_TABLE_ID}"
    print(f"[BigQuery] Connecting to: {table_ref}  (location={BQ_LOCATION})")
    credentials = _get_credentials()
    client = bigquery.Client(project=BQ_PROJECT_ID, credentials=credentials)
    query = f"""
    SELECT * EXCEPT(rn) FROM (
        SELECT *, ROW_NUMBER() OVER (
            PARTITION BY Patent_Number ORDER BY created_at DESC
        ) AS rn
        FROM `{table_ref}`
    ) WHERE rn = 1
    """
    df = client.query(query, location=BQ_LOCATION).to_dataframe()
    print(f"[BigQuery] Loaded {len(df)} rows")

    bq_to_code = {
        "Drug_Name": "Drug Name", "Patent_Number": "Patent Number",
        "Jurisdiction": "Jurisdiction", "Tag": "Tag",
        "Blocking_Category": "Blocking Category", "Reason": "Reason",
        "Step_1_Claim_Category": "Step 1 Claim Category",
        "Step_2_Matched_Elements": "Step 2 Matched Elements",
        "S2_Active_Ingredient__Form": "S2 Active Ingredient Form",
        "S2_Formulation_Details": "S2 Formulation Details",
        "S2_Route_of_Administration": "S2 Route of Administration",
        "S2_Device_Description": "S2 Device Description",
        "S2_Combination_TechProcess": "S2 Combination TechProcess",
        "Step_3_Technical_Barrier": "Step 3 Technical Barrier",
        "Step_3_Confidence": "Step 3 Confidence",
        "Step_3_Evidence_Type": "Step 3 Evidence Type",
        "Step_3_Evidence_Summary": "Step 3 Evidence Summary",
        "Step_4_Blocking_Indicator": "Step 4 Blocking Indicator",
        "Step_4_Confidence": "Step 4 Confidence",
        "Step_4_Regulatory_Failure_if_Removed": "Step 4 Regulatory Failure if Removed",
        "Step_4_Bridging_Studies_Required": "Step 4 Bridging Studies Required",
        "Step_4_Formulation_Consistent_Across_Phases": "Step 4 Formulation Consistent Across Phases",
        "Step_4_Reason": "Step 4 Reason",
        "Step_5_Novel__Difficult": "Step 5 Novel Difficult",
        "Step_5_Novelty_Signal": "Step 5 Novelty Signal",
        "Step_5_FirstinClass": "Step 5 FirstinClass",
        "Step_5_Prior_Failed_Attempts": "Step 5 Prior Failed Attempts",
        "Step_5_Complex_Implementation": "Step 5 Complex Implementation",
        "Step_5_Confidence": "Step 5 Confidence",
        "Step_5_Reason": "Step 5 Reason",
        "Filing_Date": "Filing Date", "Grant_Date": "Grant Date",
        "PTE_months": "PTE months", "Pediatric_Exclusivity": "Pediatric Exclusivity",
        "Phase": "Phase", "Launch_Date": "Launch Date",
        "Approval_Date": "Approval Date", "Approval_Date_Source": "Approval Date Source",
        "Est_Approval_Year": "Est Approval Year",
        "Exclusivity_Year": "Exclusivity Year",
        "Controlling_Patent_Expiry_Year": "Controlling Patent Expiry Year",
        "Years_to_Entry": "Years to Entry",
        "Avg_Years_to_Entry": "Avg Years to Entry",
        "Score": "Score", "Avg_Years_to_Entry_US__EP": "Avg Years to Entry US EP",
        "IP_Dimension_1_Score": "IP Dimension 1 Score",
        "Source_File": "Source File", "Type": "Type",
        "No_Of_Forecasted_Patents": "No Of Forecasted Patents",
    }
    df.rename(columns=bq_to_code, inplace=True)
    return df


# ── Circumvention Prompts (identical to original) ────────────────────────────

CIRCUMVENTION_PROMPT = """
You are a pharmaceutical patent attorney specializing in 505(b)(2) regulatory strategy
and patent design-around analysis.

Below are the most relevant excerpts from non-blocking patent(s) in the "{claim_category}"
category for the drug "{drug}":

--- PATENT EXCERPTS ---
{chunks}
--- END EXCERPTS ---

Patents in this category: {patent_numbers}
Drug: {drug}

Objective: Identify potential non-infringing product designs that avoid the innovator's
patent claims.

Evaluation Steps:
1. Identify the key claim limitations in the independent claims for each patent.
2. For the "{claim_category}" category, evaluate whether alternative solutions exist
   that omit at least one key limitation.
3. Consider known approaches from FDA Drugs@FDA approval packages, FDA Orange Book
   listed patents, and scientific literature.
4. Assess whether a non-infringing product modification could still be regulatorily
   viable under a 505(b)(2) pathway.

For context, here are example design-around strategies by category:
- Formulation: Alternative excipient or stabilization chemistry
- API Ratio / Dosing: Different clinically justified ratio outside scope of dosing claims
- Delivery Device: Mechanically distinct injector design
- Manufacturing: Different synthesis pathway (process patents are often narrow)
- Method of Use: New therapeutic indication

Respond ONLY with a valid JSON object — no markdown, no explanation outside the JSON:
{{
  "claim_category": "{claim_category}",
  "key_claim_limitations": ["<limitation 1>", "<limitation 2>", "<limitation 3>"],
  "design_around_strategies": [
    {{
      "strategy": "<concise description of the design-around approach>",
      "rationale": "<why this avoids infringement — which claim limitation is omitted>",
      "feasibility": "High / Medium / Low",
      "regulatory_pathway": "<how this could work under 505(b)(2) or other pathway>",
      "prior_art_support": "<any known precedent from FDA approvals, Orange Book, or literature>"
    }}
  ],
  "white_space_opportunities": ["<opportunity 1>", "<opportunity 2>"],
  "overall_circumvention_difficulty": "Easy / Moderate / Difficult",
  "summary": "<2-3 sentence overall assessment>"
}}
"""

CIRCUMVENTION_SEARCH_PROMPT = """
You are a pharmaceutical patent analyst. Analyse 505(b)(2) design-around opportunities
for the drug "{drug}" in the "{claim_category}" patent category.

Specifically consider:
1. FDA Drugs@FDA: alternative formulations, dosing regimens, delivery devices, or new indications
2. FDA Orange Book: listed patents and expiry dates to identify product features NOT protected
3. Scientific literature: {drug} + {claim_category_lower} alternatives

Return JSON:
{{
  "fda_precedents": ["<relevant FDA approval or product>"],
  "orange_book_gaps": ["<product feature not covered by listed patents>"],
  "literature_alternatives": ["<published alternative approach>"],
  "regulatory_viability": "<assessment of whether non-infringing modifications are feasible>"
}}

If no information found for a field, return an empty list for that field.
"""

CIRCUMVENTION_ONLINE_PROMPT = """
You are a pharmaceutical patent attorney specialising in 505(b)(2) regulatory
strategy and patent design-around analysis.

No patent text is available in the local database for the "{claim_category}"
category patents of "{drug}" (patent numbers: {patent_numbers}).

Using your knowledge of publicly available information — FDA Drugs@FDA, the FDA
Orange Book, scientific literature, and patent databases — perform a full
circumvention analysis for this category.

Return ONLY a valid JSON object:
{{
  "claim_category": "{claim_category}",
  "key_claim_limitations": ["<limitation 1>", "<limitation 2>", "<limitation 3>"],
  "design_around_strategies": [
    {{
      "strategy": "<description>",
      "rationale": "<why this avoids infringement>",
      "feasibility": "High / Medium / Low",
      "regulatory_pathway": "<505(b)(2) pathway>",
      "prior_art_support": "<any known precedent>"
    }}
  ],
  "white_space_opportunities": ["<opportunity 1>", "<opportunity 2>"],
  "overall_circumvention_difficulty": "Easy / Moderate / Difficult",
  "fda_precedents": ["<relevant FDA approval>"],
  "orange_book_gaps": ["<gap>"],
  "literature_alternatives": ["<alternative>"],
  "regulatory_viability": "<assessment>",
  "summary": "<2-3 sentence assessment>"
}}
"""

# ── Thicket Score ─────────────────────────────────────────────────────────────
THICKET_SCORE_LABELS = {
    5: "Exceptional – No meaningful secondary patent fence",
    4: "Strong – High probability of design-around",
    3: "Moderate – Requires structured strategy",
    2: "Weak – Circumvention and litigation become costly",
    1: "Poor – Dense Patent Thicket",
}


def compute_diversity_score(active_areas: int) -> int:
    if active_areas <= 2: return 5
    if active_areas == 3: return 3
    if active_areas <= 5: return 2
    return 1


def compute_density_score(adjusted_count: float) -> int:
    if adjusted_count <= 5: return 5
    if adjusted_count <= 15: return 3
    if adjusted_count <= 30: return 2
    return 1


def compute_final_score(total_patents, combined_total, adjusted_count, active_areas):
    diversity_score = compute_diversity_score(active_areas)
    effective_adjusted = max(0, adjusted_count - 1 if diversity_score == 5 else adjusted_count)
    density_score = compute_density_score(effective_adjusted)
    base_score = min(density_score, diversity_score)
    if density_score <= 2 and diversity_score <= 2:
        base_score -= 1
    validation_pct = (total_patents / combined_total * 100) if combined_total > 0 else 100.0
    if validation_pct < 50:
        base_score += 1
    final_score = max(1, min(5, base_score))
    return {
        "diversity_score": diversity_score, "density_score": density_score,
        "effective_adjusted_count": effective_adjusted, "base_score": base_score,
        "validation_pct": round(validation_pct, 1), "final_score": final_score,
        "final_label": THICKET_SCORE_LABELS[final_score],
    }


def patent_count_label(n):
    if n == 0:   return "No Patents"
    if n <= 5:   return "Low Density Patent Thicket (0–5)"
    if n <= 15:  return "Moderate Density Patent Thicket (6–15)"
    if n <= 30:  return "High Density Patent Thicket (16–30)"
    return "Dense / Multi-Layer Patent Thicket (>30)"

def diversity_label(areas):
    if areas == 0:  return "No Active Areas"
    if areas <= 2:  return "Low Diversity (1–2 areas)"
    if areas == 3:  return "Moderate Diversity (3 areas)"
    if areas <= 5:  return "High Diversity (4–5 areas)"
    return "Dense / Multi-Domain Patent Thicket (>5 areas)"


# ── AlloyDB Helpers ───────────────────────────────────────────────────────────

_alloydb_client = AlloyDBClient() if _HAS_ALLOYDB else _AlloyDBStub()

def get_chroma_clients():
    return [_alloydb_client]

def collection_name(drug: str) -> str:
    return f"patents_{drug.strip().replace(' ', '_')}"

def fetch_relevant_chunks(client, drug, source_file, sections, top_k=12):
    if not _HAS_ALLOYDB:
        return []
    if not source_file.strip():
        return []
    exact_filename = source_file.strip()

    def _query_collection(collection):
        docs = []
        try:
            results = collection.get(
                where={"$and": [{"filename": {"$eq": exact_filename}}, {"chunk_index": {"$gte": 0}}]},
                include=["documents", "metadatas"],
            )
            docs = results.get("documents", [])
        except Exception:
            pass
        if not docs:
            try:
                results = collection.get(
                    where={"filename": {"$eq": exact_filename}},
                    include=["documents", "metadatas"],
                )
                docs = [d for d, m in zip(results.get("documents", []), results.get("metadatas", []))
                        if m.get("chunk_index", -1) >= 0]
            except Exception as e:
                print(f"    [WARN] Query failed: {e}")
        return docs

    def _rank_and_slice(docs):
        def relevance_score(text):
            t = text.lower()
            return sum(1 for kw in sections if kw in t)
        return sorted(docs, key=relevance_score, reverse=True)[:top_k]

    primary_coll_name = collection_name(drug)
    try:
        primary_coll = _alloydb_client.get_collection(primary_coll_name)
        docs = _query_collection(primary_coll)
        if docs:
            print(f"    [INFO] Found {len(docs)} chunks in '{primary_coll_name}' for '{exact_filename}'")
            return _rank_and_slice(docs)
    except Exception as e:
        print(f"    [WARN] Could not query collection '{primary_coll_name}': {e}")

    print(f"    [INFO] '{exact_filename}' not found in '{primary_coll_name}' — no fallback scan.")
    return []


# ── Claude Helpers ────────────────────────────────────────────────────────────

_claude_client = None

def _get_claude_client() -> AnthropicVertex:
    global _claude_client
    if _claude_client is None:
        _claude_client = AnthropicVertex(region=GCP_REGION, project_id=PROJECT_ID)
    return _claude_client


def call_claude(prompt: str, max_retries: int = 3) -> dict:
    client = _get_claude_client()
    text = ""
    for attempt in range(max_retries + 1):
        try:
            resp = client.messages.create(
                model=CLAUDE_MODEL, max_tokens=8192,
                messages=[{"role": "user", "content": prompt}],
            )
            text = (resp.content[0].text or "").strip()
            text = re.sub(r"^```(?:json)?", "", text).strip()
            text = re.sub(r"```$", "", text).strip()
            return json.loads(text)
        except RateLimitError:
            if attempt == max_retries: raise
            wait = 2.0 * (2 ** attempt) * (1 + random.uniform(0, 0.25))
            print(f"    [Claude] Rate limited, retry {attempt+1}/{max_retries} in {wait:.1f}s")
            time.sleep(wait)
        except json.JSONDecodeError:
            return _repair_json(text)
        except Exception as e:
            if attempt == max_retries: raise
            time.sleep(2.0 * (2 ** attempt))
    return {}


def _call_claude_raw(prompt: str, max_retries: int = 3) -> str:
    client = _get_claude_client()
    for attempt in range(max_retries + 1):
        try:
            resp = client.messages.create(
                model=CLAUDE_MODEL, max_tokens=8192,
                messages=[{"role": "user", "content": prompt}],
            )
            return (resp.content[0].text or "").strip()
        except RateLimitError:
            if attempt == max_retries: raise
            time.sleep(2.0 * (2 ** attempt) * (1 + random.uniform(0, 0.25)))
        except Exception as e:
            if attempt == max_retries: raise
            time.sleep(2.0 * (2 ** attempt))
    return ""


def _extract_json_from_response(text):
    if not text: return None
    text = re.sub(r"^```(?:json)?", "", text.strip()).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else None


def _repair_json(text):
    try: return json.loads(text)
    except json.JSONDecodeError:
        text = re.sub(r",\s*}", "}", text)
        text = re.sub(r",\s*]", "]", text)
        try: return json.loads(text)
        except json.JSONDecodeError: return {"raw_text": text}


# ── Circumvention Analysis ────────────────────────────────────────────────────

import concurrent.futures
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=16)
CATEGORY_CONCURRENCY = 8


async def _analyse_one_category(drug_name, category, patents, chroma_client):
    print(f"\n[Circumvention] Category: {category} ({len(patents)} patent(s))")
    patent_numbers = [p["patent_number"] for p in patents]

    all_chunks = []
    for p in patents:
        chunks = fetch_relevant_chunks(
            chroma_client, p["drug_name"], p["source_file"],
            sections=["independent claim", "claim 1", "claims", "background",
                      "summary of invention", "abstract", "example",
                      "formulation", "device", "method", "dosing", "manufacturing"],
            top_k=8,
        )
        if chunks:
            all_chunks.extend(chunks[:6])

    loop = asyncio.get_event_loop()

    if all_chunks:
        chunks_text = "\n\n---\n\n".join(f"[Chunk {i+1}]\n{c}" for i, c in enumerate(all_chunks[:15]))
        prompt = CIRCUMVENTION_PROMPT.format(
            claim_category=category, drug=drug_name, chunks=chunks_text,
            patent_numbers=", ".join(str(x) for x in patent_numbers),
        )
        try:
            result = await loop.run_in_executor(_executor, call_claude, prompt)
        except Exception as e:
            print(f"    [WARN] Claude scoring error: {e}")
            result = {"claim_category": category, "design_around_strategies": [], "summary": f"Error: {e}"}

        # FDA/literature search
        search_prompt = CIRCUMVENTION_SEARCH_PROMPT.format(
            drug=drug_name, claim_category=category, claim_category_lower=category.lower(),
        )
        try:
            search_text = await loop.run_in_executor(_executor, _call_claude_raw, search_prompt)
            search_json = _extract_json_from_response(search_text)
            if search_json:
                search_data = _repair_json(search_json)
                if isinstance(search_data, dict):
                    result["fda_precedents"] = search_data.get("fda_precedents", [])
                    result["orange_book_gaps"] = search_data.get("orange_book_gaps", [])
                    result["literature_alternatives"] = search_data.get("literature_alternatives", [])
                    result["regulatory_viability"] = search_data.get("regulatory_viability", "")
        except Exception as e:
            print(f"    [WARN] FDA/lit search error: {e}")
            result.setdefault("fda_precedents", [])
            result.setdefault("orange_book_gaps", [])
            result.setdefault("literature_alternatives", [])
            result.setdefault("regulatory_viability", f"Search error: {e}")
    else:
        # Fallback: no chunks
        print(f"    [INFO] No AlloyDB chunks — generating via Claude online knowledge...")
        online_prompt = CIRCUMVENTION_ONLINE_PROMPT.format(
            claim_category=category, drug=drug_name,
            patent_numbers=", ".join(str(x) for x in patent_numbers),
        )
        try:
            online_text = await loop.run_in_executor(_executor, _call_claude_raw, online_prompt)
            online_json = _extract_json_from_response(online_text)
            if online_json:
                result = _repair_json(online_json)
                if not isinstance(result, dict):
                    result = json.loads(result)
            else:
                raise ValueError("Empty response")
        except Exception as e:
            print(f"    [WARN] Online circumvention error: {e}")
            result = {
                "claim_category": category, "key_claim_limitations": [],
                "design_around_strategies": [], "white_space_opportunities": [],
                "overall_circumvention_difficulty": "N/A",
                "fda_precedents": [], "orange_book_gaps": [],
                "literature_alternatives": [], "regulatory_viability": f"Error: {e}",
                "summary": "Patent text unavailable and online analysis failed. Manual review required.",
            }

    result["patent_numbers"] = patent_numbers
    result["patent_count"] = len(patents)
    print(f"    Found {len(result.get('design_around_strategies', []))} strategies")
    return category, result


async def run_circumvention_analysis(drug_name, patents_by_category, chroma_client):
    t0 = time.time()
    print(f"\n[Circumvention] Analysing for {drug_name} (Claude)...")
    semaphore = asyncio.Semaphore(CATEGORY_CONCURRENCY)

    async def _bounded(cat, pats):
        async with semaphore:
            return await _analyse_one_category(drug_name, cat, pats, chroma_client)

    tasks = [_bounded(cat, pats) for cat, pats in patents_by_category.items()]
    results_list = await asyncio.gather(*tasks)
    category_results = {cat: res for cat, res in results_list}
    elapsed = round(time.time() - t0, 1)
    print(f"\n[Circumvention] Completed for {drug_name} in {elapsed}s")
    return {
        "drug_name": drug_name,
        "categories_analysed": list(category_results.keys()),
        "results_by_category": category_results,
        "analysis_date": datetime.now().strftime("%Y-%m-%d"),
        "search_time_seconds": elapsed,
    }


def get_circumvention_for_drugs(non_blocking_df, chroma_client, max_patents_per_category=10):
    cat_col = None
    for col_name in ["Step 1 Claim Category", "Patent Type"]:
        if col_name in non_blocking_df.columns:
            cat_col = col_name
            break
    if cat_col is None:
        print("[Circumvention] WARNING: No category column found.")
        return {}

    drugs_categories = {}
    for _, row in non_blocking_df.iterrows():
        drug = str(row.get("Drug Name", "")).strip()
        pn = str(row.get("Patent Number", "")).strip()
        cat = str(row.get(cat_col, "")).strip()
        source_file = str(row.get("Source File", "")).strip()
        if not drug or not pn or cat in ("", "nan", "None"): continue
        if not source_file or source_file in ("nan", "None"): continue
        drugs_categories.setdefault(drug, {}).setdefault(cat, []).append(
            {"patent_number": pn, "drug_name": drug, "source_file": source_file})

    for drug in drugs_categories:
        for cat in drugs_categories[drug]:
            if len(drugs_categories[drug][cat]) > max_patents_per_category:
                drugs_categories[drug][cat] = drugs_categories[drug][cat][:max_patents_per_category]

    async def _run_all():
        tasks = {d: run_circumvention_analysis(d, pbc, chroma_client)
                 for d, pbc in drugs_categories.items()}
        names = list(tasks.keys())
        results = await asyncio.gather(*tasks.values())
        return dict(zip(names, results))

    return asyncio.run(_run_all())


# ── BigQuery Writers ──────────────────────────────────────────────────────────

BQ_CIRC_TABLE  = "Circumvention_Table_Claude"
BQ_SCORE_TABLE = "Patent_Thicket_Score_Table_Claude"

def _get_credentials():
    if CREDENTIALS_PATH and os.path.exists(CREDENTIALS_PATH):
        return service_account.Credentials.from_service_account_file(CREDENTIALS_PATH)
    return None

def _get_bq_client():
    return bigquery.Client(project=BQ_PROJECT_ID, credentials=_get_credentials())


def _flatten_circumvention(circumvention_by_drug):
    """Flatten circumvention results into rows (used by both BQ and JSON export)."""
    rows = []
    for drug_name, circ_data in circumvention_by_drug.items():
        analysis_date = circ_data.get("analysis_date", "")
        for category, cat_result in circ_data.get("results_by_category", {}).items():
            strategies = cat_result.get("design_around_strategies", [])
            common = dict(
                Drug_Name=drug_name, Patent_Category=category,
                Patents=", ".join(str(x) for x in cat_result.get("patent_numbers", [])),
                Num_Patents=int(cat_result.get("patent_count", 0)),
                Overall_Difficulty=cat_result.get("overall_circumvention_difficulty", "N/A"),
                Key_Claim_Limitations="; ".join(str(x) for x in cat_result.get("key_claim_limitations", [])),
                White_Space_Opportunities="; ".join(str(x) for x in cat_result.get("white_space_opportunities", [])),
                FDA_Precedents="; ".join(str(x) for x in cat_result.get("fda_precedents", [])),
                Orange_Book_Gaps="; ".join(str(x) for x in cat_result.get("orange_book_gaps", [])),
                Literature_Alternatives="; ".join(str(x) for x in cat_result.get("literature_alternatives", [])),
                Regulatory_Viability=cat_result.get("regulatory_viability", ""),
                Summary=cat_result.get("summary", ""),
                Analysis_Date=analysis_date, Model_Used="claude-sonnet-4-6",
            )
            if not strategies:
                rows.append({**common, "Strategy": "No strategies identified",
                             "Rationale": "", "Feasibility": "",
                             "Regulatory_Pathway": "", "Prior_Art_Support": ""})
            else:
                for s in strategies:
                    rows.append({**common, "Strategy": s.get("strategy", ""),
                                 "Rationale": s.get("rationale", ""),
                                 "Feasibility": s.get("feasibility", ""),
                                 "Regulatory_Pathway": s.get("regulatory_pathway", ""),
                                 "Prior_Art_Support": s.get("prior_art_support", "")})
    return rows


def _flatten_scores(drug_scores):
    """Flatten score results into rows (used by both BQ and JSON export)."""
    rows = []
    for sd in drug_scores:
        for jd in sd.get("jurisdiction_scores", []):
            rows.append({
                "Drug_Name": sd["drug"], "Jurisdiction": jd["jurisdiction"],
                "Combined_Total": jd["combined_total"], "Adjusted_Count": jd["adjusted_count"],
                "Active_Technology_Areas": jd["active_areas"],
                "Active_Categories": jd["active_categories"],
                "Density_Interpretation": jd["density_label"],
                "Diversity_Interpretation": jd["diversity_label"],
                "Density_Score": jd["density_score"], "Diversity_Score": jd["diversity_score"],
                "Base_Score": jd["base_score"], "Validation_Pct": jd["validation_pct"],
                "Final_Score": jd["final_score"], "Score_Label": jd["final_label"],
                "Model_Used": "claude-sonnet-4-6",
            })
        avg_final = sd["avg_final_score"]
        avg_final_int = max(1, min(5, round(avg_final)))
        rows.append({
            "Drug_Name": sd["drug"], "Jurisdiction": "Final Score (Average)",
            "Combined_Total": None, "Adjusted_Count": None,
            "Active_Technology_Areas": None, "Active_Categories": "",
            "Density_Interpretation": "", "Diversity_Interpretation": "",
            "Density_Score": None, "Diversity_Score": None,
            "Base_Score": None, "Validation_Pct": None,
            "Final_Score": avg_final, "Score_Label": THICKET_SCORE_LABELS.get(avg_final_int, ""),
            "Model_Used": "claude-sonnet-4-6",
        })
    return rows


def write_results_to_json(circumvention_by_drug, drug_scores):
    """Always write results to local JSON files for orchestrator consumption."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if circumvention_by_drug:
        circ_rows = _flatten_circumvention(circumvention_by_drug)
        path = os.path.join(OUTPUT_DIR, "circumvention_claude.json")
        with open(path, "w") as f:
            json.dump(circ_rows, f, indent=2, default=str)
        print(f"[JSON] Circumvention → {path} ({len(circ_rows)} rows)")

    if drug_scores:
        score_rows = _flatten_scores(drug_scores)
        path = os.path.join(OUTPUT_DIR, "scores_claude.json")
        with open(path, "w") as f:
            json.dump(score_rows, f, indent=2, default=str)
        print(f"[JSON] Scores → {path} ({len(score_rows)} rows)")


def write_circumvention_to_bq(circumvention_by_drug):
    rows = _flatten_circumvention(circumvention_by_drug)
    if not rows:
        print("[BQ] No circumvention rows to write."); return
    df_circ = pd.DataFrame(rows).drop_duplicates()
    df_circ["created_at"] = pd.Timestamp.now(tz="UTC")
    table_ref = f"{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{BQ_CIRC_TABLE}"
    client = _get_bq_client()
    try:
        existing = client.query(f"SELECT DISTINCT Drug_Name, Patent_Category FROM `{table_ref}`").to_dataframe()
        if not existing.empty:
            existing["_key"] = existing["Drug_Name"].str.strip() + "|" + existing["Patent_Category"].str.strip()
            df_circ["_key"] = df_circ["Drug_Name"].str.strip() + "|" + df_circ["Patent_Category"].str.strip()
            df_circ = df_circ[~df_circ["_key"].isin(set(existing["_key"]))].drop(columns=["_key"])
    except Exception:
        pass
    if df_circ.empty:
        print("[BQ] All rows already exist."); return
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND, autodetect=True,
        schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION],
    )
    client.load_table_from_dataframe(df_circ, table_ref, job_config=job_config, location=BQ_LOCATION).result()
    print(f"[BQ] {BQ_CIRC_TABLE}: {len(df_circ)} rows written")


def write_score_to_bq(drug_scores, refresh=False):
    rows = _flatten_scores(drug_scores)
    if not rows:
        print("[BQ] No score rows to write."); return
    df_score = pd.DataFrame(rows).drop_duplicates()
    df_score["created_at"] = pd.Timestamp.now(tz="UTC")
    table_ref = f"{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{BQ_SCORE_TABLE}"
    client = _get_bq_client()
    if refresh:
        for dn in df_score["Drug_Name"].dropna().unique():
            try:
                client.query(f"DELETE FROM `{table_ref}` WHERE Drug_Name = @drug",
                             job_config=bigquery.QueryJobConfig(
                                 query_parameters=[bigquery.ScalarQueryParameter("drug", "STRING", dn)]
                             )).result()
            except Exception: pass
    else:
        try:
            existing = client.query(f"SELECT DISTINCT Drug_Name, Jurisdiction FROM `{table_ref}`").to_dataframe()
            if not existing.empty:
                existing["_key"] = existing["Drug_Name"].str.strip() + "|" + existing["Jurisdiction"].str.strip()
                df_score["_key"] = df_score["Drug_Name"].str.strip() + "|" + df_score["Jurisdiction"].str.strip()
                df_score = df_score[~df_score["_key"].isin(set(existing["_key"]))].drop(columns=["_key"])
        except Exception: pass
    if df_score.empty:
        print("[BQ] All rows already exist."); return
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND, autodetect=True,
        schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION],
    )
    client.load_table_from_dataframe(df_score, table_ref, job_config=job_config, location=BQ_LOCATION).result()
    print(f"[BQ] {BQ_SCORE_TABLE}: {len(df_score)} rows written")


# ── Main ──────────────────────────────────────────────────────────────────────
EXCLUDED_CATEGORIES = {"Composition Of Matter"}

def process_patents(skip_circumvention=False, drug_filter=None, refresh_scores=False,
                    rerun=False, max_patents_per_category=10, no_bq=False):
    """
    Main entry point. Returns a dict with circumvention_by_drug and drug_scores
    so the orchestrator can capture results directly when importing this module.
    """
    # ── Load from BigQuery ────────────────────────────────────────────────
    df = load_data_from_bigquery()
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates()

    for col in ["Tag", "Step 1 Claim Category", "Phase", "Drug Name"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    df["Step 1 Claim Category"] = df["Step 1 Claim Category"].str.title()
    if "Jurisdiction" in df.columns:
        df["Jurisdiction"] = df["Jurisdiction"].astype(str).str.strip()
    if "Type" in df.columns:
        df["Type"] = df["Type"].astype(str).str.strip()

    forecasted_col = None
    for c in df.columns:
        if c.lower().replace(" ", "").replace("_", "") in (
            "noofforecastedpatents", "numberofforecastedpatents",
            "forecastedpatents", "noofforecasted",
        ):
            forecasted_col = c; break
    if forecasted_col is None and "No Of Forecasted Patents" in df.columns:
        forecasted_col = "No Of Forecasted Patents"
    if forecasted_col:
        df["No Of Forecasted Patents"] = pd.to_numeric(df[forecasted_col], errors="coerce").fillna(0).astype(int)
    else:
        df["No Of Forecasted Patents"] = 0

    all_categories = sorted(
        c for c in df["Step 1 Claim Category"].replace("nan", pd.NA).dropna().unique()
        if c not in EXCLUDED_CATEGORIES
    )

    non_blocking = df[df["Tag"].str.upper() == "NON-BLOCKING"].copy()
    if drug_filter:
        non_blocking = non_blocking[non_blocking["Drug Name"].str.lower() == drug_filter.lower()].copy()
        if non_blocking.empty:
            print(f"No NON-BLOCKING rows for drug: '{drug_filter}'"); return {}

    if non_blocking.empty:
        print("No NON-BLOCKING rows found."); return {}

    # GLP-1 filter (optional)
    if _HAS_COG and glp1_universe:
        if drug_filter:
            if not glp1_universe.require_allowed_drug(drug_filter):
                return {}
        else:
            before_names = sorted(non_blocking["Drug Name"].dropna().unique())
            allowed_names = glp1_universe.filter_allowed_drugs(before_names)
            if len(allowed_names) != len(before_names):
                allowed_norm = {glp1_universe.normalize(d) for d in allowed_names}
                non_blocking = non_blocking[
                    non_blocking["Drug Name"].apply(lambda d: glp1_universe.normalize(d) in allowed_norm)
                ].copy()
                if non_blocking.empty:
                    print("No GLP-1 drugs remain."); return {}

    drugs = non_blocking["Drug Name"].unique()

    # Checkpoint (optional — skip if cog not available)
    completed_drugs = set()
    _save_ckpt = lambda: None
    if _HAS_COG and not (refresh_scores or rerun):
        try:
            from cog import gcs_cache
            ckpt = gcs_cache.read_json("ipd3_checkpoints_claude", "completed_drugs_claude.json")
            if ckpt and isinstance(ckpt, list):
                completed_drugs = set(ckpt)
                print(f"[CHECKPOINT] {len(completed_drugs)} drugs already done (Claude)")
            def _save_ckpt():
                try: gcs_cache.write_json("ipd3_checkpoints_claude", "completed_drugs_claude.json", sorted(completed_drugs))
                except Exception: pass
        except Exception:
            pass

    drug_scores = []
    for drug in drugs:
        if drug in completed_drugs:
            print(f"\n  [SKIP] {drug} — already completed"); continue

        drug_df = non_blocking[non_blocking["Drug Name"] == drug].copy()
        forecasted_drug_df = df[(df["Drug Name"] == drug) & (df["Type"].str.lower() == "forecasted")].copy()
        filtered_cats = [c for c in all_categories if c not in EXCLUDED_CATEGORIES]

        def compute_block_scores(block_df, forecasted_source_df):
            category_counts = block_df.groupby("Step 1 Claim Category").size().to_dict()
            category_counts = {cat: category_counts.get(cat, 0) for cat in filtered_cats}
            forecasted_counts = {}
            if not forecasted_source_df.empty:
                forecasted_counts = forecasted_source_df.groupby("Step 1 Claim Category")["No Of Forecasted Patents"].max().to_dict()
            forecasted_counts = {cat: int(forecasted_counts.get(cat, 0)) for cat in filtered_cats}
            total_patents = sum(category_counts.values())
            forecasted_total = sum(forecasted_counts.values())
            combined_total = total_patents + forecasted_total
            adjusted_count = total_patents + (forecasted_total * 0.5)
            active_cat_names = [cat for cat in filtered_cats if category_counts[cat] > 0]
            active_areas = len(active_cat_names)
            score_info = compute_final_score(total_patents, combined_total, adjusted_count, active_areas)
            return (score_info, active_cat_names, combined_total, adjusted_count, active_areas)

        jurisdiction_scores = []
        if "Jurisdiction" in drug_df.columns:
            jurisdictions = sorted(
                j for j in drug_df["Jurisdiction"].dropna().unique()
                if j not in ("", "nan", "None") and j.upper() != "WO"
            )
            for jur in jurisdictions:
                jur_df = drug_df[drug_df["Jurisdiction"] == jur].copy()
                jur_fc = (forecasted_drug_df[forecasted_drug_df["Jurisdiction"] == jur].copy()
                          if "Jurisdiction" in forecasted_drug_df.columns else forecasted_drug_df.copy())
                si, jc, jcm, jadj, ja = compute_block_scores(jur_df, jur_fc)
                jurisdiction_scores.append({
                    "drug": drug, "jurisdiction": jur,
                    "combined_total": jcm, "adjusted_count": jadj,
                    "active_areas": ja, "active_categories": ", ".join(jc) if jc else "None",
                    "density_label": patent_count_label(si["effective_adjusted_count"]),
                    "diversity_label": diversity_label(ja),
                    "density_score": si["density_score"], "diversity_score": si["diversity_score"],
                    "base_score": si["base_score"], "validation_pct": si["validation_pct"],
                    "final_score": si["final_score"], "final_label": si["final_label"],
                })

        avg_final = (round(sum(js["final_score"] for js in jurisdiction_scores) / len(jurisdiction_scores), 1)
                     if jurisdiction_scores else 0)

        non_wo_df = drug_df[drug_df["Jurisdiction"].str.upper() != "WO"] if "Jurisdiction" in drug_df.columns else drug_df
        _counts = non_wo_df.groupby("Step 1 Claim Category").size().to_dict()
        drug_scores.append({
            "drug": drug,
            "total_patents": sum(_counts.get(c, 0) for c in all_categories if c not in EXCLUDED_CATEGORIES),
            "active_areas": sum(1 for c in all_categories if c not in EXCLUDED_CATEGORIES and _counts.get(c, 0) > 0),
            "active_categories": ", ".join(c for c in all_categories if c not in EXCLUDED_CATEGORIES and _counts.get(c, 0) > 0) or "None",
            "combined_total": sum(js["combined_total"] for js in jurisdiction_scores),
            "adjusted_count": sum(js["adjusted_count"] for js in jurisdiction_scores),
            "avg_final_score": avg_final,
            "jurisdiction_scores": jurisdiction_scores,
        })
        completed_drugs.add(drug)
        _save_ckpt()

    # ── Circumvention ─────────────────────────────────────────────────────
    circumvention_by_drug = {}
    processed_drugs = {ds["drug"] for ds in drug_scores}
    if not skip_circumvention and processed_drugs:
        print(f"\n{'='*60}\nRunning circumvention analysis (Claude)...\n{'='*60}")
        nb_to_analyse = non_blocking[non_blocking["Drug Name"].isin(processed_drugs)]
        chroma_client = get_chroma_clients()[0]
        circumvention_by_drug = get_circumvention_for_drugs(
            nb_to_analyse, chroma_client, max_patents_per_category=max_patents_per_category)

    # ── Always write local JSON ───────────────────────────────────────────
    write_results_to_json(circumvention_by_drug, drug_scores)

    # ── Write to BQ (unless --no-bq) ─────────────────────────────────────
    if not no_bq:
        try:
            if circumvention_by_drug:
                write_circumvention_to_bq(circumvention_by_drug)
            write_score_to_bq(drug_scores, refresh=refresh_scores)
        except Exception as e:
            print(f"[BQ] Write failed ({e}) — results saved to JSON only")
    else:
        print("[NO-BQ] Skipping BigQuery writes. Results in local JSON only.")

    print(f"\nDone! Model: {CLAUDE_MODEL}")
    print(f"  Local JSON output: {OUTPUT_DIR}/")
    return {"circumvention_by_drug": circumvention_by_drug, "drug_scores": drug_scores}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Patent Thicket Analysis & Circumvention (505(b)(2)) — CLAUDE VERSION"
    )
    parser.add_argument("drug", nargs="?", default=None, help="Single drug name")
    parser.add_argument("--skip-circumvention", action="store_true")
    parser.add_argument("--refresh-scores", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--max-patents-per-category", type=int, default=10)
    parser.add_argument("--no-bq", action="store_true",
                        help="Skip all BigQuery writes; output local JSON only")
    args = parser.parse_args()

    process_patents(
        args.skip_circumvention, drug_filter=args.drug,
        refresh_scores=args.refresh_scores or args.rerun,
        rerun=args.rerun,
        max_patents_per_category=args.max_patents_per_category,
        no_bq=args.no_bq,
    )
