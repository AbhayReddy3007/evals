"""
IPD3 Orchestrator — Run Gemini & Claude Pipelines, Then LLM-as-Judge Evaluation

This orchestrator:
  1. Runs ipd3bq__6_.py (Gemini) — writes to BQ + local JSON
  2. Runs ipd3bq_claude.py (Claude) — writes to BQ + local JSON
  3. Loads both results from LOCAL JSON files (no BQ queries needed)
  4. Compares circumvention strategies & thicket scores using Claude Sonnet 4.6 as judge
  5. Writes evaluation to Excel (default) or BQ (with --to-bq)

No BigQuery jobs are created for evaluation. Pipelines write JSON locally
and the orchestrator reads from those files.

Usage:
    python ipd3bq_orchestrator.py                              # run all, eval → Excel
    python ipd3bq_orchestrator.py --drug Semaglutide           # single drug
    python ipd3bq_orchestrator.py --skip-run                   # skip pipelines, eval only
    python ipd3bq_orchestrator.py --skip-gemini                # skip Gemini, run Claude + eval
    python ipd3bq_orchestrator.py --skip-claude                # skip Claude, run Gemini + eval
    python ipd3bq_orchestrator.py --to-bq                      # write eval results to BQ too
    python ipd3bq_orchestrator.py --no-bq                      # pass --no-bq to pipelines too
"""

import os
import re
import sys
import json
import time
import random
import argparse
import subprocess
from datetime import datetime
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from anthropic import AnthropicVertex, RateLimitError

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

try:
    from json_repair import repair_json as _repair_json_lib
except ImportError:
    _repair_json_lib = None

# ── Config ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

BQ_PROJECT_ID = os.getenv("BQ_PROJECT_ID", "cognito-prod-394707")
BQ_DATASET_ID = os.getenv("BQ_DATASET_ID", "cognito_prod_datamart")
BQ_LOCATION   = os.getenv("BQ_LOCATION", "asia-south1")
GCP_REGION    = os.getenv("GCP_REGION", "us-east5")
PROJECT_ID    = os.getenv("BQ_PROJECT_ID", "cognito-prod-394707")

CREDENTIALS_PATH = os.environ.get("CREDENTIALS_PATH") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
if CREDENTIALS_PATH and not os.path.isabs(CREDENTIALS_PATH):
    CREDENTIALS_PATH = os.path.join(SCRIPT_DIR, CREDENTIALS_PATH)

GCS_BUCKET    = os.getenv("GCS_BUCKET", "cognito-gcs")
GCS_BASE_PATH = os.getenv("GCS_EVAL_PATH", "Cognito_new/eval_reports")

CLAUDE_MODEL   = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
MAX_WORKERS    = 6
LLM_MAX_TOKENS = 16384

OUTPUT_DIR = os.getenv("IPD3_OUTPUT_DIR", os.path.join(SCRIPT_DIR, "ipd3_output"))

EVAL_TABLE = "IPD3_Eval_Table"


# ── Claude Client ─────────────────────────────────────────────────────────────

def get_claude_client() -> AnthropicVertex:
    return AnthropicVertex(region=GCP_REGION, project_id=PROJECT_ID)


def _call_claude(client: AnthropicVertex, prompt: str, max_retries: int = 3) -> Dict[str, Any]:
    for attempt in range(max_retries + 1):
        try:
            resp = client.messages.create(
                model=CLAUDE_MODEL, max_tokens=LLM_MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text.strip()
            text = re.sub(r"^```(?:json)?", "", text).strip()
            text = re.sub(r"```$", "", text).strip()
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                if _repair_json_lib:
                    return json.loads(_repair_json_lib(text))
                text = re.sub(r",\s*}", "}", text)
                text = re.sub(r",\s*]", "]", text)
                return json.loads(text)
        except RateLimitError:
            if attempt == max_retries: raise
            wait = 2.0 * (2 ** attempt) * (1 + random.uniform(0, 0.25))
            print(f"    [Claude] Rate limited, retry {attempt+1}/{max_retries} in {wait:.1f}s")
            time.sleep(wait)
        except Exception as e:
            if attempt == max_retries:
                return {"error": str(e)}
            time.sleep(2.0 * (2 ** attempt))
    return {}


# ── Step 1 & 2: Run Pipelines ────────────────────────────────────────────────

def _find_script(name):
    """Find a script by name in SCRIPT_DIR."""
    candidates = [name, name.replace("(", "_").replace(")", "_")]
    for c in candidates:
        p = os.path.join(SCRIPT_DIR, c)
        if os.path.exists(p):
            return p
    return os.path.join(SCRIPT_DIR, name)


def run_gemini_pipeline(drug=None, extra_args=None):
    script = _find_script("ipd3bq__6_.py")
    cmd = [sys.executable, script]
    if drug: cmd.append(drug)
    if extra_args: cmd.extend(extra_args)

    print(f"\n{'='*60}")
    print(f"STEP 1: Running Gemini pipeline")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[WARN] Gemini pipeline exited with code {result.returncode}")
    return result.returncode


def run_claude_pipeline(drug=None, extra_args=None):
    script = _find_script("ipd3bq_claude.py")
    cmd = [sys.executable, script]
    if drug: cmd.append(drug)
    if extra_args: cmd.extend(extra_args)

    print(f"\n{'='*60}")
    print(f"STEP 2: Running Claude pipeline")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[WARN] Claude pipeline exited with code {result.returncode}")
    return result.returncode


# ── Step 3: Load Results from LOCAL JSON ──────────────────────────────────────

def _load_json_safe(path: str) -> List[Dict]:
    """Load a JSON file or return empty list if missing/corrupt."""
    if not os.path.exists(path):
        print(f"[WARN] File not found: {path}")
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        print(f"[JSON] Loaded {len(data)} rows from {path}")
        return data
    except Exception as e:
        print(f"[WARN] Failed to load {path}: {e}")
        return []


def load_circumvention_results(drug=None) -> pd.DataFrame:
    """Load Gemini + Claude circumvention results from local JSON, joined by Drug+Category."""
    gemini_rows = _load_json_safe(os.path.join(OUTPUT_DIR, "circumvention_gemini.json"))
    claude_rows = _load_json_safe(os.path.join(OUTPUT_DIR, "circumvention_claude.json"))

    if not gemini_rows and not claude_rows:
        return pd.DataFrame()

    g_df = pd.DataFrame(gemini_rows) if gemini_rows else pd.DataFrame()
    c_df = pd.DataFrame(claude_rows) if claude_rows else pd.DataFrame()

    # Prefix columns
    if not g_df.empty:
        g_key = g_df[["Drug_Name", "Patent_Category"]].copy()
        g_renamed = g_df.rename(columns={c: f"gemini_{c}" for c in g_df.columns if c not in ("Drug_Name", "Patent_Category")})
    else:
        g_renamed = pd.DataFrame()

    if not c_df.empty:
        c_renamed = c_df.rename(columns={c: f"claude_{c}" for c in c_df.columns if c not in ("Drug_Name", "Patent_Category")})
    else:
        c_renamed = pd.DataFrame()

    # Join
    if not g_renamed.empty and not c_renamed.empty:
        merged = pd.merge(g_renamed, c_renamed, on=["Drug_Name", "Patent_Category"], how="outer")
    elif not g_renamed.empty:
        merged = g_renamed
    else:
        merged = c_renamed

    if drug and not merged.empty:
        merged = merged[merged["Drug_Name"].str.lower() == drug.lower()]

    print(f"[LOAD] {len(merged)} circumvention comparison rows")
    return merged


def load_score_results(drug=None) -> pd.DataFrame:
    """Load Gemini + Claude score results from local JSON, joined by Drug+Jurisdiction."""
    gemini_rows = _load_json_safe(os.path.join(OUTPUT_DIR, "scores_gemini.json"))
    claude_rows = _load_json_safe(os.path.join(OUTPUT_DIR, "scores_claude.json"))

    if not gemini_rows and not claude_rows:
        return pd.DataFrame()

    g_df = pd.DataFrame(gemini_rows) if gemini_rows else pd.DataFrame()
    c_df = pd.DataFrame(claude_rows) if claude_rows else pd.DataFrame()

    if not g_df.empty:
        g_renamed = g_df.rename(columns={c: f"gemini_{c}" for c in g_df.columns if c not in ("Drug_Name", "Jurisdiction")})
    else:
        g_renamed = pd.DataFrame()

    if not c_df.empty:
        c_renamed = c_df.rename(columns={c: f"claude_{c}" for c in c_df.columns if c not in ("Drug_Name", "Jurisdiction")})
    else:
        c_renamed = pd.DataFrame()

    if not g_renamed.empty and not c_renamed.empty:
        merged = pd.merge(g_renamed, c_renamed, on=["Drug_Name", "Jurisdiction"], how="outer")
    elif not g_renamed.empty:
        merged = g_renamed
    else:
        merged = c_renamed

    if drug and not merged.empty:
        merged = merged[merged["Drug_Name"].str.lower() == drug.lower()]

    print(f"[LOAD] {len(merged)} score comparison rows")
    return merged


# ── LLM-as-Judge Prompts ─────────────────────────────────────────────────────

CIRCUMVENTION_EVAL_PROMPT = """
You are a senior pharmaceutical patent attorney acting as an impartial LLM judge.

Two independent LLM systems analysed circumvention / 505(b)(2) design-around strategies
for the same drug and patent category:

  - System A: Gemini 2.5 Flash (primary pipeline)
  - System B: Claude Sonnet 4.6 (secondary pipeline)

Drug: {drug}
Patent Category: {patent_category}
Patents: {patents}

--- GEMINI OUTPUT ---
Difficulty: {gemini_difficulty}
Strategy: {gemini_strategy}
Rationale: {gemini_rationale}
Feasibility: {gemini_feasibility}
Regulatory Pathway: {gemini_regulatory_pathway}
Prior Art Support: {gemini_prior_art_support}
Key Claim Limitations: {gemini_key_claim_limitations}
White Space: {gemini_white_space}
FDA Precedents: {gemini_fda_precedents}
Summary: {gemini_summary}
--- END GEMINI ---

--- CLAUDE OUTPUT ---
Difficulty: {claude_difficulty}
Strategy: {claude_strategy}
Rationale: {claude_rationale}
Feasibility: {claude_feasibility}
Regulatory Pathway: {claude_regulatory_pathway}
Prior Art Support: {claude_prior_art_support}
Key Claim Limitations: {claude_key_claim_limitations}
White Space: {claude_white_space}
FDA Precedents: {claude_fda_precedents}
Summary: {claude_summary}
--- END CLAUDE ---

Evaluate both outputs on these dimensions:
1. **Faithfulness**: Does the output avoid hallucinated references, fabricated patent
   details, or unsupported assertions? Score LOW if the system invents specific patent
   claims, cites non-existent FDA approvals, or presents fabricated prior art as fact.
2. **Relevance**: Is the analysis specifically targeted to this drug and patent category?
   Do the identified claim limitations and design-around strategies actually address the
   patents listed, rather than providing generic boilerplate? Score LOW if the output
   could apply to any drug/category interchangeably without meaningful customisation.
3. **Grounding**: Can the key claims, limitations, and strategies in the output be traced
   back to the specific patent excerpts / source data provided in the input? For each
   strategy or limitation stated, is there a clear link to a passage in the patent chunks?
   Score LOW if the output makes assertions that cannot be mapped to any provided source
   text, even if the assertions happen to be factually plausible.
   NOTE: Faithfulness asks "did it make things up?"; Grounding asks "did it use the source
   data provided, and can we trace its claims back to those sources?".
4. **Accuracy**: Are claim limitations and strategies technically/legally sound?
5. **Completeness**: Does the analysis cover all viable circumvention approaches?
6. **Feasibility Assessment**: Are feasibility ratings well-justified?
7. **Regulatory Viability**: Are 505(b)(2) pathways realistic?
8. **Prior Art Quality**: Are FDA/Orange Book/literature references relevant and specific?

Respond ONLY with valid JSON:
{{
  "agreement_level": "<full|partial|none>",
  "difficulty_agreement": <true or false>,
  "gemini_faithfulness_score": <integer 1-5, where 5=no hallucinations, 1=mostly hallucinated>,
  "claude_faithfulness_score": <integer 1-5>,
  "gemini_relevance_score": <integer 1-5, where 5=highly specific to this drug/category, 1=generic boilerplate>,
  "claude_relevance_score": <integer 1-5>,
  "gemini_grounding_score": <integer 1-5, where 5=every claim traceable to source data, 1=no link to sources>,
  "claude_grounding_score": <integer 1-5>,
  "gemini_accuracy_score": <integer 1-5>,
  "claude_accuracy_score": <integer 1-5>,
  "gemini_completeness_score": <integer 1-5>,
  "claude_completeness_score": <integer 1-5>,
  "gemini_feasibility_score": <integer 1-5>,
  "claude_feasibility_score": <integer 1-5>,
  "gemini_regulatory_score": <integer 1-5>,
  "claude_regulatory_score": <integer 1-5>,
  "gemini_prior_art_score": <integer 1-5>,
  "claude_prior_art_score": <integer 1-5>,
  "faithfulness_notes": "<1-2 sentences: which system had more hallucinated content and why>",
  "relevance_notes": "<1-2 sentences: which system was more specific vs generic and why>",
  "grounding_notes": "<1-2 sentences: which system's output was more traceable to the provided patent excerpts>",
  "preferred_system": "<gemini|claude|tie>",
  "preference_reason": "<1-2 sentences>",
  "discrepancy_explanation": "<1-2 sentences or null>",
  "combined_assessment": "<2-3 sentence overall assessment>",
  "recommended_strategies": ["<best strategy 1>", "<best strategy 2>"]
}}
"""

SCORE_EVAL_PROMPT = """
You are a senior pharmaceutical patent attorney acting as an impartial LLM judge.

Two independent LLM pipelines computed patent thicket scores for the same drug and jurisdiction.

Drug: {drug}
Jurisdiction: {jurisdiction}

--- GEMINI SCORES ---
Final Score: {gemini_Final_Score} ({gemini_Score_Label})
Density Score: {gemini_Density_Score} | Diversity Score: {gemini_Diversity_Score}
Combined Total: {gemini_Combined_Total} | Adjusted Count: {gemini_Adjusted_Count}
Density Interpretation: {gemini_Density_Interpretation}
Diversity Interpretation: {gemini_Diversity_Interpretation}
--- END GEMINI ---

--- CLAUDE SCORES ---
Final Score: {claude_Final_Score} ({claude_Score_Label})
Density Score: {claude_Density_Score} | Diversity Score: {claude_Diversity_Score}
Combined Total: {claude_Combined_Total} | Adjusted Count: {claude_Adjusted_Count}
Density Interpretation: {claude_Density_Interpretation}
Diversity Interpretation: {claude_Diversity_Interpretation}
--- END CLAUDE ---

The thicket scoring formula is deterministic. Differences indicate the underlying
patent classification or filtering diverged.

Evaluate on:
1. **Faithfulness**: Are the score interpretations (labels, density/diversity readings)
   consistent with the numeric values shown? Does either system misrepresent what the
   numbers mean?
2. **Relevance**: Are the density/diversity interpretations specific to this drug and
   jurisdiction, or are they generic descriptions that ignore the actual patent landscape?
3. **Grounding**: Are the numeric scores (density, diversity, combined total) derived from
   verifiable patent counts and categories, or do they appear to use unsubstantiated inputs?
   Can the adjusted count and active areas be traced to actual patent data?

Respond ONLY with valid JSON:
{{
  "scores_match": <true or false>,
  "final_score_delta": <integer: gemini - claude>,
  "gemini_faithfulness_score": <integer 1-5, where 5=interpretations fully match the numbers>,
  "claude_faithfulness_score": <integer 1-5>,
  "gemini_relevance_score": <integer 1-5, where 5=interpretation is specific to this drug/jurisdiction>,
  "claude_relevance_score": <integer 1-5>,
  "gemini_grounding_score": <integer 1-5, where 5=scores traceable to patent data>,
  "claude_grounding_score": <integer 1-5>,
  "faithfulness_notes": "<1 sentence on whether labels match numeric scores>",
  "relevance_notes": "<1 sentence on specificity of interpretations>",
  "grounding_notes": "<1 sentence on whether numeric inputs are substantiated>",
  "discrepancy_explanation": "<explanation>",
  "data_consistency_flag": "<consistent|minor_divergence|major_divergence>",
  "recommended_final_score": <integer 1-5>,
  "recommended_label": "<thicket label>",
  "assessment": "<1-2 sentence assessment>"
}}
"""

OVERALL_SYNTHESIS_PROMPT = """
You are a senior pharmaceutical patent attorney providing a final synthesis of a
dual-LLM patent thicket evaluation.

Drug: {drug}

Summary statistics:
- Circumvention categories evaluated: {num_categories}
- Gemini preferred: {gemini_preferred}
- Claude preferred: {claude_preferred}
- Tied: {tied}
- Average Gemini accuracy: {avg_gemini_accuracy}
- Average Claude accuracy: {avg_claude_accuracy}
- Average Gemini faithfulness: {avg_gemini_faithfulness}
- Average Claude faithfulness: {avg_claude_faithfulness}
- Average Gemini relevance: {avg_gemini_relevance}
- Average Claude relevance: {avg_claude_relevance}
- Average Gemini grounding: {avg_gemini_grounding}
- Average Claude grounding: {avg_claude_grounding}
- Score agreement rate: {score_agreement_pct}%

Respond ONLY with valid JSON:
{{
  "overall_preferred_system": "<gemini|claude|tie>",
  "confidence": "<high|medium|low>",
  "faithfulness_winner": "<gemini|claude|tie>",
  "faithfulness_summary": "<1-2 sentences comparing hallucination levels>",
  "relevance_winner": "<gemini|claude|tie>",
  "relevance_summary": "<1-2 sentences comparing specificity to drug/patents>",
  "grounding_winner": "<gemini|claude|tie>",
  "grounding_summary": "<1-2 sentences comparing traceability to source patent data>",
  "key_strengths_gemini": ["<strength 1>", "<strength 2>"],
  "key_strengths_claude": ["<strength 1>", "<strength 2>"],
  "key_weaknesses_gemini": ["<weakness 1>"],
  "key_weaknesses_claude": ["<weakness 1>"],
  "recommendation": "<2-3 sentence recommendation>",
  "combined_thicket_assessment": "<2-3 sentence overall assessment>"
}}
"""


# ── Evaluation Logic ──────────────────────────────────────────────────────────

def _safe_get(row, key, default="N/A"):
    v = row.get(key)
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return default
    return str(v)[:500]


def evaluate_circumvention_row(client, row):
    prompt = CIRCUMVENTION_EVAL_PROMPT.format(
        drug=_safe_get(row, "Drug_Name"),
        patent_category=_safe_get(row, "Patent_Category"),
        patents=_safe_get(row, "gemini_Patents") or _safe_get(row, "claude_Patents"),
        gemini_difficulty=_safe_get(row, "gemini_Overall_Difficulty"),
        gemini_strategy=_safe_get(row, "gemini_Strategy"),
        gemini_rationale=_safe_get(row, "gemini_Rationale"),
        gemini_feasibility=_safe_get(row, "gemini_Feasibility"),
        gemini_regulatory_pathway=_safe_get(row, "gemini_Regulatory_Pathway"),
        gemini_prior_art_support=_safe_get(row, "gemini_Prior_Art_Support"),
        gemini_key_claim_limitations=_safe_get(row, "gemini_Key_Claim_Limitations"),
        gemini_white_space=_safe_get(row, "gemini_White_Space_Opportunities"),
        gemini_fda_precedents=_safe_get(row, "gemini_FDA_Precedents"),
        gemini_summary=_safe_get(row, "gemini_Summary"),
        claude_difficulty=_safe_get(row, "claude_Overall_Difficulty"),
        claude_strategy=_safe_get(row, "claude_Strategy"),
        claude_rationale=_safe_get(row, "claude_Rationale"),
        claude_feasibility=_safe_get(row, "claude_Feasibility"),
        claude_regulatory_pathway=_safe_get(row, "claude_Regulatory_Pathway"),
        claude_prior_art_support=_safe_get(row, "claude_Prior_Art_Support"),
        claude_key_claim_limitations=_safe_get(row, "claude_Key_Claim_Limitations"),
        claude_white_space=_safe_get(row, "claude_White_Space_Opportunities"),
        claude_fda_precedents=_safe_get(row, "claude_FDA_Precedents"),
        claude_summary=_safe_get(row, "claude_Summary"),
    )
    return _call_claude(client, prompt)


def evaluate_score_row(client, row):
    # Build the prompt with the exact column names from the merged DF
    prompt = SCORE_EVAL_PROMPT.format(
        drug=_safe_get(row, "Drug_Name"),
        jurisdiction=_safe_get(row, "Jurisdiction"),
        **{k: _safe_get(row, k) for k in [
            "gemini_Final_Score", "gemini_Score_Label",
            "gemini_Density_Score", "gemini_Diversity_Score",
            "gemini_Combined_Total", "gemini_Adjusted_Count",
            "gemini_Density_Interpretation", "gemini_Diversity_Interpretation",
            "claude_Final_Score", "claude_Score_Label",
            "claude_Density_Score", "claude_Diversity_Score",
            "claude_Combined_Total", "claude_Adjusted_Count",
            "claude_Density_Interpretation", "claude_Diversity_Interpretation",
        ]}
    )
    return _call_claude(client, prompt)


def run_overall_synthesis(client, drug, circ_evals, score_evals):
    gemini_pref = sum(1 for e in circ_evals if e.get("preferred_system") == "gemini")
    claude_pref = sum(1 for e in circ_evals if e.get("preferred_system") == "claude")
    tied = sum(1 for e in circ_evals if e.get("preferred_system") == "tie")

    def _avg(evals, key):
        vals = [e.get(key, 0) for e in evals if e.get(key)]
        return round(sum(vals) / len(vals), 1) if vals else 0

    avg_g_acc = _avg(circ_evals, "gemini_accuracy_score")
    avg_c_acc = _avg(circ_evals, "claude_accuracy_score")
    avg_g_faith = _avg(circ_evals, "gemini_faithfulness_score")
    avg_c_faith = _avg(circ_evals, "claude_faithfulness_score")
    avg_g_rel = _avg(circ_evals, "gemini_relevance_score")
    avg_c_rel = _avg(circ_evals, "claude_relevance_score")
    avg_g_gnd = _avg(circ_evals, "gemini_grounding_score")
    avg_c_gnd = _avg(circ_evals, "claude_grounding_score")

    score_matches = sum(1 for e in score_evals if e.get("scores_match"))
    score_total = len(score_evals) or 1

    prompt = OVERALL_SYNTHESIS_PROMPT.format(
        drug=drug, num_categories=len(circ_evals),
        gemini_preferred=gemini_pref, claude_preferred=claude_pref, tied=tied,
        avg_gemini_accuracy=avg_g_acc, avg_claude_accuracy=avg_c_acc,
        avg_gemini_faithfulness=avg_g_faith, avg_claude_faithfulness=avg_c_faith,
        avg_gemini_relevance=avg_g_rel, avg_claude_relevance=avg_c_rel,
        avg_gemini_grounding=avg_g_gnd, avg_claude_grounding=avg_c_gnd,
        score_agreement_pct=round(score_matches / score_total * 100, 1),
    )
    return _call_claude(client, prompt)


# ── Save Results ──────────────────────────────────────────────────────────────

def _get_credentials():
    """Service-account credentials matching generate_tolerability_report.py pattern."""
    if CREDENTIALS_PATH and os.path.exists(CREDENTIALS_PATH):
        from google.oauth2 import service_account
        return service_account.Credentials.from_service_account_file(CREDENTIALS_PATH)
    return None


def upload_to_gcs(local_path: str) -> str:
    """Upload a file to GCS and return the gs:// URI."""
    from google.cloud import storage as gcs_storage

    credentials = _get_credentials()
    client = gcs_storage.Client(project=BQ_PROJECT_ID, credentials=credentials)
    bucket = client.bucket(GCS_BUCKET)

    filename = os.path.basename(local_path)
    blob_name = f"{GCS_BASE_PATH}/{filename}"
    gcs_uri = f"gs://{GCS_BUCKET}/{blob_name}"

    print(f"[GCS] Uploading → {gcs_uri}")
    bucket.blob(blob_name).upload_from_filename(
        local_path,
        content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    return gcs_uri


def save_eval_to_excel(rows, drug=None):
    """Build a clean, presentable multi-sheet comparison Excel and upload to GCS."""
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = os.path.join(OUTPUT_DIR, f"ipd3_eval_{drug or 'all'}_{ts}.xlsx")

    wb = Workbook()

    # ── Styles ────────────────────────────────────────────────────────────
    hdr_font = Font(name="Arial", bold=True, color="FFFFFF", size=11)
    hdr_fill = PatternFill("solid", fgColor="2F5496")
    title_font = Font(name="Arial", bold=True, size=14, color="1F3864")
    sub_font = Font(name="Arial", bold=True, size=11, color="2F5496")
    cell_font = Font(name="Arial", size=10)
    bold_font = Font(name="Arial", bold=True, size=10)
    center = Alignment(horizontal="center", vertical="center", wrap_text=True)
    left = Alignment(horizontal="left", vertical="center", wrap_text=True)
    thin = Border(
        left=Side("thin"), right=Side("thin"),
        top=Side("thin"), bottom=Side("thin"),
    )
    green_fill = PatternFill("solid", fgColor="E2EFDA")
    red_fill = PatternFill("solid", fgColor="FCE4EC")
    yellow_fill = PatternFill("solid", fgColor="FFF9C4")
    light_gray = PatternFill("solid", fgColor="F5F5F5")

    def _score_fill(val):
        try:
            v = int(float(val))
        except (ValueError, TypeError):
            return None
        if v >= 4: return green_fill
        if v == 3: return yellow_fill
        if v <= 2: return red_fill
        return None

    def _write_header(ws, row, headers, col_start=1):
        for i, h in enumerate(headers):
            c = ws.cell(row=row, column=col_start + i, value=h)
            c.font, c.fill, c.alignment, c.border = hdr_font, hdr_fill, center, thin

    def _write_row(ws, row, values, col_start=1, fonts=None, fills=None):
        for i, v in enumerate(values):
            c = ws.cell(row=row, column=col_start + i, value=v)
            c.font = (fonts[i] if fonts and i < len(fonts) else cell_font)
            c.alignment = center if i > 0 else left
            c.border = thin
            if fills and i < len(fills) and fills[i]:
                c.fill = fills[i]

    def _auto_width(ws, min_w=10, max_w=40):
        for col_cells in ws.columns:
            length = max(len(str(c.value or "")) for c in col_cells)
            ws.column_dimensions[get_column_letter(col_cells[0].column)].width = \
                min(max(length + 2, min_w), max_w)

    # ── Split rows by eval_type ───────────────────────────────────────────
    circ_rows = [r for r in rows if r.get("eval_type") == "circumvention"]
    score_rows = [r for r in rows if r.get("eval_type") == "thicket_score"]
    synth_rows = [r for r in rows if r.get("eval_type") == "overall_synthesis"]

    # ══════════════════════════════════════════════════════════════════════
    # SHEET 1: Summary
    # ══════════════════════════════════════════════════════════════════════
    ws = wb.active
    ws.title = "Summary"
    ws.cell(row=1, column=1, value="IPD3 Evaluation — Gemini vs Claude").font = title_font
    ws.cell(row=2, column=1, value=f"Drug: {drug or 'All'}  |  Date: {ts}  |  Judge: {CLAUDE_MODEL}").font = sub_font
    ws.merge_cells("A1:F1")
    ws.merge_cells("A2:F2")

    # Preference counts
    g_pref = sum(1 for r in circ_rows if r.get("eval_preferred_system") == "gemini")
    c_pref = sum(1 for r in circ_rows if r.get("eval_preferred_system") == "claude")
    tied = sum(1 for r in circ_rows if r.get("eval_preferred_system") == "tie")

    def _avg_score(rows_list, key):
        vals = [r.get(key) for r in rows_list if r.get(key) is not None]
        nums = []
        for v in vals:
            try: nums.append(float(v))
            except (ValueError, TypeError): pass
        return round(sum(nums) / len(nums), 1) if nums else "—"

    r = 4
    _write_header(ws, r, ["Metric", "Gemini", "Claude", "Winner"])
    metrics = [
        ("Categories Evaluated", len(circ_rows), len(circ_rows), ""),
        ("Preferred System", g_pref, c_pref,
         "Gemini" if g_pref > c_pref else "Claude" if c_pref > g_pref else "Tie"),
        ("Tied", tied, tied, ""),
        ("Avg Faithfulness", _avg_score(circ_rows, "eval_gemini_faithfulness_score"),
         _avg_score(circ_rows, "eval_claude_faithfulness_score"), ""),
        ("Avg Relevance", _avg_score(circ_rows, "eval_gemini_relevance_score"),
         _avg_score(circ_rows, "eval_claude_relevance_score"), ""),
        ("Avg Grounding", _avg_score(circ_rows, "eval_gemini_grounding_score"),
         _avg_score(circ_rows, "eval_claude_grounding_score"), ""),
        ("Avg Accuracy", _avg_score(circ_rows, "eval_gemini_accuracy_score"),
         _avg_score(circ_rows, "eval_claude_accuracy_score"), ""),
        ("Avg Completeness", _avg_score(circ_rows, "eval_gemini_completeness_score"),
         _avg_score(circ_rows, "eval_claude_completeness_score"), ""),
        ("Avg Feasibility", _avg_score(circ_rows, "eval_gemini_feasibility_score"),
         _avg_score(circ_rows, "eval_claude_feasibility_score"), ""),
        ("Avg Regulatory", _avg_score(circ_rows, "eval_gemini_regulatory_score"),
         _avg_score(circ_rows, "eval_claude_regulatory_score"), ""),
        ("Avg Prior Art", _avg_score(circ_rows, "eval_gemini_prior_art_score"),
         _avg_score(circ_rows, "eval_claude_prior_art_score"), ""),
        ("Score Rows Match", sum(1 for r in score_rows if str(r.get("eval_scores_match")).lower() == "true"),
         len(score_rows), ""),
    ]
    # Compute winners for avg metrics
    for i, (metric, g, c, w) in enumerate(metrics):
        if w == "" and isinstance(g, (int, float)) and isinstance(c, (int, float)) and g != c:
            metrics[i] = (metric, g, c, "Gemini" if g > c else "Claude")
        elif w == "":
            metrics[i] = (metric, g, c, "Tie" if isinstance(g, (int, float)) else "")

    for i, (metric, g, c, winner) in enumerate(metrics):
        row_num = r + 1 + i
        fill_g = green_fill if winner == "Gemini" else None
        fill_c = green_fill if winner == "Claude" else None
        _write_row(ws, row_num, [metric, g, c, winner],
                   fonts=[bold_font, cell_font, cell_font, bold_font],
                   fills=[light_gray, fill_g, fill_c, None])

    # Synthesis notes
    if synth_rows:
        synth_start = r + len(metrics) + 3
        ws.cell(row=synth_start, column=1, value="Per-Drug Synthesis").font = sub_font
        for si, sr in enumerate(synth_rows):
            row_num = synth_start + 1 + si
            ws.cell(row=row_num, column=1, value=sr.get("Drug_Name", "")).font = bold_font
            ws.cell(row=row_num, column=2, value=str(sr.get("synth_recommendation", ""))).font = cell_font
            ws.cell(row=row_num, column=2).alignment = left
            ws.merge_cells(start_row=row_num, start_column=2, end_row=row_num, end_column=6)

    _auto_width(ws)

    # ══════════════════════════════════════════════════════════════════════
    # SHEET 2: Circumvention Comparison
    # ══════════════════════════════════════════════════════════════════════
    if circ_rows:
        ws2 = wb.create_sheet("Circumvention")
        headers = [
            "Drug", "Category",
            "Gemini Strategy", "Claude Strategy",
            "Agreement", "Winner",
            "G\nFaith", "C\nFaith",
            "G\nGround", "C\nGround",
            "G\nRelev", "C\nRelev",
            "G\nAccur", "C\nAccur",
            "G\nCompl", "C\nCompl",
            "G\nFeasib", "C\nFeasib",
            "G\nRegul", "C\nRegul",
            "Reason",
        ]
        _write_header(ws2, 1, headers)

        for i, r_data in enumerate(circ_rows):
            rn = i + 2
            g_faith = r_data.get("eval_gemini_faithfulness_score")
            c_faith = r_data.get("eval_claude_faithfulness_score")
            g_gnd = r_data.get("eval_gemini_grounding_score")
            c_gnd = r_data.get("eval_claude_grounding_score")
            g_rel = r_data.get("eval_gemini_relevance_score")
            c_rel = r_data.get("eval_claude_relevance_score")
            g_acc = r_data.get("eval_gemini_accuracy_score")
            c_acc = r_data.get("eval_claude_accuracy_score")
            g_comp = r_data.get("eval_gemini_completeness_score")
            c_comp = r_data.get("eval_claude_completeness_score")
            g_feas = r_data.get("eval_gemini_feasibility_score")
            c_feas = r_data.get("eval_claude_feasibility_score")
            g_reg = r_data.get("eval_gemini_regulatory_score")
            c_reg = r_data.get("eval_claude_regulatory_score")
            winner = r_data.get("eval_preferred_system", "")

            # Full strategy text from pipeline outputs
            g_strat = str(r_data.get("gemini_Strategy") or "")
            c_strat = str(r_data.get("claude_Strategy") or "")
            # Full reason — no truncation
            reason = str(r_data.get("eval_preference_reason") or "")

            vals = [
                r_data.get("Drug_Name", ""),
                r_data.get("Patent_Category", ""),
                g_strat, c_strat,
                r_data.get("eval_agreement_level", ""),
                winner,
                g_faith, c_faith, g_gnd, c_gnd,
                g_rel, c_rel, g_acc, c_acc,
                g_comp, c_comp, g_feas, c_feas,
                g_reg, c_reg,
                reason,
            ]
            fills = [
                None, None, None, None, None,
                green_fill if winner in ("gemini", "claude") else yellow_fill,
                _score_fill(g_faith), _score_fill(c_faith),
                _score_fill(g_gnd), _score_fill(c_gnd),
                _score_fill(g_rel), _score_fill(c_rel),
                _score_fill(g_acc), _score_fill(c_acc),
                _score_fill(g_comp), _score_fill(c_comp),
                _score_fill(g_feas), _score_fill(c_feas),
                _score_fill(g_reg), _score_fill(c_reg),
                None,
            ]
            _write_row(ws2, rn, vals, fills=fills)

        _auto_width(ws2, min_w=8, max_w=30)
        # Strategy + Reason columns need more width
        ws2.column_dimensions["C"].width = 55  # Gemini Strategy
        ws2.column_dimensions["D"].width = 55  # Claude Strategy
        ws2.column_dimensions["U"].width = 60  # Reason

    # ══════════════════════════════════════════════════════════════════════
    # SHEET 3: Score Comparison
    # ══════════════════════════════════════════════════════════════════════
    if score_rows:
        ws3 = wb.create_sheet("Thicket Scores")
        headers = [
            "Drug", "Jurisdiction",
            "Gemini\nFinal Score", "Claude\nFinal Score",
            "Match?", "Delta",
            "G\nFaith", "C\nFaith",
            "G\nGround", "C\nGround",
            "G\nRelev", "C\nRelev",
            "Recommended\nScore", "Consistency", "Assessment",
        ]
        _write_header(ws3, 1, headers)

        for i, r_data in enumerate(score_rows):
            rn = i + 2
            match = str(r_data.get("eval_scores_match", "")).lower() == "true"
            vals = [
                r_data.get("Drug_Name", ""),
                r_data.get("Jurisdiction", ""),
                r_data.get("gemini_Final_Score", ""),
                r_data.get("claude_Final_Score", ""),
                "Yes" if match else "No",
                r_data.get("eval_final_score_delta", ""),
                r_data.get("eval_gemini_faithfulness_score", ""),
                r_data.get("eval_claude_faithfulness_score", ""),
                r_data.get("eval_gemini_grounding_score", ""),
                r_data.get("eval_claude_grounding_score", ""),
                r_data.get("eval_gemini_relevance_score", ""),
                r_data.get("eval_claude_relevance_score", ""),
                r_data.get("eval_recommended_final_score", ""),
                r_data.get("eval_data_consistency_flag", ""),
                str(r_data.get("eval_assessment", "")),
            ]
            fills = [
                None, None,
                _score_fill(vals[2]), _score_fill(vals[3]),
                green_fill if match else red_fill, None,
                _score_fill(vals[6]), _score_fill(vals[7]),
                _score_fill(vals[8]), _score_fill(vals[9]),
                _score_fill(vals[10]), _score_fill(vals[11]),
                _score_fill(vals[12]), None, None,
            ]
            _write_row(ws3, rn, vals, fills=fills)

        _auto_width(ws3, min_w=8, max_w=30)
        ws3.column_dimensions["O"].width = 60

    # ══════════════════════════════════════════════════════════════════════
    # SHEET 4: Judge Notes (faithfulness & relevance details)
    # ══════════════════════════════════════════════════════════════════════
    if circ_rows:
        ws4 = wb.create_sheet("Judge Notes")
        headers = [
            "Drug", "Category", "Winner",
            "Faithfulness Notes", "Grounding Notes",
            "Relevance Notes", "Discrepancy", "Combined Assessment",
        ]
        _write_header(ws4, 1, headers)

        for i, r_data in enumerate(circ_rows):
            rn = i + 2
            vals = [
                r_data.get("Drug_Name", ""),
                r_data.get("Patent_Category", ""),
                r_data.get("eval_preferred_system", ""),
                str(r_data.get("eval_faithfulness_notes", "")),
                str(r_data.get("eval_grounding_notes", "")),
                str(r_data.get("eval_relevance_notes", "")),
                str(r_data.get("eval_discrepancy_explanation", "")),
                str(r_data.get("eval_combined_assessment", "")),
            ]
            _write_row(ws4, rn, vals)

        _auto_width(ws4, min_w=12, max_w=65)

    # ── Save & upload ─────────────────────────────────────────────────────
    wb.save(fname)
    print(f"[EXCEL] Eval results → {fname}")

    try:
        gcs_uri = upload_to_gcs(fname)
        print(f"[GCS] {gcs_uri}")
    except Exception as e:
        print(f"[WARN] GCS upload failed: {e} — file saved locally at {fname}")

    return fname


def save_eval_to_bq(rows):
    if not rows: return
    from google.cloud import bigquery
    credentials = _get_credentials()
    client = bigquery.Client(project=BQ_PROJECT_ID, credentials=credentials, location=BQ_LOCATION)
    table_ref = f"{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{EVAL_TABLE}"
    df = pd.DataFrame(rows)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).replace({"None": None, "nan": None, "NaN": None})
    df["created_at"] = pd.Timestamp.now(tz="UTC")
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND, autodetect=True,
        schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION],
    )
    client.load_table_from_dataframe(df, table_ref, job_config=job_config, location=BQ_LOCATION).result()
    print(f"[BQ] {EVAL_TABLE}: {len(df)} rows written")


# ── Main Orchestrator ─────────────────────────────────────────────────────────

def orchestrate(drug=None, skip_run=False, skip_gemini=False, skip_claude=False,
                to_bq=False, extra_args=None, no_bq_pipelines=False):
    start = time.time()

    print("=" * 70)
    print("IPD3 ORCHESTRATOR — Gemini vs Claude Patent Thicket Evaluation")
    print(f"  Drug filter : {drug or 'ALL'}")
    print(f"  Judge model : {CLAUDE_MODEL}")
    print(f"  Eval output : {'BQ + Excel' if to_bq else 'Excel only'}")
    print(f"  JSON dir    : {OUTPUT_DIR}")
    print("=" * 70)

    # ── Step 1 & 2: Run pipelines ────────────────────────────────────────
    if not skip_run:
        pipeline_extra = list(extra_args or [])
        if no_bq_pipelines:
            pipeline_extra.append("--no-bq")

        if not skip_gemini:
            run_gemini_pipeline(drug, pipeline_extra)
        else:
            print("\n⏭️  Gemini pipeline skipped")

        if not skip_claude:
            run_claude_pipeline(drug, pipeline_extra)
        else:
            print("\n⏭️  Claude pipeline skipped")
    else:
        print("\n⏭️  Both pipelines skipped (--skip-run)")

    # ── Step 3: Load results from local JSON ─────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 3: Loading results from local JSON files")
    print(f"  Directory: {OUTPUT_DIR}")
    print(f"{'='*60}")

    circ_df = load_circumvention_results(drug)
    score_df = load_score_results(drug)

    if circ_df.empty and score_df.empty:
        print("\n[ERROR] No results found. Ensure pipelines ran successfully and")
        print(f"  JSON files exist in {OUTPUT_DIR}/")
        print(f"  Expected: circumvention_gemini.json, circumvention_claude.json,")
        print(f"            scores_gemini.json, scores_claude.json")
        return {"error": "No data", "output_rows": []}

    # ── Step 4: LLM-as-Judge evaluation ──────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 4: LLM-as-Judge Evaluation (Claude Sonnet 4.6)")
    print(f"  Circumvention rows: {len(circ_df)}")
    print(f"  Score rows: {len(score_df)}")
    print(f"{'='*60}")

    client = get_claude_client()
    eval_timestamp = datetime.now().isoformat()
    all_output_rows = []
    circ_evals = []
    score_evals = []

    # ── Evaluate circumvention ────────────────────────────────────────────
    circ_rows = circ_df.to_dict("records") if not circ_df.empty else []
    if circ_rows:
        print(f"\nEvaluating {len(circ_rows)} circumvention comparisons ...")

        def _eval_circ(row):
            try:
                dn = row.get("Drug_Name", "?")
                cat = row.get("Patent_Category", "?")
                print(f"  [{dn}/{cat}] Evaluating ...")
                result = evaluate_circumvention_row(client, row)
                circ_evals.append(result)
                out = dict(row)
                out["eval_timestamp"] = eval_timestamp
                out["eval_type"] = "circumvention"
                for k, v in result.items():
                    out[f"eval_{k}"] = "; ".join(str(x) for x in v) if isinstance(v, list) else v
                return out
            except Exception as e:
                print(f"  [ERROR] {row.get('Drug_Name')}/{row.get('Patent_Category')}: {e}")
                out = dict(row)
                out["eval_timestamp"] = eval_timestamp
                out["eval_error"] = str(e)
                return out

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(_eval_circ, r): r for r in circ_rows}
            for f in as_completed(futures):
                out = f.result()
                if out: all_output_rows.append(out)

    # ── Evaluate scores ──────────────────────────────────────────────────
    score_rows = score_df.to_dict("records") if not score_df.empty else []
    if score_rows:
        print(f"\nEvaluating {len(score_rows)} score comparisons ...")

        def _eval_score(row):
            try:
                dn = row.get("Drug_Name", "?")
                jur = row.get("Jurisdiction", "?")
                print(f"  [{dn}/{jur}] Evaluating scores ...")
                result = evaluate_score_row(client, row)
                score_evals.append(result)
                out = dict(row)
                out["eval_timestamp"] = eval_timestamp
                out["eval_type"] = "thicket_score"
                for k, v in result.items():
                    out[f"eval_{k}"] = v
                return out
            except Exception as e:
                print(f"  [ERROR] {row.get('Drug_Name')}/{row.get('Jurisdiction')}: {e}")
                out = dict(row)
                out["eval_timestamp"] = eval_timestamp
                out["eval_error"] = str(e)
                return out

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(_eval_score, r): r for r in score_rows}
            for f in as_completed(futures):
                out = f.result()
                if out: all_output_rows.append(out)

    # ── Overall synthesis per drug ────────────────────────────────────────
    drugs_evaluated = {r.get("Drug_Name") for r in all_output_rows if r.get("Drug_Name")}
    for d in sorted(drugs_evaluated):
        print(f"\n  Generating synthesis for {d} ...")
        d_circ = [e for e, r in zip(circ_evals, circ_rows) if r.get("Drug_Name") == d] if circ_rows else []
        d_score = [e for e, r in zip(score_evals, score_rows) if r.get("Drug_Name") == d] if score_rows else []
        try:
            synth = run_overall_synthesis(client, d, d_circ, d_score)
            all_output_rows.append({
                "Drug_Name": d, "eval_type": "overall_synthesis",
                "eval_timestamp": eval_timestamp,
                **{f"synth_{k}": ("; ".join(v) if isinstance(v, list) else v) for k, v in synth.items()},
            })
        except Exception as e:
            print(f"  [ERROR] Synthesis for {d}: {e}")

    # ── Summary ───────────────────────────────────────────────────────────
    gemini_pref = sum(1 for e in circ_evals if e.get("preferred_system") == "gemini")
    claude_pref = sum(1 for e in circ_evals if e.get("preferred_system") == "claude")
    tied = sum(1 for e in circ_evals if e.get("preferred_system") == "tie")
    score_matches = sum(1 for e in score_evals if e.get("scores_match"))

    def _avg(evals, key):
        vals = [e.get(key, 0) for e in evals if e.get(key)]
        return round(sum(vals) / len(vals), 1) if vals else 0

    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"  Circumvention categories : {len(circ_evals)}")
    print(f"    Gemini preferred       : {gemini_pref}")
    print(f"    Claude preferred       : {claude_pref}")
    print(f"    Tied                   : {tied}")
    print(f"  Faithfulness (avg)")
    print(f"    Gemini                 : {_avg(circ_evals, 'gemini_faithfulness_score')}")
    print(f"    Claude                 : {_avg(circ_evals, 'claude_faithfulness_score')}")
    print(f"  Relevance (avg)")
    print(f"    Gemini                 : {_avg(circ_evals, 'gemini_relevance_score')}")
    print(f"    Claude                 : {_avg(circ_evals, 'claude_relevance_score')}")
    print(f"  Grounding (avg)")
    print(f"    Gemini                 : {_avg(circ_evals, 'gemini_grounding_score')}")
    print(f"    Claude                 : {_avg(circ_evals, 'claude_grounding_score')}")
    print(f"  Accuracy (avg)")
    print(f"    Gemini                 : {_avg(circ_evals, 'gemini_accuracy_score')}")
    print(f"    Claude                 : {_avg(circ_evals, 'claude_accuracy_score')}")
    print(f"  Score rows evaluated     : {len(score_evals)}")
    print(f"    Scores match           : {score_matches}/{len(score_evals)}")
    print(f"  Drugs evaluated          : {len(drugs_evaluated)}")
    print(f"  Total eval rows          : {len(all_output_rows)}")
    print(f"  Elapsed                  : {time.time()-start:.1f}s")
    print("=" * 60)

    # ── Save ──────────────────────────────────────────────────────────────
    save_eval_to_excel(all_output_rows, drug)
    if to_bq:
        try:
            save_eval_to_bq(all_output_rows)
        except Exception as e:
            print(f"[BQ] Write failed ({e}) — results already saved to Excel")

    return {
        "output_rows": all_output_rows,
        "summary": {
            "gemini_preferred": gemini_pref, "claude_preferred": claude_pref,
            "tied": tied, "score_matches": score_matches,
            "drugs_evaluated": len(drugs_evaluated),
        },
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="IPD3 Orchestrator — Run Gemini & Claude, Then LLM-as-Judge Evaluation"
    )
    parser.add_argument("--drug", default=None, help="Drug name (omit = all)")
    parser.add_argument("--to-bq", action="store_true", help="Also write eval to BigQuery")
    parser.add_argument("--no-bq", action="store_true",
                        help="Pass --no-bq to sub-pipelines (skip BQ writes in pipelines)")
    parser.add_argument("--skip-run", action="store_true", help="Skip pipelines, eval only")
    parser.add_argument("--skip-gemini", action="store_true")
    parser.add_argument("--skip-claude", action="store_true")
    parser.add_argument("--skip-circumvention", action="store_true")
    parser.add_argument("--refresh-scores", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--max-patents-per-category", type=int, default=1000)
    parser.add_argument("--csv-input", default=None,
                        help="Path to local CSV/Excel export of Master_LOE table. "
                             "Passed to both pipelines so they skip BigQuery reads.")
    args = parser.parse_args()

    extra = []
    if args.skip_circumvention: extra.append("--skip-circumvention")
    if args.refresh_scores: extra.append("--refresh-scores")
    if args.rerun: extra.append("--rerun")
    if args.max_patents_per_category != 1000:
        extra.extend(["--max-patents-per-category", str(args.max_patents_per_category)])
    if args.csv_input:
        extra.extend(["--csv-input", args.csv_input])

    result = orchestrate(
        drug=args.drug, skip_run=args.skip_run,
        skip_gemini=args.skip_gemini, skip_claude=args.skip_claude,
        to_bq=args.to_bq, extra_args=extra or None,
        no_bq_pipelines=args.no_bq,
    )
    if "error" in result:
        print(f"Error: {result['error']}")
        sys.exit(1)
