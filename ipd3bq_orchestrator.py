"""
IPD3 Orchestrator — Run Gemini & Claude Pipelines, Then LLM-as-Judge Evaluation

This orchestrator:
  1. Runs ipd3bq__6_.py (Gemini) — writes to Circumvention_Table / Patent_Thicket_Score_Table
  2. Runs ipd3bq_claude.py (Claude) — writes to Circumvention_Table_Claude / Patent_Thicket_Score_Table_Claude
  3. Loads both sets of results from BigQuery
  4. Compares circumvention strategies & thicket scores using Claude Sonnet 4.6 as LLM-as-Judge
  5. Writes evaluation results to BigQuery (IPD3_Eval_Table) and/or Excel

Modelled after dimension2_eval_standalone.py for the LLM-as-Judge pattern.

Usage:
    python ipd3bq_orchestrator.py                              # run all, save eval to BQ
    python ipd3bq_orchestrator.py --drug Semaglutide           # single drug
    python ipd3bq_orchestrator.py --drug Semaglutide --no-bq   # eval → Excel only
    python ipd3bq_orchestrator.py --skip-run                   # skip pipeline runs, just eval
    python ipd3bq_orchestrator.py --skip-gemini                # skip Gemini, only run Claude + eval
    python ipd3bq_orchestrator.py --skip-claude                # skip Claude, only run Gemini + eval

Setup (same env vars as ipd3bq__6_.py + dimension2_eval_standalone.py):
    pip install anthropic[vertex] google-cloud-bigquery google-auth pandas openpyxl json-repair python-dotenv
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
from dotenv import load_dotenv
from google.cloud import bigquery
from google.oauth2 import service_account
from anthropic import AnthropicVertex, RateLimitError

try:
    from json_repair import repair_json as _repair_json_lib
except ImportError:
    _repair_json_lib = None

load_dotenv(override=True)

# ── Config ─────────────────────────────────────────────────────────────────────

BQ_PROJECT_ID = os.getenv("BQ_PROJECT_ID", "cognito-prod-394707")
BQ_DATASET_ID = os.getenv("BQ_DATASET_ID", "cognito_prod_datamart")
BQ_LOCATION   = os.getenv("BQ_LOCATION", "asia-south1")
GCP_REGION    = os.getenv("GCP_REGION", "us-east5")
PROJECT_ID    = os.getenv("BQ_PROJECT_ID", "cognito-prod-394707")

CLAUDE_MODEL   = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
MAX_WORKERS    = 6
LLM_MAX_TOKENS = 8192

# BQ table names — Gemini pipeline
GEMINI_CIRC_TABLE  = "Circumvention_Table"
GEMINI_SCORE_TABLE = "Patent_Thicket_Score_Table"

# BQ table names — Claude pipeline
CLAUDE_CIRC_TABLE  = "Circumvention_Table_Claude"
CLAUDE_SCORE_TABLE = "Patent_Thicket_Score_Table_Claude"

# Eval output
EVAL_TABLE = "IPD3_Eval_Table"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ── BigQuery Client ───────────────────────────────────────────────────────────

def _get_credentials():
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    if creds_path and os.path.exists(creds_path):
        return service_account.Credentials.from_service_account_file(creds_path)
    if os.path.exists("service.json"):
        return service_account.Credentials.from_service_account_file("service.json")
    return None


def _get_bq_client() -> bigquery.Client:
    credentials = _get_credentials()
    return bigquery.Client(project=BQ_PROJECT_ID, credentials=credentials)


# ── Claude Client ─────────────────────────────────────────────────────────────

def get_claude_client() -> AnthropicVertex:
    return AnthropicVertex(region=GCP_REGION, project_id=PROJECT_ID)


def _call_claude(client: AnthropicVertex, prompt: str, max_retries: int = 3) -> Dict[str, Any]:
    """Call Claude Sonnet 4.6 and return parsed JSON."""
    for attempt in range(max_retries + 1):
        try:
            resp = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=LLM_MAX_TOKENS,
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
                # Manual repair
                text = re.sub(r",\s*}", "}", text)
                text = re.sub(r",\s*]", "]", text)
                return json.loads(text)
        except RateLimitError:
            if attempt == max_retries:
                raise
            wait = 2.0 * (2 ** attempt) * (1 + random.uniform(0, 0.25))
            print(f"    [Claude] Rate limited, retry {attempt+1}/{max_retries} in {wait:.1f}s")
            time.sleep(wait)
        except Exception as e:
            if attempt == max_retries:
                return {"error": str(e)}
            wait = 2.0 * (2 ** attempt)
            print(f"    [Claude] Error ({e}), retry {attempt+1}/{max_retries}")
            time.sleep(wait)
    return {}


# ── Step 1 & 2: Run Pipelines ────────────────────────────────────────────────

def run_gemini_pipeline(drug: Optional[str] = None, extra_args: List[str] = None):
    """Run ipd3bq__6_.py (Gemini) as a subprocess."""
    script = os.path.join(SCRIPT_DIR, "ipd3bq__6_.py")
    if not os.path.exists(script):
        # Try alternate name without parens
        for candidate in ["ipd3bq__6_.py", "ipd3bq_(6).py", "ipd3bq.py"]:
            p = os.path.join(SCRIPT_DIR, candidate)
            if os.path.exists(p):
                script = p
                break

    cmd = [sys.executable, script]
    if drug:
        cmd.append(drug)
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'='*60}")
    print(f"STEP 1: Running Gemini pipeline")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"[WARN] Gemini pipeline exited with code {result.returncode}")
    return result.returncode


def run_claude_pipeline(drug: Optional[str] = None, extra_args: List[str] = None):
    """Run ipd3bq_claude.py (Claude) as a subprocess."""
    script = os.path.join(SCRIPT_DIR, "ipd3bq_claude.py")

    cmd = [sys.executable, script]
    if drug:
        cmd.append(drug)
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'='*60}")
    print(f"STEP 2: Running Claude pipeline")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"[WARN] Claude pipeline exited with code {result.returncode}")
    return result.returncode


# ── Step 3: Load Results from BQ ──────────────────────────────────────────────

def load_circumvention_results(drug: Optional[str] = None) -> pd.DataFrame:
    """Load both Gemini and Claude circumvention results, joined by Drug+Category."""
    client = _get_bq_client()

    drug_filter = f"AND LOWER(TRIM(g.Drug_Name)) = LOWER('{drug}')" if drug else ""

    query = f"""
    WITH gemini AS (
        SELECT * EXCEPT(rn) FROM (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY Drug_Name, Patent_Category
                ORDER BY created_at DESC
            ) AS rn
            FROM `{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{GEMINI_CIRC_TABLE}`
        ) WHERE rn = 1
    ),
    claude AS (
        SELECT * EXCEPT(rn) FROM (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY Drug_Name, Patent_Category
                ORDER BY created_at DESC
            ) AS rn
            FROM `{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{CLAUDE_CIRC_TABLE}`
        ) WHERE rn = 1
    )
    SELECT
        COALESCE(g.Drug_Name, c.Drug_Name)            AS Drug_Name,
        COALESCE(g.Patent_Category, c.Patent_Category) AS Patent_Category,
        g.Patents           AS gemini_patents,
        g.Num_Patents       AS gemini_num_patents,
        g.Overall_Difficulty AS gemini_difficulty,
        g.Strategy          AS gemini_strategy,
        g.Rationale         AS gemini_rationale,
        g.Feasibility       AS gemini_feasibility,
        g.Regulatory_Pathway AS gemini_regulatory_pathway,
        g.Prior_Art_Support  AS gemini_prior_art_support,
        g.Key_Claim_Limitations AS gemini_key_claim_limitations,
        g.White_Space_Opportunities AS gemini_white_space,
        g.FDA_Precedents    AS gemini_fda_precedents,
        g.Orange_Book_Gaps  AS gemini_orange_book_gaps,
        g.Literature_Alternatives AS gemini_literature_alts,
        g.Summary           AS gemini_summary,
        c.Patents           AS claude_patents,
        c.Num_Patents       AS claude_num_patents,
        c.Overall_Difficulty AS claude_difficulty,
        c.Strategy          AS claude_strategy,
        c.Rationale         AS claude_rationale,
        c.Feasibility       AS claude_feasibility,
        c.Regulatory_Pathway AS claude_regulatory_pathway,
        c.Prior_Art_Support  AS claude_prior_art_support,
        c.Key_Claim_Limitations AS claude_key_claim_limitations,
        c.White_Space_Opportunities AS claude_white_space,
        c.FDA_Precedents    AS claude_fda_precedents,
        c.Orange_Book_Gaps  AS claude_orange_book_gaps,
        c.Literature_Alternatives AS claude_literature_alts,
        c.Summary           AS claude_summary
    FROM gemini g
    FULL OUTER JOIN claude c
        ON TRIM(g.Drug_Name) = TRIM(c.Drug_Name)
       AND TRIM(g.Patent_Category) = TRIM(c.Patent_Category)
    WHERE 1=1 {drug_filter}
    ORDER BY Drug_Name, Patent_Category
    """
    print("[BQ] Loading circumvention results (Gemini + Claude) ...")
    df = client.query(query, location=BQ_LOCATION).to_dataframe()
    print(f"[BQ] {len(df)} circumvention rows loaded")
    return df


def load_score_results(drug: Optional[str] = None) -> pd.DataFrame:
    """Load both Gemini and Claude thicket scores, joined by Drug+Jurisdiction."""
    client = _get_bq_client()

    drug_filter = f"AND LOWER(TRIM(g.Drug_Name)) = LOWER('{drug}')" if drug else ""

    query = f"""
    WITH gemini AS (
        SELECT * EXCEPT(rn) FROM (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY Drug_Name, Jurisdiction
                ORDER BY created_at DESC
            ) AS rn
            FROM `{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{GEMINI_SCORE_TABLE}`
        ) WHERE rn = 1
    ),
    claude AS (
        SELECT * EXCEPT(rn) FROM (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY Drug_Name, Jurisdiction
                ORDER BY created_at DESC
            ) AS rn
            FROM `{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{CLAUDE_SCORE_TABLE}`
        ) WHERE rn = 1
    )
    SELECT
        COALESCE(g.Drug_Name, c.Drug_Name)         AS Drug_Name,
        COALESCE(g.Jurisdiction, c.Jurisdiction)    AS Jurisdiction,
        g.Final_Score       AS gemini_final_score,
        g.Score_Label       AS gemini_score_label,
        g.Density_Score     AS gemini_density_score,
        g.Diversity_Score   AS gemini_diversity_score,
        g.Combined_Total    AS gemini_combined_total,
        g.Adjusted_Count    AS gemini_adjusted_count,
        g.Density_Interpretation  AS gemini_density_interp,
        g.Diversity_Interpretation AS gemini_diversity_interp,
        c.Final_Score       AS claude_final_score,
        c.Score_Label       AS claude_score_label,
        c.Density_Score     AS claude_density_score,
        c.Diversity_Score   AS claude_diversity_score,
        c.Combined_Total    AS claude_combined_total,
        c.Adjusted_Count    AS claude_adjusted_count,
        c.Density_Interpretation  AS claude_density_interp,
        c.Diversity_Interpretation AS claude_diversity_interp
    FROM gemini g
    FULL OUTER JOIN claude c
        ON TRIM(g.Drug_Name) = TRIM(c.Drug_Name)
       AND TRIM(g.Jurisdiction) = TRIM(c.Jurisdiction)
    WHERE 1=1 {drug_filter}
    ORDER BY Drug_Name, Jurisdiction
    """
    print("[BQ] Loading thicket score results (Gemini + Claude) ...")
    df = client.query(query, location=BQ_LOCATION).to_dataframe()
    print(f"[BQ] {len(df)} score rows loaded")
    return df


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
White Space Opportunities: {gemini_white_space}
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
White Space Opportunities: {claude_white_space}
FDA Precedents: {claude_fda_precedents}
Summary: {claude_summary}
--- END CLAUDE ---

Evaluate both outputs on these dimensions:
1. **Accuracy**: Are the identified claim limitations and design-around strategies
   technically and legally sound?
2. **Completeness**: Does the analysis cover all viable circumvention approaches?
3. **Feasibility Assessment**: Are feasibility ratings well-justified?
4. **Regulatory Viability**: Are the suggested 505(b)(2) pathways realistic?
5. **Prior Art Quality**: Are FDA/Orange Book/literature references relevant and specific?

Respond ONLY with valid JSON:
{{
  "agreement_level": "<full|partial|none>",
  "difficulty_agreement": <true or false>,
  "gemini_accuracy_score": <integer 1-5, where 5=excellent>,
  "claude_accuracy_score": <integer 1-5>,
  "gemini_completeness_score": <integer 1-5>,
  "claude_completeness_score": <integer 1-5>,
  "gemini_feasibility_score": <integer 1-5>,
  "claude_feasibility_score": <integer 1-5>,
  "gemini_regulatory_score": <integer 1-5>,
  "claude_regulatory_score": <integer 1-5>,
  "gemini_prior_art_score": <integer 1-5>,
  "claude_prior_art_score": <integer 1-5>,
  "preferred_system": "<gemini|claude|tie>",
  "preference_reason": "<1-2 sentences why one system's output is preferred or why they tie>",
  "discrepancy_explanation": "<1-2 sentences on where and why the outputs differ, or null if full agreement>",
  "combined_assessment": "<2-3 sentence overall assessment synthesising the best insights from both systems>",
  "recommended_strategies": ["<best strategy 1 from either system>", "<best strategy 2>"]
}}
"""

SCORE_EVAL_PROMPT = """
You are a senior pharmaceutical patent attorney acting as an impartial LLM judge.

Two independent LLM pipelines computed patent thicket scores for the same drug and jurisdiction.
The scoring methodology is deterministic (based on patent counts and diversity), but the
circumvention analysis that feeds into the interpretation may differ.

Drug: {drug}
Jurisdiction: {jurisdiction}

--- GEMINI SCORES ---
Final Score: {gemini_final_score} ({gemini_score_label})
Density Score: {gemini_density_score} | Diversity Score: {gemini_diversity_score}
Combined Total: {gemini_combined_total} | Adjusted Count: {gemini_adjusted_count}
Density Interpretation: {gemini_density_interp}
Diversity Interpretation: {gemini_diversity_interp}
--- END GEMINI ---

--- CLAUDE SCORES ---
Final Score: {claude_final_score} ({claude_score_label})
Density Score: {claude_density_score} | Diversity Score: {claude_diversity_score}
Combined Total: {claude_combined_total} | Adjusted Count: {claude_adjusted_count}
Density Interpretation: {claude_density_interp}
Diversity Interpretation: {claude_diversity_interp}
--- END CLAUDE ---

The thicket scoring formula is deterministic: identical input data should produce identical scores.
Differences indicate that the underlying patent classification or filtering diverged.

Respond ONLY with valid JSON:
{{
  "scores_match": <true or false>,
  "final_score_delta": <integer: gemini_final - claude_final>,
  "discrepancy_explanation": "<explanation of why scores differ, or 'Scores match — identical input processing' if they match>",
  "data_consistency_flag": "<consistent|minor_divergence|major_divergence>",
  "recommended_final_score": <integer 1-5 that you believe is most accurate>,
  "recommended_label": "<corresponding thicket label>",
  "assessment": "<1-2 sentence assessment of the scoring reliability>"
}}
"""

OVERALL_SYNTHESIS_PROMPT = """
You are a senior pharmaceutical patent attorney providing a final synthesis of a
dual-LLM patent thicket evaluation.

Drug: {drug}

Summary statistics from the evaluation:
- Circumvention categories evaluated: {num_categories}
- Categories where Gemini was preferred: {gemini_preferred}
- Categories where Claude was preferred: {claude_preferred}
- Categories that tied: {tied}
- Average Gemini accuracy: {avg_gemini_accuracy}
- Average Claude accuracy: {avg_claude_accuracy}
- Score agreement rate: {score_agreement_pct}%

Respond ONLY with valid JSON:
{{
  "overall_preferred_system": "<gemini|claude|tie>",
  "confidence": "<high|medium|low>",
  "key_strengths_gemini": ["<strength 1>", "<strength 2>"],
  "key_strengths_claude": ["<strength 1>", "<strength 2>"],
  "key_weaknesses_gemini": ["<weakness 1>"],
  "key_weaknesses_claude": ["<weakness 1>"],
  "recommendation": "<2-3 sentence recommendation on which system to trust and why>",
  "combined_thicket_assessment": "<2-3 sentence overall patent thicket assessment for this drug, synthesising both systems>"
}}
"""


# ── Evaluation Logic ──────────────────────────────────────────────────────────

def evaluate_circumvention_row(client: AnthropicVertex, row: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate one circumvention category (Gemini vs Claude) using LLM judge."""
    prompt = CIRCUMVENTION_EVAL_PROMPT.format(
        drug=row.get("Drug_Name", ""),
        patent_category=row.get("Patent_Category", ""),
        patents=row.get("gemini_patents") or row.get("claude_patents", ""),
        gemini_difficulty=row.get("gemini_difficulty", "N/A"),
        gemini_strategy=(row.get("gemini_strategy") or "")[:500],
        gemini_rationale=(row.get("gemini_rationale") or "")[:500],
        gemini_feasibility=row.get("gemini_feasibility", "N/A"),
        gemini_regulatory_pathway=(row.get("gemini_regulatory_pathway") or "")[:300],
        gemini_prior_art_support=(row.get("gemini_prior_art_support") or "")[:300],
        gemini_key_claim_limitations=(row.get("gemini_key_claim_limitations") or "")[:300],
        gemini_white_space=(row.get("gemini_white_space") or "")[:300],
        gemini_fda_precedents=(row.get("gemini_fda_precedents") or "")[:300],
        gemini_summary=(row.get("gemini_summary") or "")[:500],
        claude_difficulty=row.get("claude_difficulty", "N/A"),
        claude_strategy=(row.get("claude_strategy") or "")[:500],
        claude_rationale=(row.get("claude_rationale") or "")[:500],
        claude_feasibility=row.get("claude_feasibility", "N/A"),
        claude_regulatory_pathway=(row.get("claude_regulatory_pathway") or "")[:300],
        claude_prior_art_support=(row.get("claude_prior_art_support") or "")[:300],
        claude_key_claim_limitations=(row.get("claude_key_claim_limitations") or "")[:300],
        claude_white_space=(row.get("claude_white_space") or "")[:300],
        claude_fda_precedents=(row.get("claude_fda_precedents") or "")[:300],
        claude_summary=(row.get("claude_summary") or "")[:500],
    )
    return _call_claude(client, prompt)


def evaluate_score_row(client: AnthropicVertex, row: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate one thicket score row (Gemini vs Claude) using LLM judge."""
    prompt = SCORE_EVAL_PROMPT.format(
        drug=row.get("Drug_Name", ""),
        jurisdiction=row.get("Jurisdiction", ""),
        gemini_final_score=row.get("gemini_final_score", "N/A"),
        gemini_score_label=row.get("gemini_score_label", "N/A"),
        gemini_density_score=row.get("gemini_density_score", "N/A"),
        gemini_diversity_score=row.get("gemini_diversity_score", "N/A"),
        gemini_combined_total=row.get("gemini_combined_total", "N/A"),
        gemini_adjusted_count=row.get("gemini_adjusted_count", "N/A"),
        gemini_density_interp=row.get("gemini_density_interp", "N/A"),
        gemini_diversity_interp=row.get("gemini_diversity_interp", "N/A"),
        claude_final_score=row.get("claude_final_score", "N/A"),
        claude_score_label=row.get("claude_score_label", "N/A"),
        claude_density_score=row.get("claude_density_score", "N/A"),
        claude_diversity_score=row.get("claude_diversity_score", "N/A"),
        claude_combined_total=row.get("claude_combined_total", "N/A"),
        claude_adjusted_count=row.get("claude_adjusted_count", "N/A"),
        claude_density_interp=row.get("claude_density_interp", "N/A"),
        claude_diversity_interp=row.get("claude_diversity_interp", "N/A"),
    )
    return _call_claude(client, prompt)


def run_overall_synthesis(client: AnthropicVertex, drug: str,
                          circ_evals: List[Dict], score_evals: List[Dict]) -> Dict[str, Any]:
    """Produce an overall synthesis for one drug."""
    gemini_pref = sum(1 for e in circ_evals if e.get("preferred_system") == "gemini")
    claude_pref = sum(1 for e in circ_evals if e.get("preferred_system") == "claude")
    tied = sum(1 for e in circ_evals if e.get("preferred_system") == "tie")

    gemini_acc = [e.get("gemini_accuracy_score", 0) for e in circ_evals if e.get("gemini_accuracy_score")]
    claude_acc = [e.get("claude_accuracy_score", 0) for e in circ_evals if e.get("claude_accuracy_score")]
    avg_g = round(sum(gemini_acc) / len(gemini_acc), 1) if gemini_acc else 0
    avg_c = round(sum(claude_acc) / len(claude_acc), 1) if claude_acc else 0

    score_matches = sum(1 for e in score_evals if e.get("scores_match"))
    score_total = len(score_evals) or 1
    score_agree_pct = round(score_matches / score_total * 100, 1)

    prompt = OVERALL_SYNTHESIS_PROMPT.format(
        drug=drug,
        num_categories=len(circ_evals),
        gemini_preferred=gemini_pref,
        claude_preferred=claude_pref,
        tied=tied,
        avg_gemini_accuracy=avg_g,
        avg_claude_accuracy=avg_c,
        score_agreement_pct=score_agree_pct,
    )
    return _call_claude(client, prompt)


# ── Build Output Rows ─────────────────────────────────────────────────────────

def build_circ_eval_row(input_row: Dict, eval_result: Dict, timestamp: str) -> Dict:
    out = dict(input_row)
    out["eval_timestamp"] = timestamp
    out["eval_type"] = "circumvention"
    for k, v in eval_result.items():
        if isinstance(v, list):
            out[f"eval_{k}"] = "; ".join(str(x) for x in v)
        else:
            out[f"eval_{k}"] = v
    return out


def build_score_eval_row(input_row: Dict, eval_result: Dict, timestamp: str) -> Dict:
    out = dict(input_row)
    out["eval_timestamp"] = timestamp
    out["eval_type"] = "thicket_score"
    for k, v in eval_result.items():
        out[f"eval_{k}"] = v
    return out


# ── Save to BQ ────────────────────────────────────────────────────────────────

def save_eval_to_bq(rows: List[Dict]):
    if not rows:
        return
    client = _get_bq_client()
    table_ref = f"{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{EVAL_TABLE}"

    df = pd.DataFrame(rows)
    # Convert all columns to string-safe types for autodetect
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).replace({"None": None, "nan": None, "NaN": None})

    df["created_at"] = pd.Timestamp.now(tz="UTC")

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        autodetect=True,
        schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION],
    )
    job = client.load_table_from_dataframe(df, table_ref, job_config=job_config,
                                           location=BQ_LOCATION)
    job.result()
    print(f"[BQ] {EVAL_TABLE}: {len(df)} eval rows written → {table_ref}")


# ── Main Orchestrator ─────────────────────────────────────────────────────────

def orchestrate(drug: Optional[str] = None, skip_run: bool = False,
                skip_gemini: bool = False, skip_claude: bool = False,
                save_to_bq: bool = True, extra_args: List[str] = None):
    start = time.time()

    print("=" * 70)
    print("IPD3 ORCHESTRATOR — Gemini vs Claude Patent Thicket Evaluation")
    print(f"  Drug filter : {drug or 'ALL'}")
    print(f"  Judge model : {CLAUDE_MODEL}")
    print(f"  Save to BQ  : {save_to_bq}")
    print(f"  Skip runs   : {skip_run}  |  Skip Gemini: {skip_gemini}  |  Skip Claude: {skip_claude}")
    print("=" * 70)

    # ── Step 1 & 2: Run pipelines ────────────────────────────────────────
    if not skip_run:
        if not skip_gemini:
            run_gemini_pipeline(drug, extra_args)
        else:
            print("\n⏭️  Gemini pipeline skipped (--skip-gemini)")

        if not skip_claude:
            run_claude_pipeline(drug, extra_args)
        else:
            print("\n⏭️  Claude pipeline skipped (--skip-claude)")
    else:
        print("\n⏭️  Both pipelines skipped (--skip-run)")

    # ── Step 3: Load results ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 3: Loading results from BigQuery for evaluation")
    print(f"{'='*60}")

    circ_df = load_circumvention_results(drug)
    score_df = load_score_results(drug)

    if circ_df.empty and score_df.empty:
        print("[ERROR] No results found in either table. Ensure pipelines ran successfully.")
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

    # ── Evaluate circumvention strategies ──────────────────────────────
    circ_rows = circ_df.to_dict("records")
    circ_evals = []

    if circ_rows:
        print(f"\nEvaluating {len(circ_rows)} circumvention comparisons ...")

        def _eval_circ(row):
            try:
                drug_name = row.get("Drug_Name", "?")
                cat = row.get("Patent_Category", "?")
                print(f"  [{drug_name}/{cat}] Evaluating ...")
                result = evaluate_circumvention_row(client, row)
                circ_evals.append(result)
                return build_circ_eval_row(row, result, eval_timestamp)
            except Exception as e:
                print(f"  [ERROR] {row.get('Drug_Name')}/{row.get('Patent_Category')}: {e}")
                out = dict(row)
                out["eval_timestamp"] = eval_timestamp
                out["eval_error"] = str(e)
                return out

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(_eval_circ, r): r for r in circ_rows}
            for future in as_completed(futures):
                out = future.result()
                if out:
                    all_output_rows.append(out)

    # ── Evaluate thicket scores ───────────────────────────────────────────
    score_rows = score_df.to_dict("records")
    score_evals = []

    if score_rows:
        print(f"\nEvaluating {len(score_rows)} score comparisons ...")

        def _eval_score(row):
            try:
                drug_name = row.get("Drug_Name", "?")
                jur = row.get("Jurisdiction", "?")
                print(f"  [{drug_name}/{jur}] Evaluating scores ...")
                result = evaluate_score_row(client, row)
                score_evals.append(result)
                return build_score_eval_row(row, result, eval_timestamp)
            except Exception as e:
                print(f"  [ERROR] {row.get('Drug_Name')}/{row.get('Jurisdiction')}: {e}")
                out = dict(row)
                out["eval_timestamp"] = eval_timestamp
                out["eval_error"] = str(e)
                return out

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(_eval_score, r): r for r in score_rows}
            for future in as_completed(futures):
                out = future.result()
                if out:
                    all_output_rows.append(out)

    # ── Overall synthesis per drug ────────────────────────────────────────
    drugs_evaluated = set()
    for row in all_output_rows:
        d = row.get("Drug_Name")
        if d:
            drugs_evaluated.add(d)

    synthesis_rows = []
    for d in sorted(drugs_evaluated):
        print(f"\n  Generating overall synthesis for {d} ...")
        d_circ = [e for e, r in zip(circ_evals, circ_rows) if r.get("Drug_Name") == d]
        d_score = [e for e, r in zip(score_evals, score_rows) if r.get("Drug_Name") == d]
        try:
            synth = run_overall_synthesis(client, d, d_circ, d_score)
            synthesis_rows.append({
                "Drug_Name": d,
                "eval_type": "overall_synthesis",
                "eval_timestamp": eval_timestamp,
                **{f"synth_{k}": ("; ".join(v) if isinstance(v, list) else v)
                   for k, v in synth.items()},
            })
        except Exception as e:
            print(f"  [ERROR] Synthesis for {d}: {e}")

    all_output_rows.extend(synthesis_rows)

    # ── Summary ───────────────────────────────────────────────────────────
    gemini_pref = sum(1 for e in circ_evals if e.get("preferred_system") == "gemini")
    claude_pref = sum(1 for e in circ_evals if e.get("preferred_system") == "claude")
    tied = sum(1 for e in circ_evals if e.get("preferred_system") == "tie")
    score_matches = sum(1 for e in score_evals if e.get("scores_match"))

    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"  Circumvention categories evaluated : {len(circ_evals)}")
    print(f"    Gemini preferred : {gemini_pref}")
    print(f"    Claude preferred : {claude_pref}")
    print(f"    Tied             : {tied}")
    print(f"  Score rows evaluated               : {len(score_evals)}")
    print(f"    Scores match     : {score_matches}/{len(score_evals)}")
    print(f"  Drugs evaluated                    : {len(drugs_evaluated)}")
    print(f"  Total eval rows                    : {len(all_output_rows)}")
    print(f"  Elapsed                            : {time.time()-start:.1f}s")
    print("=" * 60)

    # ── Save ──────────────────────────────────────────────────────────────
    if save_to_bq:
        save_eval_to_bq(all_output_rows)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"ipd3_eval_{drug or 'all'}_{ts}.xlsx"
        pd.DataFrame(all_output_rows).to_excel(fname, index=False)
        print(f"  Excel output : {fname}")

    return {
        "output_rows": all_output_rows,
        "circ_evals": circ_evals,
        "score_evals": score_evals,
        "synthesis": synthesis_rows,
        "summary": {
            "gemini_preferred": gemini_pref,
            "claude_preferred": claude_pref,
            "tied": tied,
            "score_matches": score_matches,
            "total_score_evals": len(score_evals),
            "drugs_evaluated": len(drugs_evaluated),
        },
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="IPD3 Orchestrator — Run Gemini & Claude, then LLM-as-Judge Evaluation"
    )
    parser.add_argument("--drug", default=None, help="Drug name (omit = all drugs)")
    parser.add_argument("--no-bq", action="store_true", help="Save eval to Excel instead of BigQuery")
    parser.add_argument("--skip-run", action="store_true", help="Skip pipeline runs, only do evaluation")
    parser.add_argument("--skip-gemini", action="store_true", help="Skip Gemini pipeline run")
    parser.add_argument("--skip-claude", action="store_true", help="Skip Claude pipeline run")
    parser.add_argument("--skip-circumvention", action="store_true",
                        help="Pass --skip-circumvention to both pipelines")
    parser.add_argument("--refresh-scores", action="store_true",
                        help="Pass --refresh-scores to both pipelines")
    parser.add_argument("--rerun", action="store_true",
                        help="Pass --rerun to both pipelines")
    parser.add_argument("--max-patents-per-category", type=int, default=10,
                        help="Max patents per category (passed to both pipelines)")
    args = parser.parse_args()

    # Build extra args to forward to sub-pipelines
    extra = []
    if args.skip_circumvention:
        extra.append("--skip-circumvention")
    if args.refresh_scores:
        extra.append("--refresh-scores")
    if args.rerun:
        extra.append("--rerun")
    if args.max_patents_per_category != 10:
        extra.extend(["--max-patents-per-category", str(args.max_patents_per_category)])

    result = orchestrate(
        drug=args.drug,
        skip_run=args.skip_run,
        skip_gemini=args.skip_gemini,
        skip_claude=args.skip_claude,
        save_to_bq=not args.no_bq,
        extra_args=extra or None,
    )
    if "error" in result:
        print(f"Error: {result['error']}")
        sys.exit(1)
