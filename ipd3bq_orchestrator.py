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

BQ_PROJECT_ID = os.getenv("BQ_PROJECT_ID", "cognito-prod-394707")
BQ_DATASET_ID = os.getenv("BQ_DATASET_ID", "cognito_prod_datamart")
BQ_LOCATION   = os.getenv("BQ_LOCATION", "asia-south1")
GCP_REGION    = os.getenv("GCP_REGION", "us-east5")
PROJECT_ID    = os.getenv("BQ_PROJECT_ID", "cognito-prod-394707")

CLAUDE_MODEL   = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
MAX_WORKERS    = 6
LLM_MAX_TOKENS = 8192

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
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
1. Accuracy: Are claim limitations and strategies technically/legally sound?
2. Completeness: Does the analysis cover all viable circumvention approaches?
3. Feasibility Assessment: Are feasibility ratings well-justified?
4. Regulatory Viability: Are 505(b)(2) pathways realistic?
5. Prior Art Quality: Are FDA/Orange Book/literature references relevant?

Respond ONLY with valid JSON:
{{
  "agreement_level": "<full|partial|none>",
  "difficulty_agreement": <true or false>,
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

Respond ONLY with valid JSON:
{{
  "scores_match": <true or false>,
  "final_score_delta": <integer: gemini - claude>,
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
- Score agreement rate: {score_agreement_pct}%

Respond ONLY with valid JSON:
{{
  "overall_preferred_system": "<gemini|claude|tie>",
  "confidence": "<high|medium|low>",
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
    gemini_acc = [e.get("gemini_accuracy_score", 0) for e in circ_evals if e.get("gemini_accuracy_score")]
    claude_acc = [e.get("claude_accuracy_score", 0) for e in circ_evals if e.get("claude_accuracy_score")]
    avg_g = round(sum(gemini_acc) / len(gemini_acc), 1) if gemini_acc else 0
    avg_c = round(sum(claude_acc) / len(claude_acc), 1) if claude_acc else 0
    score_matches = sum(1 for e in score_evals if e.get("scores_match"))
    score_total = len(score_evals) or 1

    prompt = OVERALL_SYNTHESIS_PROMPT.format(
        drug=drug, num_categories=len(circ_evals),
        gemini_preferred=gemini_pref, claude_preferred=claude_pref, tied=tied,
        avg_gemini_accuracy=avg_g, avg_claude_accuracy=avg_c,
        score_agreement_pct=round(score_matches / score_total * 100, 1),
    )
    return _call_claude(client, prompt)


# ── Save Results ──────────────────────────────────────────────────────────────

def save_eval_to_excel(rows, drug=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = os.path.join(OUTPUT_DIR, f"ipd3_eval_{drug or 'all'}_{ts}.xlsx")
    pd.DataFrame(rows).to_excel(fname, index=False)
    print(f"[EXCEL] Eval results → {fname}")
    return fname


def save_eval_to_bq(rows):
    if not rows: return
    from google.cloud import bigquery
    from google.oauth2 import service_account
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    creds = (service_account.Credentials.from_service_account_file(creds_path)
             if creds_path and os.path.exists(creds_path) else None)
    client = bigquery.Client(project=BQ_PROJECT_ID, credentials=creds)
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

    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"  Circumvention categories : {len(circ_evals)}")
    print(f"    Gemini preferred       : {gemini_pref}")
    print(f"    Claude preferred       : {claude_pref}")
    print(f"    Tied                   : {tied}")
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
    parser.add_argument("--max-patents-per-category", type=int, default=10)
    args = parser.parse_args()

    extra = []
    if args.skip_circumvention: extra.append("--skip-circumvention")
    if args.refresh_scores: extra.append("--refresh-scores")
    if args.rerun: extra.append("--rerun")
    if args.max_patents_per_category != 10:
        extra.extend(["--max-patents-per-category", str(args.max_patents_per_category)])

    result = orchestrate(
        drug=args.drug, skip_run=args.skip_run,
        skip_gemini=args.skip_gemini, skip_claude=args.skip_claude,
        to_bq=args.to_bq, extra_args=extra or None,
        no_bq_pipelines=args.no_bq,
    )
    if "error" in result:
        print(f"Error: {result['error']}")
        sys.exit(1)
