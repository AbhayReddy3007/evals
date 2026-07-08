"""
drug_filter.py
──────────────
Single source of truth for which drugs the pipeline is allowed to process.

Scope is defined by one canonical BigQuery query (GLP-1 / glucagon-like
peptide-1 agonists only — antagonists explicitly excluded):

    SELECT DISTINCT cleaned_generic_name
    FROM `{PROJECT_ID}.{DATASET_ID}.vw_drug_details_full`
    WHERE
    (
        UPPER(cleaned_Target) LIKE '%GLUCAGON LIKE PEPTIDE 1%'
        OR UPPER(cleaned_Target) LIKE '%GLP-1%'
        OR UPPER(cleaned_Target) LIKE '%GLUCAGON LIKE PEPTIDE-1%'
        OR (data_source = 'IPD' AND Mechanism_of_Action = 'Glucagon-like peptide-1 (GLP-1) agonist')
    )
    AND Mechanism_of_Action IS NOT NULL
    AND LOWER(Mechanism_of_Action) NOT LIKE '%antagonist%'

`cleaned_generic_name` is the canonical drug name used everywhere downstream
(GCS patent folders, BigQuery `Drug Name` / `drug_name` columns, etc.).

Every part of the pipeline that either (a) discovers "all drugs" on its own,
or (b) accepts a single drug name from the command line, should run that
name through this module so that non-GLP-1 drugs are never processed —
even if they exist in older GCS folders, BigQuery tables, or are typed in
by mistake.

Usage:
    from cog import drug_filter            # from top-level scripts
    from . import drug_filter               # from inside cog/

    allowed = drug_filter.fetch_allowed_drugs()
    drug_filter.is_allowed_drug("Semaglutide")
    drug_filter.filter_allowed_drugs(["Semaglutide", "Imatinib"])
    drug_filter.require_allowed_drug("Imatinib")   # -> False, prints why
"""

import os
import re
import threading
from typing import Dict, Iterable, List, Optional, Set

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

PROJECT_ID = (
    os.getenv("BQ_PROJECT_ID")
    or os.getenv("PROJECT_ID")
    or "cognito-prod-394707"
)
DATASET_ID = os.getenv("BQ_DATASET_ID", "cognito_prod_datamart")
SOURCE_VIEW = os.getenv("DRUG_UNIVERSE_VIEW", "vw_drug_details_full")

CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

# Set DRUG_FILTER_DISABLED=1 only for local debugging against a sandbox
# project that doesn't have vw_drug_details_full. Never disable in prod.
_DISABLED = os.getenv("DRUG_FILTER_DISABLED", "").strip() in ("1", "true", "True")

GLP1_DRUG_QUERY_TEMPLATE = """
SELECT DISTINCT cleaned_generic_name
FROM `{project}.{dataset}.{view}`
WHERE
(
    UPPER(cleaned_Target) LIKE '%GLUCAGON LIKE PEPTIDE 1%'
    OR UPPER(cleaned_Target) LIKE '%GLP-1%'
    OR UPPER(cleaned_Target) LIKE '%GLUCAGON LIKE PEPTIDE-1%'
    OR (
        data_source = 'IPD'
        AND Mechanism_of_Action = 'Glucagon-like peptide-1 (GLP-1) agonist'
    )
)
AND Mechanism_of_Action IS NOT NULL
AND LOWER(Mechanism_of_Action) NOT LIKE '%antagonist%'
"""

_lock = threading.Lock()
_cached_norm: Optional[Set[str]] = None          # normalized name -> membership
_cached_display: Optional[Dict[str, str]] = None  # normalized -> original casing


def get_glp1_query(project_id: str = None, dataset_id: str = None, view: str = None) -> str:
    """Return the canonical GLP-1 drug-universe query, fully qualified."""
    return GLP1_DRUG_QUERY_TEMPLATE.format(
        project=project_id or PROJECT_ID,
        dataset=dataset_id or DATASET_ID,
        view=view or SOURCE_VIEW,
    ).strip()


def normalize(name: str) -> str:
    """Fuzzy-normalize a drug name for matching (lowercase, strip spaces/-/_).

    Matches the convention already used by cog/gcs_lister.py so GLP-1 query
    results line up with GCS folder names, BQ Drug Name values, etc.
    regardless of spacing/casing/hyphenation differences.
    """
    return re.sub(r"[\s\-_]+", "", str(name or "").lower().strip())


def safe_name(drug_name: str, lowercase: bool = False) -> str:
    """Canonical GCS / filesystem-safe drug name.

    Preserves the canonical drug name from BigQuery as closely as possible.
    In particular the ``+`` in combination drugs like
    ``Cagrilintide+Semaglutide`` is kept.

    Only characters that are genuinely unsafe in GCS paths or Linux
    filesystems are replaced with ``_``:  ``/ \\ < > : " | ? *`` and
    null bytes.

    For AlloyDB collection names (which cannot contain ``+``), use
    ``safe_collection_name()`` instead.
    """
    s = str(drug_name or "").strip()
    s = re.sub(r'[/\\\\<>:"|?*\x00]', "_", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")
    if lowercase:
        s = s.lower()
    return s


def safe_collection_name(drug_name: str) -> str:
    """Strict alphanumeric-safe name for AlloyDB collection identifiers.

    Replaces every character that is NOT alphanumeric, underscore, or
    hyphen with ``_``.  Existing collections were already indexed with
    this convention (e.g. ``patents_Cagrilintide_Semaglutide``).
    """
    s = str(drug_name or "").strip()
    s = re.sub(r"[^a-zA-Z0-9_-]", "_", s)
    s = re.sub(r"[_-]{2,}", "_", s)
    s = s.strip("_-")
    return s


def _get_bq_client(project_id: str = None):
    from google.cloud import bigquery
    try:
        from google.oauth2 import service_account
    except ImportError:
        service_account = None

    proj = project_id or PROJECT_ID
    if CREDENTIALS_PATH and service_account and os.path.exists(CREDENTIALS_PATH):
        creds = service_account.Credentials.from_service_account_file(CREDENTIALS_PATH)
        return bigquery.Client(credentials=creds, project=proj)
    return bigquery.Client(project=proj)


def fetch_allowed_drugs(force_refresh: bool = False, project_id: str = None,
                        dataset_id: str = None) -> Set[str]:
    """
    Return the set of normalized drug names allowed by the canonical GLP-1
    query. Cached for the lifetime of the process (BigQuery is hit at most
    once per run unless force_refresh=True).
    """
    global _cached_norm, _cached_display

    if _DISABLED:
        return None  # caller treats None as "no filtering"

    with _lock:
        if _cached_norm is not None and not force_refresh:
            return _cached_norm

        query = get_glp1_query(project_id, dataset_id)
        try:
            client = _get_bq_client(project_id)
            rows = client.query(query).result()
        except Exception as e:
            print(f"[DRUG FILTER] WARNING: could not load the GLP-1 drug universe "
                  f"from BigQuery ({e}). Proceeding WITHOUT drug filtering for this "
                  f"run — re-run once BigQuery access is restored.")
            return None

        norm_set: Set[str] = set()
        display: Dict[str, str] = {}
        for row in rows:
            raw = row["cleaned_generic_name"] if "cleaned_generic_name" in row.keys() else None
            if not raw:
                continue
            raw = str(raw).strip()
            if not raw:
                continue
            norm = normalize(raw)
            norm_set.add(norm)
            display.setdefault(norm, raw)

        _cached_norm = norm_set
        _cached_display = display
        print(f"[DRUG FILTER] {len(norm_set)} allowed GLP-1 drug name(s) loaded "
              f"from {project_id or PROJECT_ID}.{dataset_id or DATASET_ID}.{SOURCE_VIEW}")
        return norm_set


def is_allowed_drug(drug_name: str, allowed: Optional[Set[str]] = None) -> bool:
    """True if drug_name matches (fuzzy) a name returned by the GLP-1 query.

    If the allowed-list can't be loaded (BQ unreachable), fails OPEN (returns
    True) so a transient BQ outage doesn't silently skip a whole pipeline run
    — the warning printed by fetch_allowed_drugs() makes this visible.
    """
    if not drug_name:
        return False
    if allowed is None:
        allowed = fetch_allowed_drugs()
    if allowed is None:  # filtering disabled or BQ unreachable
        return True
    return normalize(drug_name) in allowed


def filter_allowed_drugs(drug_names: Iterable[str],
                         allowed: Optional[Set[str]] = None) -> List[str]:
    """Filter an iterable of drug names down to only GLP-1-allowed ones.

    Preserves original casing/order. Prints which names were excluded.
    """
    if allowed is None:
        allowed = fetch_allowed_drugs()
    if allowed is None:  # filtering disabled or BQ unreachable
        return list(drug_names)

    kept, dropped = [], []
    for d in drug_names:
        if d and normalize(d) in allowed:
            kept.append(d)
        else:
            dropped.append(d)

    if dropped:
        print(f"[DRUG FILTER] Excluded {len(dropped)} non-GLP-1 drug(s) "
              f"(not in the canonical query result): {sorted(set(dropped))}")
    return kept


def require_allowed_drug(drug_name: str) -> bool:
    """
    For single-drug CLI entry points. Returns True if processing should
    continue, False if the caller should skip/exit. Always prints a clear
    reason either way it short-circuits.
    """
    if is_allowed_drug(drug_name):
        return True
    print(
        f"[DRUG FILTER] '{drug_name}' is not a GLP-1 agonist per the canonical "
        f"BigQuery query (vw_drug_details_full) — skipping. Any other drug is "
        f"out of scope for this pipeline."
    )
    return False
