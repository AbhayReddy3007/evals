"""
indexer.py
──────────
Handles:
  - Downloading single patent PDFs from GCS
  - Uploading PDFs to Gemini Files API and extracting full text
  - Extracting filing/grant dates from cover page (in parallel with text extraction)
    → Supports scanned/image-only PDFs via OCR-aware Gemini Vision prompts
    → 3-tier fallback: vision → text extraction → OCR-focused vision
  - Chunking text and generating embeddings
  - Storing chunks + sentinel records in AlloyDB (pgvector)
    → Dates stored in EVERY chunk's metadata and the sentinel upfront.
      No backfill step needed — copies automatically carry dates.
  - Cross-collection deduplication (copy instead of re-index)
  - **Parallel indexing**: multiple patents processed concurrently via
    asyncio.Semaphore-bounded tasks.
"""

import asyncio
import hashlib
import json
import random
import re
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from google import genai
from google.genai import types

from .gcs_lister import get_gcs_client, GCS_BUCKET_NAME
from .alloydb_client import AlloyDBClient

# ─────────────────────────────────────────────
# Gemini + AlloyDB clients
# ─────────────────────────────────────────────

import os


class _LazyGenaiClient:
    """Lazily instantiate the real genai.Client on first attribute access.

    Importing this module no longer requires GOOGLE_API_KEY/GEMINI_API_KEY to
    be present — the key is only required the first time the client is
    actually used. All existing call sites (gemini_client.models...,
    gemini_client.aio..., gemini_client.files...) work unchanged because
    attribute access is transparently forwarded to the real client.
    """
    _client = None

    def _resolve(self):
        if _LazyGenaiClient._client is None:
            key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not key:
                raise RuntimeError(
                    "GOOGLE_API_KEY or GEMINI_API_KEY must be set to use the "
                    "Gemini client."
                )
            _LazyGenaiClient._client = genai.Client(api_key=key)
        return _LazyGenaiClient._client

    def __getattr__(self, name):
        return getattr(self._resolve(), name)


gemini_client = _LazyGenaiClient()

chroma_client = AlloyDBClient()
print("[ALLOYDB] Client initialized")

# ─────────────────────────────────────────────
# Progress tracking — GCS-backed (survives container restarts)
# ─────────────────────────────────────────────

from . import gcs_cache
from . import drug_filter

_PROGRESS_SUBFOLDER = "indexer_progress"


def _progress_filename(drug_name: str) -> str:
    """Return the progress filename for a given drug."""
    safe = drug_filter.safe_name(drug_name)
    return f"progress_{safe}.json"


def _load_progress(drug_name: str) -> set:
    """Load the set of filenames already completed for this drug batch from GCS."""
    try:
        data = gcs_cache.read_json(
            _PROGRESS_SUBFOLDER,
            _progress_filename(drug_name),
        )
        if data is not None:
            completed = set(data.get("completed", []))
            print(f"[RESUME] Found progress for '{drug_name}': {len(completed)} patents already done")
            return completed
    except Exception:
        pass
    return set()


def _save_progress(drug_name: str, completed: set):
    """Save the set of completed filenames for this drug batch to GCS."""
    try:
        gcs_cache.write_json(
            _PROGRESS_SUBFOLDER,
            _progress_filename(drug_name),
            {"completed": sorted(completed)},
        )
    except Exception as e:
        print(f"[RESUME] Failed to save progress for '{drug_name}': {e}")


def _mark_completed(drug_name: str, filename: str, completed: set):
    """Mark a single patent as completed and persist immediately."""
    completed.add(filename)
    _save_progress(drug_name, completed)


def _clear_progress(drug_name: str):
    """Remove progress file when the entire batch finishes successfully."""
    try:
        deleted = gcs_cache.delete_blob(
            _PROGRESS_SUBFOLDER,
            _progress_filename(drug_name),
        )
        if deleted:
            print(f"[RESUME] Cleared progress for '{drug_name}' — batch complete")
    except Exception:
        pass

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

CHUNK_SIZE_CHARS           = 2000
OVERLAP_CHARS              = 400
_MAX_UPLOAD_RETRIES        = 3
_MAX_EMBED_RETRIES         = 3
_GEMINI_FILE_SIZE_LIMIT_MB = 2000

# Max concurrent patent-processing tasks per drug.
# Each task does: GCS download → Gemini upload → text extraction →
# date extraction → embedding → AlloyDB write.
# 10 tasks run in parallel for each drug, controlled by asyncio.Semaphore.
# Override via INDEXER_CONCURRENCY env var if needed.
MAX_CONCURRENCY = int(os.getenv("INDEXER_CONCURRENCY", "10"))

# Cover page render DPI — higher values produce sharper images for
# Gemini Vision to read small-font fields like "(22) Filed:" and
# "(45) Date of Patent:".  150 was often too low for dense USPTO
# cover pages; 250 works reliably.
COVER_PAGE_DPI = int(os.getenv("COVER_PAGE_DPI", "300"))

DATE_EXTRACTION_PROMPT = """You are a patent document parser. This image is the cover page of a patent or patent application (it may be a scanned image with no embedded text — use OCR to read it).

Extract ONLY these two dates:
1. Filing date: look for ANY of these labels (in any language):
   - "(22) Filed:" (US patents)
   - "(22) International Filing Date:" (WO/PCT applications)
   - "(22) Date de dépôt international:" (French WO)
   - "(22) Fecha de presentación:" or "Fecha de depósito:" (Spanish/MX patents)
   - "(22) Data de Depósito:" or "Data de apresentação:" (Brazilian/PT patents)
   - "Filing Date:", "Date Filed:", "PCT Filed:"
   - Any field labelled with INID code (22)
2. Grant/Publication date: look for ANY of these labels (in any language):
   - "(45) Date of Patent:" (US granted patents)
   - "(43) International Publication Date:" (WO/PCT applications)
   - "(43) Date de la publication internationale:" (French WO)
   - "(43) Fecha de publicación internacional:" (Spanish WO/MX)
   - "(43) Data da Publicação Internacional:" (Brazilian/PT WO)
   - "(45) Дата публикации:" (Eurasian/Russian EA patents)
   - "(43) Дата международной публикации:" (Russian WO)
   - "Grant Date:", "Published:", "Publication Date:", "Data de Publicação:"
   - Any field labelled with INID code (43) or (45)

Rules:
- Return ONLY the dates for THIS patent/application (not cited prior art references)
- For WO/PCT, EP, BR, MX, EA, JP applications: use the publication date as grant_date (or null if not found)
- If a date is missing or unclear -> use null
- Convert any date format (e.g. "Dec. 14, 2023", "14.12.2023", "14 December 2023", "14.12.2023", "2023.12.14") to YYYY-MM-DD format

Return ONLY valid JSON with no markdown, no explanation:
{
  "filing_date": "YYYY-MM-DD or null",
  "grant_date":  "YYYY-MM-DD or null"
}
"""

# Fallback prompt: extract dates from raw text of the cover page
DATE_EXTRACTION_TEXT_PROMPT = """You are a patent document parser. Extract ONLY the filing date and grant/publication date from the text below.

Look for (in any language):
- Filing date (INID code 22): "(22) Filed:", "(22) International Filing Date:", "Filing Date:", "Date Filed:", "PCT Filed:", "Fecha de presentación:", "Data de Depósito:", "Дата подачи:"
- Grant/Publication date (INID code 43 or 45): "(45) Date of Patent:", "(43) International Publication Date:", "Grant Date:", "Published:", "Publication Date:", "Fecha de publicación:", "Data de Publicação:", "Дата публикации:"

Rules:
- Return ONLY the dates for THIS patent/application (not cited prior art references or foreign patent documents)
- For WO/PCT, EP, BR, MX, EA, JP applications: use the publication date as grant_date (or null if not found)
- If a date is missing or unclear -> use null
- Format all dates as YYYY-MM-DD

Return ONLY valid JSON with no markdown, no explanation:
{
  "filing_date": "YYYY-MM-DD or null",
  "grant_date":  "YYYY-MM-DD or null"
}

TEXT:
"""

# OCR-focused prompt for scanned/image-only patents (used as final fallback)
_OCR_DATE_PROMPT = """This is a scanned image of a patent cover page. Please carefully read ALL text in the image using OCR.

The document may be a US patent, a WO/PCT international application, or another type.

Then extract ONLY these two dates:
1. Filing date — look for:
   - "(22) Filed:" (US patents — usually left side)
   - "(22) International Filing Date:" (WO/PCT — usually near the top)
2. Grant/Publication date — look for:
   - "(45) Date of Patent:" (US patents — usually right side near the top)
   - "(43) International Publication Date:" (WO/PCT — usually near the top)

Convert any dates you find to YYYY-MM-DD format.
Dates may appear as "Dec. 14, 2023", "14 December 2023", "14.12.2023", etc.

Return ONLY valid JSON:
{"filing_date": "YYYY-MM-DD or null", "grant_date": "YYYY-MM-DD or null"}
"""


# ─────────────────────────────────────────────
# Date helpers
# ─────────────────────────────────────────────

def _clean_date(val) -> Optional[str]:
    """
    Normalize a date value from Gemini's JSON response.

    Handles common Gemini quirks:
      - String "null" / "None" / "N/A" → None
      - Empty string → None
      - Validates YYYY-MM-DD format
      - Salvages non-ISO formats like "Dec. 14, 2023"
    """
    if val is None:
        return None
    if isinstance(val, str):
        stripped = val.strip()
        if stripped.lower() in ("null", "none", "n/a", "unknown", ""):
            return None
        # Validate YYYY-MM-DD pattern
        if re.match(r"^\d{4}-\d{2}-\d{2}$", stripped):
            return stripped
        # Try to salvage partial dates like "Dec. 14, 2023" or "14.12.2023"
        try:
            from datetime import datetime
            for fmt in (
                "%B %d, %Y", "%b. %d, %Y", "%b %d, %Y",
                "%m/%d/%Y", "%d/%m/%Y",
                "%d.%m.%Y",   # European dot-separated: 14.12.2023
                "%Y.%m.%d",   # ISO-ish: 2023.12.14
                "%d-%m-%Y",   # dash-separated: 14-12-2023
            ):
                try:
                    dt = datetime.strptime(stripped, fmt)
                    return dt.strftime("%Y-%m-%d")
                except ValueError:
                    continue
        except Exception:
            pass
        print(f"[WARN] Could not parse date value: '{stripped}'")
        return None
    return None


def _has_valid_dates(meta: dict) -> bool:
    """Check whether metadata has at least one non-empty date."""
    filing = meta.get("filing_date", "")
    grant  = meta.get("grant_date", "")
    return bool(filing and filing not in ("", "null", "None")) or \
           bool(grant and grant not in ("", "null", "None"))


def _parse_gemini_date_response(raw: str) -> Dict:
    """
    Parse a Gemini response that should contain date JSON.

    Handles:
      - Clean JSON
      - JSON wrapped in markdown code fences
      - Empty responses (returns nulls)
    """
    if not raw or not raw.strip():
        return {"filing_date": None, "grant_date": None}

    cleaned = raw.strip()
    if "```" in cleaned:
        cleaned = re.sub(r"```(?:json)?", "", cleaned).replace("```", "").strip()

    # Try to extract JSON from the response even if there's surrounding text
    # Use re.DOTALL so { ... } matches across newlines
    match = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)

    try:
        dates = json.loads(cleaned)
        return {
            "filing_date": _clean_date(dates.get("filing_date")),
            "grant_date":  _clean_date(dates.get("grant_date")),
        }
    except (json.JSONDecodeError, AttributeError):
        return {"filing_date": None, "grant_date": None}


async def _call_gemini_for_dates(contents: list, filename: str) -> Dict:
    """
    Call Gemini to extract dates, with automatic retry.

    First tries with response_mime_type="application/json" (structured output).
    If that returns empty, retries WITHOUT the constraint so Gemini can
    produce free-form text that we parse manually.
    """
    # Attempt 1: structured JSON output
    try:
        response = await gemini_client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0,
                max_output_tokens=256,
            ),
        )
        raw = response.text.strip() if response.text else ""
        if raw:
            dates = _parse_gemini_date_response(raw)
            if dates.get("filing_date") or dates.get("grant_date"):
                return dates
        print(f"  [DATES] Structured JSON response was empty for {filename} — retrying without constraint")
    except Exception as e:
        print(f"  [DATES] Structured call failed for {filename}: {e} — retrying without constraint")

    # Attempt 2: free-form (no response_mime_type constraint)
    try:
        response = await gemini_client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=256,
            ),
        )
        raw = response.text.strip() if response.text else ""
        if raw:
            dates = _parse_gemini_date_response(raw)
            return dates
    except Exception as e:
        print(f"  [DATES] Free-form call also failed for {filename}: {e}")

    return {"filing_date": None, "grant_date": None}


# ─────────────────────────────────────────────
# GCS download
# ─────────────────────────────────────────────

def download_single_patent_pdf(blob_name: str, filename: str, drug_name: str) -> Optional[dict]:
    if not GCS_BUCKET_NAME:
        print("[GCS] GCS_BUCKET_NAME not set — cannot download")
        return None
    try:
        client     = get_gcs_client()
        bucket     = client.bucket(GCS_BUCKET_NAME)
        blob       = bucket.blob(blob_name)
        tmp_dir    = Path(tempfile.mkdtemp(prefix=f"patents_{drug_name}_"))
        local_path = tmp_dir / filename
        blob.download_to_filename(str(local_path))
        print(f"[GCS] Downloaded {filename}")
        return {"filename": filename, "path": str(local_path), "tmp_dir": str(tmp_dir)}
    except Exception as e:
        print(f"[GCS] Failed to download {filename}: {e}")
        return None


# ─────────────────────────────────────────────
# Gemini file upload + text extraction
# ─────────────────────────────────────────────

async def upload_pdf_to_gemini(file_path: str) -> Optional[object]:
    path = Path(file_path)
    print(f"[UPLOAD] Uploading {path.name}...")

    file_size_mb = path.stat().st_size / (1024 * 1024)
    print(f"[UPLOAD] File size: {file_size_mb:.1f} MB")
    if file_size_mb > _GEMINI_FILE_SIZE_LIMIT_MB:
        print(f"[ERROR] {path.name} is {file_size_mb:.1f} MB — exceeds {_GEMINI_FILE_SIZE_LIMIT_MB} MB limit.")
        return None

    loop = asyncio.get_running_loop()

    for attempt in range(1, _MAX_UPLOAD_RETRIES + 1):
        try:
            uploaded_file = await loop.run_in_executor(
                None,
                lambda: gemini_client.files.upload(
                    file=file_path,
                    config=dict(mime_type="application/pdf"),
                )
            )

            max_wait, wait_time = 60, 0
            while uploaded_file.state == "PROCESSING" and wait_time < max_wait:
                await asyncio.sleep(2 + random.uniform(0, 0.5))
                _file_name = uploaded_file.name  # capture by value to avoid closure bug
                uploaded_file = await loop.run_in_executor(
                    None, lambda n=_file_name: gemini_client.files.get(name=n)
                )
                wait_time += 2
                print(f"[UPLOAD] Processing... ({wait_time}s)")

            if uploaded_file.state == "FAILED":
                print(f"[ERROR] Gemini failed to process {path.name}")
                return None

            print(f"[UPLOAD] Ready: {path.name}")
            return uploaded_file

        except Exception as e:
            if attempt == _MAX_UPLOAD_RETRIES:
                print(f"[ERROR] Upload failed for {path.name} after {_MAX_UPLOAD_RETRIES} attempts: {e}")
                return None
            backoff = (2 ** attempt) + random.uniform(0, 1)
            print(f"[UPLOAD] Attempt {attempt} failed: {e} — retrying in {backoff:.1f}s")
            await asyncio.sleep(backoff)

    return None


async def extract_text_via_gemini(uploaded_file: object, filename: str) -> Optional[str]:
    """
    Extract full plain text from uploaded PDF via Gemini.
    max_output_tokens=65536 ensures claims section at end of patent is captured.
    """
    print(f"[TEXT EXTRACTION] Extracting text from {filename}...")
    try:
        response = await gemini_client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                uploaded_file,
                "Extract ALL text from this patent document exactly as it appears. "
                "Include every section: cover page, patent number, all dates, "
                "inventors, assignee, claims, description, abstract. "
                "Return only plain text with no commentary or formatting.",
            ],
            config=types.GenerateContentConfig(temperature=0, max_output_tokens=65536),
        )

        try:
            finish_reason = response.candidates[0].finish_reason
            if str(finish_reason) in ("MAX_TOKENS", "2"):
                print(
                    f"[WARNING] Text extraction for {filename} hit MAX_TOKENS — "
                    "document may be partially indexed. Consider splitting the PDF."
                )
        except (IndexError, AttributeError):
            pass

        text = response.text
        print(f"[TEXT EXTRACTION] Extracted {len(text)} characters from {filename}")
        return text

    except Exception as e:
        print(f"[ERROR] Text extraction failed for {filename}: {e}")
        return None


async def cleanup_uploaded_file(uploaded_file: object):
    loop = asyncio.get_running_loop()
    _file_name = uploaded_file.name  # capture by value
    try:
        await loop.run_in_executor(
            None, lambda: gemini_client.files.delete(name=_file_name)
        )
        print(f"[UPLOAD] Cleaned up {_file_name}")
    except Exception as e:
        print(f"[WARNING] Could not clean up {_file_name}: {e}")


# ─────────────────────────────────────────────
# Fallback text extraction for image-only PDFs
# ─────────────────────────────────────────────

def extract_text_via_pymupdf(file_path: str, filename: str) -> Optional[str]:
    """
    Extract text from a PDF using PyMuPDF's built-in text extractor.

    Returns the concatenated text from all pages, or None if the PDF
    is image-only (no embedded text layer) or extraction fails.
    """
    try:
        import fitz
    except ImportError:
        print(f"[PYMUPDF] pymupdf not available — cannot extract text for {filename}")
        return None

    try:
        doc = fitz.open(file_path)
        all_text = []
        for page_num in range(len(doc)):
            page_text = doc[page_num].get_text("text")
            if page_text and page_text.strip():
                all_text.append(page_text.strip())
        doc.close()

        combined = "\n\n".join(all_text)
        if len(combined.strip()) < 100:
            print(f"[PYMUPDF] {filename}: text layer too short ({len(combined)} chars) — likely image-only PDF")
            return None

        print(f"[PYMUPDF] {filename}: extracted {len(combined)} chars from text layer")
        return combined
    except Exception as e:
        print(f"[PYMUPDF] Text extraction failed for {filename}: {e}")
        return None


def render_all_pages_as_pngs(
    file_path: str, dpi: int = 200, max_pages: int = 50
) -> List[bytes]:
    """
    Render all pages (up to max_pages) of a PDF as PNG byte strings.

    Uses a moderate DPI (200) to balance OCR quality with file size,
    since we're sending potentially many pages to Gemini Vision.
    """
    try:
        import fitz
    except ImportError:
        raise ImportError("pymupdf is required for OCR text extraction")

    doc = fitz.open(file_path)
    pages_to_render = min(max_pages, len(doc))
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pngs = []
    for i in range(pages_to_render):
        pix = doc[i].get_pixmap(matrix=mat, alpha=False)
        pngs.append(pix.tobytes("png"))
    doc.close()
    print(f"[OCR RENDER] Rendered {pages_to_render} page(s) at {dpi} DPI for {Path(file_path).name}")
    return pngs


_OCR_TEXT_EXTRACTION_PROMPT = (
    "This is a scanned patent document (image-only PDF with no embedded text). "
    "Please carefully OCR every page and extract ALL text exactly as it appears. "
    "Include every section: cover page, patent number, all dates, inventors, "
    "assignee, claims, description, abstract. "
    "Return only plain text with no commentary or formatting."
)


async def extract_text_via_ocr(file_path: str, filename: str) -> Optional[str]:
    """
    OCR fallback for image-only PDFs.

    Renders all pages as PNGs and sends them to Gemini Vision in batches
    to extract the full text content. Pages are batched (up to 10 per call)
    to stay within Gemini's input limits.
    """
    loop = asyncio.get_running_loop()

    try:
        png_list = await loop.run_in_executor(
            None, render_all_pages_as_pngs, file_path
        )
    except ImportError as e:
        print(f"[OCR] {e}")
        return None
    except Exception as e:
        print(f"[OCR] Page rendering failed for {filename}: {e}")
        return None

    if not png_list:
        return None

    print(f"[OCR] Sending {len(png_list)} page(s) to Gemini Vision for {filename}...")

    # Process pages in batches to avoid hitting input size limits
    BATCH_SIZE = 10
    all_text_parts = []

    for batch_start in range(0, len(png_list), BATCH_SIZE):
        batch = png_list[batch_start:batch_start + BATCH_SIZE]
        batch_end = batch_start + len(batch)

        contents = []
        for png_bytes in batch:
            contents.append(
                types.Part.from_bytes(data=png_bytes, mime_type="image/png")
            )

        if len(png_list) > BATCH_SIZE:
            contents.append(
                f"{_OCR_TEXT_EXTRACTION_PROMPT}\n\n"
                f"These are pages {batch_start + 1}-{batch_end} of {len(png_list)}."
            )
        else:
            contents.append(_OCR_TEXT_EXTRACTION_PROMPT)

        try:
            response = await gemini_client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=65536,
                ),
            )
            batch_text = response.text if response.text else ""
            if batch_text.strip():
                all_text_parts.append(batch_text.strip())
                print(
                    f"[OCR] Batch pages {batch_start + 1}-{batch_end}: "
                    f"extracted {len(batch_text)} chars"
                )
            else:
                print(f"[OCR] Batch pages {batch_start + 1}-{batch_end}: no text returned")

        except Exception as e:
            print(f"[OCR] Batch pages {batch_start + 1}-{batch_end} failed: {e}")

    if not all_text_parts:
        print(f"[OCR] No text extracted from any page of {filename}")
        return None

    combined = "\n\n".join(all_text_parts)
    print(f"[OCR] {filename}: extracted {len(combined)} chars total via OCR")
    return combined


# ─────────────────────────────────────────────
# Date extraction
# ─────────────────────────────────────────────

def render_cover_page_as_png(file_path: str, dpi: int = COVER_PAGE_DPI) -> Optional[bytes]:
    """
    Render page 1 of a PDF as a PNG byte string using PyMuPDF.

    Raises ImportError explicitly if pymupdf is missing so the caller
    gets a clear signal (instead of silently returning None).
    """
    try:
        import fitz  # pymupdf
    except ImportError:
        raise ImportError(
            "pymupdf is required for cover-page date extraction. "
            "Install it with:  pip install pymupdf --break-system-packages"
        )

    doc       = fitz.open(file_path)
    page      = doc[0]
    mat       = fitz.Matrix(dpi / 72, dpi / 72)
    pix       = page.get_pixmap(matrix=mat, alpha=False)
    png_bytes = pix.tobytes("png")
    doc.close()
    return png_bytes


def render_cover_pages_as_pngs(
    file_path: str, dpi: int = COVER_PAGE_DPI, max_pages: int = 2
) -> List[bytes]:
    """
    Render first N pages of a PDF as PNG byte strings using PyMuPDF.

    Returns a list of PNG byte strings (one per page).
    Some patents split cover info (e.g., filing date on page 2),
    so we render multiple pages to improve extraction reliability.
    """
    try:
        import fitz  # pymupdf
    except ImportError:
        raise ImportError(
            "pymupdf is required for cover-page date extraction. "
            "Install it with:  pip install pymupdf --break-system-packages"
        )

    doc = fitz.open(file_path)
    pages_to_render = min(max_pages, len(doc))
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pngs = []
    for i in range(pages_to_render):
        pix = doc[i].get_pixmap(matrix=mat, alpha=False)
        pngs.append(pix.tobytes("png"))
    doc.close()
    return pngs


async def extract_dates_from_pdf(file_path: str, filename: str) -> Dict:
    """
    Extract filing/grant dates from a patent PDF.

    Strategy (4-tier fallback):
      1. Upload the raw PDF to Gemini and ask it to extract dates directly.
      2. If the PDF upload fails or returns no dates, render the first 2
         cover pages as high-DPI PNGs and send them to Gemini Vision.
      3. If vision fails and PyMuPDF can extract text → text-based prompt.
      4. If PDF is image-only (no text layer) → OCR-focused vision prompt
         with the rendered PNGs.
    """
    if not file_path or not Path(file_path).exists():
        print(f"[DATE EXTRACTION] No local file for {filename} — dates will be null")
        return {"filing_date": None, "grant_date": None}

    loop = asyncio.get_running_loop()

    # ── Step 1: Native PDF upload to Gemini ──────────────────────────
    print(f"[DATE EXTRACTION] Trying native PDF upload for {filename}...")
    try:
        pdf_bytes = Path(file_path).read_bytes()
        max_bytes = 2 * 1024 * 1024
        if len(pdf_bytes) > max_bytes:
            try:
                import fitz
                doc = fitz.open(file_path)
                new_doc = fitz.open()
                for i in range(min(2, len(doc))):
                    new_doc.insert_pdf(doc, from_page=i, to_page=i)
                pdf_bytes = new_doc.tobytes()
                new_doc.close()
                doc.close()
                print(f"[DATE EXTRACTION] Trimmed PDF to first 2 pages ({len(pdf_bytes)} bytes)")
            except ImportError:
                if len(pdf_bytes) > 5 * 1024 * 1024:
                    print(f"[DATE EXTRACTION] PDF too large for native upload without pymupdf, skipping")
                    pdf_bytes = None
            except Exception as e:
                print(f"[DATE EXTRACTION] PDF trim failed: {e}, using full PDF")

        if pdf_bytes:
            dates = await _call_gemini_for_dates(
                contents=[
                    types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
                    DATE_EXTRACTION_PROMPT,
                ],
                filename=filename,
            )
            if dates.get("filing_date") or dates.get("grant_date"):
                print(f"[DATE EXTRACTION] {filename} -> Filed: {dates['filing_date']} | Granted: {dates['grant_date']} (native PDF)")
                return dates
            print(f"[DATE EXTRACTION] Native PDF upload returned no dates for {filename} — trying vision fallback")
    except Exception as e:
        print(f"[DATE EXTRACTION] Native PDF upload failed for {filename}: {e} — trying vision fallback")

    # ── Step 2: Multi-page vision-based extraction ───────────────────
    print(f"[DATE EXTRACTION] Rendering cover pages of {filename} at {COVER_PAGE_DPI} DPI...")
    png_list: List[bytes] = []
    try:
        png_list = await loop.run_in_executor(
            None, render_cover_pages_as_pngs, file_path
        )
    except ImportError as e:
        print(f"[ERROR] {e}")
    except Exception as e:
        print(f"[DATE EXTRACTION] Cover page render failed for {filename}: {e}")

    if png_list:
        contents = []
        for i, png_bytes in enumerate(png_list):
            contents.append(
                types.Part.from_bytes(data=png_bytes, mime_type="image/png")
            )
        contents.append(DATE_EXTRACTION_PROMPT)

        dates = await _call_gemini_for_dates(
            contents=contents,
            filename=filename,
        )
        if dates.get("filing_date") or dates.get("grant_date"):
            print(f"[DATE EXTRACTION] {filename} -> Filed: {dates['filing_date']} | Granted: {dates['grant_date']} (vision)")
            return dates
        print(f"[DATE EXTRACTION] Vision returned no dates for {filename} — trying text/OCR fallback")

    # ── Step 3 & 4: Text extraction then OCR ─────────────────────────
    return await _extract_dates_fallback(file_path, filename, png_list)


async def _extract_dates_fallback(
    file_path: str, filename: str, png_list: Optional[List[bytes]] = None
) -> Dict:
    """
    Fallback date extraction for when the primary methods fail.
    """
    cover_text = ""
    pymupdf_available = True
    try:
        import fitz
        doc = fitz.open(file_path)
        for page_num in range(min(2, len(doc))):
            cover_text += doc[page_num].get_text("text") + "\n"
        doc.close()
    except ImportError:
        pymupdf_available = False
    except Exception as e:
        print(f"[DATE EXTRACTION] Text extraction from PDF failed for {filename}: {e}")

    if cover_text.strip():
        dates = await _call_gemini_for_dates(
            contents=[DATE_EXTRACTION_TEXT_PROMPT + cover_text[:3000]],
            filename=filename,
        )
        if dates.get("filing_date") or dates.get("grant_date"):
            print(f"[DATE EXTRACTION] {filename} -> Filed: {dates['filing_date']} | Granted: {dates['grant_date']} (text fallback)")
            return dates
    else:
        print(f"[DATE EXTRACTION] PDF is image-only (no text layer) — using OCR fallback for {filename}")

    if not png_list:
        if not pymupdf_available:
            print(f"[ERROR] pymupdf not available and no cover image — cannot extract dates for {filename}")
            return {"filing_date": None, "grant_date": None}
        try:
            loop = asyncio.get_running_loop()
            png_list = await loop.run_in_executor(
                None, render_cover_pages_as_pngs, file_path
            )
        except Exception as e:
            print(f"[ERROR] Could not render cover pages for OCR fallback: {e}")
            return {"filing_date": None, "grant_date": None}

    if not png_list:
        return {"filing_date": None, "grant_date": None}

    contents = []
    for png_bytes in png_list:
        contents.append(
            types.Part.from_bytes(data=png_bytes, mime_type="image/png")
        )
    contents.append(_OCR_DATE_PROMPT)

    dates = await _call_gemini_for_dates(
        contents=contents,
        filename=filename,
    )
    print(f"[DATE EXTRACTION] {filename} -> Filed: {dates.get('filing_date')} | Granted: {dates.get('grant_date')} (OCR fallback)")
    return dates


# ─────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────

def chunk_text(
    text:       str,
    chunk_size: int = CHUNK_SIZE_CHARS,
    overlap:    int = OVERLAP_CHARS,
) -> List[str]:
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size})")

    chunks, start = [], 0
    while start < len(text):
        end   = start + chunk_size
        chunk = text[start:end]
        if end < len(text):
            bp = max(chunk.rfind("."), chunk.rfind("\n"))
            if bp > chunk_size // 2:
                chunk = chunk[: bp + 1]
                end   = start + bp + 1
        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks


# ─────────────────────────────────────────────
# Embeddings
# ─────────────────────────────────────────────

async def generate_embeddings(
    texts: List[str],
    model: str = "gemini-embedding-001",
) -> List[List[float]]:
    print(f"[EMBEDDINGS] Generating for {len(texts)} chunks...")
    loop = asyncio.get_running_loop()

    try:
        embeddings = []
        for i in range(0, len(texts), 100):
            batch = texts[i : i + 100]

            for attempt in range(1, _MAX_EMBED_RETRIES + 1):
                try:
                    result = await loop.run_in_executor(
                        None,
                        lambda b=batch: gemini_client.models.embed_content(
                            model=model,
                            contents=b,
                            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
                        ),
                    )
                    for emb in result.embeddings:
                        embeddings.append(emb.values)
                    break

                except Exception as e:
                    if attempt == _MAX_EMBED_RETRIES:
                        raise RuntimeError(
                            f"Embedding batch {i}-{i+len(batch)} failed after "
                            f"{_MAX_EMBED_RETRIES} attempts: {e}"
                        ) from e
                    backoff = (2 ** attempt) + random.uniform(0, 1)
                    print(f"[EMBEDDINGS] Attempt {attempt} failed — retrying in {backoff:.1f}s")
                    await asyncio.sleep(backoff)

        print(f"[EMBEDDINGS] Generated {len(embeddings)} embeddings")
        return embeddings

    except Exception as e:
        print(f"[ERROR] Embedding generation failed: {e}")
        return []


# ─────────────────────────────────────────────
# AlloyDB helpers
# ─────────────────────────────────────────────

# Lock to serialize AlloyDB writes — concurrent coroutine writes
# can still cause issues with connection management.
_chroma_write_lock = asyncio.Lock()


def sanitize_collection_name(drug_name: str) -> str:
    safe = drug_filter.safe_collection_name(drug_name)
    safe = safe.ljust(3, "x")
    safe = safe[:55]
    safe = re.sub(r"[^a-zA-Z0-9]+$", "", safe)
    return f"patents_{safe}"


def get_or_create_collection(drug_name: str):
    name = sanitize_collection_name(drug_name)
    print(f"[ALLOYDB] Collection name: {name}")
    try:
        col = chroma_client.get_collection(name=name)
        print(f"[ALLOYDB] Using existing: {name}")
    except Exception:
        col = chroma_client.create_collection(
            name=name,
            metadata={"description": f"Patent embeddings for {drug_name}"},
        )
        print(f"[ALLOYDB] Created new: {name}")
    return col


def sentinel_exists(collection, filename: str) -> bool:
    try:
        sid    = hashlib.md5(filename.encode()).hexdigest() + "_complete"
        result = collection.get(ids=[sid])
        return bool(result["ids"])
    except Exception:
        return False


def get_dates_from_chromadb(collection, filename: str) -> dict:
    """
    Read filing/grant dates for a patent from AlloyDB.

    Priority:
      1. Sentinel record (chunk_index = -1)
      2. First chunk (chunk_index = 0) — fallback, also carries dates
    """
    file_hash = hashlib.md5(filename.encode()).hexdigest()

    try:
        sid    = file_hash + "_complete"
        result = collection.get(ids=[sid], include=["metadatas"])
        if result["metadatas"]:
            meta   = result["metadatas"][0]
            filing = meta.get("filing_date") or None
            grant  = meta.get("grant_date")  or None
            if filing or grant:
                return {"filing_date": filing, "grant_date": grant}
    except Exception as e:
        print(f"[DATES] Sentinel read failed for {filename}: {e}")

    try:
        result = collection.get(ids=[f"{file_hash}_chunk_0"], include=["metadatas"])
        if result["metadatas"]:
            meta   = result["metadatas"][0]
            filing = meta.get("filing_date") or None
            grant  = meta.get("grant_date")  or None
            if filing or grant:
                print(f"[DATES] Dates read from chunk_0 for {filename}")
                return {"filing_date": filing, "grant_date": grant}
    except Exception as e:
        print(f"[DATES] Chunk-0 read failed for {filename}: {e}")

    return {"filing_date": None, "grant_date": None}


# ─────────────────────────────────────────────
# Core indexing
# ─────────────────────────────────────────────

async def index_text(
    drug_name: str,
    filename:  str,
    text:      str,
    collection,
    dates:     dict = None,
) -> bool:
    """
    Chunk and index a patent into AlloyDB.

    Dates are stored in every chunk's metadata and the sentinel record.
    AlloyDB writes are serialized via _chroma_write_lock.
    """
    file_hash   = hashlib.md5(filename.encode()).hexdigest()
    sentinel_id = f"{file_hash}_complete"

    try:
        if collection.get(ids=[sentinel_id])["ids"]:
            print(f"[INDEXING] Already fully indexed: {filename}")
            return True
    except Exception:
        pass

    try:
        stale = collection.get(where={"filename": filename}, include=["ids"])
        if stale["ids"]:
            print(f"[INDEXING] Incomplete index for {filename} — clearing and re-indexing")
            async with _chroma_write_lock:
                collection.delete(where={"filename": filename})
    except Exception:
        pass

    chunks = chunk_text(text)
    print(f"[CHUNKING] {filename} -> {len(chunks)} chunks")
    if not chunks:
        return False

    embeddings = await generate_embeddings(chunks)
    if not embeddings:
        return False

    dates       = dates or {}
    filing_date = _clean_date(dates.get("filing_date")) or ""
    grant_date  = _clean_date(dates.get("grant_date"))  or ""

    async with _chroma_write_lock:
        collection.add(
            documents  = chunks,
            embeddings = embeddings,
            metadatas  = [
                {
                    "filename":     filename,
                    "drug":         drug_name,
                    "chunk_index":  i,
                    "total_chunks": len(chunks),
                    "filing_date":  filing_date,
                    "grant_date":   grant_date,
                }
                for i in range(len(chunks))
            ],
            ids=[f"{file_hash}_chunk_{i}" for i in range(len(chunks))],
        )

        collection.add(
            documents  = ["__index_complete__"],
            embeddings = [[0.0] * len(embeddings[0])],
            metadatas  = [{
                "filename":     filename,
                "drug":         drug_name,
                "chunk_index":  -1,
                "total_chunks": len(chunks),
                "filing_date":  filing_date,
                "grant_date":   grant_date,
            }],
            ids=[sentinel_id],
        )

    print(
        f"[INDEXING] {filename} — {len(chunks)} chunks stored | "
        f"Filed: {filing_date or 'unknown'} | Granted: {grant_date or 'unknown'}"
    )
    return True


# ─────────────────────────────────────────────
# Cross-collection deduplication
# ─────────────────────────────────────────────

def find_in_any_collection(filename: str) -> Optional[str]:
    sid = hashlib.md5(filename.encode()).hexdigest() + "_complete"
    for col in chroma_client.list_collections():
        if not col.name.startswith("patents_"):
            continue
        try:
            existing = chroma_client.get_collection(col.name)
            if existing.get(ids=[sid])["ids"]:
                print(f"[CROSS-CHECK] '{filename}' found in '{col.name}'")
                return col.name
        except Exception:
            continue
    return None


async def copy_from_collection(
    filename:         str,
    source_name:      str,
    target_collection,
    target_drug:      str,
) -> bool:
    """
    Copies all chunks + sentinel from source to target collection.
    Dates are embedded in chunk metadata so they transfer automatically.
    """
    try:
        source    = chroma_client.get_collection(source_name)
        file_hash = hashlib.md5(filename.encode()).hexdigest()

        chunks = source.get(
            where={
                "$and": [
                    {"filename":    {"$eq": filename}},
                    {"chunk_index": {"$gte": 0}},
                ]
            },
            include=["documents", "metadatas", "embeddings"],
        )
        if chunks["ids"]:
            async with _chroma_write_lock:
                target_collection.add(
                    documents  = chunks["documents"],
                    embeddings = chunks["embeddings"],
                    metadatas  = [{**m, "drug": target_drug} for m in chunks["metadatas"]],
                    ids        = chunks["ids"],
                )
            print(f"[COPY] {len(chunks['ids'])} chunks -> '{target_collection.name}'")

        sentinel_id = f"{file_hash}_complete"
        sentinel    = source.get(
            ids=[sentinel_id], include=["documents", "metadatas", "embeddings"]
        )
        if sentinel["ids"]:
            async with _chroma_write_lock:
                target_collection.add(
                    documents  = sentinel["documents"],
                    embeddings = sentinel["embeddings"],
                    metadatas  = [{**sentinel["metadatas"][0], "drug": target_drug}],
                    ids        = [sentinel_id],
                )
            print(f"[COPY] Sentinel copied for '{filename}'")

        return True

    except Exception as e:
        print(f"[COPY] Failed to copy '{filename}' from '{source_name}': {e}")
        return False


# ─────────────────────────────────────────────
# Fix dates for already-indexed patents
# ─────────────────────────────────────────────

async def fix_dates_for_file(
    filename:   str,
    file_path:  str,
    collection,
) -> bool:
    """
    Re-extract dates for a single patent and update all its records
    in AlloyDB in-place (no re-embedding needed).
    """
    file_hash   = hashlib.md5(filename.encode()).hexdigest()
    sentinel_id = f"{file_hash}_complete"

    try:
        result = collection.get(ids=[sentinel_id], include=["metadatas"])
        if result["metadatas"] and _has_valid_dates(result["metadatas"][0]):
            meta = result["metadatas"][0]
            print(
                f"[FIX-DATES] {filename} already has dates: "
                f"Filed={meta.get('filing_date')} | Granted={meta.get('grant_date')}"
            )
            return True
    except Exception:
        pass

    dates  = await extract_dates_from_pdf(file_path, filename)
    filing = _clean_date(dates.get("filing_date")) or ""
    grant  = _clean_date(dates.get("grant_date"))  or ""

    if not filing and not grant:
        print(f"[FIX-DATES] Still could not extract dates for {filename}")
        return False

    try:
        all_records = collection.get(
            where={"filename": filename},
            include=["metadatas"],
        )
        if all_records["ids"]:
            updated_metas = []
            for m in all_records["metadatas"]:
                m["filing_date"] = filing
                m["grant_date"]  = grant
                updated_metas.append(m)

            async with _chroma_write_lock:
                collection.update(
                    ids       = all_records["ids"],
                    metadatas = updated_metas,
                )
            print(
                f"[FIX-DATES] {filename}: {len(all_records['ids'])} records updated — "
                f"Filed={filing} | Granted={grant}"
            )
            return True
    except Exception as e:
        print(f"[FIX-DATES] Update failed for {filename}: {e}")

    return False


# ─────────────────────────────────────────────
# Single-patent processing task (used by parallel runner)
# ─────────────────────────────────────────────

async def _process_single_patent(
    ref:        dict,
    drug_name:  str,
    collection,
    reindex:    bool,
    semaphore:  asyncio.Semaphore,
    completed:  set = None,
) -> dict:
    """
    Process one patent end-to-end, bounded by *semaphore*.

    Returns {"filename": str, "path": str | None, "tmp_dir": str | None}.
    """
    filename = ref["filename"]

    if completed is not None and filename in completed:
        print(f"[RESUME] {filename} — already completed in previous run, skipping")
        return {"filename": filename, "path": None, "tmp_dir": None}

    async with semaphore:
        if not reindex and sentinel_exists(collection, filename):
            existing_dates = get_dates_from_chromadb(collection, filename)
            filing = existing_dates.get("filing_date")
            if filing and filing not in ("", "null", "None"):
                print(f"[SKIP] {filename} — already indexed (Filed: {filing})")
                if completed is not None:
                    _mark_completed(drug_name, filename, completed)
                return {"filename": filename, "path": None, "tmp_dir": None}

            print(f"[FIX-DATES] {filename} — indexed but missing filing date, re-extracting...")
            loop = asyncio.get_running_loop()
            pf = await loop.run_in_executor(
                None,
                download_single_patent_pdf, ref["blob_name"], filename, drug_name,
            )
            if pf:
                try:
                    fixed = await fix_dates_for_file(filename, pf["path"], collection)
                    if fixed:
                        print(f"[FIX-DATES] {filename} — dates updated successfully")
                    else:
                        print(f"[FIX-DATES] {filename} — could not extract dates (will retry on next run)")
                except Exception as e:
                    print(f"[FIX-DATES] {filename} — error: {e}")
                finally:
                    if pf.get("tmp_dir"):
                        shutil.rmtree(pf["tmp_dir"], ignore_errors=True)
            else:
                print(f"[FIX-DATES] {filename} — could not download for date re-extraction")

            if completed is not None:
                _mark_completed(drug_name, filename, completed)
            return {"filename": filename, "path": None, "tmp_dir": None}

        if not reindex:
            source_col = find_in_any_collection(filename)
            if source_col:
                print(f"[COPY] {filename} — found in '{source_col}', copying (dates included)...")
                await copy_from_collection(filename, source_col, collection, drug_name)
                if completed is not None:
                    _mark_completed(drug_name, filename, completed)
                return {"filename": filename, "path": None, "tmp_dir": None}

        print(f"[INDEX] {filename} — downloading...")
        loop = asyncio.get_running_loop()
        pf = await loop.run_in_executor(
            None,
            download_single_patent_pdf, ref["blob_name"], filename, drug_name,
        )
        if not pf:
            print(f"[WARNING] Could not download {filename}")
            return {"filename": filename, "path": None, "tmp_dir": None}

        try:
            uploaded_file = await upload_pdf_to_gemini(pf["path"])
            if not uploaded_file:
                print(f"[WARNING] Upload failed for {filename}")
                return pf

            text, dates = await asyncio.gather(
                extract_text_via_gemini(uploaded_file, filename),
                extract_dates_from_pdf(pf["path"], filename),
            )

            # ── Fallback chain for image-only / scanned PDFs ─────────────
            if not text:
                print(f"[FALLBACK] Gemini text extraction returned nothing for {filename} — trying PyMuPDF text layer...")
                text = extract_text_via_pymupdf(pf["path"], filename)

            if not text:
                print(f"[FALLBACK] No text layer found for {filename} — trying Gemini Vision OCR on rendered pages...")
                text = await extract_text_via_ocr(pf["path"], filename)

            if text:
                await index_text(drug_name, filename, text, collection, dates=dates)
                if completed is not None:
                    _mark_completed(drug_name, filename, completed)
            else:
                print(f"[WARNING] No text extracted from {filename} — all methods failed (Gemini / PyMuPDF / OCR)")

            await cleanup_uploaded_file(uploaded_file)
            await asyncio.sleep(0.5 + random.uniform(0, 0.5))

        except Exception as e:
            print(f"[ERROR] Processing failed for {filename}: {e}")

        finally:
            if pf.get("tmp_dir"):
                shutil.rmtree(pf["tmp_dir"], ignore_errors=True)
                print(f"[GCS] Cleaned up temp dir for {filename}")
                pf["tmp_dir"] = None

        return pf


# ─────────────────────────────────────────────
# Main public function
# ─────────────────────────────────────────────

async def run_indexing(
    drug_name:  str,
    pdf_refs:   List[dict],
    collection,
    reindex:    bool = False,
    max_concurrency: int = MAX_CONCURRENCY,
) -> List[dict]:
    """
    Index multiple patents **in parallel** (bounded by *max_concurrency*).

    Default concurrency is 10 — meaning 10 patents per drug are processed
    simultaneously (download → upload → text+dates → embed → store).
    Override via the INDEXER_CONCURRENCY env var or by passing max_concurrency
    directly.

    Crash-resilient: progress is tracked in a JSON file on disk.
    If the process crashes mid-batch, restarting will skip already-completed patents.

    For each PDF ref:
      1. Skip if already completed in a previous (crashed) run
      2. Skip if already indexed AND has valid dates
      3. If indexed but missing filing date → re-download, re-extract, update in-place
      4. Copy from another collection if found — dates transfer automatically
      5. Download → upload → extract text + dates in parallel → index

    AlloyDB writes are serialized internally via _chroma_write_lock.

    Args:
        drug_name:       Drug name string
        pdf_refs:        List of {"filename": str, "blob_name": str} from gcs_lister
        collection:      AlloyDB collection object
        reindex:         If True, force re-indexing even if sentinel exists
        max_concurrency: Max patents processed in parallel (default: 10)

    Returns:
        List of {"filename": str, "path": str | None, "tmp_dir": str | None}
    """
    completed    = _load_progress(drug_name) if not reindex else set()
    already_done = sum(1 for r in pdf_refs if r["filename"] in completed)
    remaining    = len(pdf_refs) - already_done

    print(
        f"\n[INDEXER] '{drug_name}': {len(pdf_refs)} patent(s), "
        f"{max_concurrency} running in parallel..."
    )
    if already_done > 0:
        print(
            f"[RESUME] Skipping {already_done} already-completed patents, "
            f"{remaining} remaining"
        )

    semaphore = asyncio.Semaphore(max_concurrency)

    tasks = [
        _process_single_patent(
            ref, drug_name, collection, reindex, semaphore,
            completed=completed,
        )
        for ref in pdf_refs
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    downloaded_files: List[dict] = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            filename = pdf_refs[i]["filename"]
            print(f"[ERROR] Task for {filename} raised: {result}")
            downloaded_files.append({"filename": filename, "path": None, "tmp_dir": None})
        else:
            downloaded_files.append(result)

    all_filenames = {r["filename"] for r in pdf_refs}
    if all_filenames.issubset(completed):
        _clear_progress(drug_name)

    return downloaded_files
