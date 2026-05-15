"""
api.py — FastAPI REST API for the image-detector / crawler.

Run with:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Interactive docs:
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime
from enum import Enum
from typing import List, Optional

import requests
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from classifier import classify_image
from constants import DEFAULT_TABLE_SCORE_THRESHOLD, ImageResult
from database import (
    clear_classification_cache,
    delete_run,
    get_cache_stats,
    get_run_summary,
    list_run_ids,
    load_classification_cache,
    load_results_db,
    save_classification_cache,
    update_run_result_label,
    write_results_db,
)
from orchestrator import crawl_site, format_results_csv_bytes

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Image Detector API",
    description=(
        "Scan web pages, discover images, and classify each as **table** or "
        "**normal**. Backed by the same engine that powers the Streamlit webapp."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class CrawlMode(str, Enum):
    page = "page"
    site = "site"
    urls = "urls"
    paginated_listing = "paginated_listing"


class ScanRequest(BaseModel):
    """Request body for POST /scan."""

    url: str = Field(..., description="Target URL to scan.", examples=["https://example.com/page"])
    crawl_mode: CrawlMode = Field(
        CrawlMode.page,
        description="Scan scope: single page, whole site, specified URLs, or paginated listing.",
    )
    target_urls: Optional[List[str]] = Field(
        None,
        description="List of specific URLs to scan (only used when crawl_mode='urls').",
    )
    listing_urls: Optional[List[str]] = Field(
        None,
        description="Additional listing base URLs (only used when crawl_mode='paginated_listing').",
    )
    max_pages: int = Field(40, ge=1, description="Maximum pages to scan (site / listing modes).")
    run_id: Optional[str] = Field(
        None,
        description="Custom run ID. Auto-generated if omitted.",
    )
    fast_mode: bool = Field(False, description="Enable fast OCR mode.")
    turbo_mode: bool = Field(True, description="Enable turbo mode (faster, slightly lower quality).")
    table_confidence: int = Field(
        100,
        ge=50,
        le=100,
        description="Table confidence threshold (50-100). Higher = stricter.",
    )
    flag_uncertain: bool = Field(False, description="Flag images near the threshold as uncertain.")
    render_js: bool = Field(True, description="Render pages with JavaScript.")
    heartbeat_seconds: float = Field(10.0, ge=1.0, description="Heartbeat interval in seconds.")
    ocr_workers: int = Field(
        0,
        ge=0,
        description="OCR worker threads. 0 = auto (based on CPU count).",
    )
    disable_cache: bool = Field(False, description="Skip the classification cache entirely.")


class ImageResultResponse(BaseModel):
    """Single classified image result."""

    page_url: str
    image_url: str
    label: str
    score: float
    reason: str
    image_hash: str = ""


class ScanResponse(BaseModel):
    """Response from POST /scan."""

    run_id: str
    total: int
    tables: int
    normal: int
    uncertain: int
    results: List[ImageResultResponse]


class ClassifyResponse(BaseModel):
    """Response from POST /classify."""

    image_url: str
    label: str
    score: float
    reason: str
    image_hash: str


class RunSummary(BaseModel):
    run_id: str
    total: int = 0
    tables: int = 0
    normal: int = 0
    uncertain: int = 0
    created_at: str = ""


class RunDetailResponse(BaseModel):
    run_id: str
    results: List[dict]


class CacheStatsResponse(BaseModel):
    total_entries: int = 0
    tables: int = 0
    normal: int = 0
    uncertain: int = 0


class MarkNormalRequest(BaseModel):
    """Request body for PATCH /runs/{run_id}/mark-normal."""

    image_url: str = Field(..., description="URL of the image to mark as normal.")
    image_hash: str = Field(..., description="SHA-1 hash of the image bytes.")
    fast_mode: bool = False
    turbo_mode: bool = True
    table_confidence: int = Field(100, ge=50, le=100)
    flag_uncertain: bool = False
    disable_cache: bool = False


class MessageResponse(BaseModel):
    message: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_workers(requested: int) -> int:
    if requested > 0:
        return requested
    return max(1, min(4, (os.cpu_count() or 4)))


def _image_result_to_response(r: ImageResult) -> ImageResultResponse:
    return ImageResultResponse(
        page_url=r.page_url,
        image_url=r.image_url,
        label=r.label,
        score=round(r.score, 4),
        reason=r.reason,
        image_hash=r.image_hash,
    )


# ---------------------------------------------------------------------------
# POST /scan — run a full crawl + classify
# ---------------------------------------------------------------------------

@app.post(
    "/scan",
    response_model=ScanResponse,
    summary="Scan a URL",
    description="Crawl the target URL (single page, whole site, specific URLs, or paginated listing), "
    "discover images, classify each, and return results.",
    tags=["Scanning"],
)
def scan(body: ScanRequest):
    run_id = body.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    threshold_multiplier = float(body.table_confidence) / 100.0
    configured_threshold = DEFAULT_TABLE_SCORE_THRESHOLD * threshold_multiplier
    workers = _resolve_workers(body.ocr_workers)

    try:
        rows: List[ImageResult] = crawl_site(
            start_url=body.url.strip(),
            run_id=run_id,
            render_js=body.render_js,
            heartbeat_seconds=body.heartbeat_seconds,
            fast_mode=body.fast_mode,
            turbo_mode=body.turbo_mode,
            table_score_threshold=configured_threshold,
            flag_uncertain=body.flag_uncertain,
            ocr_workers=workers,
            crawl_mode=body.crawl_mode.value,
            max_pages=body.max_pages,
            target_urls=body.target_urls if body.crawl_mode == CrawlMode.urls else None,
            listing_urls=body.listing_urls if body.crawl_mode == CrawlMode.paginated_listing else None,
            disable_cache=body.disable_cache,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Scan failed: {exc}")

    table_count = sum(1 for r in rows if r.label == "table")
    uncertain_count = sum(1 for r in rows if r.label == "uncertain")
    normal_count = len(rows) - table_count - uncertain_count

    return ScanResponse(
        run_id=run_id,
        total=len(rows),
        tables=table_count,
        normal=normal_count,
        uncertain=uncertain_count,
        results=[_image_result_to_response(r) for r in rows],
    )


# ---------------------------------------------------------------------------
# POST /classify — classify a single image
# ---------------------------------------------------------------------------

@app.post(
    "/classify",
    response_model=ClassifyResponse,
    summary="Classify a single image",
    description="Classify a single image (by URL or file upload) as table or normal.",
    tags=["Classification"],
)
async def classify(
    image_url: Optional[str] = Form(None, description="URL of the image to classify."),
    file: Optional[UploadFile] = File(None, description="Image file to upload and classify."),
    fast_mode: bool = Form(False),
    turbo_mode: bool = Form(True),
    table_confidence: int = Form(100, ge=50, le=100),
    flag_uncertain: bool = Form(False),
):
    if not image_url and file is None:
        raise HTTPException(
            status_code=422, detail="Provide either 'image_url' or upload a 'file'."
        )

    threshold_multiplier = float(table_confidence) / 100.0
    configured_threshold = DEFAULT_TABLE_SCORE_THRESHOLD * threshold_multiplier

    if file is not None:
        image_bytes = await file.read()
        resolved_url = f"upload://{file.filename or 'uploaded_image'}"
    else:
        try:
            resp = requests.get(image_url, timeout=(3.0, 10.0))
            resp.raise_for_status()
            image_bytes = resp.content or b""
        except requests.RequestException as exc:
            raise HTTPException(status_code=400, detail=f"Failed to fetch image URL: {exc}")
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Image URL returned an empty response.")
        resolved_url = image_url

    label, score, reason = classify_image(
        image_bytes=image_bytes,
        image_url=resolved_url,
        fast_mode=fast_mode,
        turbo_mode=turbo_mode,
        table_score_threshold=configured_threshold,
        flag_uncertain=flag_uncertain,
    )

    return ClassifyResponse(
        image_url=resolved_url,
        label=label,
        score=round(score, 4),
        reason=reason,
        image_hash=hashlib.sha1(image_bytes).hexdigest(),
    )


# ---------------------------------------------------------------------------
# GET /runs — list all runs
# ---------------------------------------------------------------------------

@app.get(
    "/runs",
    response_model=List[RunSummary],
    summary="List all runs",
    description="Returns every run ID with summary counts, newest first.",
    tags=["Runs"],
)
def list_runs():
    run_ids = list_run_ids()
    summaries = []
    for rid in run_ids:
        s = get_run_summary(rid)
        summaries.append(
            RunSummary(
                run_id=rid,
                total=s.get("total", 0),
                tables=s.get("tables", 0),
                normal=s.get("normal", 0),
                uncertain=s.get("uncertain", 0),
                created_at=s.get("created_at", ""),
            )
        )
    return summaries


# ---------------------------------------------------------------------------
# GET /runs/{run_id} — get full results for a run
# ---------------------------------------------------------------------------

@app.get(
    "/runs/{run_id}",
    response_model=RunDetailResponse,
    summary="Get run results",
    description="Returns all classified image results for the given run.",
    tags=["Runs"],
)
def get_run(run_id: str):
    results = load_results_db(run_id)
    if not results:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found or has no results.")
    return RunDetailResponse(run_id=run_id, results=results)


# ---------------------------------------------------------------------------
# DELETE /runs/{run_id} — delete a run
# ---------------------------------------------------------------------------

@app.delete(
    "/runs/{run_id}",
    response_model=MessageResponse,
    summary="Delete a run",
    description="Permanently deletes all results for the given run.",
    tags=["Runs"],
)
def remove_run(run_id: str):
    ok = delete_run(run_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found or delete failed.")
    return MessageResponse(message=f"Run '{run_id}' deleted.")


# ---------------------------------------------------------------------------
# PATCH /runs/{run_id}/mark-normal — correct a false positive
# ---------------------------------------------------------------------------

@app.patch(
    "/runs/{run_id}/mark-normal",
    response_model=MessageResponse,
    summary="Mark image as normal",
    description="Overrides a table classification to normal. Updates both the run results and the "
    "classification cache so future scans won't flag the image again.",
    tags=["Runs"],
)
def mark_normal(run_id: str, body: MarkNormalRequest):
    threshold_multiplier = float(body.table_confidence) / 100.0
    configured_threshold = DEFAULT_TABLE_SCORE_THRESHOLD * threshold_multiplier

    # Update classification cache
    if not body.disable_cache:
        cache = load_classification_cache(
            body.fast_mode, body.turbo_mode, configured_threshold, body.flag_uncertain
        )
        cache[body.image_hash] = ("normal", 0.0, "manually_marked_normal")
        save_classification_cache(
            cache, body.fast_mode, body.turbo_mode, configured_threshold, body.flag_uncertain
        )

    # Update run results row
    update_run_result_label(run_id, body.image_url, "normal", 0.0, "manually_marked_normal")

    return MessageResponse(
        message=f"Image marked as normal in run '{run_id}' and classification cache updated."
    )


# ---------------------------------------------------------------------------
# GET /cache/stats — cache statistics
# ---------------------------------------------------------------------------

@app.get(
    "/cache/stats",
    response_model=CacheStatsResponse,
    summary="Cache statistics",
    description="Returns counts of entries in the classification cache.",
    tags=["Cache"],
)
def cache_stats():
    stats = get_cache_stats()
    return CacheStatsResponse(**stats)


# ---------------------------------------------------------------------------
# DELETE /cache — clear the classification cache
# ---------------------------------------------------------------------------

@app.delete(
    "/cache",
    response_model=MessageResponse,
    summary="Clear classification cache",
    description="Deletes every entry from the classification cache.",
    tags=["Cache"],
)
def clear_cache():
    deleted = clear_classification_cache()
    return MessageResponse(message=f"Classification cache cleared ({deleted} entries deleted).")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=MessageResponse,
    summary="Health check",
    tags=["System"],
)
def health():
    return MessageResponse(message="ok")


# ---------------------------------------------------------------------------
# Run directly: python api.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
