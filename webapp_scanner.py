"""Scanner tab – extracted from webapp.py."""

import hashlib
import threading
import time
from typing import Dict, List

import requests
import streamlit as st

from constants import DEFAULT_TABLE_SCORE_THRESHOLD, ImageResult
from classifier import classify_image
from database import (
    load_classification_cache,
    save_classification_cache,
    update_run_result_label,
    write_results_db,
)
from orchestrator import (
    crawl_site,
    format_results_csv_bytes,
    write_results_csv,
)
from webapp_report import render_metrics, build_interactive_html_report


# ── "Mark as Normal" helper ──────────────────────────────────────────────────

def _apply_mark_as_normal(image_hash: str, image_url: str) -> None:
    """Update the classification cache, run_results row, and in-memory rows; optional CSV."""
    if not image_hash:
        st.error("Cannot mark as normal: image hash is unavailable for this result.")
        return

    run_cfg = st.session_state.get("_scan_run_config", {})
    fast = run_cfg.get("fast_mode", False)
    turbo = run_cfg.get("turbo_mode", False)
    threshold = run_cfg.get("table_score_threshold", DEFAULT_TABLE_SCORE_THRESHOLD)
    flag_unc = run_cfg.get("flag_uncertain", False)

    # ── Update the SQLite classification cache ────────────────────────────────
    if not run_cfg.get("disable_cache", False):
        cache = load_classification_cache(fast, turbo, threshold, flag_unc)
        cache[image_hash] = ("normal", 0.0, "manually_marked_normal")
        save_classification_cache(cache, fast, turbo, threshold, flag_unc)

    rid = st.session_state.get("_scan_run_id", "")
    if rid:
        update_run_result_label(rid, image_url, "normal", 0.0, "manually_marked_normal")

    # ── Update in-memory rows so the rest of the UI stays consistent ──────────
    current_rows: List[ImageResult] = st.session_state.get("_scan_results") or []
    updated_rows = [
        ImageResult(
            r.page_url, r.image_url, "normal", 0.0, "manually_marked_normal", r.image_hash
        )
        if r.image_url == image_url
        else r
        for r in current_rows
    ]
    st.session_state["_scan_results"] = updated_rows

    # ── Rewrite the results CSV so downloads reflect the correction ───────────
    out_path = st.session_state.get("_scan_output_path", "")
    if out_path:
        try:
            write_results_csv(out_path, updated_rows)
        except OSError:
            pass  # Non-fatal — cache was already updated.

    st.session_state["_manually_marked_normal"].add(image_url)


# ── Scanner tab rendering ────────────────────────────────────────────────────

def render_scanner_tab(
    *,
    run_id: str,
    eval_mode: str,
    url: str,
    crawl_mode: str,
    single_image_source: str,
    single_image_url: str,
    uploaded_images,
    single_image_page_url: str,
    parsed_target_urls: List[str],
    parsed_listing_urls: List[str],
    max_pages: int,
    fast_mode: bool,
    turbo_mode: bool,
    table_confidence: int,
    flag_uncertain: bool,
    heartbeat_seconds: float,
    ocr_workers: int,
    disable_cache: bool = False,
) -> None:
    """Render the full Scanner tab content."""



    progress_bar = st.progress(0)
    status_box = st.empty()
    stop_btn_box = st.empty()
    metrics_box = st.empty()
    split_box = st.empty()
    table_results_box = st.empty()
    uncertain_results_box = st.empty()
    normal_results_box = st.empty()
    table_gallery_box = st.empty()

    # ── stop-crawl session state ──────────────────────────────────────────────────
    if "_stop_event" not in st.session_state:
        st.session_state["_stop_event"] = None
    if "_crawl_running" not in st.session_state:
        st.session_state["_crawl_running"] = False

    # ── gallery / correction session state ───────────────────────────────────────
    if "_scan_results" not in st.session_state:
        st.session_state["_scan_results"] = None
    if "_scan_output_path" not in st.session_state:
        st.session_state["_scan_output_path"] = ""
    if "_scan_run_id" not in st.session_state:
        st.session_state["_scan_run_id"] = ""
    if "_scan_run_config" not in st.session_state:
        st.session_state["_scan_run_config"] = {}
    if "_manually_marked_normal" not in st.session_state:
        st.session_state["_manually_marked_normal"] = set()

    # ── Scan / evaluate button ────────────────────────────────────────────────────
    _scan_button_label = "🔍 Evaluate Image" if eval_mode == "single_image" else "🔍 Start Scan"
    _scan_clicked = st.button(
        _scan_button_label,
        type="primary",
        use_container_width=True,
        disabled=bool(st.session_state.get("_crawl_running")),
    )

    if _scan_clicked:
        stop_event = threading.Event()
        st.session_state["_stop_event"] = stop_event
        st.session_state["_crawl_running"] = True
        st.session_state["_manually_marked_normal"] = set()

        crawl_wall_start = time.monotonic()

        totals: Dict[str, float] = {
            "processed": 0,
            "total": 0,
            "tables": 0,
            "uncertain": 0,
            "normal": 0,
            "cache_hits": 0,
            "elapsed_seconds": 0.0,
            "crawl_elapsed_seconds": 0.0,
            "classified": 0,
            "classified_total": 0,
            "classify_wall_start": crawl_wall_start,
            "page_index": 0,
            "page_total": 0,
            "raw_candidates": 0,
            "unique_candidates": 0,
            "fetch_failures": 0,
            "skipped_due_to_stall": 0,
        }
        live_rows: List[dict] = []
        status_box.info("Starting scan...")

        def _on_progress(event: dict) -> None:
            # Honour stop signal.
            if stop_event.is_set():
                raise StopIteration("stop_requested")

            event_type = event.get("event", "")
            now_mono = time.monotonic()
            totals["crawl_elapsed_seconds"] = now_mono - crawl_wall_start

            if event_type == "status":
                page_url = str(event.get("page_url", "") or "")
                page_index = int(event.get("page_index", totals["page_index"]))
                page_total = int(event.get("page_total", totals["page_total"]))
                totals["page_index"] = page_index
                totals["page_total"] = page_total
                if page_url and page_total > 0:
                    status_box.info(f"Crawling page {page_index}/{page_total}: {page_url}")
                elif page_url:
                    status_box.info(f"Crawling page: {page_url}")
                else:
                    status_box.info(event.get("message", "Running..."))

            elif event_type == "discovered":
                totals["classify_wall_start"] = now_mono
                totals["raw_candidates"] = int(
                    event.get("total_candidates", totals["raw_candidates"])
                )
                totals["unique_candidates"] = int(
                    event.get("unique_candidates", totals["unique_candidates"])
                )
                totals["total"] = int(
                    event.get("unique_candidates", event.get("total_candidates", totals["total"]))
                )
                totals["classified_total"] = totals["total"]
                status_box.info(event.get("message", "Discovered image candidates."))
                split_box.caption(
                    f"Candidates: raw {int(totals['raw_candidates'])} -> "
                    f"unique {int(totals['unique_candidates'])} | "
                    f"processed {int(totals['processed'])} | "
                    f"fetch failures {int(totals['fetch_failures'])} | "
                    f"skipped stalled {int(totals['skipped_due_to_stall'])}"
                )

            elif event_type == "heartbeat":
                totals["elapsed_seconds"] = float(event.get("elapsed_seconds", 0.0))
                totals["processed"] = int(event.get("processed", totals["processed"]))
                totals["cache_hits"] = int(event.get("cache_hits", totals["cache_hits"]))
                status_box.info(
                    f"Running... {int(totals['processed'])}/{int(max(1, totals['total']))} processed | "
                    f"cache hits: {int(totals['cache_hits'])}"
                )

            elif event_type == "image_processed":
                totals["processed"] = int(event.get("processed", totals["processed"]))
                totals["total"] = int(event.get("total", totals["total"]))
                totals["classified"] = totals["processed"]
                totals["classified_total"] = totals["total"]
                totals["elapsed_seconds"] = now_mono - totals["classify_wall_start"]
                totals["cache_hits"] = totals["cache_hits"] + (1 if event.get("cached") else 0)
                label = event.get("label", "normal")
                reason = str(event.get("reason", "") or "")
                if reason.startswith("fetch_"):
                    totals["fetch_failures"] += 1
                if label == "table":
                    totals["tables"] += 1
                elif label == "uncertain":
                    totals["uncertain"] += 1
                else:
                    totals["normal"] += 1
                live_rows.append(
                    {
                        "image_url": event.get("image_url", ""),
                        "label": label,
                        "score": round(float(event.get("score", 0.0)), 4),
                        "reason": event.get("reason", ""),
                        "cached": bool(event.get("cached", False)),
                    }
                )
                total_for_progress = int(max(1, totals["total"]))
                progress_value = min(1.0, float(totals["processed"]) / float(total_for_progress))
                progress_bar.progress(progress_value)
                with metrics_box.container():
                    render_metrics(
                        int(totals["processed"]),
                        int(totals["total"]),
                        int(totals["tables"]),
                        int(totals["normal"]),
                        int(totals["cache_hits"]),
                        float(totals["elapsed_seconds"]),
                        crawl_elapsed_s=float(totals["crawl_elapsed_seconds"]),
                        classified=int(totals["classified"]),
                        classified_total=int(totals["classified_total"]),
                    )
                split_box.caption(
                    f"Candidates: raw {int(totals['raw_candidates'])} -> "
                    f"unique {int(totals['total'])} | "
                    f"processed {int(totals['processed'])} | "
                    f"fetch failures {int(totals['fetch_failures'])} | "
                    f"skipped stalled {int(totals['skipped_due_to_stall'])}"
                )

            elif event_type == "error":
                status_box.error(event.get("message", "An error occurred while scanning."))

            elif event_type == "finished":
                totals["elapsed_seconds"] = float(
                    event.get("elapsed_seconds", totals["elapsed_seconds"])
                )
                totals["raw_candidates"] = int(
                    event.get("raw_candidates", totals["raw_candidates"])
                )
                totals["unique_candidates"] = int(
                    event.get("unique_candidates", totals["unique_candidates"])
                )
                totals["processed"] = int(event.get("processed", totals["processed"]))
                totals["skipped_due_to_stall"] = int(
                    event.get("skipped_due_to_stall", totals["skipped_due_to_stall"])
                )
                status_box.success(event.get("message", "Scan finished."))

        threshold_multiplier = float(table_confidence) / 100.0
        configured_threshold = DEFAULT_TABLE_SCORE_THRESHOLD * threshold_multiplier

        if eval_mode == "single_image":
            # ── Build a list of (image_bytes, resolved_url) pairs ─────────────
            image_items: List[tuple] = []
            if single_image_source == "url":
                single_url = single_image_url.strip()
                try:
                    response = requests.get(single_url, timeout=(3.0, 10.0))
                    response.raise_for_status()
                    image_bytes = response.content or b""
                except requests.RequestException as exc:
                    st.error(f"Failed to fetch image URL: {exc}")
                    st.stop()
                if not image_bytes:
                    st.error("Image URL returned an empty response body.")
                    st.stop()
                image_items.append((image_bytes, single_url))
            else:
                if not uploaded_images:
                    st.error("No images uploaded.")
                    st.stop()
                for uploaded_file in uploaded_images:
                    file_bytes = uploaded_file.getvalue()
                    if not file_bytes:
                        continue
                    image_items.append((file_bytes, f"upload://{uploaded_file.name}"))
                if not image_items:
                    st.error("All uploaded images are empty.")
                    st.stop()

            num_images = len(image_items)
            status_box.info(f"Evaluating {num_images} image{'s' if num_images != 1 else ''}...")
            rows: List[ImageResult] = []
            source_page = single_image_page_url.strip()

            for idx, (image_bytes, resolved_image_url) in enumerate(image_items, 1):
                label, score, reason = classify_image(
                    image_bytes=image_bytes,
                    image_url=resolved_image_url,
                    fast_mode=fast_mode,
                    turbo_mode=turbo_mode,
                    table_score_threshold=configured_threshold,
                    flag_uncertain=flag_uncertain,
                )
                image_hash = hashlib.sha1(image_bytes).hexdigest()
                rows.append(
                    ImageResult(
                        page_url=source_page or resolved_image_url,
                        image_url=resolved_image_url,
                        label=label,
                        score=score,
                        reason=reason,
                        image_hash=image_hash,
                    )
                )
                if label == "table":
                    totals["tables"] += 1
                elif label == "uncertain":
                    totals["uncertain"] += 1
                else:
                    totals["normal"] += 1
                totals["processed"] = idx
                totals["total"] = num_images
                progress_bar.progress(float(idx) / float(num_images))

            totals["elapsed_seconds"] = 0.0
            status_box.success(
                f"Evaluation finished — {num_images} image{'s' if num_images != 1 else ''} processed."
            )
            write_results_db(run_id, rows)

        else:
            with stop_btn_box.container():
                st.info("Crawl/classify is running. Use the ⏹ Stop Crawl button to abort.")
                if st.button("⏹ Stop Crawl", type="secondary"):
                    if st.session_state.get("_stop_event") is not None:
                        st.session_state["_stop_event"].set()
            try:
                rows = crawl_site(
                    start_url=(
                        url.strip()
                        if url.strip()
                        else (parsed_target_urls[0] if parsed_target_urls else "")
                    ),
                    run_id=run_id,
                    render_js=True,
                    heartbeat_seconds=float(heartbeat_seconds),
                    fast_mode=fast_mode,
                    turbo_mode=turbo_mode,
                    table_score_threshold=configured_threshold,
                    flag_uncertain=flag_uncertain,
                    ocr_workers=int(ocr_workers),
                    crawl_mode=crawl_mode,
                    max_pages=int(max_pages),
                    progress_callback=_on_progress,
                    target_urls=parsed_target_urls if crawl_mode == "urls" else None,
                    listing_urls=parsed_listing_urls if crawl_mode == "paginated_listing" else None,
                    disable_cache=disable_cache,
                )
            except StopIteration:
                rows = []
                status_box.warning("Crawl stopped by user request.")
            stop_btn_box.empty()

        # Mark crawl as finished so the Stop button disables itself.
        st.session_state["_crawl_running"] = False
        st.session_state["_stop_event"] = None

        # Persist results so the interactive gallery stays alive across reruns.
        st.session_state["_scan_results"] = list(rows)
        st.session_state["_scan_run_id"] = run_id
        st.session_state["_scan_output_path"] = ""
        st.session_state["_scan_run_config"] = {
            "fast_mode": fast_mode,
            "turbo_mode": turbo_mode,
            "table_score_threshold": configured_threshold,
            "flag_uncertain": flag_uncertain,
            "disable_cache": disable_cache,
        }

        progress_bar.progress(1.0)

        table_count = sum(1 for r in rows if r.label == "table")
        uncertain_count = sum(1 for r in rows if r.label == "uncertain")
        normal_count = len(rows) - table_count - uncertain_count
        with metrics_box.container():
            render_metrics(
                int(totals["processed"]),
                int(totals["unique_candidates"]) if int(totals["unique_candidates"]) > 0 else len(rows),
                table_count,
                normal_count,
                int(totals["cache_hits"]),
                float(totals["elapsed_seconds"]),
                crawl_elapsed_s=float(totals["crawl_elapsed_seconds"]),
                classified=int(totals["classified"]),
                classified_total=int(totals["classified_total"]),
            )
        split_box.caption(
            f"Candidates: raw {int(totals['raw_candidates'])} -> "
            f"unique {int(totals['unique_candidates'])} | "
            f"processed {int(totals['processed'])} | "
            f"fetch failures {int(totals['fetch_failures'])} | "
            f"skipped stalled {int(totals['skipped_due_to_stall'])}"
        )

        result_rows = [
            {
                "page_url": r.page_url,
                "image_url": r.image_url,
                "label": r.label,
                "score": round(r.score, 4),
                "reason": r.reason,
            }
            for r in rows
        ]
        table_rows_data = [r for r in result_rows if r["label"] == "table"]
        uncertain_rows_data = [r for r in result_rows if r["label"] == "uncertain"]
        normal_rows_data = [r for r in result_rows if r["label"] == "normal"]

        with table_results_box.container():
            st.subheader(f"Detected Tables ({len(table_rows_data)})")
            st.dataframe(table_rows_data, use_container_width=True, height=280)

        with normal_results_box.container():
            st.subheader(f"Detected Normal Images ({len(normal_rows_data)})")
            st.dataframe(normal_rows_data, use_container_width=True, height=280)

        if flag_uncertain:
            with uncertain_results_box.container():
                st.subheader(f"Flagged Uncertain Images ({len(uncertain_rows_data)})")
                st.dataframe(uncertain_rows_data, use_container_width=True, height=220)

        csv_bytes = format_results_csv_bytes(rows)
        st.download_button(
            label="Download Results CSV",
            data=csv_bytes,
            file_name=f"{run_id}_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
        report_target = single_image_url.strip() if eval_mode == "single_image" else url.strip()
        report_html = build_interactive_html_report(
            rows=rows,
            totals=totals,
            run_id=run_id,
            target=report_target,
        )
        st.download_button(
            label="Download Interactive HTML Report",
            data=report_html.encode("utf-8"),
            file_name=f"{run_id}_report.html",
            mime="text/html",
            use_container_width=True,
        )

        st.success(f"Completed. Results stored in `.crawler.db` (run_id `{run_id}`).")


    # ── Persistent interactive table gallery ──────────────────────────────────────
    # Renders from session state so it survives Streamlit reruns triggered by button clicks.
    _gallery_rows: List[ImageResult] = st.session_state.get("_scan_results") or []
    _marked_normal: set = st.session_state.get("_manually_marked_normal", set())

    if _gallery_rows:
        # Only show tables that haven't been manually corrected in this session.
        _table_items: List[ImageResult] = [
            r
            for r in _gallery_rows
            if r.label == "table" and r.image_url not in _marked_normal
        ]

        with table_gallery_box.container():
            _active_count = len(_table_items)
            _corrected_count = len(_marked_normal)

            st.subheader(f"Table Image Gallery — {_active_count} active")

            if _corrected_count:
                st.success(
                    f"✅ {_corrected_count} image{'s' if _corrected_count != 1 else ''} manually "
                    f"marked as normal and written to cache — future runs will skip "
                    f"{'them' if _corrected_count != 1 else 'it'}."
                )

            if not _table_items:
                if _corrected_count:
                    st.info("All detected table images in this run have been marked as normal.")
                else:
                    st.info("No table images were detected in this run.")
            else:
                st.caption(
                    "Images are loaded directly from their source URLs. "
                    "Click **Mark as Normal** on any image that was misclassified — "
                    "the correction is saved to the cache immediately."
                )
                _COLS = 3
                for _i in range(0, len(_table_items), _COLS):
                    _batch = _table_items[_i : _i + _COLS]
                    _cols = st.columns(_COLS)
                    for _col, _r in zip(_cols, _batch):
                        with _col:
                            # Display image directly from its source URL — no local file needed.
                            try:
                                st.image(_r.image_url, use_container_width=True)
                            except Exception:
                                st.warning("⚠️ Could not load image preview.")
                            st.caption(
                                f"Score: **{_r.score:.4f}**  \n"
                                f"[view image]({_r.image_url})  \n"
                                f"[source page]({_r.page_url})"
                            )
                            _btn_disabled = not bool(_r.image_hash)
                            _btn_help = (
                                "Overrides this classification to 'normal' and updates the cache "
                                "so future runs won't flag it again."
                                if _r.image_hash
                                else "Hash unavailable (fetch may have failed); cannot update cache."
                            )
                            if st.button(
                                "✅ Mark as Normal",
                                key=f"mark_{hashlib.sha1(_r.image_url.encode()).hexdigest()[:12]}",
                                use_container_width=True,
                                disabled=_btn_disabled,
                                help=_btn_help,
                            ):
                                _apply_mark_as_normal(_r.image_hash, _r.image_url)
                                st.rerun()
