from datetime import datetime
import hashlib
import html
import json
import threading
import time
from typing import Dict, List

import requests
import streamlit as st

from constants import (
    DEFAULT_TABLE_SCORE_THRESHOLD,
    ImageResult,
)
from classifier import classify_image
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
from orchestrator import (
    crawl_site,
    format_results_csv_bytes,
    write_results_csv,
)


st.set_page_config(
    page_title="Crawler",
    page_icon="🕸️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
        background: linear-gradient(90deg, #4f46e5, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        border: 1px solid rgba(100, 116, 139, 0.2);
        border-radius: 12px;
        padding: 0.9rem 1rem;
        background: rgba(15, 23, 42, 0.02);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">Lorem Ipsum</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Scan one page or a whole website, discover images, classify each image as <b>table</b> or <b>normal</b>, and export results.</div>',
    unsafe_allow_html=True,
)

default_workers = max(1, min(4, (__import__("os").cpu_count() or 4)))
default_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

with st.sidebar:
    st.header("Scan Settings")

    # ── Presets ──────────────────────────────────────────────────────────────
    st.markdown("**Presets**")
    preset = st.radio(
        "Preset",
        options=["full_precision", "full_speed", "custom"],
        format_func=lambda x: {
            "full_precision": "Full Precision",
            "full_speed": "Full Speed",
            "custom": "Custom",
        }[x],
        horizontal=True,
        label_visibility="collapsed",
    )
    if preset != "custom":
        st.caption(
            "Full Precision: slow but maximally accurate — fast OCR off, turbo off, confidence 100%. "
            if preset == "full_precision"
            else "Full Speed: fastest possible — fast OCR on, turbo on, confidence 50%."
        )
    st.divider()

    eval_mode = st.radio(
        "Evaluation mode",
        options=[("crawl", "Scan page/site"), ("single_image", "Evaluate one image only")],
        format_func=lambda option: option[1],
    )[0]
    url = ""
    crawl_mode = "page"
    single_image_source = "url"
    single_image_url = ""
    uploaded_single_image = None
    single_image_page_url = ""
    target_urls_input = ""
    listing_urls_input = ""
    max_pages = 40
    if eval_mode == "crawl":
        url = st.text_input("Target URL", placeholder="https://example.com/page")
        crawl_scope = st.radio(
            "Scan scope",
            options=[
                ("page", "Single page"),
                ("site", "Whole website (same domain)"),
                ("urls", "Specified URLs only"),
                ("paginated_listing", "Paginated listing (news / events agenda)"),
            ],
            format_func=lambda option: option[1],
        )
        crawl_mode = crawl_scope[0]
        if crawl_mode == "paginated_listing":
            st.caption(
                "Paginated listing mode: fetches all listing pages (?page=1…N), "
                "follows every card/item link on each page, and classifies images from both "
                "listing and detail pages. The total page count is auto-detected from the "
                "pagination label on the page."
            )
            listing_urls_input = st.text_area(
                "Additional listing URLs (one per line, optional)",
                placeholder="https://example.com/events\nhttps://example.com/news",
                help=(
                    "Crawl multiple paginated listings in one run. Each URL gets its own "
                    "page-count detection. The Target URL above is always included as the first listing."
                ),
                height=110,
            )
    else:
        single_image_source = st.radio(
            "Single image input",
            options=[("url", "Image URL"), ("upload", "Upload image file")],
            format_func=lambda option: option[1],
            horizontal=True,
        )[0]
        if single_image_source == "url":
            single_image_url = st.text_input(
                "Image URL", placeholder="https://example.com/image.png"
            )
        else:
            uploaded_single_image = st.file_uploader(
                "Upload image",
                type=["png", "jpg", "jpeg", "webp", "bmp", "gif", "tif", "tiff"],
            )
        single_image_page_url = st.text_input(
            "Source page URL (optional)",
            placeholder="https://example.com/page",
            help="Optional context shown in CSV as page_url.",
        )
    if crawl_mode in {"site", "paginated_listing"}:
        max_pages = int(st.number_input("Max pages to scan", min_value=1, value=40, step=1))
    if crawl_mode == "urls":
        target_urls_input = st.text_area(
            "Target URLs (one per line)",
            placeholder="https://example.com/page-1\nhttps://example.com/page-2",
            help="Only these URLs will be scanned. Links from these pages are not followed.",
            height=140,
        )
    run_id = st.text_input(
        "Run ID",
        value=default_run_id,
        help="Stored under this id in .crawler.db (run_results). CSV is built on download.",
    )

    # Derive preset overrides
    _preset_fast_mode = {"full_precision": False, "full_speed": True}.get(preset, None)
    _preset_turbo_mode = {"full_precision": False, "full_speed": True}.get(preset, None)
    _preset_confidence = {"full_precision": 100, "full_speed": 50}.get(preset, None)

    fast_mode = st.toggle(
        "Fast OCR mode",
        value=_preset_fast_mode if _preset_fast_mode is not None else True,
        disabled=(preset != "custom"),
    )
    turbo_mode = st.toggle(
        "Turbo mode (much faster, slightly lower quality)",
        value=_preset_turbo_mode if _preset_turbo_mode is not None else True,
        help="Uses aggressive skipping + lower OCR detail + faster image fetch failover.",
        disabled=(preset != "custom"),
    )
    table_confidence = st.slider(
        "Table confidence threshold",
        min_value=50,
        max_value=100,
        value=_preset_confidence if _preset_confidence is not None else 100,
        help="Higher means stricter. 100% keeps the current table detection threshold.",
        disabled=(preset != "custom"),
    )
    flag_uncertain = st.checkbox(
        "Flag uncertain images",
        value=False,
        help="Marks images as uncertain when they are very close to being labeled as a table.",
    )
    heartbeat_seconds = st.number_input("Heartbeat seconds", min_value=1.0, value=10.0, step=1.0)
    ocr_workers = st.number_input("OCR workers", min_value=1, value=default_workers, step=1)

_tab_scanner, _tab_history = st.tabs(["🕸️ Scanner", "📋 History & Cache"])

# ── History & Cache tab ────────────────────────────────────────────────────────

def _render_history_and_cache_tab() -> None:
    st.markdown(
        """
        <style>
        .hc-section-title {
            font-size: 1.15rem; font-weight: 700; margin: 1.5rem 0 0.6rem;
            color: #4f46e5;
        }
        .run-card {
            border: 1px solid rgba(100,116,139,0.18);
            border-radius: 14px;
            padding: 1rem 1.25rem;
            margin-bottom: 0.75rem;
            background: rgba(15,23,42,0.02);
            transition: box-shadow 0.15s;
        }
        .run-card:hover { box-shadow: 0 4px 18px rgba(79,70,229,0.08); }
        .run-id { font-weight: 700; font-size: 1rem; font-family: monospace; }
        .run-meta { color: #64748b; font-size: 0.8rem; margin-top: 0.15rem; }
        .badge {
            display: inline-block; border-radius: 9999px;
            padding: 0.15rem 0.6rem; font-size: 0.75rem; font-weight: 600;
            margin-right: 0.3rem;
        }
        .badge-table  { background: #dcfce7; color: #166534; }
        .badge-normal { background: #e2e8f0; color: #334155; }
        .badge-uncertain { background: #fef3c7; color: #92400e; }
        .cache-stat-card {
            border: 1px solid rgba(100,116,139,0.18);
            border-radius: 14px; padding: 1.1rem 1.4rem;
            background: rgba(15,23,42,0.02);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── Header ────────────────────────────────────────────────────────────────
    h_col, cache_col = st.columns([2, 1], gap="large")

    # ── LEFT: Run history ─────────────────────────────────────────────────────
    with h_col:
        st.markdown('<div class="hc-section-title">📋 Run History</div>', unsafe_allow_html=True)

        _run_ids = list_run_ids()
        if not _run_ids:
            st.info("No runs recorded yet. Run a scan to see history here.")
        else:
            st.caption(f"{len(_run_ids)} run{'s' if len(_run_ids) != 1 else ''} stored in `.crawler.db`")
            for _rid in _run_ids:
                _summary = get_run_summary(_rid)
                _is_active = st.session_state.get("_scan_run_id") == _rid

                badges = (
                    f'<span class="badge badge-table">{_summary["tables"]} tables</span>'
                    f'<span class="badge badge-normal">{_summary["normal"]} normal</span>'
                    + (
                        f'<span class="badge badge-uncertain">{_summary["uncertain"]} uncertain</span>'
                        if _summary["uncertain"]
                        else ""
                    )
                )
                active_marker = " ✦ active" if _is_active else ""
                st.markdown(
                    f'<div class="run-card">'
                    f'<span class="run-id">{_rid}</span>'
                    f'<span style="color:#4f46e5;font-size:0.78rem;margin-left:0.5rem">{active_marker}</span>'
                    f'<div class="run-meta">{_summary["created_at"]} &nbsp;·&nbsp; {_summary["total"]} images total</div>'
                    f'<div style="margin-top:0.5rem">{badges}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                btn_c1, btn_c2, btn_c3 = st.columns([2, 2, 1])
                with btn_c1:
                    if st.button(
                        "📂 Load into Scanner",
                        key=f"hc_load_{_rid}",
                        use_container_width=True,
                        type="primary" if not _is_active else "secondary",
                        disabled=_is_active,
                        help="Restore these results in the Scanner tab gallery.",
                    ):
                        _loaded = load_results_db(_rid)
                        if _loaded:
                            st.session_state["_scan_results"] = [
                                ImageResult(
                                    page_url=r["page_url"],
                                    image_url=r["image_url"],
                                    label=r["label"],
                                    score=r["score"],
                                    reason=r["reason"],
                                    image_hash=r.get("image_hash", ""),
                                )
                                for r in _loaded
                            ]
                            st.session_state["_scan_run_id"] = _rid
                            st.session_state["_manually_marked_normal"] = set()
                            st.toast(f"Loaded {_rid} ({len(_loaded)} images) — switch to the Scanner tab.", icon="📂")
                            st.rerun()
                        else:
                            st.warning("No results found for this run.")

                with btn_c2:
                    _rows_for_dl = load_results_db(_rid)
                    if _rows_for_dl:
                        _dl_rows = [
                            ImageResult(
                                page_url=r["page_url"],
                                image_url=r["image_url"],
                                label=r["label"],
                                score=r["score"],
                                reason=r["reason"],
                                image_hash=r.get("image_hash", ""),
                            )
                            for r in _rows_for_dl
                        ]
                        st.download_button(
                            label="⬇️ Download CSV",
                            data=format_results_csv_bytes(_dl_rows),
                            file_name=f"{_rid}_results.csv",
                            mime="text/csv",
                            key=f"hc_dl_{_rid}",
                            use_container_width=True,
                        )

                with btn_c3:
                    if st.button("🗑️ Delete", key=f"hc_del_{_rid}", use_container_width=True):
                        st.session_state[f"_hc_confirm_del_{_rid}"] = True
                        st.rerun()

                if st.session_state.get(f"_hc_confirm_del_{_rid}"):
                    st.warning(f"Permanently delete **{_rid}**? This cannot be undone.")
                    _yes_col, _no_col = st.columns(2)
                    with _yes_col:
                        if st.button("Yes, delete", key=f"hc_confirm_yes_{_rid}", type="primary"):
                            delete_run(_rid)
                            st.session_state.pop(f"_hc_confirm_del_{_rid}", None)
                            if st.session_state.get("_scan_run_id") == _rid:
                                st.session_state["_scan_results"] = None
                                st.session_state["_scan_run_id"] = ""
                            st.toast(f"Run {_rid} deleted.", icon="🗑️")
                            st.rerun()
                    with _no_col:
                        if st.button("Cancel", key=f"hc_confirm_no_{_rid}"):
                            st.session_state.pop(f"_hc_confirm_del_{_rid}", None)
                            st.rerun()

    # ── RIGHT: Cache stats & management ───────────────────────────────────────
    with cache_col:
        st.markdown('<div class="hc-section-title">🗄️ Classification Cache</div>', unsafe_allow_html=True)
        _stats = get_cache_stats()

        st.markdown(
            f'<div class="cache-stat-card">'
            f'<div style="font-size:2rem;font-weight:800;color:#4f46e5">{_stats["total_entries"]}</div>'
            f'<div style="color:#64748b;font-size:0.82rem;margin-bottom:0.75rem">cached entries</div>'
            f'<span class="badge badge-table">{_stats["tables"]} tables</span>'
            f'<span class="badge badge-normal">{_stats["normal"]} normal</span>'
            + (f'<span class="badge badge-uncertain">{_stats["uncertain"]} uncertain</span>' if _stats["uncertain"] else "")
            + f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown("")
        if _stats["total_entries"] == 0:
            st.info("Cache is empty — all images will be freshly classified on the next run.")
        else:
            st.caption(
                "The cache maps image hashes → label so identical images aren't re-classified on subsequent runs. "
                "Clear it to force full re-classification."
            )
            if st.button("🧹 Clear Entire Cache", type="secondary", use_container_width=True):
                st.session_state["_hc_confirm_clear"] = True
                st.rerun()
            if st.session_state.get("_hc_confirm_clear"):
                st.warning("All cached classifications will be deleted. Future runs will re-classify every image from scratch.")
                _cy, _cn = st.columns(2)
                with _cy:
                    if st.button("Yes, clear", key="hc_clear_yes", type="primary"):
                        _n = clear_classification_cache()
                        st.session_state.pop("_hc_confirm_clear", None)
                        st.toast(f"Cache cleared — {_n} entries removed.", icon="🧹")
                        st.rerun()
                with _cn:
                    if st.button("Cancel", key="hc_clear_no"):
                        st.session_state.pop("_hc_confirm_clear", None)
                        st.rerun()


with _tab_history:
    _render_history_and_cache_tab()

with _tab_scanner:
    st.markdown("### Output Preview")
    col1, col2 = st.columns(2)
    with col1:
        st.code(f"run_id: {run_id}", language="text")
    with col2:
        st.code(".crawler.db → run_results", language="text")

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


    def _render_metrics(
        processed: int,
        total: int,
        tables: int,
        normal: int,
        cache_hits: int,
        elapsed_s: float,
        crawl_elapsed_s: float = 0.0,
        classified: int = 0,
        classified_total: int = 0,
    ) -> None:
        c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
        classified_label = (
            f"{classified}/{classified_total}" if classified_total > 0 else str(classified)
        )
        c1.markdown(
            f'<div class="metric-card"><b>Processed</b><br>{processed}</div>',
            unsafe_allow_html=True,
        )
        c2.markdown(
            f'<div class="metric-card"><b>Total</b><br>{total}</div>', unsafe_allow_html=True
        )
        c3.markdown(
            f'<div class="metric-card"><b>Classified</b><br>{classified_label}</div>',
            unsafe_allow_html=True,
        )
        c4.markdown(
            f'<div class="metric-card"><b>Tables</b><br>{tables}</div>', unsafe_allow_html=True
        )
        c5.markdown(
            f'<div class="metric-card"><b>Normal</b><br>{normal}</div>', unsafe_allow_html=True
        )
        c6.markdown(
            f'<div class="metric-card"><b>Cache Hits</b><br>{cache_hits}</div>',
            unsafe_allow_html=True,
        )
        c7.markdown(
            f'<div class="metric-card"><b>Crawl Time (s)</b><br>{crawl_elapsed_s:.1f}</div>',
            unsafe_allow_html=True,
        )
        c8.markdown(
            f'<div class="metric-card"><b>Classify Time (s)</b><br>{elapsed_s:.1f}</div>',
            unsafe_allow_html=True,
        )


    def _build_interactive_html_report(
        rows: List[ImageResult],
        totals: Dict[str, float],
        run_id: str,
        target: str,
    ) -> str:
        rows_payload = [
            {
                "page_url": r.page_url,
                "image_url": r.image_url,
                "label": r.label,
                "score": round(float(r.score), 4),
                "reason": r.reason,
            }
            for r in rows
        ]
        payload_json = json.dumps(rows_payload, ensure_ascii=False)
        total_rows = len(rows_payload)
        table_rows = sum(1 for row in rows_payload if row["label"] == "table")
        uncertain_rows = sum(1 for row in rows_payload if row["label"] == "uncertain")
        normal_rows = total_rows - table_rows - uncertain_rows
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        avg_score = (
            (sum(float(row["score"]) for row in rows_payload) / total_rows) if total_rows else 0.0
        )
        return f"""<!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>Report - {html.escape(run_id)}</title>
      <style>
        :root {{
          --bg: #f3f7ff; --bg-2: #eef2ff; --text: #0f172a; --muted: #475569;
          --panel: rgba(255,255,255,0.9); --panel-border: rgba(148,163,184,0.26);
          --shadow: 0 10px 35px rgba(2,6,23,0.08); --primary: #2563eb;
          --primary-2: #0ea5e9; --table-head: #eff6ff; --row-alt: #f8fbff;
          --chip-table-bg: #dcfce7; --chip-table-fg: #166534;
          --chip-normal-bg: #e2e8f0; --chip-normal-fg: #334155;
          --chip-uncertain-bg: #fef3c7; --chip-uncertain-fg: #92400e;
        }}
        @media (prefers-color-scheme: dark) {{
          :root {{
            --bg: #0b1224; --bg-2: #111b34; --text: #e2e8f0; --muted: #94a3b8;
            --panel: rgba(15,23,42,0.75); --panel-border: rgba(71,85,105,0.45);
            --shadow: 0 16px 45px rgba(2,6,23,0.45); --primary: #60a5fa;
            --primary-2: #22d3ee; --table-head: #1e293b; --row-alt: #0f172a;
            --chip-table-bg: #14532d; --chip-table-fg: #86efac;
            --chip-normal-bg: #1e293b; --chip-normal-fg: #94a3b8;
            --chip-uncertain-bg: #78350f; --chip-uncertain-fg: #fde68a;
          }}
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ background: var(--bg); color: var(--text); font-family: system-ui, sans-serif; padding: 2rem; }}
        h1 {{ font-size: 1.8rem; font-weight: 700; margin-bottom: .25rem; }}
        .meta {{ color: var(--muted); font-size: .85rem; margin-bottom: 1.5rem; }}
        .stats {{ display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.5rem; }}
        .stat {{ background: var(--panel); border: 1px solid var(--panel-border);
                 border-radius: 12px; padding: .75rem 1.25rem; min-width: 110px; }}
        .stat b {{ display: block; font-size: 1.4rem; }}
        .stat span {{ font-size: .78rem; color: var(--muted); }}
        .filters {{ display: flex; gap: .5rem; flex-wrap: wrap; margin-bottom: 1rem; }}
        .filter-btn {{ padding: .4rem .9rem; border-radius: 9999px; border: 1px solid var(--panel-border);
                       cursor: pointer; background: var(--panel); color: var(--text); font-size: .82rem; }}
        .filter-btn.active {{ background: var(--primary); color: #fff; border-color: var(--primary); }}
        #search {{ padding: .4rem .8rem; border-radius: 8px; border: 1px solid var(--panel-border);
                   background: var(--panel); color: var(--text); font-size: .85rem; width: 280px; }}
        table {{ width: 100%; border-collapse: collapse; background: var(--panel);
                 border-radius: 12px; overflow: hidden; box-shadow: var(--shadow); }}
        th {{ background: var(--table-head); padding: .6rem 1rem; text-align: left;
              font-size: .8rem; text-transform: uppercase; letter-spacing: .05em; color: var(--muted); }}
        td {{ padding: .55rem 1rem; font-size: .83rem; border-top: 1px solid var(--panel-border); }}
        tr:nth-child(even) td {{ background: var(--row-alt); }}
        .chip {{ display: inline-block; border-radius: 9999px; padding: .15rem .6rem; font-size: .75rem; font-weight: 600; }}
        .chip-table {{ background: var(--chip-table-bg); color: var(--chip-table-fg); }}
        .chip-normal {{ background: var(--chip-normal-bg); color: var(--chip-normal-fg); }}
        .chip-uncertain {{ background: var(--chip-uncertain-bg); color: var(--chip-uncertain-fg); }}
        a {{ color: var(--primary); text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        .trunc {{ max-width: 260px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; display: block; }}
        #count {{ color: var(--muted); font-size: .82rem; margin-bottom: .5rem; }}
        .img-thumb {{ max-width: 80px; max-height: 48px; border-radius: 4px; object-fit: contain; }}
      </style>
    </head>
    <body>
      <h1>Crawler Report</h1>
      <p class="meta">Run ID: {html.escape(run_id)} &nbsp;|&nbsp; Target: {html.escape(target)} &nbsp;|&nbsp; Generated: {generated_at}</p>
      <div class="stats">
        <div class="stat"><b>{total_rows}</b><span>Total images</span></div>
        <div class="stat"><b>{table_rows}</b><span>Tables</span></div>
        <div class="stat"><b>{normal_rows}</b><span>Normal</span></div>
        <div class="stat"><b>{uncertain_rows}</b><span>Uncertain</span></div>
        <div class="stat"><b>{avg_score:.3f}</b><span>Avg score</span></div>
        <div class="stat"><b>{int(totals.get('pages_scanned', 0))}</b><span>Pages scanned</span></div>
        <div class="stat"><b>{int(totals.get('cache_hits', 0))}</b><span>Cache hits</span></div>
        <div class="stat"><b>{float(totals.get('elapsed_seconds', 0)):.1f}s</b><span>Classify time</span></div>
      </div>
      <div class="filters">
        <button class="filter-btn active" onclick="setFilter('all')">All ({total_rows})</button>
        <button class="filter-btn" onclick="setFilter('table')">Tables ({table_rows})</button>
        <button class="filter-btn" onclick="setFilter('normal')">Normal ({normal_rows})</button>
        <button class="filter-btn" onclick="setFilter('uncertain')">Uncertain ({uncertain_rows})</button>
        <input id="search" type="text" placeholder="Search URL or reason..." oninput="applyFilters()" />
      </div>
      <p id="count"></p>
      <table id="tbl">
        <thead><tr>
          <th>Preview</th><th>Label</th><th>Score</th>
          <th>Image URL</th><th>Page URL</th><th>Reason</th>
        </tr></thead>
        <tbody id="tbody"></tbody>
      </table>
      <script>
        const DATA = {payload_json};
        let currentFilter = 'all';
        function chip(label) {{
          const cls = label === 'table' ? 'chip-table' : label === 'uncertain' ? 'chip-uncertain' : 'chip-normal';
          return `<span class="chip ${{cls}}">${{label}}</span>`;
        }}
        function setFilter(f) {{
          currentFilter = f;
          document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
          event.target.classList.add('active');
          applyFilters();
        }}
        function applyFilters() {{
          const q = document.getElementById('search').value.toLowerCase();
          const rows = DATA.filter(r =>
            (currentFilter === 'all' || r.label === currentFilter) &&
            (!q || r.image_url.toLowerCase().includes(q) || r.page_url.toLowerCase().includes(q) || r.reason.toLowerCase().includes(q))
          );
          document.getElementById('count').textContent = rows.length + ' result(s)';
          document.getElementById('tbody').innerHTML = rows.map(r => `
            <tr>
              <td><img class="img-thumb" src="${{r.image_url}}" alt="" loading="lazy" onerror="this.style.display='none'" /></td>
              <td>${{chip(r.label)}}</td>
              <td>${{r.score.toFixed(4)}}</td>
              <td><a href="${{r.image_url}}" target="_blank" class="trunc" title="${{r.image_url}}">${{r.image_url}}</a></td>
              <td><a href="${{r.page_url}}" target="_blank" class="trunc" title="${{r.page_url}}">${{r.page_url}}</a></td>
              <td><span class="trunc" title="${{r.reason}}">${{r.reason}}</span></td>
            </tr>`).join('');
        }}
        applyFilters();
      </script>
    </body>
    </html>"""


    # ── Scan / evaluate button ────────────────────────────────────────────────────
    _scan_button_label = "🔍 Evaluate Image" if eval_mode == "single_image" else "🔍 Start Scan"
    _scan_clicked = st.button(
        _scan_button_label,
        type="primary",
        use_container_width=True,
        disabled=bool(st.session_state.get("_crawl_running")),
    )

    parsed_target_urls: List[str] = (
        [u.strip() for u in target_urls_input.splitlines() if u.strip()]
        if crawl_mode == "urls"
        else []
    )
    parsed_listing_urls: List[str] = (
        [u.strip() for u in listing_urls_input.splitlines() if u.strip()]
        if crawl_mode == "paginated_listing"
        else []
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
                    _render_metrics(
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
            status_box.info("Evaluating single image...")
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
                resolved_image_url = single_url
            else:
                image_bytes = (
                    uploaded_single_image.getvalue() if uploaded_single_image is not None else b""
                )
                if not image_bytes:
                    st.error("Uploaded image is empty.")
                    st.stop()
                upload_name = (
                    uploaded_single_image.name
                    if uploaded_single_image is not None
                    else "uploaded_image"
                )
                resolved_image_url = f"upload://{upload_name}"

            label, score, reason = classify_image(
                image_bytes=image_bytes,
                image_url=resolved_image_url,
                fast_mode=fast_mode,
                turbo_mode=turbo_mode,
                table_score_threshold=configured_threshold,
                flag_uncertain=flag_uncertain,
            )
            image_hash = hashlib.sha1(image_bytes).hexdigest()
            source_page = single_image_page_url.strip() or resolved_image_url
            rows = [
                ImageResult(
                    page_url=source_page,
                    image_url=resolved_image_url,
                    label=label,
                    score=score,
                    reason=reason,
                    image_hash=image_hash,
                )
            ]
            totals["processed"] = 1
            totals["total"] = 1
            totals["elapsed_seconds"] = 0.0
            totals["tables"] = 1 if label == "table" else 0
            totals["uncertain"] = 1 if label == "uncertain" else 0
            totals["normal"] = 1 if label == "normal" else 0
            progress_bar.progress(1.0)
            status_box.success("Single image evaluation finished.")
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
        }

        progress_bar.progress(1.0)

        table_count = sum(1 for r in rows if r.label == "table")
        uncertain_count = sum(1 for r in rows if r.label == "uncertain")
        normal_count = len(rows) - table_count - uncertain_count
        with metrics_box.container():
            _render_metrics(
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
        report_html = _build_interactive_html_report(
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