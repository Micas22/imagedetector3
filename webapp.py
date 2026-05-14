from datetime import datetime
import hashlib
import html
import json
from pathlib import Path
import threading
import time
from typing import Dict, List

import requests
import streamlit as st

from constants import (
    CACHE_FILENAME,
    DEFAULT_TABLE_SCORE_THRESHOLD,
    ImageResult,
)
from classifier import (
    classify_image,
    load_classification_cache,
    save_classification_cache,
    save_table_image,
)
from orchestrator import (
    crawl_site,
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
        format_func=lambda x: {"full_precision": "Full Precision", "full_speed": "Full Speed", "custom": "Custom"}[x],
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
                help="Crawl multiple paginated listings in one run. Each URL gets its own page-count detection. The Target URL above is always included as the first listing.",
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
            single_image_url = st.text_input("Image URL", placeholder="https://example.com/image.png")
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
    run_id = st.text_input("Run ID", value=default_run_id)
    base_run_dir = st.text_input("Base run folder", value="runs")

    output_filename = st.text_input("CSV filename", value="results.csv")
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
    save_tables = st.toggle("Save detected table images", value=True)
    heartbeat_seconds = st.number_input("Heartbeat seconds", min_value=1.0, value=10.0, step=1.0)
    ocr_workers = st.number_input("OCR workers", min_value=1, value=default_workers, step=1)

    custom_table_dir = st.text_input(
        "Optional custom table images folder",
        value="",
        help="Leave blank to use <base_run_folder>/<run_id>/table_images when saving is enabled.",
    )

run_dir = Path(base_run_dir) / run_id
output_path = str(run_dir / output_filename)
table_dir = ""
if save_tables:
    table_dir = custom_table_dir.strip() or str(run_dir / "table_images")

st.markdown("### Output Preview")
col1, col2 = st.columns(2)
with col1:
    st.code(output_path, language="text")
with col2:
    st.code(table_dir or "(table image saving disabled)", language="text")

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
if "_scan_table_dir" not in st.session_state:
    st.session_state["_scan_table_dir"] = ""
if "_scan_output_path" not in st.session_state:
    st.session_state["_scan_output_path"] = ""
if "_scan_run_config" not in st.session_state:
    st.session_state["_scan_run_config"] = {}
if "_manually_marked_normal" not in st.session_state:
    st.session_state["_manually_marked_normal"] = set()


def _apply_mark_as_normal(img_path: Path, image_url: str) -> None:
    """Read saved image bytes → update classification cache → update rows + CSV."""
    try:
        image_bytes = img_path.read_bytes()
    except OSError as exc:
        st.error(f"Could not read image file: {exc}")
        return

    image_hash = hashlib.sha1(image_bytes).hexdigest()
    run_cfg = st.session_state.get("_scan_run_config", {})
    fast = run_cfg.get("fast_mode", False)
    turbo = run_cfg.get("turbo_mode", False)
    threshold = run_cfg.get("table_score_threshold", DEFAULT_TABLE_SCORE_THRESHOLD)
    flag_unc = run_cfg.get("flag_uncertain", False)

    # ── Update the on-disk classification cache ───────────────────────────────
    cache_path = Path(CACHE_FILENAME)
    cache = load_classification_cache(cache_path, fast, turbo, threshold, flag_unc)
    cache[image_hash] = ("normal", 0.0, "manually_marked_normal")
    save_classification_cache(cache_path, cache, fast, turbo, threshold, flag_unc)

    # ── Update in-memory rows so the rest of the UI stays consistent ──────────
    current_rows: List[ImageResult] = st.session_state.get("_scan_results") or []
    updated_rows = [
        ImageResult(r.page_url, r.image_url, "normal", 0.0, "manually_marked_normal")
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
    classified_label = f"{classified}/{classified_total}" if classified_total > 0 else str(classified)
    c1.markdown(f'<div class="metric-card"><b>Processed</b><br>{processed}</div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><b>Total</b><br>{total}</div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><b>Classified</b><br>{classified_label}</div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><b>Tables</b><br>{tables}</div>', unsafe_allow_html=True)
    c5.markdown(f'<div class="metric-card"><b>Normal</b><br>{normal}</div>', unsafe_allow_html=True)
    c6.markdown(f'<div class="metric-card"><b>Cache Hits</b><br>{cache_hits}</div>', unsafe_allow_html=True)
    c7.markdown(f'<div class="metric-card"><b>Crawl Time (s)</b><br>{crawl_elapsed_s:.1f}</div>', unsafe_allow_html=True)
    c8.markdown(f'<div class="metric-card"><b>Classify Time (s)</b><br>{elapsed_s:.1f}</div>', unsafe_allow_html=True)


def _build_interactive_html_report(
    rows: List[ImageResult],
    totals: Dict[str, float],
    run_id: str,
    target: str,
    output_path: str,
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
    avg_score = (sum(float(row["score"]) for row in rows_payload) / total_rows) if total_rows else 0.0
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Report - {html.escape(run_id)}</title>
  <style>
    :root {{
      --bg: #f3f7ff;
      --bg-2: #eef2ff;
      --text: #0f172a;
      --muted: #475569;
      --panel: rgba(255, 255, 255, 0.9);
      --panel-border: rgba(148, 163, 184, 0.26);
      --shadow: 0 10px 35px rgba(2, 6, 23, 0.08);
      --primary: #2563eb;
      --primary-2: #0ea5e9;
      --table-head: #eff6ff;
      --row-alt: #f8fbff;
      --chip-table-bg: #dcfce7;
      --chip-table-fg: #166534;
      --chip-normal-bg: #e2e8f0;
      --chip-normal-fg: #334155;
      --chip-uncertain-bg: #fef3c7;
      --chip-uncertain-fg: #92400e;
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --bg: #0b1224;
        --bg-2: #111b34;
        --text: #e2e8f0;
        --muted: #94a3b8;
        --panel: rgba(15, 23, 42, 0.75);
        --panel-border: rgba(71, 85, 105, 0.45);
        --shadow: 0 16px 45px rgba(2, 6, 23, 0.45);
        --primary: #60a5fa;
        --primary-2: #22d3ee;
        --table-head: #0f172a;
        --row-alt: #111827;
        --chip-table-bg: #14532d;
        --chip-table-fg: #dcfce7;
        --chip-normal-bg: #334155;
        --chip-normal-fg: #e2e8f0;
        --chip-uncertain-bg: #78350f;
        --chip-uncertain-fg: #fde68a;
      }}
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, Segoe UI, Roboto, Arial, sans-serif;
      color: var(--text);
      background: radial-gradient(circle at 5% 5%, #bfdbfe 0%, transparent 30%),
        radial-gradient(circle at 95% 10%, #a5f3fc 0%, transparent 30%),
        linear-gradient(180deg, var(--bg) 0%, var(--bg-2) 100%);
      min-height: 100vh;
      padding: 24px;
    }}
    .shell {{ max-width: 1320px; margin: 0 auto; }}
    .hero {{
      background: linear-gradient(115deg, rgba(37, 99, 235, 0.92), rgba(14, 165, 233, 0.9));
      color: #ffffff;
      border-radius: 18px;
      box-shadow: var(--shadow);
      padding: 20px 22px;
      margin-bottom: 16px;
    }}
    .hero h1 {{ margin: 0 0 8px; font-size: 30px; letter-spacing: -0.02em; }}
    .hero .sub {{ opacity: 0.95; font-size: 14px; }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
      gap: 8px;
      margin-top: 12px;
      font-size: 13px;
    }}
    .meta-item {{
      background: rgba(255, 255, 255, 0.16);
      border: 1px solid rgba(255, 255, 255, 0.24);
      border-radius: 10px;
      padding: 8px 10px;
      backdrop-filter: blur(6px);
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 12px;
      margin-bottom: 14px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--panel-border);
      border-radius: 14px;
      padding: 12px 14px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(8px);
    }}
    .card .k {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; }}
    .card .v {{ font-size: 28px; font-weight: 700; margin-top: 4px; line-height: 1.1; }}
    .toolbar {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
      padding: 12px;
      margin-bottom: 10px;
      border-radius: 14px;
      background: var(--panel);
      border: 1px solid var(--panel-border);
      box-shadow: var(--shadow);
    }}
    .toolbar input, .toolbar select, .toolbar button {{
      padding: 10px 12px;
      border: 1px solid var(--panel-border);
      border-radius: 10px;
      font-size: 13px;
      background: rgba(255, 255, 255, 0.82);
      color: #0f172a;
    }}
    @media (prefers-color-scheme: dark) {{
      .toolbar input, .toolbar select, .toolbar button {{
        background: rgba(15, 23, 42, 0.85);
        color: #e2e8f0;
      }}
    }}
    .toolbar input {{ min-width: 260px; flex: 1; }}
    .toolbar button {{
      background: linear-gradient(120deg, var(--primary), var(--primary-2));
      color: white;
      border: none;
      font-weight: 600;
      cursor: pointer;
      transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}
    .toolbar button:hover {{ transform: translateY(-1px); box-shadow: 0 8px 20px rgba(37, 99, 235, 0.3); }}
    .table-wrap {{
      overflow: auto;
      border-radius: 14px;
      border: 1px solid var(--panel-border);
      box-shadow: var(--shadow);
      background: var(--panel);
    }}
    table {{ width: 100%; border-collapse: collapse; min-width: 900px; }}
    th, td {{ border-bottom: 1px solid var(--panel-border); padding: 10px; vertical-align: top; font-size: 13px; }}
    th {{
      position: sticky;
      top: 0;
      z-index: 2;
      background: var(--table-head);
      cursor: pointer;
      user-select: none;
      text-align: left;
      white-space: nowrap;
    }}
    tbody tr:nth-child(even) {{ background: var(--row-alt); }}
    tbody tr:hover {{ background: rgba(14, 165, 233, 0.12); }}
    .url {{ max-width: 410px; word-break: break-word; }}
    .url a {{ color: var(--primary); text-decoration: none; }}
    .url a:hover {{ text-decoration: underline; }}
    .badge {{ display: inline-block; border-radius: 999px; padding: 4px 10px; font-size: 12px; font-weight: 600; }}
    .b-table {{ background: var(--chip-table-bg); color: var(--chip-table-fg); }}
    .b-normal {{ background: var(--chip-normal-bg); color: var(--chip-normal-fg); }}
    .b-uncertain {{ background: var(--chip-uncertain-bg); color: var(--chip-uncertain-fg); }}
    .score-wrap {{
      min-width: 140px;
      display: grid;
      grid-template-columns: 58px 1fr;
      gap: 8px;
      align-items: center;
    }}
    .score-bar {{
      height: 8px;
      border-radius: 999px;
      background: rgba(148, 163, 184, 0.25);
      overflow: hidden;
    }}
    .score-fill {{
      height: 100%;
      border-radius: 999px;
      background: linear-gradient(90deg, #22c55e, #0ea5e9, #6366f1);
    }}
    .foot {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 12px;
      display: flex;
      justify-content: space-between;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      border: 1px solid var(--panel-border);
      border-radius: 999px;
      padding: 5px 10px;
      background: var(--panel);
    }}
    .empty {{
      padding: 26px;
      text-align: center;
      color: var(--muted);
      font-size: 14px;
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <h1>Report</h1>
      <div class="sub">Visual audit of image classification output with live filtering and client-side export.</div>
      <div class="meta">
        <div class="meta-item"><b>Run ID</b><br />{html.escape(run_id)}</div>
        <div class="meta-item"><b>Target</b><br />{html.escape(target)}</div>
        <div class="meta-item"><b>Generated</b><br />{html.escape(generated_at)}</div>
        <div class="meta-item"><b>CSV Source</b><br />{html.escape(output_path)}</div>
      </div>
    </section>

    <section class="cards">
      <div class="card"><div class="k">Visible Rows</div><div id="metric-total" class="v">{total_rows}</div></div>
      <div class="card"><div class="k">Tables</div><div id="metric-table" class="v">{table_rows}</div></div>
      <div class="card"><div class="k">Uncertain</div><div id="metric-uncertain" class="v">{uncertain_rows}</div></div>
      <div class="card"><div class="k">Normal</div><div id="metric-normal" class="v">{normal_rows}</div></div>
      <div class="card"><div class="k">Avg Score</div><div id="metric-avg-score" class="v">{avg_score:.3f}</div></div>
      <div class="card"><div class="k">Elapsed</div><div class="v">{float(totals.get("elapsed_seconds", 0.0)):.1f}s</div></div>
      <div class="card"><div class="k">Cache Hits</div><div class="v">{int(totals.get("cache_hits", 0))}</div></div>
    </section>

    <section class="toolbar">
      <input id="search" placeholder="Search by URL, label, score, reason..." />
      <select id="labelFilter">
        <option value="">All labels</option>
        <option value="table">table</option>
        <option value="uncertain">uncertain</option>
        <option value="normal">normal</option>
      </select>
      <select id="scoreFilter">
        <option value="">Any score</option>
        <option value="gte_0.25">Score >= 0.25</option>
        <option value="gte_0.5">Score >= 0.50</option>
        <option value="gte_0.75">Score >= 0.75</option>
      </select>
      <button id="downloadFilteredCsv">Download filtered CSV</button>
      <span class="pill" id="visibleCount"></span>
    </section>

    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th data-sort="page_url">Page URL</th>
            <th data-sort="image_url">Image URL</th>
            <th data-sort="label">Label</th>
            <th data-sort="score">Score</th>
            <th data-sort="reason">Reason</th>
          </tr>
        </thead>
        <tbody id="tbody"></tbody>
      </table>
      <div id="emptyState" class="empty" style="display:none;">No rows match your current filters.</div>
    </div>

    <div class="foot">
      <span>Tip: click table headers to sort ascending/descending.</span>
      <span id="sortStatus">Sorted by score (desc)</span>
    </div>
  </div>

  <script>
    const rows = {payload_json};
    let sortKey = "score";
    let sortAsc = false;

    function labelBadge(label) {{
      const cls = label === "table" ? "b-table" : (label === "uncertain" ? "b-uncertain" : "b-normal");
      return `<span class="badge ${{cls}}">${{label}}</span>`;
    }}

    function escapeHtml(v) {{
      return String(v ?? "").replace(/[&<>"']/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}}[c]));
    }}

    function filteredRows() {{
      const q = document.getElementById("search").value.toLowerCase().trim();
      const filter = document.getElementById("labelFilter").value;
      const scoreFilter = document.getElementById("scoreFilter").value;
      return rows.filter(r => {{
        if (filter && r.label !== filter) return false;
        if (scoreFilter === "gte_0.25" && Number(r.score) < 0.25) return false;
        if (scoreFilter === "gte_0.5" && Number(r.score) < 0.5) return false;
        if (scoreFilter === "gte_0.75" && Number(r.score) < 0.75) return false;
        if (!q) return true;
        return [r.page_url, r.image_url, r.reason, r.label, String(r.score)]
          .join(" ")
          .toLowerCase()
          .includes(q);
      }});
    }}

    function sortRows(list) {{
      return list.slice().sort((a, b) => {{
        const av = a[sortKey];
        const bv = b[sortKey];
        let cmp = 0;
        if (sortKey === "score") cmp = Number(av) - Number(bv);
        else cmp = String(av).localeCompare(String(bv));
        return sortAsc ? cmp : -cmp;
      }});
    }}

    function render() {{
      const tbody = document.getElementById("tbody");
      const data = sortRows(filteredRows());
      const emptyState = document.getElementById("emptyState");
      tbody.innerHTML = data.map(r => `
        <tr>
          <td class="url"><a href="${{escapeHtml(r.page_url)}}" target="_blank" rel="noreferrer">${{escapeHtml(r.page_url)}}</a></td>
          <td class="url"><a href="${{escapeHtml(r.image_url)}}" target="_blank" rel="noreferrer">${{escapeHtml(r.image_url)}}</a></td>
          <td>${{labelBadge(escapeHtml(r.label))}}</td>
          <td>
            <div class="score-wrap">
              <span>${{Number(r.score).toFixed(4)}}</span>
              <div class="score-bar"><div class="score-fill" style="width:${{Math.max(0, Math.min(100, Number(r.score) * 100))}}%"></div></div>
            </div>
          </td>
          <td class="url">${{escapeHtml(r.reason)}}</td>
        </tr>`).join("");
      emptyState.style.display = data.length ? "none" : "block";
      document.getElementById("visibleCount").textContent = `Showing ${{data.length}} / ${{rows.length}} rows`;
      document.getElementById("metric-total").textContent = String(data.length);
      document.getElementById("metric-table").textContent = String(data.filter(r => r.label === "table").length);
      document.getElementById("metric-uncertain").textContent = String(data.filter(r => r.label === "uncertain").length);
      document.getElementById("metric-normal").textContent = String(data.filter(r => r.label === "normal").length);
      const avg = data.length ? (data.reduce((acc, r) => acc + Number(r.score), 0) / data.length) : 0;
      document.getElementById("metric-avg-score").textContent = avg.toFixed(3);
      document.getElementById("sortStatus").textContent = `Sorted by ${{sortKey}} (${{sortAsc ? "asc" : "desc"}})`;
    }}

    document.getElementById("search").addEventListener("input", render);
    document.getElementById("labelFilter").addEventListener("change", render);
    document.getElementById("scoreFilter").addEventListener("change", render);
    document.querySelectorAll("th[data-sort]").forEach(th => {{
      th.addEventListener("click", () => {{
        const key = th.getAttribute("data-sort");
        if (sortKey === key) sortAsc = !sortAsc;
        else {{ sortKey = key; sortAsc = true; }}
        render();
      }});
    }});

    document.getElementById("downloadFilteredCsv").addEventListener("click", () => {{
      const data = sortRows(filteredRows());
      const header = ["page_url","image_url","label","score","reason"];
      const csvRows = [header.join(",")].concat(data.map(r => header.map(k => {{
        const val = String(r[k] ?? "").replace(/"/g, '""');
        return /[",\\n]/.test(val) ? `"${{val}}"` : val;
      }}).join(",")));
      const blob = new Blob([csvRows.join("\\n")], {{ type: "text/csv;charset=utf-8;" }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "filtered_results.csv";
      a.click();
      URL.revokeObjectURL(url);
    }});

    render();
  </script>
</body>
</html>"""


col_start, col_stop = st.columns([3, 1])
with col_start:
    start_clicked = st.button("Start Scan", type="primary", use_container_width=True)
with col_stop:
    stop_clicked = st.button(
        "⏹ Stop Crawl",
        use_container_width=True,
        disabled=not st.session_state.get("_crawl_running", False),
        help="Gracefully abort the current crawl/classify run.",
    )

if stop_clicked and st.session_state.get("_stop_event") is not None:
    st.session_state["_stop_event"].set()
    st.warning("Stop signal sent — the crawl will finish its current image and then halt.")

if start_clicked:
    parsed_target_urls = [line.strip() for line in target_urls_input.splitlines() if line.strip()] if crawl_mode == "urls" else []
    parsed_listing_urls = [line.strip() for line in listing_urls_input.splitlines() if line.strip()] if crawl_mode == "paginated_listing" else []

    if eval_mode == "crawl" and crawl_mode == "urls" and not parsed_target_urls:
        st.error("Provide at least one URL in 'Target URLs'.")
        st.stop()
    if eval_mode == "crawl" and crawl_mode != "urls" and not url.strip():
        st.error("Target URL is required.")
        st.stop()
    if eval_mode == "single_image" and single_image_source == "url" and not single_image_url.strip():
        st.error("Image URL is required.")
        st.stop()
    if eval_mode == "single_image" and single_image_source == "upload" and uploaded_single_image is None:
        st.error("Upload an image file.")
        st.stop()

    run_dir.mkdir(parents=True, exist_ok=True)

    # Clear any gallery state from a previous run so stale corrections don't bleed in.
    st.session_state["_scan_results"] = None
    st.session_state["_manually_marked_normal"] = set()

    # Initialise stop signal for this run.
    stop_event = threading.Event()
    st.session_state["_stop_event"] = stop_event
    st.session_state["_crawl_running"] = True

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

        # Honour stop signal — raise to break out of the executor loop in crawl_site.
        if stop_event.is_set():
            raise StopIteration("stop_requested")

        event_type = event.get("event", "")

        # Snapshot wall-clock times so every branch can use them.
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
            # Crawling phase is done; classification is starting now.
            totals["classify_wall_start"] = now_mono
            totals["raw_candidates"] = int(event.get("total_candidates", totals["raw_candidates"]))
            totals["unique_candidates"] = int(event.get("unique_candidates", totals["unique_candidates"]))
            totals["total"] = int(event.get("unique_candidates", event.get("total_candidates", totals["total"])))
            totals["classified_total"] = totals["total"]
            status_box.info(event.get("message", "Discovered image candidates."))
            split_box.caption(
                f"Candidates: raw {int(totals['raw_candidates'])} -> unique {int(totals['unique_candidates'])} | "
                f"processed {int(totals['processed'])} | fetch failures {int(totals['fetch_failures'])} | "
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
                f"Candidates: raw {int(totals['raw_candidates'])} -> unique {int(totals['total'])} | "
                f"processed {int(totals['processed'])} | fetch failures {int(totals['fetch_failures'])} | "
                f"skipped stalled {int(totals['skipped_due_to_stall'])}"
            )
        elif event_type == "error":
            status_box.error(event.get("message", "An error occurred while scanning."))
        elif event_type == "finished":
            totals["elapsed_seconds"] = float(event.get("elapsed_seconds", totals["elapsed_seconds"]))
            totals["raw_candidates"] = int(event.get("raw_candidates", totals["raw_candidates"]))
            totals["unique_candidates"] = int(event.get("unique_candidates", totals["unique_candidates"]))
            totals["processed"] = int(event.get("processed", totals["processed"]))
            totals["skipped_due_to_stall"] = int(event.get("skipped_due_to_stall", totals["skipped_due_to_stall"]))
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
            image_bytes = uploaded_single_image.getvalue() if uploaded_single_image is not None else b""
            if not image_bytes:
                st.error("Uploaded image is empty.")
                st.stop()
            upload_name = uploaded_single_image.name if uploaded_single_image is not None else "uploaded_image"
            resolved_image_url = f"upload://{upload_name}"

        label, score, reason = classify_image(
            image_bytes=image_bytes,
            image_url=resolved_image_url,
            fast_mode=fast_mode,
            turbo_mode=turbo_mode,
            table_score_threshold=configured_threshold,
            flag_uncertain=flag_uncertain,
        )
        source_page = single_image_page_url.strip() or resolved_image_url
        rows = [ImageResult(page_url=source_page, image_url=resolved_image_url, label=label, score=score, reason=reason)]
        totals["processed"] = 1
        totals["total"] = 1
        totals["elapsed_seconds"] = 0.0
        totals["tables"] = 1 if label == "table" else 0
        totals["uncertain"] = 1 if label == "uncertain" else 0
        totals["normal"] = 1 if label == "normal" else 0
        if table_dir and label == "table":
            save_table_image(Path(table_dir), resolved_image_url, image_bytes)
        progress_bar.progress(1.0)
        status_box.success("Single image evaluation finished.")
    else:
        with stop_btn_box.container():
            st.info("Crawl/classify is running. Use the ⏹ Stop Crawl button to abort.")
        try:
            rows = crawl_site(
                start_url=(url.strip() if url.strip() else (parsed_target_urls[0] if parsed_target_urls else "")),
                render_js=True,
                save_table_dir=table_dir or None,
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

    write_results_csv(output_path, rows)

    # Persist results so the interactive gallery stays alive across reruns
    # triggered by "Mark as Normal" button clicks.
    st.session_state["_scan_results"] = list(rows)
    st.session_state["_scan_table_dir"] = table_dir
    st.session_state["_scan_output_path"] = output_path
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
        f"Candidates: raw {int(totals['raw_candidates'])} -> unique {int(totals['unique_candidates'])} | "
        f"processed {int(totals['processed'])} | fetch failures {int(totals['fetch_failures'])} | "
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
    table_rows = [r for r in result_rows if r["label"] == "table"]
    uncertain_rows = [r for r in result_rows if r["label"] == "uncertain"]
    normal_rows = [r for r in result_rows if r["label"] == "normal"]

    with table_results_box.container():
        st.subheader(f"Detected Tables ({len(table_rows)})")
        st.dataframe(table_rows, use_container_width=True, height=280)

    with normal_results_box.container():
        st.subheader(f"Detected Normal Images ({len(normal_rows)})")
        st.dataframe(normal_rows, use_container_width=True, height=280)

    if flag_uncertain:
        with uncertain_results_box.container():
            st.subheader(f"Flagged Uncertain Images ({len(uncertain_rows)})")
            st.dataframe(uncertain_rows, use_container_width=True, height=220)

    csv_bytes = Path(output_path).read_bytes()
    st.download_button(
        label="Download Results CSV",
        data=csv_bytes,
        file_name=Path(output_path).name,
        mime="text/csv",
        use_container_width=True,
    )
    report_target = single_image_url.strip() if eval_mode == "single_image" else url.strip()
    report_html = _build_interactive_html_report(
        rows=rows,
        totals=totals,
        run_id=run_id,
        target=report_target,
        output_path=output_path,
    )
    st.download_button(
        label="Download Interactive HTML Report",
        data=report_html.encode("utf-8"),
        file_name=f"{run_id}_report.html",
        mime="text/html",
        use_container_width=True,
    )

    if table_dir and Path(table_dir).exists():
        table_files = sorted(
            [p for p in Path(table_dir).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}]
        )
        with table_gallery_box.container():
            st.subheader(f"Saved Table Image Gallery ({len(table_files)})")
            if table_files:
                # Gallery with interactive "Mark as Normal" buttons is rendered
                # below (outside this block) so it persists across reruns.
                pass
            else:
                st.info("No table images were saved for this run.")

    st.success(f"Completed. Results saved to `{output_path}`.")

# ── Persistent interactive table gallery (renders from session state, survives reruns) ──
_gallery_rows: List[ImageResult] = st.session_state.get("_scan_results") or []
_gallery_table_dir: str = st.session_state.get("_scan_table_dir", "")
_marked_normal: set = st.session_state.get("_manually_marked_normal", set())

if _gallery_rows and _gallery_table_dir and Path(_gallery_table_dir).exists():
    _IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
    _all_files = sorted(p for p in Path(_gallery_table_dir).iterdir() if p.suffix.lower() in _IMAGE_EXTS)

    # Build a lookup: url_digest (14-char sha1 prefix used in filename) → file path
    _digest_to_file: Dict[str, Path] = {}
    for _p in _all_files:
        if _p.stem.startswith("table_"):
            _digest_to_file[_p.stem[len("table_"):]] = _p

    # Map each table-labelled result to its saved file (if any)
    _table_items = []  # list of (ImageResult, Path | None)
    for _r in _gallery_rows:
        if _r.label != "table" and _r.image_url not in _marked_normal:
            continue
        if _r.image_url in _marked_normal:
            continue  # already corrected — skip from gallery
        _url_digest = hashlib.sha1(_r.image_url.encode("utf-8")).hexdigest()[:14]
        _file = _digest_to_file.get(_url_digest)
        if _file is not None:
            _table_items.append((_r, _file))

    with table_gallery_box.container():
        _active_count = len(_table_items)
        _saved_count = len(_all_files)
        _corrected_count = len(_marked_normal)

        st.subheader(f"Table Image Gallery — {_active_count} active / {_saved_count} saved")

        if _corrected_count:
            st.success(
                f"✅ {_corrected_count} image{'s' if _corrected_count != 1 else ''} manually marked as normal "
                f"and written to cache — future runs will skip {'them' if _corrected_count != 1 else 'it'}."
            )

        if not _table_items:
            if _corrected_count:
                st.info("All detected table images in this run have been marked as normal.")
            else:
                st.info("No table images were saved for this run.")
        else:
            st.caption("Click **Mark as Normal** on any image that was misclassified. The correction is saved to the cache immediately.")
            _COLS = 3
            for _i in range(0, len(_table_items), _COLS):
                _batch = _table_items[_i: _i + _COLS]
                _cols = st.columns(_COLS)
                for _col, (_row, _img_file) in zip(_cols, _batch):
                    with _col:
                        st.image(str(_img_file), use_container_width=True)
                        st.caption(
                            f"Score: **{_row.score:.4f}**  \n"
                            f"[view image]({_row.image_url})  \n"
                            f"[source page]({_row.page_url})"
                        )
                        if st.button(
                            "✅ Mark as Normal",
                            key=f"mark_{_img_file.stem}",
                            use_container_width=True,
                            help="Overrides this classification to 'normal' and updates the cache so future runs won't flag it again.",
                        ):
                            _apply_mark_as_normal(_img_file, _row.image_url)
                            st.rerun()