"""Report builder & metrics renderer – extracted from webapp.py."""

import html
import json
from datetime import datetime
from typing import Dict, List

import streamlit as st

from constants import ImageResult


def render_metrics(
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


def build_interactive_html_report(
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
