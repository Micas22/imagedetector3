"""History & Cache tab – extracted from webapp.py."""

from typing import List

import streamlit as st

from constants import ImageResult
from database import (
    clear_classification_cache,
    delete_run,
    get_cache_stats,
    get_run_summary,
    list_run_ids,
    load_results_db,
)
from orchestrator import format_results_csv_bytes


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
