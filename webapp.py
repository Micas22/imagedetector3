from datetime import datetime
from typing import List

import streamlit as st

from webapp_history import _render_history_and_cache_tab
from webapp_scanner import render_scanner_tab


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
    flag_uncertain = False
    heartbeat_seconds = 10.0
    ocr_workers = default_workers
    st.divider()
    disable_cache = st.toggle(
        "Disable cache",
        value=False,
        help=(
            "Completely turns off the classification cache. "
            "Images won't be looked up in the cache and results won't be saved to it."
        ),
    )

# ── Parse multi-line URL inputs ──────────────────────────────────────────────
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

# ── Tabs ─────────────────────────────────────────────────────────────────────
_tab_scanner, _tab_history = st.tabs(["🕸️ Scanner", "📋 History & Cache"])

with _tab_history:
    _render_history_and_cache_tab()

with _tab_scanner:
    render_scanner_tab(
        run_id=run_id,
        eval_mode=eval_mode,
        url=url,
        crawl_mode=crawl_mode,
        single_image_source=single_image_source,
        single_image_url=single_image_url,
        uploaded_single_image=uploaded_single_image,
        single_image_page_url=single_image_page_url,
        parsed_target_urls=parsed_target_urls,
        parsed_listing_urls=parsed_listing_urls,
        max_pages=max_pages,
        fast_mode=fast_mode,
        turbo_mode=turbo_mode,
        table_confidence=table_confidence,
        flag_uncertain=flag_uncertain,
        heartbeat_seconds=float(heartbeat_seconds),
        ocr_workers=int(ocr_workers),
        disable_cache=disable_cache,
    )