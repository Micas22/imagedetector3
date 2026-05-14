import argparse
import concurrent.futures
import csv
import hashlib
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import requests

from classifier import classify_image
from constants import (
    DEFAULT_TABLE_SCORE_THRESHOLD,
    NON_HTML_EXTENSIONS,
    USER_AGENT,
    ImageResult,
)
from database import (
    load_classification_cache,
    save_classification_cache,
    write_results_db,
)
from fetchers import JsRenderer, fetch_html, fetch_image_bytes, shared_session
from parsers import (
    extract_images,
    extract_links,
    extract_listing_item_links,
    extract_pagination_page_count,
    normalize_page_url,
)


# ---------------------------------------------------------------------------
# Core crawl orchestration
# ---------------------------------------------------------------------------

def crawl_site(
    start_url: str,
    run_id: str = "",
    render_js: bool = True,
    heartbeat_seconds: float = 10.0,
    fast_mode: bool = False,
    turbo_mode: bool = False,
    table_score_threshold: float = DEFAULT_TABLE_SCORE_THRESHOLD,
    flag_uncertain: bool = False,
    ocr_workers: int = 1,
    crawl_mode: str = "page",
    max_pages: int = 40,
    progress_callback: Optional[Callable[[dict], None]] = None,
    target_urls: Optional[Sequence[str]] = None,
    listing_urls: Optional[Sequence[str]] = None,
) -> List[ImageResult]:
    """
    Crawl *start_url* and classify every image found.

    Returns a list of ImageResult objects (each carries page_url, image_url,
    label, score, reason, and image_hash).  Results are also persisted to the
    SQLite database when *run_id* is provided.
    """
    normalized_start = normalize_page_url(start_url)
    if normalized_start is None:
        raise ValueError("Invalid start URL. Example: https://example.com")

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    renderer: Optional[JsRenderer] = None
    if render_js:
        renderer = JsRenderer()

    visited_images: set = set()
    results: List[ImageResult] = []
    cache_hits = 0
    start_time = time.time()
    last_heartbeat = start_time

    classification_cache = load_classification_cache(
        fast_mode,
        turbo_mode,
        table_score_threshold=table_score_threshold,
        flag_uncertain=flag_uncertain,
    )

    if crawl_mode == "site":
        scan_scope = "whole_site"
    elif crawl_mode == "urls":
        scan_scope = "specified_urls"
    elif crawl_mode == "paginated_listing":
        scan_scope = "paginated_listing"
    else:
        scan_scope = "single_page"
    start_msg = (
        f"[status] scan started | page={start_url} | scope={scan_scope} | render_js={render_js}"
    )
    print(start_msg, flush=True)
    if progress_callback is not None:
        progress_callback({"event": "status", "message": start_msg})

    parsed_start = urlparse(normalized_start)
    domain = parsed_start.netloc
    pages_to_scan: List[str] = []
    page_html: Dict[str, str] = {}
    page_image_pairs: List[Tuple[str, str]] = []

    if crawl_mode == "site":
        queue: List[str] = [normalized_start]
        seen_pages = {normalized_start}
        while queue and len(pages_to_scan) < max(1, int(max_pages)):
            page_url = queue.pop(0)
            if re.search(r"([?&]|/)(page|p)=(\d+)", page_url, flags=re.IGNORECASE):
                continue
            _queue_path_lower = urlparse(page_url).path.lower()
            _queue_ext = Path(_queue_path_lower).suffix
            if _queue_ext and _queue_ext in NON_HTML_EXTENSIONS:
                continue
            fetch_msg = f"[status] fetching page -> {page_url}"
            print(fetch_msg, flush=True)
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "status",
                        "message": fetch_msg,
                        "phase": "fetching_page",
                        "page_url": page_url,
                        "pages_scanned_so_far": len(pages_to_scan),
                        "queue_remaining": len(queue),
                    }
                )
            if renderer is not None:
                html = renderer.fetch(page_url)
            else:
                html = fetch_html(session, page_url)
            if html is None:
                print(f"[warn] failed fetch: {page_url}")
                continue
            pages_to_scan.append(page_url)
            page_html[page_url] = html
            links = extract_links(page_url, html, domain)
            print(f"[debug] {page_url} produced {len(links)} links")
            for link in links:
                print(f"    -> {link}")
                if link in seen_pages:
                    print(f"       SKIPPED (already seen)")
                    continue
                seen_pages.add(link)
                queue.append(link)
                print(f"       ADDED TO QUEUE")

    elif crawl_mode == "urls":
        normalized_targets: List[str] = []
        seen_targets: set = set()
        for raw_url in target_urls or []:
            normalized_target = normalize_page_url(raw_url)
            if normalized_target is None or normalized_target in seen_targets:
                continue
            seen_targets.add(normalized_target)
            normalized_targets.append(normalized_target)

        if not normalized_targets:
            raise ValueError("No valid URLs provided for crawl mode 'urls'.")

        total_targets = len(normalized_targets)
        for idx, target_url in enumerate(normalized_targets, start=1):
            fetch_msg = f"[status] fetching target page -> {target_url}"
            print(fetch_msg, flush=True)
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "status",
                        "message": fetch_msg,
                        "phase": "fetching_target_page",
                        "page_url": target_url,
                        "page_index": idx,
                        "page_total": total_targets,
                    }
                )
            if renderer is not None:
                html = renderer.fetch(target_url)
            else:
                html = fetch_html(session, target_url)
            if html is None:
                print(f"[warn] failed fetch: {target_url}")
                continue
            pages_to_scan.append(target_url)
            page_html[target_url] = html

    elif crawl_mode == "paginated_listing":
        all_listing_base_urls: List[str] = [normalized_start]
        seen_listing_urls: set = {normalized_start}
        for raw_extra in listing_urls or []:
            norm_extra = normalize_page_url(raw_extra)
            if norm_extra and norm_extra not in seen_listing_urls:
                seen_listing_urls.add(norm_extra)
                all_listing_base_urls.append(norm_extra)

        listing_page_htmls: Dict[str, str] = {}
        item_links_seen: set = set()
        item_links: List[str] = []

        def _harvest_items(listing_url: str, lhtml: str) -> None:
            parsed_lu = urlparse(listing_url)
            lu_domain = parsed_lu.netloc
            for link in extract_listing_item_links(listing_url, lhtml, lu_domain):
                if link not in item_links_seen:
                    item_links_seen.add(link)
                    item_links.append(link)

        def _fetch_page(url: str) -> Optional[str]:
            if renderer is not None:
                return renderer.fetch(url)
            return fetch_html(session, url)

        total_listings = len(all_listing_base_urls)
        for listing_idx, base_listing_url in enumerate(all_listing_base_urls, start=1):
            listing_label = (
                f"listing {listing_idx}/{total_listings}" if total_listings > 1 else "listing"
            )
            fetch_msg = (
                f"[status] paginated_listing: fetching {listing_label} page 1 -> {base_listing_url}"
            )
            print(fetch_msg, flush=True)
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "status",
                        "message": fetch_msg,
                        "phase": "fetching_listing_page",
                        "page_url": base_listing_url,
                        "page_index": 1,
                        "listing_index": listing_idx,
                        "listing_total": total_listings,
                    }
                )

            first_html = _fetch_page(base_listing_url)
            if first_html is None:
                warn_msg = (
                    f"[warn] paginated_listing: failed to fetch first page of {listing_label}: "
                    f"{base_listing_url}; skipping this listing"
                )
                print(warn_msg, flush=True)
                if progress_callback is not None:
                    progress_callback({"event": "status", "message": warn_msg})
                continue

            total_pages = extract_pagination_page_count(first_html)
            if total_pages is None or total_pages < 1:
                total_pages = 1
                print(
                    f"[info] paginated_listing ({listing_label}): no pagination label found; "
                    "treating as 1 page",
                    flush=True,
                )
            else:
                print(
                    f"[info] paginated_listing ({listing_label}): detected {total_pages} page(s)",
                    flush=True,
                )

            total_pages = min(total_pages, max(1, int(max_pages)))
            listing_page_htmls[base_listing_url] = first_html
            _harvest_items(base_listing_url, first_html)

            for page_num in range(2, total_pages + 1):
                parsed_base = urlparse(base_listing_url)
                existing_qs = dict(parse_qsl(parsed_base.query, keep_blank_values=True))
                existing_qs["page"] = str(page_num)
                listing_page_url = urlunparse(parsed_base._replace(query=urlencode(existing_qs)))
                fetch_msg = (
                    f"[status] paginated_listing: fetching {listing_label} "
                    f"page {page_num}/{total_pages} -> {listing_page_url}"
                )
                print(fetch_msg, flush=True)
                if progress_callback is not None:
                    progress_callback(
                        {
                            "event": "status",
                            "message": fetch_msg,
                            "phase": "fetching_listing_page",
                            "page_url": listing_page_url,
                            "page_index": page_num,
                            "page_total": total_pages,
                            "listing_index": listing_idx,
                            "listing_total": total_listings,
                        }
                    )
                lhtml = _fetch_page(listing_page_url)
                if lhtml is None:
                    print(
                        f"[warn] paginated_listing: failed fetch {listing_label} "
                        f"page {page_num}: {listing_page_url}",
                        flush=True,
                    )
                    continue
                listing_page_htmls[listing_page_url] = lhtml
                _harvest_items(listing_page_url, lhtml)

        if not listing_page_htmls:
            fail_msg = "[status] paginated_listing: failed to fetch any listing page; stopping"
            print(fail_msg, flush=True)
            if progress_callback is not None:
                progress_callback({"event": "error", "message": fail_msg})
            if renderer is not None:
                renderer.close()
            return []

        print(
            f"[info] paginated_listing: discovered {len(item_links)} unique item link(s) "
            f"across {len(listing_page_htmls)} listing page(s) "
            f"({total_listings} base URL(s))",
            flush=True,
        )

        for lurl, lhtml in listing_page_htmls.items():
            pages_to_scan.append(lurl)
            page_html[lurl] = lhtml

        total_items = len(item_links)
        for idx, item_url in enumerate(item_links, start=1):
            fetch_msg = (
                f"[status] paginated_listing: fetching item page {idx}/{total_items} -> {item_url}"
            )
            print(fetch_msg, flush=True)
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "status",
                        "message": fetch_msg,
                        "phase": "fetching_item_page",
                        "page_url": item_url,
                        "page_index": idx,
                        "page_total": total_items,
                    }
                )
            ihtml = _fetch_page(item_url)
            if ihtml is None:
                print(f"[warn] paginated_listing: failed fetch item page: {item_url}", flush=True)
                continue
            pages_to_scan.append(item_url)
            page_html[item_url] = ihtml

    else:
        # Single-page mode
        fetch_msg = f"[status] fetching target page -> {normalized_start}"
        print(fetch_msg, flush=True)
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "status",
                    "message": fetch_msg,
                    "phase": "fetching_target_page",
                    "page_url": normalized_start,
                    "page_index": 1,
                    "page_total": 1,
                }
            )
        if renderer is not None:
            html = renderer.fetch(normalized_start)
        else:
            html = fetch_html(session, normalized_start)
        if html is None:
            fail_msg = "[status] page fetch failed or non-html; stopping"
            print(fail_msg, flush=True)
            if progress_callback is not None:
                progress_callback({"event": "error", "message": fail_msg})
            if renderer is not None:
                renderer.close()
            return []
        pages_to_scan = [normalized_start]
        page_html[normalized_start] = html

    if not pages_to_scan:
        fail_msg = "[status] no crawlable pages found on target; stopping"
        print(fail_msg, flush=True)
        if progress_callback is not None:
            progress_callback({"event": "error", "message": fail_msg})
        if renderer is not None:
            renderer.close()
        return []

    for page_url in pages_to_scan:
        image_urls = extract_images(page_url, page_html[page_url])
        for image_url in image_urls:
            page_image_pairs.append((page_url, image_url))

    found_msg = (
        f"[status] found {len(page_image_pairs)} raw image candidates "
        f"across {len(pages_to_scan)} page(s)"
    )
    print(found_msg, flush=True)

    unique_image_items: List[Tuple[str, str]] = []
    for page_url, image_url in page_image_pairs:
        if image_url in visited_images:
            continue
        visited_images.add(image_url)
        unique_image_items.append((page_url, image_url))

    unique_total = len(unique_image_items)
    unique_msg = f"[status] deduplicated to {unique_total} unique image URL(s)"
    print(unique_msg, flush=True)
    if progress_callback is not None:
        progress_callback(
            {
                "event": "discovered",
                "message": f"{found_msg} | {unique_msg}",
                "total_candidates": len(page_image_pairs),
                "unique_candidates": unique_total,
                "pages_scanned": len(pages_to_scan),
            }
        )

    cache_lock = threading.Lock()
    in_flight_hashes: Dict[str, threading.Event] = {}
    workers = max(1, int(ocr_workers))

    def process_image(
        page_url: str, image_url: str
    ) -> Optional[Tuple[str, str, str, float, str, str, bool]]:
        """Returns (page_url, image_url, label, score, reason, image_hash, was_cached)."""
        local_session = shared_session()
        content, fetch_reason = fetch_image_bytes(
            local_session, image_url, page_url, turbo_mode=turbo_mode
        )
        if not content:
            return page_url, image_url, "normal", 0.0, fetch_reason, "", False

        image_hash = hashlib.sha1(content).hexdigest()
        owns_classification = False
        wait_event: Optional[threading.Event] = None

        with cache_lock:
            cached = classification_cache.get(image_hash)
            if cached is None:
                wait_event = in_flight_hashes.get(image_hash)
                if wait_event is None:
                    wait_event = threading.Event()
                    in_flight_hashes[image_hash] = wait_event
                    owns_classification = True

        if cached is not None:
            return page_url, image_url, cached[0], cached[1], cached[2], image_hash, True

        if not owns_classification and wait_event is not None:
            wait_event.wait()
            with cache_lock:
                cached_after_wait = classification_cache.get(image_hash)
            if cached_after_wait is not None:
                return (
                    page_url,
                    image_url,
                    cached_after_wait[0],
                    cached_after_wait[1],
                    cached_after_wait[2],
                    image_hash,
                    True,
                )
            owns_classification = True
            with cache_lock:
                replacement_event = in_flight_hashes.get(image_hash)
                if replacement_event is None or replacement_event.is_set():
                    wait_event = threading.Event()
                    in_flight_hashes[image_hash] = wait_event
                else:
                    wait_event = replacement_event

        try:
            label, score, reason = classify_image(
                content,
                image_url=image_url,
                fast_mode=fast_mode,
                turbo_mode=turbo_mode,
                table_score_threshold=table_score_threshold,
                flag_uncertain=flag_uncertain,
            )
            if label == "table" and score <= table_score_threshold:
                label = "normal"
                reason = f"{reason}_guarded_threshold"
            with cache_lock:
                classification_cache[image_hash] = (label, score, reason)
            return page_url, image_url, label, score, reason, image_hash, False
        finally:
            if wait_event is not None and owns_classification:
                with cache_lock:
                    current_event = in_flight_hashes.get(image_hash)
                    if current_event is wait_event:
                        in_flight_hashes.pop(image_hash, None)
                wait_event.set()

    completed = 0
    pending: Dict[concurrent.futures.Future, str] = {}
    stalled_seconds = 0.0
    max_stalled_seconds = max(45.0, float(heartbeat_seconds) * 6.0)
    skipped_due_to_stall = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for page_url, image_url in unique_image_items:
            future = executor.submit(process_image, page_url, image_url)
            pending[future] = image_url

        active_futures = set(pending.keys())
        while active_futures:
            done, not_done = concurrent.futures.wait(
                active_futures,
                timeout=float(heartbeat_seconds),
                return_when=concurrent.futures.FIRST_COMPLETED,
            )

            if not done:
                stalled_seconds += float(heartbeat_seconds)
            else:
                stalled_seconds = 0.0

            for future in done:
                active_futures.discard(future)
                completed += 1
                outcome = future.result()
                if outcome is not None:
                    p_url, i_url, label, score, reason, image_hash, was_cached = outcome
                    if was_cached:
                        cache_hits += 1
                    results.append(ImageResult(p_url, i_url, label, score, reason, image_hash))
                    if progress_callback is not None:
                        progress_callback(
                            {
                                "event": "image_processed",
                                "processed": completed,
                                "total": len(unique_image_items),
                                "page_url": p_url,
                                "label": label,
                                "score": score,
                                "reason": reason,
                                "image_url": i_url,
                                "cached": was_cached,
                            }
                        )

            now = time.time()
            if now - last_heartbeat >= heartbeat_seconds:
                elapsed = now - start_time
                print(
                    f"[heartbeat] running {elapsed:.1f}s | "
                    f"processed={completed}/{len(unique_image_items)} | "
                    f"classified={len(results)}",
                    flush=True,
                )
                if progress_callback is not None:
                    progress_callback(
                        {
                            "event": "heartbeat",
                            "elapsed_seconds": elapsed,
                            "processed": completed,
                            "total": len(unique_image_items),
                            "classified": len(results),
                            "cache_hits": cache_hits,
                        }
                    )
                last_heartbeat = now

            if stalled_seconds >= max_stalled_seconds and active_futures:
                skipped_due_to_stall = len(active_futures)
                warn_msg = (
                    f"[status] classification stalled for {stalled_seconds:.0f}s; "
                    f"skipping {skipped_due_to_stall} remaining image(s)"
                )
                print(warn_msg, flush=True)
                if progress_callback is not None:
                    progress_callback({"event": "error", "message": warn_msg})
                for future in list(active_futures):
                    future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                break

    if renderer is not None:
        renderer.close()

    save_classification_cache(
        classification_cache,
        fast_mode,
        turbo_mode,
        table_score_threshold=table_score_threshold,
        flag_uncertain=flag_uncertain,
    )

    # Persist results to the database.
    if run_id:
        write_results_db(run_id, results)

    total_elapsed = time.time() - start_time
    done_msg = (
        f"[status] scan finished in {total_elapsed:.1f}s | pages={len(pages_to_scan)} "
        f"| raw_candidates={len(page_image_pairs)} | unique_images={unique_total} "
        f"| processed={len(results)} | skipped_stalled={skipped_due_to_stall} "
        f"| cache_hits={cache_hits}"
    )
    print(done_msg, flush=True)
    if progress_callback is not None:
        progress_callback(
            {
                "event": "finished",
                "message": done_msg,
                "elapsed_seconds": total_elapsed,
                "images": len(results),
                "pages_scanned": len(pages_to_scan),
                "raw_candidates": len(page_image_pairs),
                "unique_candidates": unique_total,
                "processed": len(results),
                "skipped_due_to_stall": skipped_due_to_stall,
                "cache_hits": cache_hits,
            }
        )

    return results


# ---------------------------------------------------------------------------
# CSV export  (for CLI download; not the primary storage)
# ---------------------------------------------------------------------------

def write_results_csv(path: str, rows: List[ImageResult]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["page_url", "image_url", "label", "score", "reason"])
        for row in rows:
            writer.writerow(
                [row.page_url, row.image_url, row.label, f"{row.score:.4f}", row.reason]
            )


# ---------------------------------------------------------------------------
# Interactive prompt helper
# ---------------------------------------------------------------------------

def ask_yes_no(prompt: str, default: bool) -> bool:
    suffix = "Y/n" if default else "y/N"
    raw = input(f"{prompt} [{suffix}]: ").strip().lower()
    if not raw:
        return default
    return raw in {"y", "yes"}


def configure_run_from_prompt() -> Tuple[str, bool, str, float, bool, bool, int, str, int, List[str], List[str]]:
    print("Interactive mode: enter options for this run.")
    url = input("Target URL: ").strip()
    while not url:
        url = input("Target URL (required): ").strip()

    scan_whole_site = ask_yes_no(
        "Scan whole website (same domain) instead of only this page?", False
    )
    crawl_mode = "site" if scan_whole_site else "page"
    target_urls: List[str] = []
    if crawl_mode == "page" and ask_yes_no("Scan multiple specific URLs only?", False):
        crawl_mode = "urls"
        print("Paste URLs one per line. Submit an empty line when done.")
        while True:
            candidate = input("URL: ").strip()
            if not candidate:
                break
            target_urls.append(candidate)
    if crawl_mode == "page" and ask_yes_no(
        "Crawl a paginated news/events listing (follows all cards across pages)?", False
    ):
        crawl_mode = "paginated_listing"
    listing_urls: List[str] = []
    if crawl_mode == "paginated_listing" and ask_yes_no(
        "Add more paginated listing URLs to crawl together?", False
    ):
        print(
            "Paste listing URLs one per line (in addition to the target URL above). "
            "Submit an empty line when done."
        )
        while True:
            candidate = input("Listing URL: ").strip()
            if not candidate:
                break
            listing_urls.append(candidate)
    max_pages = 40
    if crawl_mode in {"site"}:
        max_pages_raw = input("Max pages to scan on this site (default 40)").strip()
        if max_pages_raw:
            try:
                max_pages = max(1, int(max_pages_raw))
            except ValueError:
                max_pages = 40

    render_js = ask_yes_no(
        "Use JavaScript rendering (recommended for modern sites)?", True
    )
    fast_mode = ask_yes_no(
        "Enable fast OCR mode (about 2x faster, slightly less accurate)?", False
    )
    turbo_mode = ask_yes_no(
        "Enable TURBO mode (much faster, lower OCR quality + aggressive skipping)?", True
    )
    heartbeat_raw = input("Heartbeat seconds (default 10): ").strip()
    heartbeat_seconds = 10.0
    if heartbeat_raw:
        try:
            heartbeat_seconds = max(1.0, float(heartbeat_raw))
        except ValueError:
            heartbeat_seconds = 10.0
    default_workers = max(1, min(4, (os.cpu_count() or 4)))
    workers_raw = input(f"OCR workers (default {default_workers}): ").strip()
    ocr_workers = default_workers
    if workers_raw:
        try:
            ocr_workers = max(1, int(workers_raw))
        except ValueError:
            ocr_workers = default_workers

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / run_id
    output_path = str(run_dir / "results.csv")
    return (
        url,
        render_js,
        output_path,
        heartbeat_seconds,
        fast_mode,
        turbo_mode,
        ocr_workers,
        crawl_mode,
        max_pages,
        target_urls,
        listing_urls,
    )


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scan one webpage, discover images, and classify each as normal or table screenshot."
        )
    )
    parser.add_argument("url", nargs="?", help="Target page URL, e.g. https://example.com/page")
    parser.add_argument("--output", default="", help="Output CSV path")
    parser.add_argument(
        "--render-js",
        action="store_true",
        help="Render pages with JavaScript (needed for some websites that lazy-load images).",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=10.0,
        help="How often to print a running heartbeat line.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Enable faster OCR mode by halving max OCR image size (may reduce accuracy).",
    )
    parser.add_argument(
        "--turbo",
        action="store_true",
        help=(
            "Aggressive speed mode: lower OCR quality, skip obvious non-tables, "
            "and fail image fetches faster."
        ),
    )
    parser.add_argument(
        "--ocr-workers",
        type=int,
        default=max(1, min(4, (os.cpu_count() or 4))),
        help="Number of parallel OCR workers to use.",
    )
    parser.add_argument(
        "--crawl-mode",
        choices=["page", "site", "urls", "paginated_listing"],
        default="page",
        help=(
            "Scan one page, crawl same-domain pages across the site, "
            "scan only specified URLs, or crawl a paginated listing."
        ),
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=40,
        help="Maximum pages to scan for --crawl-mode=site.",
    )
    parser.add_argument(
        "--target-urls",
        nargs="+",
        default=[],
        help="List of exact URLs to scan when --crawl-mode=urls.",
    )
    parser.add_argument(
        "--listing-urls",
        nargs="+",
        default=[],
        help=(
            "Additional paginated listing base URLs to crawl alongside the main URL "
            "when --crawl-mode=paginated_listing."
        ),
    )
    args = parser.parse_args()

    if args.url:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path("runs") / run_id
        output_path = args.output or str(run_dir / "results.csv")
        target_url = args.url
        render_js = args.render_js
        heartbeat_seconds = args.heartbeat_seconds
        fast_mode = args.fast
        turbo_mode = args.turbo
        ocr_workers = args.ocr_workers
        crawl_mode = args.crawl_mode
        max_pages = args.max_pages
        target_urls = args.target_urls
        listing_urls = args.listing_urls
    else:
        (
            target_url,
            render_js,
            output_path,
            heartbeat_seconds,
            fast_mode,
            turbo_mode,
            ocr_workers,
            crawl_mode,
            max_pages,
            target_urls,
            listing_urls,
        ) = configure_run_from_prompt()
        run_id = Path(output_path).parent.name  # e.g. "20240101_120000"

    print(f"[run] results CSV: {output_path}", flush=True)
    print(f"[run] results DB:  .crawler.db", flush=True)
    print(f"[run] fast mode:   {fast_mode}", flush=True)
    print(f"[run] turbo mode:  {turbo_mode}", flush=True)
    print(f"[run] ocr workers: {ocr_workers}", flush=True)
    print(f"[run] crawl mode:  {crawl_mode}", flush=True)
    if crawl_mode in {"site", "paginated_listing"}:
        print(f"[run] max pages:   {max_pages}", flush=True)
    if crawl_mode == "urls":
        print(f"[run] target urls: {len(target_urls)}", flush=True)
    if crawl_mode == "paginated_listing" and listing_urls:
        print(f"[run] extra listing urls: {len(listing_urls)}", flush=True)

    rows = crawl_site(
        start_url=target_url,
        run_id=run_id,
        render_js=render_js,
        heartbeat_seconds=heartbeat_seconds,
        fast_mode=fast_mode,
        turbo_mode=turbo_mode,
        ocr_workers=ocr_workers,
        crawl_mode=crawl_mode,
        max_pages=max_pages,
        target_urls=None if crawl_mode != "urls" else target_urls,
        listing_urls=listing_urls if crawl_mode == "paginated_listing" else None,
    )
    # Write CSV for easy inspection; DB write already happened inside crawl_site.
    write_results_csv(output_path, rows)

    table_count = sum(1 for r in rows if r.label == "table")
    normal_count = len(rows) - table_count
    print(f"Processed images: {len(rows)}")
    print(f"Table:  {table_count}")
    print(f"Normal: {normal_count}")
    print(f"CSV results: {output_path}")
    print(f"DB results:  .crawler.db  (run_id={run_id})")


if __name__ == "__main__":
    main()
