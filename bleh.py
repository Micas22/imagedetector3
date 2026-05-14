import argparse
import csv
import concurrent.futures
import hashlib
import io
import os
import re
import threading
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import cv2
import numpy as np
import requests
from bs4 import BeautifulSoup
from requests.exceptions import SSLError
from table_transformer_adapter import classify_with_table_transformer

try:
    from urllib3.exceptions import InsecureRequestWarning
except Exception:  # Optional; used only to silence verify=False warning.
    InsecureRequestWarning = None

try:
    from playwright.sync_api import sync_playwright
except Exception:  # Optional dependency unless JS rendering is enabled.
    sync_playwright = None

try:
    from paddleocr import PaddleOCR
except Exception:  # Optional dependency.
    PaddleOCR = None


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff")
NON_HTML_EXTENSIONS = (
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".rar", ".7z", ".tar", ".gz",
    ".mp3", ".mp4", ".avi", ".mov", ".wmv", ".flv",
    ".exe", ".dmg", ".pkg", ".deb", ".rpm",
    ".csv", ".json", ".xml", ".txt", ".rtf",
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".svg", ".ico",
)
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
DEFAULT_TABLE_SCORE_THRESHOLD = 0.85
CLASSIFIER_VERSION = "v69"
TABLE_URL_HINT_WORDS = ("table", "tabela")
CACHE_FILENAME = ".crawler_classification_cache.csv"
TABLE_WORDS = {
    "total",
    "subtotal",
    "amount",
    "price",
    "qty",
    "quantity",
    "date",
    "description",
    "item",
    "balance",
    "invoice",
    "debit",
    "credit",
    "fase",
    "tipo",
    "datas",
    "valor",
    "valores",
    "descricao",
    "quantidade",
}
_ocr_lock = threading.Lock()
_ocr_engine = None
_ocr_mkldnn_disabled = False
_thread_local = threading.local()


@dataclass
class ImageResult:
    page_url: str
    image_url: str
    label: str
    score: float
    reason: str


def normalize_image_url(page_url: str, src: str) -> Optional[str]:
    if not src:
        return None
    src = src.strip()
    if src.startswith("data:"):
        return None
    full = urljoin(page_url, src)
    parsed = urlparse(full)
    if parsed.scheme not in ("http", "https"):
        return None
    # Accept only URLs whose path ends with a known image extension.
    # Reject everything else (plain-text strings, page paths, etc.).
    if not parsed.path.lower().endswith(IMAGE_EXTENSIONS):
        return None
    return full


def extract_images(page_url: str, html: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    images: List[str] = []

    for img in soup.find_all("img"):
        for attr in ("src", "data-src", "data-original", "data-lazy-src"):
            normalized = normalize_image_url(page_url, img.get(attr, ""))
            if normalized:
                images.append(normalized)

        for srcset_attr in ("srcset", "data-srcset"):
            srcset = img.get(srcset_attr)
            if srcset:
                for item in srcset.split(","):
                    candidate = item.strip().split(" ")[0]
                    normalized_set = normalize_image_url(page_url, candidate)
                    if normalized_set:
                        images.append(normalized_set)

    for source in soup.find_all("source"):
        for srcset_attr in ("srcset", "data-srcset"):
            srcset = source.get(srcset_attr)
            if srcset:
                for item in srcset.split(","):
                    candidate = item.strip().split(" ")[0]
                    normalized_set = normalize_image_url(page_url, candidate)
                    if normalized_set:
                        images.append(normalized_set)

    for tag in soup.find_all(style=True):
        style = tag.get("style") or ""
        for match in re.findall(r"url\(([^)]+)\)", style, flags=re.IGNORECASE):
            cleaned = match.strip(" '\"")
            normalized = normalize_image_url(page_url, cleaned)
            if normalized:
                images.append(normalized)

    # Only scrape Open Graph / Twitter image meta tags — not all meta content.
    for meta in soup.find_all("meta", attrs={"content": True}):
        prop = (meta.get("property") or meta.get("name") or "").lower()
        if prop not in ("og:image", "twitter:image", "og:image:url", "twitter:image:src"):
            continue
        content = meta.get("content", "")
        normalized = normalize_image_url(page_url, content)
        if normalized:
            images.append(normalized)

    return images


def extract_links(page_url: str, html: str, domain: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: List[str] = []
    for anchor in soup.find_all("a", href=True):
        href = (anchor.get("href") or "").strip()
        if not href or href.startswith("#") or href.startswith("mailto:") or href.startswith("tel:"):
            continue
        full = urljoin(page_url, href)
        parsed = urlparse(full)
        if parsed.scheme not in ("http", "https"):
            continue
        # Skip non-HTML resources (PDFs, documents, archives, media, images, etc.)
        link_path_lower = (parsed.path or "").lower()
        link_ext = Path(link_path_lower).suffix
        if link_ext and link_ext in NON_HTML_EXTENSIONS:
            continue
        def normalize_domain(netloc: str) -> str:
            return netloc.lower().replace("www.", "")

        if normalize_domain(parsed.netloc) != normalize_domain(domain):
            continue

        path = parsed.path or '/'
        if path != '/':
            path = path.rstrip('/')

        normalized = f"{parsed.scheme}://{parsed.netloc}{path}"
        if parsed.query:
            normalized = f"{normalized}?{parsed.query}"
        links.append(normalized)
    return links

def _normalize_netloc(netloc: str) -> str:
    """Strip www. and lowercase for loose domain comparison."""
    return netloc.lower().lstrip("www.")


def extract_listing_item_links(listing_url: str, html: str, domain: str) -> List[str]:
    parsed_listing = urlparse(listing_url)
    listing_path = parsed_listing.path or "/"
    listing_root = f"{parsed_listing.scheme}://{parsed_listing.netloc}{listing_path}"
    norm_domain = _normalize_netloc(domain)
    soup = BeautifulSoup(html, "html.parser")
    links: List[str] = []
    seen = set()

    for anchor in soup.find_all("a", href=True):
        href = (anchor.get("href") or "").strip()
        if not href or href.startswith("#") or href.startswith("mailto:") or href.startswith("tel:"):
            continue
        full = urljoin(listing_url, href)
        parsed = urlparse(full)
        if parsed.scheme not in ("http", "https"):
            continue
        # Skip non-HTML resources (PDFs, documents, archives, media, etc.)
        link_path_lower = (parsed.path or "").lower()
        link_ext = Path(link_path_lower).suffix
        if link_ext and link_ext in NON_HTML_EXTENSIONS:
            continue
        # Allow www/non-www variants of the same domain.
        if _normalize_netloc(parsed.netloc) != norm_domain:
            continue
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path or '/'}"
        if parsed.query:
            normalized = f"{normalized}?{parsed.query}"

        # Skip self links (the listing page itself).
        if normalized == listing_url or normalized == listing_root:
            continue
        # Skip pagination controls (?page=N on the same path).
        query_map = dict(parse_qsl(parsed.query, keep_blank_values=True))
        if (parsed.path or "/") == listing_path and "page" in {k.lower() for k in query_map.keys()}:
            continue
        # Skip the bare domain root — it's never a listing item.
        if (parsed.path or "/") in ("/", ""):
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        links.append(normalized)
    return links

def extract_pagination_page_count(html: str) -> int | None:
    soup = BeautifulSoup(html, "html.parser")

    # ── PASS 1: ".pagination--select__label" with "de N" text ─────────────────
    for label in soup.select(".pagination--select__label"):
        text = label.get_text(" ", strip=True)
        print(f"[debug] pagination--select__label =", repr(text))
        match = re.search(r"de\s+(\d+)", text, re.IGNORECASE)
        if match:
            total = int(match.group(1))
            print(f"[debug] extracted total_pages (de N label) = {total}")
            return total

    # ── PASS 2: any pagination__item <a> that is just a digit ─────────────────
    page_numbers = [
        int(a.get_text(strip=True))
        for a in soup.select("li.pagination__item a")
        if a.get_text(strip=True).isdigit()
    ]
    if page_numbers:
        total = max(page_numbers)
        print(f"[debug] extracted total_pages (pagination__item digits) = {total}")
        return total

    # ── PASS 3: rel="last" link  <a rel="last" href="?page=N"> ───────────────
    last_link = soup.find("a", rel=lambda v: v and "last" in v)
    if last_link:
        href = last_link.get("href", "")
        m = re.search(r"[?&]page=(\d+)", href, re.IGNORECASE)
        if m:
            total = int(m.group(1))
            print(f"[debug] extracted total_pages (rel=last) = {total}")
            return total

    # ── PASS 4: aria-label="Page N" / aria-label="Go to page N" buttons ──────
    aria_pages = []
    for tag in soup.find_all(attrs={"aria-label": True}):
        m = re.search(r"\bpage\s+(\d+)\b", tag["aria-label"], re.IGNORECASE)
        if m:
            aria_pages.append(int(m.group(1)))
    if aria_pages:
        total = max(aria_pages)
        print(f"[debug] extracted total_pages (aria-label page N) = {total}")
        return total

    # ── PASS 5: generic pagination container — collect all visible digit links ─
    # Look inside common pagination wrappers for the highest digit.
    pag_selectors = [
        "nav[aria-label*='agina' i]",   # "Paginação", "Pagination", …
        "nav[aria-label*='pagin' i]",
        ".pagination", ".pager", ".paginator",
        "[class*='pagination']", "[class*='pager']",
        "ul.pages", "ol.pages",
    ]
    for sel in pag_selectors:
        container = soup.select_one(sel)
        if not container:
            continue
        nums = [
            int(t)
            for tag in container.find_all(["a", "span", "button", "li"])
            for t in [tag.get_text(strip=True)]
            if t.isdigit() and int(t) > 0
        ]
        if nums:
            total = max(nums)
            print(f"[debug] extracted total_pages (container '{sel}') = {total}")
            return total

    # ── PASS 6: "?page=N" / "/page/N" in any href on the page ────────────────
    all_page_nums = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # query-string style: ?page=38
        m = re.search(r"[?&]page=(\d+)", href, re.IGNORECASE)
        if m:
            all_page_nums.append(int(m.group(1)))
            continue
        # path style: /page/38  or  /p/38
        m = re.search(r"/p(?:age)?/(\d+)", href, re.IGNORECASE)
        if m:
            all_page_nums.append(int(m.group(1)))
    if all_page_nums:
        total = max(all_page_nums)
        print(f"[debug] extracted total_pages (href scan) = {total}")
        return total

    # ── PASS 7: plain text patterns like "Page 1 of 38" / "1 de 38" ──────────
    full_text = soup.get_text(" ", strip=True)
    for pattern in [
        r"\bof\s+(\d+)\b",          # "Page 1 of 38"
        r"\bde\s+(\d+)\b",          # "1 de 38"
        r"\bvan\s+(\d+)\b",         # Dutch "1 van 38"
        r"\bvon\s+(\d+)\b",         # German "Seite 1 von 38"
        r"\bdi\s+(\d+)\b",          # Italian "1 di 38"
        r"/\s*(\d+)\s*pages?",      # "1 / 38 pages"
    ]:
        matches = re.findall(pattern, full_text, re.IGNORECASE)
        if matches:
            total = max(int(v) for v in matches)
            if total > 1:
                print(f"[debug] extracted total_pages (text pattern '{pattern}') = {total}")
                return total

    print("[debug] could not parse page count (no patterns matched)")
    return None

def normalize_page_url(raw_url: str) -> Optional[str]:
    raw = (raw_url or "").strip()
    if not raw:
        return None
    parsed = urlparse(raw)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return None
    path = parsed.path or '/'
    if path != '/':
        path = path.rstrip('/')
    normalized = f"{parsed.scheme}://{parsed.netloc}{path}"
    if parsed.query:
        normalized = f"{normalized}?{parsed.query}"
    return normalized


def fetch_html(session: requests.Session, url: str, timeout: int = 30, retries: int = 2) -> Optional[str]:
    for attempt in range(1, retries + 1):
        try:
            response = session.get(
                url,
                timeout=(8, timeout),  # connect, read
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0 Safari/537.36"
                    )
                },
            )

            print(f"[fetch] {url} -> {response.status_code}")

            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "").lower()

            if "text/html" not in content_type:
                print(f"[skip] non-html content: {content_type} -> {url}")
                return None

            return response.text

        except requests.exceptions.Timeout:
            print(f"[timeout] attempt {attempt}/{retries} {url}")
        except requests.exceptions.HTTPError as e:
            print(f"[http error] {url} -> {e}")
            return None  # Don't retry HTTP errors (4xx/5xx)
        except requests.exceptions.RequestException as e:
            print(f"[request error] attempt {attempt}/{retries} {url} -> {e}")

        if attempt < retries:
            time.sleep(1.5 * attempt)

    return None

class JsRenderer:
    def __init__(self, timeout_ms: int = 30000) -> None:
        if sync_playwright is None:
            raise RuntimeError(
                "Playwright is not installed. Install with 'pip install playwright' and run 'playwright install chromium'."
            )

        self.timeout_ms = timeout_ms
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=True)
        self._context = self._browser.new_context(user_agent=USER_AGENT)

    def fetch(self, url: str, retries: int = 2) -> Optional[str]:
        for attempt in range(1, retries + 1):
            page = self._context.new_page()
            try:
                print(f"[js] navigating (attempt {attempt}) -> {url}")
                page.goto(
                    url,
                    wait_until="domcontentloaded",
                    timeout=self.timeout_ms,
                )
                # Give JS frameworks time to render their content.
                page.wait_for_timeout(2000)
                html = page.content()
                if not html or len(html) < 500:
                    print(f"[warn] suspicious HTML size -> {url} ({len(html) if html else 0})")
                    if attempt < retries:
                        continue
                return html
            except Exception as e:
                print(f"[js error] attempt {attempt}/{retries} {url} -> {repr(e)}")
                if attempt == retries:
                    return None
                time.sleep(1.5 * attempt)
            finally:
                page.close()
        return None

    def close(self) -> None:
        try:
            self._context.close()
        except Exception as e:
            print("[js close] context error:", repr(e))

        try:
            self._browser.close()
        except Exception as e:
            print("[js close] browser error:", repr(e))

        try:
            self._playwright.stop()
        except Exception as e:
            print("[js close] playwright stop error:", repr(e))

def _is_valid_image_bytes(content: bytes) -> bool:
    if not content:
        return False
    try:
        arr = np.frombuffer(content, dtype=np.uint8)
        if arr.size == 0:
            return False
        decoded = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        return decoded is not None
    except Exception:
        return False


def fetch_image_bytes(
    session: requests.Session,
    image_url: str,
    page_url: str,
    timeout: Tuple[float, float] = (4, 12),
    turbo_mode: bool = False,
) -> Tuple[Optional[bytes], str]:
    last_reason = "fetch_unknown_error"
    connect_timeout, read_timeout = timeout
    request_headers = {
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Referer": page_url,
        "Origin": f"{urlparse(page_url).scheme}://{urlparse(page_url).netloc}",
    }
    if turbo_mode:
        connect_timeout, read_timeout = (1.5, 3.5)
        max_attempts = 2
    else:
        max_attempts = 4
    retryable_statuses = {408, 425, 429, 500, 502, 503, 504}
    for attempt in range(1, max_attempts + 1):
        this_timeout = (
            max(1.0, connect_timeout + (attempt - 1) * 1.0),
            max(2.0, read_timeout + (attempt - 1) * 2.0),
        )
        insecure_tls_fallback_used = False
        try:
            response = session.get(image_url, timeout=this_timeout, headers=request_headers, allow_redirects=True)
        except SSLError:
            # Some hosts present cert chains that fail verification on local Python installs.
            # Retry once with TLS verification disabled as a best-effort fallback.
            try:
                with warnings.catch_warnings():
                    if InsecureRequestWarning is not None:
                        warnings.simplefilter("ignore", InsecureRequestWarning)
                    response = session.get(
                        image_url,
                        timeout=this_timeout,
                        headers=request_headers,
                        allow_redirects=True,
                        verify=False,
                    )
                insecure_tls_fallback_used = True
            except requests.RequestException:
                last_reason = "fetch_request_exception_sslerror"
                if attempt < max_attempts:
                    time.sleep(0.2 * attempt)
                continue
        except requests.Timeout:
            last_reason = "fetch_timeout"
            if attempt < max_attempts:
                time.sleep(0.2 * attempt)
            continue
        except requests.RequestException as exc:
            exc_name = exc.__class__.__name__.lower()
            last_reason = f"fetch_request_exception_{exc_name}"
            if attempt < max_attempts:
                time.sleep(0.2 * attempt)
            continue

        status = int(response.status_code)
        content_type = response.headers.get("Content-Type", "").lower()
        content = response.content or b""
        if status != 200:
            last_reason = f"fetch_http_{status}"
            if status not in retryable_statuses:
                break
        elif not content:
            last_reason = "fetch_empty_body"
        elif "image/" in content_type or content_type == "":
            if insecure_tls_fallback_used:
                return content, "ok_insecure_tls_fallback"
            return content, "ok"
        elif _is_valid_image_bytes(content):
            # Some CDNs mislabel content-type but still return an image.
            if insecure_tls_fallback_used:
                return content, "ok_content_type_mismatch_insecure_tls_fallback"
            return content, "ok_content_type_mismatch"
        else:
            sanitized_content_type = re.sub(r"[^a-z0-9+./-]", "_", content_type)[:48] or "unknown"
            last_reason = f"fetch_non_image_content_type_{sanitized_content_type}"

        if attempt < max_attempts:
            time.sleep(0.2 * attempt)

    return None, last_reason


def safe_to_cv(image_bytes: bytes) -> Optional[np.ndarray]:
    try:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        if arr.size == 0:
            return None
        decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return decoded
    except Exception:
        return None


def _get_ocr_engine():
    if PaddleOCR is None:
        return None
    engine = getattr(_thread_local, "ocr_engine", None)
    if engine is None:
        init_variants = [
            {"lang": "en", "enable_mkldnn": False, "use_textline_orientation": True},
            {"lang": "en", "enable_mkldnn": False, "use_angle_cls": True},
            {"lang": "en", "enable_mkldnn": False},
            {"lang": "en", "use_textline_orientation": True},
            {"lang": "en", "use_angle_cls": True},
            {"lang": "en"},
            {},
        ]
        for kwargs in init_variants:
            try:
                engine = PaddleOCR(**kwargs)
                break
            except (TypeError, ValueError):
                engine = None
        _thread_local.ocr_engine = engine
    return engine


def _reset_ocr_engine() -> None:
    _thread_local.ocr_engine = None


def shared_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def _disable_mkldnn_if_possible() -> None:
    """Disable MKLDNN/oneDNN to avoid Paddle runtime crashes on some Windows setups."""
    global _ocr_mkldnn_disabled
    if _ocr_mkldnn_disabled:
        return
    os.environ["FLAGS_use_mkldnn"] = "0"
    try:
        import paddle

        paddle.set_flags({"FLAGS_use_mkldnn": False})
    except Exception:
        pass
    _ocr_mkldnn_disabled = True


def _run_ocr(ocr_engine, img: np.ndarray):
    if hasattr(ocr_engine, "predict"):
        return ocr_engine.predict(img)
    if hasattr(ocr_engine, "ocr"):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*Please use `predict` instead\..*",
                category=DeprecationWarning,
            )
            return ocr_engine.ocr(img)
    raise RuntimeError("ocr_engine_missing_methods")


def _is_mkldnn_runtime_error(exc: Exception) -> bool:
    msg = str(exc)
    return (
        "ConvertPirAttribute2RuntimeAttribute" in msg
        or "onednn_instruction.cc" in msg
        or "oneDNN" in msg
    )


def _cluster_axis(values: List[float], tolerance: float) -> int:
    if not values:
        return 0
    sorted_values = sorted(values)
    clusters = 1
    anchor = sorted_values[0]
    for value in sorted_values[1:]:
        if abs(value - anchor) > tolerance:
            clusters += 1
            anchor = value
    return clusters


def _detect_table_grid_signal(img: np.ndarray) -> float:
    """Estimate whether image contains a table-like ruled region."""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except Exception:
        return 0.0

    h, w = gray.shape[:2]
    if h < 120 or w < 120:
        return 0.0

    # Adaptive threshold highlights dark ruling lines on light backgrounds.
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        4,
    )

    # Extract long horizontal and vertical line candidates.
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(12, w // 24), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(12, h // 24)))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    intersections = cv2.bitwise_and(horizontal, vertical)
    h_pixels = float(np.count_nonzero(horizontal))
    v_pixels = float(np.count_nonzero(vertical))
    x_pixels = float(np.count_nonzero(intersections))
    image_pixels = float(max(1, h * w))

    h_ratio = h_pixels / image_pixels
    v_ratio = v_pixels / image_pixels
    x_ratio = x_pixels / image_pixels

    # Count distinct line runs to avoid treating a single box as a table.
    h_proj = (np.sum(horizontal > 0, axis=1) > max(6, w * 0.02)).astype(np.uint8)
    v_proj = (np.sum(vertical > 0, axis=0) > max(6, h * 0.02)).astype(np.uint8)
    h_runs = int(np.count_nonzero((h_proj[1:] == 1) & (h_proj[:-1] == 0)))
    v_runs = int(np.count_nonzero((v_proj[1:] == 1) & (v_proj[:-1] == 0)))

    run_signal = min(1.0, (max(0, h_runs - 1) * max(0, v_runs - 1)) / 20.0)
    density_signal = min(1.0, (h_ratio + v_ratio) / 0.07)
    intersection_signal = min(1.0, x_ratio / 0.0035)

    return max(0.0, min(1.0, 0.35 * run_signal + 0.35 * density_signal + 0.30 * intersection_signal))


def alignment_consistency(centers_x, tolerance):
    centers_x = sorted(centers_x)
    clusters = []

    for x in centers_x:
        placed = False
        for c in clusters:
            if abs(x - c[0]) < tolerance:
                c.append(x)
                placed = True
                break
        if not placed:
            clusters.append([x])

    if not clusters:
        return 0.0

    sizes = [len(c) for c in clusters]
    return max(sizes) / sum(sizes)


def _flatten_ocr_lines(raw_ocr) -> List[Tuple[List[List[float]], str, float]]:
    items: List[Tuple[List[List[float]], str, float]] = []
    if not raw_ocr:
        return items

    def _coerce_quad_points(poly) -> Optional[List[List[float]]]:
        if poly is None:
            return None
        pts = poly.tolist() if hasattr(poly, "tolist") else poly
        if not isinstance(pts, (list, tuple)) or len(pts) < 4:
            return None
        out: List[List[float]] = []
        for pt in list(pts)[:4]:
            coords = pt.tolist() if hasattr(pt, "tolist") else pt
            if not isinstance(coords, (list, tuple)) or len(coords) < 2:
                return None
            out.append([float(coords[0]), float(coords[1])])
        return out if len(out) == 4 else None

    chunks = raw_ocr if isinstance(raw_ocr, list) else [raw_ocr]
    for chunk in chunks:
        if not chunk:
            continue
        # Newer PaddleOCR `predict()` often returns dict-like results.
        if isinstance(chunk, dict) or hasattr(chunk, "keys"):
            payload = chunk if isinstance(chunk, dict) else dict(chunk)
            polys = payload.get("rec_polys") or payload.get("dt_polys") or []
            texts = payload.get("rec_texts") or []
            scores = payload.get("rec_scores")
            limit = min(len(polys), len(texts))
            for i in range(limit):
                points = _coerce_quad_points(polys[i])
                text = str(texts[i]).strip()
                if scores is None:
                    conf = 1.0
                else:
                    try:
                        conf = float(scores[i])
                    except (TypeError, ValueError, IndexError):
                        conf = 0.0
                if text and points:
                    items.append((points, text, conf))
            continue

        # Legacy PaddleOCR `ocr()` returns list-based line tuples.
        for line in chunk:
            if not line or len(line) < 2:
                continue
            box, data = line[0], line[1]
            points = _coerce_quad_points(box)
            if not points:
                continue
            if not isinstance(data, (list, tuple)) or len(data) < 2:
                continue
            text = str(data[0]).strip()
            try:
                conf = float(data[1])
            except (TypeError, ValueError):
                conf = 0.0
            if text:
                items.append((points, text, conf))
    return items


def classify_image(
    image_bytes: bytes,
    image_url: str = "",
    fast_mode: bool = False,
    turbo_mode: bool = False,
    table_score_threshold: float = DEFAULT_TABLE_SCORE_THRESHOLD,
    flag_uncertain: bool = False,
) -> Tuple[str, float, str]:
    # Route table detection to Microsoft Table Transformer (detection model).
    # Keep the original function signature so webapp/crawler callers remain compatible.
    label, score, reason = classify_with_table_transformer(
        image_bytes=image_bytes,
        image_url=image_url,
        table_score_threshold=table_score_threshold,
    )
    if label == "table" and score <= table_score_threshold:
        # Preserve adapter-provided table decision even if numeric score is slightly below threshold.
        # The adapter already performs its own structure verification.
        return "normal", score, f"{reason}_guarded_threshold"
    if flag_uncertain and label != "table" and score >= max(0.0, table_score_threshold * 0.9):
        return "uncertain", score, f"{reason}_near_threshold"
    return label, score, reason

    img = safe_to_cv(image_bytes)
    if img is None:
        return "normal", 0.0, "decode_failed"

    h, w = img.shape[:2]
    if h < 120 or w < 120:
        return "normal", 0.05, "too_small"
    aspect_ratio = float(max(h, w)) / float(max(1, min(h, w)))
    if turbo_mode and aspect_ratio >= 6.5 and min(h, w) < 280:
        # Extreme thin banners/icons are very unlikely to be useful tables.
        return "normal", 0.02, "turbo_extreme_aspect_skip"
    if turbo_mode:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if float(np.std(gray)) < 14.0:
            return "normal", 0.02, "turbo_low_variance_skip"
    grid_region_signal = _detect_table_grid_signal(img)

    # Large images are much slower for OCR; downscale while preserving layout.
    if turbo_mode:
        max_ocr_side = 560 if fast_mode else 1000
    else:
        max_ocr_side = 800 if fast_mode else 1600
    max_side = max(h, w)
    if max_side > max_ocr_side:
        scale = float(max_ocr_side) / float(max_side)
        resized_w = max(1, int(w * scale))
        resized_h = max(1, int(h * scale))
        img = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    ocr_engine = _get_ocr_engine()
    if ocr_engine is None:
        return "normal", 0.0, "paddleocr_not_installed"

    # PaddleOCR runtime can hang/crash when accessed concurrently across threads.
    # Serialize OCR calls even if image downloading/classification orchestration is parallel.
    with _ocr_lock:
        try:
            ocr_result = _run_ocr(ocr_engine, img)
        except Exception as first_error:
            # Some Paddle builds crash with oneDNN enabled on CPU. Retry once after disabling it.
            if _is_mkldnn_runtime_error(first_error):
                try:
                    _disable_mkldnn_if_possible()
                    _reset_ocr_engine()
                    ocr_engine = _get_ocr_engine()
                    if ocr_engine is None:
                        return "normal", 0.0, "paddleocr_not_installed"
                    ocr_result = _run_ocr(ocr_engine, img)
                except Exception:
                    return "normal", 0.0, "ocr_failed_mkldnn_runtime"
            else:
                return "normal", 0.0, "ocr_failed"

    lines = _flatten_ocr_lines(ocr_result)
    if not lines:
        return "normal", 0.05, "ocr_no_text"

    min_conf = 0.45
    lines = [entry for entry in lines if entry[2] >= min_conf]
    if not lines:
        return "normal", 0.06, "ocr_low_confidence"

    centers_x: List[float] = []
    centers_y: List[float] = []
    alnum_tokens = 0
    numericish_tokens = 0
    table_keyword_hits = 0
    alpha_chars = 0
    uppercase_alpha_chars = 0
    long_alpha_tokens = 0
    for box, text, _conf in lines:
        x_vals = [pt[0] for pt in box]
        y_vals = [pt[1] for pt in box]
        centers_x.append((min(x_vals) + max(x_vals)) * 0.5)
        centers_y.append((min(y_vals) + max(y_vals)) * 0.5)
        text_l = text.lower()
        tokens = re.findall(r"[a-zA-Z0-9$€£%.,:/-]+", text_l)
        for token in tokens:
            alnum_tokens += 1
            if re.search(r"\d", token):
                numericish_tokens += 1
            if token.isalpha() and len(token) >= 6:
                long_alpha_tokens += 1
            if token in TABLE_WORDS:
                table_keyword_hits += 1
        for ch in text:
            if ch.isalpha():
                alpha_chars += 1
                if ch.isupper():
                    uppercase_alpha_chars += 1

    row_tolerance = max(8.0, h * 0.018)
    col_tolerance = max(8.0, w * 0.018)
    row_count = _cluster_axis(centers_y, tolerance=row_tolerance)
    col_count = _cluster_axis(centers_x, tolerance=col_tolerance)
    line_count = len(lines)
    numeric_ratio = numericish_tokens / max(1, alnum_tokens)
    uppercase_ratio = uppercase_alpha_chars / max(1, alpha_chars)
    long_alpha_ratio = long_alpha_tokens / max(1, alnum_tokens)
    alignment_signal = alignment_consistency(centers_x, tolerance=col_tolerance)
    col_stability = column_stability(
        centers_x,
        centers_y,
        col_tolerance,
        row_tolerance
    )

    grid_likeness = min(1.0, (row_count * col_count) / max(1.0, line_count * 1.4))
    structure_signal = min(1.0, (max(0, row_count - 1) * max(0, col_count - 1)) / 24.0)
    density_signal = min(1.0, line_count / 45.0)
    keyword_signal = min(1.0, table_keyword_hits / 3.0)
    numeric_signal = min(1.0, numeric_ratio / 0.55)
    stability_signal = col_stability

    score = (
        0.28 * numeric_signal +
        0.22 * keyword_signal +
        0.18 * structure_signal +
        0.10 * grid_likeness +
        0.10 * alignment_signal +
        0.10 * stability_signal +
        0.02 * density_signal
    )
    score += 0.12 * grid_region_signal
    parsed_image = urlparse(image_url or "")
    image_path_l = (parsed_image.path or "").lower()
    image_name_l = Path(image_path_l).name
    has_table_url_hint = any(
        hint_word in image_name_l or hint_word in image_path_l
        for hint_word in TABLE_URL_HINT_WORDS
    )
    if has_table_url_hint:
        # Strongly prefer "table/tabela"-named assets, but never auto-classify.
        score += 0.22
    score = max(0.0, min(1.0, score))

    # Require stronger geometric/table-like structure before labeling as table.
    has_strong_structure = (
        row_count >= 4
        and col_count >= 3
        and line_count >= 10
        and grid_likeness >= 0.42
        and structure_signal >= 0.22
    )
    has_table_content = (
        numeric_ratio >= 0.18
        or table_keyword_hits >= 2
        or (numeric_ratio >= 0.12 and numericish_tokens >= 4 and col_stability >= 0.6)
        or (has_table_url_hint and numeric_ratio >= 0.08)
    )
    # Allow mixed-content images (table + surrounding paragraph text) to pass when
    # table content signals are clear but full-frame grid structure is diluted.
    has_partial_table_structure = (
        row_count >= 3
        and col_count >= 3
        and line_count >= 7
        and col_stability >= 0.52
        and structure_signal >= 0.08
        and (numeric_ratio >= 0.12 or table_keyword_hits >= 2 or grid_region_signal >= 0.44)
    )
    has_hint_boosted_structure = (
        has_table_url_hint
        and row_count >= 2
        and col_count >= 2
        and line_count >= 6
        and col_stability >= 0.42
        and structure_signal >= 0.05
    )
    looks_like_poster_banner = (
        uppercase_ratio >= 0.68
        and long_alpha_ratio >= 0.30
        and numericish_tokens <= 3
        and table_keyword_hits == 0
        and grid_region_signal < 0.12
        and structure_signal < 0.20
    )
    is_table = (
        score > table_score_threshold
        and has_table_content
        and not (looks_like_poster_banner and not has_strong_structure)
        and (col_stability > (0.25 if has_table_url_hint else 0.35) or grid_region_signal >= 0.58)
        and (has_strong_structure or has_partial_table_structure or has_hint_boosted_structure)
    )
    uncertain_margin = max(0.003, table_score_threshold * 0.08)
    is_uncertain = (
        flag_uncertain
        and not is_table
        and score <= table_score_threshold
        and (table_score_threshold - score) <= uncertain_margin
        and has_table_content
        and col_stability > (0.22 if has_table_url_hint else 0.3)
    )
    reason = (
        f"ocr_lines{line_count}_rows{row_count}_cols{col_count}_"
        f"numratio{numeric_ratio:.3f}_keywords{table_keyword_hits}_"
        f"align{alignment_signal:.3f}_grid{grid_region_signal:.3f}_"
        f"upratio{uppercase_ratio:.3f}_longalpha{long_alpha_ratio:.3f}_"
        f"urlhint{int(has_table_url_hint)}_"
        f"score{score:.4f}_threshold{table_score_threshold:.4f}"
    )
    if is_table:
        return "table", score, reason
    if is_uncertain:
        return "uncertain", score, f"{reason}_near_threshold"
    return "normal", score, reason



def save_table_image(save_dir: Path, image_url: str, image_bytes: bytes) -> Optional[Path]:
    parsed = urlparse(image_url)
    ext = Path(parsed.path).suffix.lower()
    if ext not in IMAGE_EXTENSIONS:
        ext = ".jpg"
    digest = hashlib.sha1(image_url.encode("utf-8")).hexdigest()[:14]
    name = f"table_{digest}{ext}"
    out_path = save_dir / name
    try:
        out_path.write_bytes(image_bytes)
        return out_path
    except OSError:
        return None

def column_stability(centers_x, centers_y, col_tolerance, row_tolerance):
    """
    Strong column consistency:
    Checks if columns align at consistent x positions across rows.
    """

    if not centers_x or not centers_y:
        return 0.0

    # Step 1: cluster rows
    points = sorted(zip(centers_y, centers_x))
    rows = []
    current = [points[0][1]]
    anchor_y = points[0][0]

    for y, x in points[1:]:
        if abs(y - anchor_y) <= row_tolerance:
            current.append(x)
        else:
            rows.append(current)
            current = [x]
            anchor_y = y

    if current:
        rows.append(current)

    if len(rows) < 3:
        return 0.0  # not enough rows to be a table

    # Step 2: build global column anchors
    all_x = sorted(centers_x)
    col_anchors = []

    for x in all_x:
        placed = False
        for c in col_anchors:
            if abs(x - c) <= col_tolerance:
                placed = True
                break
        if not placed:
            col_anchors.append(x)

    if len(col_anchors) < 2:
        return 0.0

    # Step 3: check how consistently rows hit these anchors
    row_hits = []

    for row in rows:
        hits = 0
        for anchor in col_anchors:
            for x in row:
                if abs(x - anchor) <= col_tolerance:
                    hits += 1
                    break
        row_hits.append(hits / len(col_anchors))

    # Step 4: stability = consistency across rows
    mean = np.mean(row_hits)
    std = np.std(row_hits)

    stability = mean * (1.0 - std)
    return float(max(0.0, min(1.0, stability)))

def load_classification_cache(
    cache_path: Path,
    fast_mode: bool,
    turbo_mode: bool = False,
    table_score_threshold: float = DEFAULT_TABLE_SCORE_THRESHOLD,
    flag_uncertain: bool = False,
) -> dict:
    cache: dict = {}
    if not cache_path.exists():
        return cache
    try:
        with cache_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row:
                    continue
                if row.get("classifier_version") != CLASSIFIER_VERSION:
                    continue
                if row.get("threshold") != f"{table_score_threshold:.4f}":
                    continue
                if row.get("fast_mode", "0") != ("1" if fast_mode else "0"):
                    continue
                if row.get("turbo_mode", "0") != ("1" if turbo_mode else "0"):
                    continue
                if row.get("flag_uncertain", "0") != ("1" if flag_uncertain else "0"):
                    continue
                image_hash = row.get("image_hash", "")
                if not image_hash:
                    continue
                try:
                    score = float(row.get("score", "0"))
                except ValueError:
                    continue
                cache[image_hash] = (row.get("label", "normal"), score, row.get("reason", "cache_miss"))
    except OSError:
        return {}
    return cache


def save_classification_cache(
    cache_path: Path,
    cache: dict,
    fast_mode: bool,
    turbo_mode: bool = False,
    table_score_threshold: float = DEFAULT_TABLE_SCORE_THRESHOLD,
    flag_uncertain: bool = False,
) -> None:
    try:
        with cache_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "image_hash",
                    "label",
                    "score",
                    "reason",
                    "classifier_version",
                    "threshold",
                    "fast_mode",
                    "turbo_mode",
                    "flag_uncertain",
                ]
            )
            for image_hash, (label, score, reason) in cache.items():
                writer.writerow(
                    [
                        image_hash,
                        label,
                        f"{score:.6f}",
                        reason,
                        CLASSIFIER_VERSION,
                        f"{table_score_threshold:.4f}",
                        "1" if fast_mode else "0",
                        "1" if turbo_mode else "0",
                        "1" if flag_uncertain else "0",
                    ]
                )
    except OSError:
        return


def crawl_site(
    start_url: str,
    render_js: bool = True,
    save_table_dir: Optional[str] = None,
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
    normalized_start = normalize_page_url(start_url)
    if normalized_start is None:
        raise ValueError("Invalid start URL. Example: https://example.com")

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    renderer: Optional[JsRenderer] = None
    if render_js:
        renderer = JsRenderer()
    table_dir: Optional[Path] = None
    if save_table_dir:
        table_dir = Path(save_table_dir)
        table_dir.mkdir(parents=True, exist_ok=True)

    visited_images = set()
    results: List[ImageResult] = []
    table_saved = 0
    cache_hits = 0
    start_time = time.time()
    last_heartbeat = start_time
    cache_path = Path(CACHE_FILENAME)
    classification_cache = load_classification_cache(
        cache_path,
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
    start_msg = f"[status] scan started | page={start_url} | scope={scan_scope} | render_js={render_js}"
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
            # Avoid wasting crawl budget on pagination URLs.
            # Sites often include page-n links that don't lead to meaningful new content.
            if re.search(r"([?&]|/)(page|p)=(\d+)", page_url, flags=re.IGNORECASE):
                continue
            # Skip non-HTML resources that somehow made it into the queue.
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
        seen_targets = set()
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
                print(f"[warn] failed fetch: {page_url}")
                continue
            pages_to_scan.append(target_url)
            page_html[target_url] = html
    elif crawl_mode == "paginated_listing":
        # ── Paginated listing crawl ────────────────────────────────────────────
        # Supports one or more listing base URLs (normalized_start is always the
        # first; extra ones come from listing_urls).  For each listing URL:
        # 1. Fetch page 1 and auto-detect total pages (many fallback strategies).
        # 2. Iterate ?page=2…N, harvesting card/item links via extract_listing_item_links.
        # 3. Fetch every discovered detail page and add it to pages_to_scan.
        # Image extraction happens in the shared loop below, same as other modes.

        # Build the ordered list of listing base URLs (deduplicated).
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
            listing_label = f"listing {listing_idx}/{total_listings}" if total_listings > 1 else "listing"

            fetch_msg = f"[status] paginated_listing: fetching {listing_label} page 1 -> {base_listing_url}"
            print(fetch_msg, flush=True)
            if progress_callback is not None:
                progress_callback({
                    "event": "status",
                    "message": fetch_msg,
                    "phase": "fetching_listing_page",
                    "page_url": base_listing_url,
                    "page_index": 1,
                    "listing_index": listing_idx,
                    "listing_total": total_listings,
                })

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
                # Build paginated URL: set or replace ?page=N while keeping other query params.
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
                    progress_callback({
                        "event": "status",
                        "message": fetch_msg,
                        "phase": "fetching_listing_page",
                        "page_url": listing_page_url,
                        "page_index": page_num,
                        "page_total": total_pages,
                        "listing_index": listing_idx,
                        "listing_total": total_listings,
                    })

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

        # Add listing pages themselves to pages_to_scan (their images count too).
        for lurl, lhtml in listing_page_htmls.items():
            pages_to_scan.append(lurl)
            page_html[lurl] = lhtml

        # Fetch each item/detail page.
        total_items = len(item_links)
        for idx, item_url in enumerate(item_links, start=1):
            fetch_msg = f"[status] paginated_listing: fetching item page {idx}/{total_items} -> {item_url}"
            print(fetch_msg, flush=True)
            if progress_callback is not None:
                progress_callback({
                    "event": "status",
                    "message": fetch_msg,
                    "phase": "fetching_item_page",
                    "page_url": item_url,
                    "page_index": idx,
                    "page_total": total_items,
                })

            ihtml = _fetch_page(item_url)

            if ihtml is None:
                print(f"[warn] paginated_listing: failed fetch item page: {item_url}", flush=True)
                continue

            pages_to_scan.append(item_url)
            page_html[item_url] = ihtml

    else:
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

    found_msg = f"[status] found {len(page_image_pairs)} raw image candidates across {len(pages_to_scan)} page(s)"
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

    def process_image(page_url: str, image_url: str) -> Optional[Tuple[str, str, str, float, str, bytes, bool]]:
        local_session = shared_session()
        content, fetch_reason = fetch_image_bytes(local_session, image_url, page_url, turbo_mode=turbo_mode)
        if not content:
            return page_url, image_url, "normal", 0.0, fetch_reason, b"", False

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
            return page_url, image_url, cached[0], cached[1], cached[2], content, True

        if not owns_classification and wait_event is not None:
            wait_event.wait()
            with cache_lock:
                cached_after_wait = classification_cache.get(image_hash)
            if cached_after_wait is not None:
                return page_url, image_url, cached_after_wait[0], cached_after_wait[1], cached_after_wait[2], content, True
            # Fallback if the owner thread failed unexpectedly before caching.
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
            # Defensive guard: never allow table labels below the configured threshold.
            if label == "table" and score <= table_score_threshold:
                label = "normal"
                reason = f"{reason}_guarded_threshold"
            with cache_lock:
                classification_cache[image_hash] = (label, score, reason)
            return page_url, image_url, label, score, reason, content, False
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
                    page_url, image_url, label, score, reason, content, was_cached = outcome
                    if was_cached:
                        cache_hits += 1
                    results.append(ImageResult(page_url, image_url, label, score, reason))
                    if label == "table" and table_dir is not None:
                        saved = save_table_image(table_dir, image_url, content)
                        if saved is not None:
                            table_saved += 1
                    if progress_callback is not None:
                        progress_callback(
                            {
                                "event": "image_processed",
                                "processed": completed,
                                "total": len(unique_image_items),
                                "page_url": page_url,
                                "label": label,
                                "score": score,
                                "reason": reason,
                                "image_url": image_url,
                                "cached": was_cached,
                                "tables_saved": table_saved,
                            }
                        )

            now = time.time()
            if now - last_heartbeat >= heartbeat_seconds:
                elapsed = now - start_time
                print(
                    f"[heartbeat] running {elapsed:.1f}s | processed={completed}/{len(unique_image_items)} | classified={len(results)}",
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
                            "tables_saved": table_saved,
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
        cache_path,
        classification_cache,
        fast_mode,
        turbo_mode,
        table_score_threshold=table_score_threshold,
        flag_uncertain=flag_uncertain,
    )

    total_elapsed = time.time() - start_time
    done_msg = (
        f"[status] scan finished in {total_elapsed:.1f}s | pages={len(pages_to_scan)} "
        f"| raw_candidates={len(page_image_pairs)} | unique_images={unique_total} | processed={len(results)} "
        f"| skipped_stalled={skipped_due_to_stall} | tables_saved={table_saved} | cache_hits={cache_hits}"
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
                "tables_saved": table_saved,
                "cache_hits": cache_hits,
            }
        )

    return results


def write_results_csv(path: str, rows: List[ImageResult]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["page_url", "image_url", "label", "score", "reason"])
        for row in rows:
            writer.writerow([row.page_url, row.image_url, row.label, f"{row.score:.4f}", row.reason])


def ask_yes_no(prompt: str, default: bool) -> bool:
    suffix = "Y/n" if default else "y/N"
    raw = input(f"{prompt} [{suffix}]: ").strip().lower()
    if not raw:
        return default
    return raw in {"y", "yes"}


def configure_run_from_prompt() -> Tuple[str, bool, str, str, float, bool, bool, int, str, int, List[str], List[str]]:
    print("Interactive mode: enter options for this run.")
    url = input("Target URL: ").strip()
    while not url:
        url = input("Target URL (required): ").strip()

    scan_whole_site = ask_yes_no("Scan whole website (same domain) instead of only this page?", False)
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
    if crawl_mode == "page" and ask_yes_no("Crawl a paginated news/events listing (follows all cards across pages)?", False):
        crawl_mode = "paginated_listing"
    listing_urls: List[str] = []
    if crawl_mode == "paginated_listing" and ask_yes_no("Add more paginated listing URLs to crawl together?", False):
        print("Paste listing URLs one per line (in addition to the target URL above). Submit an empty line when done.")
        while True:
            candidate = input("Listing URL: ").strip()
            if not candidate:
                break
            listing_urls.append(candidate)
    max_pages = 40
    if crawl_mode in {"site"}:
        max_pages_prompt = (
            "Max pages to scan on this site (default 40)"
        )
        max_pages_raw = input(max_pages_prompt).strip()
        if max_pages_raw:
            try:
                max_pages = max(1, int(max_pages_raw))
            except ValueError:
                max_pages = 40

    render_js = ask_yes_no("Use JavaScript rendering (recommended for modern sites)?", True)
    fast_mode = ask_yes_no("Enable fast OCR mode (about 2x faster, slightly less accurate)?", False)
    turbo_mode = ask_yes_no("Enable TURBO mode (much faster, lower OCR quality + aggressive skipping)?", True)
    save_tables = ask_yes_no("Save detected table images to a folder?", True)
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
    table_dir = str(run_dir / "table_images") if save_tables else ""
    return (
        url,
        render_js,
        output_path,
        table_dir,
        heartbeat_seconds,
        fast_mode,
        turbo_mode,
        ocr_workers,
        crawl_mode,
        max_pages,
        target_urls,
        listing_urls,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan one webpage, discover images, and classify each as normal or table screenshot."
    )
    parser.add_argument("url", nargs="?", help="Target page URL, e.g. https://example.com/page")
    parser.add_argument("--output", default="", help="Output CSV path")
    parser.add_argument(
        "--render-js",
        action="store_true",
        help="Render pages with JavaScript (needed for some websites that lazy-load images).",
    )
    parser.add_argument(
        "--save-table-images",
        default="",
        help="If set, saves images classified as table to this folder path.",
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
        help="Aggressive speed mode: lower OCR quality, skip obvious non-tables, and fail image fetches faster.",
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
        help="Scan one page, crawl same-domain pages across the site, scan only specified URLs, or crawl a paginated listing (news/events) following all cards.",
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
        help="Additional paginated listing base URLs to crawl alongside the main URL when --crawl-mode=paginated_listing.",
    )
    args = parser.parse_args()

    if args.url:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path("runs") / run_id
        output_path = args.output or str(run_dir / "results.csv")
        table_dir = args.save_table_images or str(run_dir / "table_images")
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
            table_dir,
            heartbeat_seconds,
            fast_mode,
            turbo_mode,
            ocr_workers,
            crawl_mode,
            max_pages,
            target_urls,
            listing_urls,
        ) = configure_run_from_prompt()

    print(f"[run] results file: {output_path}", flush=True)
    print(f"[run] table images folder: {table_dir}", flush=True)
    print(f"[run] fast mode: {fast_mode}", flush=True)
    print(f"[run] turbo mode: {turbo_mode}", flush=True)
    print(f"[run] ocr workers: {ocr_workers}", flush=True)
    print(f"[run] crawl mode: {crawl_mode}", flush=True)
    if crawl_mode in {"site", "paginated_listing"}:
        print(f"[run] max pages: {max_pages}", flush=True)
    if crawl_mode == "urls":
        print(f"[run] target urls: {len(target_urls)}", flush=True)
    if crawl_mode == "paginated_listing" and listing_urls:
        print(f"[run] additional listing urls: {len(listing_urls)}", flush=True)

    rows = crawl_site(
        start_url=target_url,
        render_js=render_js,
        save_table_dir=table_dir or None,
        heartbeat_seconds=heartbeat_seconds,
        fast_mode=fast_mode,
        turbo_mode=turbo_mode,
        ocr_workers=ocr_workers,
        crawl_mode=crawl_mode,
        max_pages=max_pages,
        target_urls=None if crawl_mode != "urls" else target_urls,
        listing_urls=listing_urls if crawl_mode == "paginated_listing" else None,
    )
    write_results_csv(output_path, rows)

    table_count = sum(1 for r in rows if r.label == "table")
    normal_count = len(rows) - table_count
    print(f"Processed images: {len(rows)}")
    print(f"Table: {table_count}")
    print(f"Normal: {normal_count}")
    print(f"Saved results: {output_path}")
    print(f"Saved table images folder: {table_dir}")


if __name__ == "__main__":
    main()
