import re
from pathlib import Path
from typing import List, Optional
from urllib.parse import parse_qsl, urljoin, urlparse

from bs4 import BeautifulSoup

from constants import IMAGE_EXTENSIONS, NON_HTML_EXTENSIONS


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
