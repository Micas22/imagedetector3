import re
import time
import warnings
from typing import Optional, Tuple
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import SSLError

from constants import USER_AGENT

try:
    from urllib3.exceptions import InsecureRequestWarning
except Exception:  # Optional; used only to silence verify=False warning.
    InsecureRequestWarning = None

try:
    from playwright.sync_api import sync_playwright
except Exception:  # Optional dependency unless JS rendering is enabled.
    sync_playwright = None


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


def fetch_image_bytes(
    session: requests.Session,
    image_url: str,
    page_url: str,
    timeout: Tuple[float, float] = (4, 12),
    turbo_mode: bool = False,
) -> Tuple[Optional[bytes], str]:


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

    last_reason = "fetch_unknown_error"
    connect_timeout, read_timeout = timeout
    parsed_page = urlparse(page_url)
    request_headers = {
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Referer": page_url,
        "Origin": f"{parsed_page.scheme}://{parsed_page.netloc}",
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


def shared_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    adapter = HTTPAdapter(pool_connections=20, pool_maxsize=20)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session
