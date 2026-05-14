import threading
from dataclasses import dataclass, field
from typing import Optional

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

# Single SQLite file used for both the classification cache and run results.
# Replaces the old .crawler_classification_cache.csv and per-run results.csv files.
DB_FILENAME = ".crawler.db"

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

# OCR thread-local state (shared between classifier.py internals)
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
    # SHA-1 hex digest of the raw image bytes — stored in the DB so the webapp
    # can issue "mark as normal" corrections without re-fetching the image.
    image_hash: str = field(default="")
