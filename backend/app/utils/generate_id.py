import uuid
import unicodedata
from backend.app.core.config import get_settings


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip().lower()


def generate_id(text: str) -> str:
    return str(uuid.uuid5(get_settings().UUID_NAMESPACE, normalize_text(text)))
