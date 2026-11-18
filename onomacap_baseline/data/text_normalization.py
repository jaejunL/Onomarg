"""WavCaps-compatible target normalization used by the reproduction."""
import re
import unicodedata


_PUNCTUATION = re.compile(r"[(,.!?;:|*\"\\]")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    text = _PUNCTUATION.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()
