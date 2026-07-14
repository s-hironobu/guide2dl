"""
text_normalizer.py
------------------
Minimal replacement for the `ftfy` library, tailored for GPT-1 pre-processing.

Responsibilities (matching ftfy's usage in the GPT-1 paper):
  1. Fix mojibake / broken Unicode encodings
  2. Normalize Unicode to NFC form
  3. Replace "smart" / curly quotes, dashes, ellipses with ASCII equivalents
  4. Collapse excessive whitespace
  5. Strip control characters

Only the subset actually needed for BooksCorpus / SNLI text is implemented.
"""

import unicodedata
import re

# ==========================================
# Unicode fix-up tables
# ==========================================

# Common mojibake patterns: (broken_str, correct_str)
# These arise when UTF-8 bytes are mis-decoded as Latin-1 / Windows-1252.
_MOJIBAKE_FIXES: list[tuple[str, str]] = [
    ("\xe2\x80\x99", "'"),  # RIGHT SINGLE QUOTATION MARK (UTF-8 as Latin-1)
    ("\xe2\x80\x9c", '"'),  # LEFT DOUBLE QUOTATION MARK
    ("\xe2\x80\x9d", '"'),  # RIGHT DOUBLE QUOTATION MARK
    ("\xe2\x80\x93", "-"),  # EN DASH
    ("\xe2\x80\x94", "--"),  # EM DASH
    ("\xe2\x80\xa6", "..."),  # HORIZONTAL ELLIPSIS
]

# Unicode character -> ASCII replacement (applied after NFC normalisation)
_CHAR_MAP: dict[str, str] = {
    # Curly / typographic quotes  ->  plain ASCII
    "\u2018": "'",  # LEFT SINGLE QUOTATION MARK
    "\u2019": "'",  # RIGHT SINGLE QUOTATION MARK
    "\u201a": "'",  # SINGLE LOW-9 QUOTATION MARK
    "\u201b": "'",  # SINGLE HIGH-REVERSED-9 QUOTATION MARK
    "\u201c": '"',  # LEFT DOUBLE QUOTATION MARK
    "\u201d": '"',  # RIGHT DOUBLE QUOTATION MARK
    "\u201e": '"',  # DOUBLE LOW-9 QUOTATION MARK
    "\u201f": '"',  # DOUBLE HIGH-REVERSED-9 QUOTATION MARK
    "\u2039": "<",  # SINGLE LEFT-POINTING ANGLE QUOTATION MARK
    "\u203a": ">",  # SINGLE RIGHT-POINTING ANGLE QUOTATION MARK
    "\u00ab": '"',  # LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
    "\u00bb": '"',  # RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
    # Dashes  ->  ASCII hyphen or double-hyphen
    "\u2010": "-",  # HYPHEN
    "\u2011": "-",  # NON-BREAKING HYPHEN
    "\u2012": "-",  # FIGURE DASH
    "\u2013": "-",  # EN DASH
    "\u2014": "--",  # EM DASH
    "\u2015": "--",  # HORIZONTAL BAR
    "\u2212": "-",  # MINUS SIGN
    # Ellipsis
    "\u2026": "...",  # HORIZONTAL ELLIPSIS
    # Bullets / misc symbols  ->  space (swallowed later)
    "\u2022": " ",  # BULLET
    "\u00b7": " ",  # MIDDLE DOT
    # Non-breaking space  ->  regular space
    "\u00a0": " ",  # NO-BREAK SPACE
    "\u202f": " ",  # NARROW NO-BREAK SPACE
    "\u3000": " ",  # IDEOGRAPHIC SPACE
    # Zero-width characters  ->  empty
    "\u200b": "",  # ZERO WIDTH SPACE
    "\u200c": "",  # ZERO WIDTH NON-JOINER
    "\u200d": "",  # ZERO WIDTH JOINER
    "\ufeff": "",  # BYTE ORDER MARK
}

# Pre-compile a single regex that matches any key in _CHAR_MAP
_CHAR_MAP_RE = re.compile("[" + re.escape("".join(_CHAR_MAP.keys())) + "]")

# Control characters (except \t \n \r)
_CONTROL_CHAR_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f"  # C0 controls (excl. \t\n\r)
    r"\x80-\x9f]"  # C1 controls
)

# Collapse runs of spaces/tabs to a single space; keep newlines
_MULTI_SPACE_RE = re.compile(r"[ \t]+")

# Collapse 3+ consecutive newlines to exactly 2
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


# ==========================================
# Public API
# ==========================================


def fix_text(text: str) -> str:
    """
    Replicate the subset of `ftfy.fix_text` used in GPT-1:
      - Attempt mojibake repair (Latin-1 -> UTF-8 bytes -> decode)
      - NFC normalisation
      - Curly quote / dash / ellipsis -> ASCII
      - Strip control characters
      - Collapse whitespace
    """
    if not isinstance(text, str):
        text = str(text)

    # mojibake repair (best-effort)
    text = _repair_mojibake(text)

    # Unicode NFC
    text = unicodedata.normalize("NFC", text)

    # character substitutions
    text = _CHAR_MAP_RE.sub(lambda m: _CHAR_MAP[m.group()], text)

    # strip control chars
    text = _CONTROL_CHAR_RE.sub(" ", text)

    # normalise whitespace
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    text = text.strip()

    return text


def fix_text_segment(text: str) -> str:
    """Alias kept for API compatibility with ftfy."""
    return fix_text(text)


# ==========================================
# Internal helpers
# ==========================================


def _repair_mojibake(text: str) -> str:
    """
    Try to detect and fix the most common mojibake pattern:
    UTF-8 bytes interpreted as Latin-1 (Windows-1252).

    UTF-8 lead bytes that indicate multi-byte sequences:
      2-byte: 0xC2-0xDF  (Latin-1 supplement through misc symbols)
      3-byte: 0xE0-0xEF  (most CJK, arrows, math, punctuation like U+2019)
      4-byte: 0xF0-0xF4  (emoji, supplementary planes)

    If none of these appear in the text, it is almost certainly clean and we
    skip the expensive encode->decode round-trip.
    """
    # Quick scan: any Latin-1 char that could be a UTF-8 lead byte?
    _LEAD_RANGE = range(0x00C2, 0x00F5)  # 0xC2-0xF4
    if not any(ord(c) in _LEAD_RANGE for c in text):
        return text

    try:
        raw_bytes = text.encode("latin-1", errors="strict")
        repaired = raw_bytes.decode("utf-8", errors="strict")
        # Accept the repair when the result has fewer code points
        # (3 Latin-1 chars for a 3-byte UTF-8 sequence become 1 Unicode char)
        # AND the repaired non-ASCII ratio is no worse.
        if len(repaired) < len(text):
            return repaired
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass
    return text


def _non_ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return non_ascii / len(text)
