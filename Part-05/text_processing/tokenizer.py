"""
tokenizer.py
------------
Minimal replacement for the `spaCy` English tokenizer used in GPT-1.

Covers the subset of spaCy behaviour actually needed:
  - Split on whitespace
  - Separate punctuation that flanks tokens  (e.g. "Hello," -> ["Hello", ","])
  - Handle English contractions            (e.g. "don't" -> ["do", "n't"])
  - Keep intra-word hyphens intact         (e.g. "state-of-the-art")
  - Sentence boundary detection (optional) for paragraph-level splitting

Only ASCII / Latin-script English text is in scope (matching SNLI + BooksCorpus).
"""

import re
from typing import Iterator

#
# Contraction table  (word -> [part1, part2])
#

# spaCy splits on the apostrophe position, keeping the root and suffix.
# We map lower-cased contracted forms; casing is preserved via the lookup
# against the original token (see _split_contraction).

_CONTRACTIONS: dict[str, list[str]] = {
    # be
    "i'm": ["i", "'m"],
    "you're": ["you", "'re"],
    "he's": ["he", "'s"],
    "she's": ["she", "'s"],
    "it's": ["it", "'s"],
    "we're": ["we", "'re"],
    "they're": ["they", "'re"],
    "that's": ["that", "'s"],
    "there's": ["there", "'s"],
    "here's": ["here", "'s"],
    "who's": ["who", "'s"],
    "what's": ["what", "'s"],
    # have
    "i've": ["i", "'ve"],
    "you've": ["you", "'ve"],
    "we've": ["we", "'ve"],
    "they've": ["they", "'ve"],
    "i'd": ["i", "'d"],
    "you'd": ["you", "'d"],
    "he'd": ["he", "'d"],
    "she'd": ["she", "'d"],
    "we'd": ["we", "'d"],
    "they'd": ["they", "'d"],
    # will
    "i'll": ["i", "'ll"],
    "you'll": ["you", "'ll"],
    "he'll": ["he", "'ll"],
    "she'll": ["she", "'ll"],
    "we'll": ["we", "'ll"],
    "they'll": ["they", "'ll"],
    # negations
    "can't": ["can", "n't"],
    "cannot": ["can", "not"],
    "couldn't": ["could", "n't"],
    "didn't": ["did", "n't"],
    "doesn't": ["does", "n't"],
    "don't": ["do", "n't"],
    "hadn't": ["had", "n't"],
    "hasn't": ["has", "n't"],
    "haven't": ["have", "n't"],
    "isn't": ["is", "n't"],
    "mightn't": ["might", "n't"],
    "mustn't": ["must", "n't"],
    "needn't": ["need", "n't"],
    "shan't": ["sha", "n't"],
    "shouldn't": ["should", "n't"],
    "wasn't": ["was", "n't"],
    "weren't": ["were", "n't"],
    "won't": ["will", "n't"],  # irregular
    "wouldn't": ["would", "n't"],
    "ain't": ["ai", "n't"],
    # misc
    "'s": ["'s"],  # possessive / is (keep as-is after split)
    "'re": ["'re"],
    "'ve": ["'ve"],
    "'ll": ["'ll"],
    "'d": ["'d"],
    "'m": ["'m"],
    "n't": ["n't"],
}

# ==========================================
# Punctuation split patterns
# ==========================================

# Characters that should always be split off as their own token when they
# appear at the START of a raw token.
_PREFIX_PUNCT = re.compile(r'^(["\(\[\{\*`#@\$£€¥\^~<])')

# Characters that should always be split off as their own token when they
# appear at the END of a raw token.
_SUFFIX_PUNCT = re.compile(r'(["\)\]\}\*`!?,;:\.\u2026%>])$')

# Standalone punctuation tokens (single character or repeated same char)
_PUNCT_RE = re.compile(r"^[^\w\s]+$")

# Sentence-final punctuation that ends a sentence when followed by space+upper
_SENT_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


# ==========================================
# Word tokenizer
# ==========================================


class Tokenizer:
    """
    Rule-based English word tokenizer.

    Usage
    -----
    >>> tok = Tokenizer()
    >>> tok.tokenize("I don't know. Can't you see?")
    ['I', 'do', "n't", 'know', '.', 'Ca', "n't", 'you', 'see', '?']
    """

    def tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        for word in text.split():
            tokens.extend(self._tokenize_word(word))
        return tokens

    def _tokenize_word(self, word: str) -> list[str]:
        """Recursively split prefixes, suffixes, and contractions."""
        if not word:
            return []

        # --- prefix punctuation ---
        m = _PREFIX_PUNCT.match(word)
        if m and len(word) > 1:
            return [m.group(1)] + self._tokenize_word(word[1:])

        # --- suffix punctuation ---
        m = _SUFFIX_PUNCT.search(word)
        if m and len(word) > 1:
            stem = word[: m.start(1)]
            punct = m.group(1)
            # Keep trailing period attached to abbreviations (e.g. "U.S.")
            if punct == "." and ("." in stem or len(stem) <= 2):
                pass  # fall through
            else:
                return self._tokenize_word(stem) + [punct]

        # --- known full-word contractions FIRST (e.g. "don't", "won't", "can't") ---
        # These are checked before generic suffix splitting so that irregular
        # forms like "won't" -> ["will", "n't"] and "can't" -> ["can", "n't"]
        # take priority over the naive suffix loop below.
        parts = _split_contraction(word)
        if parts is not None:
            return parts

        # --- generic suffix splitting for unknown words (e.g. "Alice's") ---
        for suffix in ("n't", "'s", "'re", "'ve", "'ll", "'d", "'m"):
            lower = word.lower()
            if lower.endswith(suffix) and len(word) > len(suffix):
                stem = word[: -len(suffix)]
                return self._tokenize_word(stem) + [suffix]

        return [word]

    def tokenize_sentences(self, text: str) -> list[list[str]]:
        """Split *text* into sentences, then tokenize each."""
        sentences = _split_sentences(text)
        return [self.tokenize(s) for s in sentences if s.strip()]


# ==========================================
# Sentence splitter
# ==========================================


def _split_sentences(text: str) -> list[str]:
    """
    Very lightweight sentence boundary detection.

    Rules:
      - Split on `.`, `!`, `?` followed by whitespace + uppercase letter.
      - Don't split on common abbreviations (Mr., Dr., etc.)
      - Don't split mid-word (e.g. "U.S.A.")
    """
    # Known title abbreviations - do not split after these
    _NO_SPLIT = re.compile(
        r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|"
        r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec|"
        r"St|Ave|Blvd|Dept|Corp|Inc|Ltd|Co|Ph|D)\.$",
        re.IGNORECASE,
    )

    sentences: list[str] = []
    current_parts: list[str] = []

    # Split at likely sentence boundaries
    parts = _SENT_BOUNDARY_RE.split(text)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Check if the previous chunk ended with an abbreviation
        if current_parts and _NO_SPLIT.search(current_parts[-1]):
            current_parts.append(part)
        else:
            if current_parts:
                sentences.append(" ".join(current_parts))
                current_parts = []
            current_parts.append(part)

    if current_parts:
        sentences.append(" ".join(current_parts))

    return sentences if sentences else [text]


# ==========================================
# Contraction splitting helper
# ==========================================


def _split_contraction(token: str) -> list[str] | None:
    """
    If *token* (case-insensitive) is a known contraction, return its parts
    preserving the original casing of the first part.  Returns None if not
    a contraction.
    """
    lower = token.lower()
    if lower not in _CONTRACTIONS:
        return None

    parts = _CONTRACTIONS[lower]
    if len(parts) == 1:
        return [token]  # e.g. standalone "'s"

    # Preserve casing of the root (first part) from the original token.
    root_len = len(parts[0])
    restored_root = _match_case(token[:root_len], parts[0])
    return [restored_root] + parts[1:]


def _match_case(original: str, template: str) -> str:
    """Apply the case pattern of *original* to *template*."""
    if original.isupper():
        return template.upper()
    if original.istitle():
        return template.capitalize()
    return template  # default: keep template as-is (usually lowercase)
