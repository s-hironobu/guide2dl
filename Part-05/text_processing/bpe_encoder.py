"""
bpe_encoder.py
--------------
Byte Pair Encoding (BPE) tokenizer - minimal re-implementation of the
encoder used in the original GPT-1 code release.

Reference: Sennrich et al. (2016) "Neural Machine Translation of Rare Words
with Subword Units"  https://arxiv.org/abs/1508.07909

The interface mirrors the original OpenAI GPT-1 `encoder.py`:

    enc = BPEEncoder.from_files(vocab_path, merges_path)
    ids  = enc.encode("Hello, world!")   -> list[int]
    text = enc.decode(ids)               -> "Hello, world!"

It can also be *trained from scratch* on a corpus via `BPETrainer`.

Design choices (matching GPT-1):
  - Vocabulary size target: 40,000 merges above the initial character vocab.
  - End-of-word marker: `</w>` appended to the last character of every word.
  - All text is lower-cased before encoding (matching GPT-1 pre-processing).
  - Rare-token fallback: unknown characters map to `<unk>`.
"""

import re
import json
import collections
from pathlib import Path
from typing import Optional


# ==========================================
# Constants
# ==========================================

END_OF_WORD = "</w>"
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
START_TOKEN = "<s>"
END_TOKEN = "</s>"
DELIM_TOKEN = "$"  # GPT-1 uses '$' as the entailment delimiter

_SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, START_TOKEN, END_TOKEN, DELIM_TOKEN]

# Regex that splits raw text into "words" before BPE is applied.
# Matches: contractions, possessives, words, and individual punctuation chars.
_WORD_RE = re.compile(
    r"n't|'s|'re|'ve|'ll|'d|'m"  # contractions (greedy, before \w+)
    r"|\w+"  # runs of word chars
    r"|[^\w\s]"  # single non-word, non-space char
    r"|\s+",  # whitespace (usually skipped)
    re.IGNORECASE,
)


# ==========================================
# BPEEncoder  (inference / encode + decode)
# ==========================================


class BPEEncoder:
    """
    Encode/decode text using a trained BPE vocabulary.

    Parameters
    ----------
    encoder : dict[str, int]   token-string -> token-id
    bpe_merges : list[tuple[str, str]]   ordered merge rules
    """

    def __init__(
        self,
        encoder: dict[str, int],
        bpe_merges: list[tuple[str, str]],
    ) -> None:
        self.encoder: dict[str, int] = encoder
        self.decoder: dict[int, str] = {v: k for k, v in encoder.items()}
        self.bpe_ranks: dict[tuple[str, str], int] = {
            pair: idx for idx, pair in enumerate(bpe_merges)
        }
        self._cache: dict[str, list[str]] = {}

    # Public API
    def encode(self, text: str) -> list[int]:
        """Return a list of BPE token ids for *text*."""
        ids: list[int] = []
        for token in self._tokenize_to_words(text):
            for bpe_token in self._apply_bpe(token):
                ids.append(self.encoder.get(bpe_token, self.encoder.get(UNK_TOKEN, 0)))
        return ids

    def encode_with_special(
        self,
        premise: str,
        hypothesis: str,
    ) -> list[int]:
        """
        Encode a (premise, hypothesis) pair using the GPT-1 NLI format:
            <s> premise_tokens $ hypothesis_tokens </s>
        """
        start = self.encoder[START_TOKEN]
        end = self.encoder[END_TOKEN]
        delim = self.encoder[DELIM_TOKEN]
        return (
            [start] + self.encode(premise) + [delim] + self.encode(hypothesis) + [end]
        )

    def decode(self, ids: list[int]) -> str:
        """Convert a list of ids back to a string."""
        tokens = [self.decoder.get(i, UNK_TOKEN) for i in ids]
        text = " ".join(tokens)
        # Remove the end-of-word marker and the space that precedes it
        text = text.replace(f" {END_OF_WORD}", "").replace(END_OF_WORD, "")
        return text

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    # Serialisation
    def save(self, vocab_path: str | Path, merges_path: str | Path) -> None:
        vocab_path = Path(vocab_path)
        merges_path = Path(merges_path)
        vocab_path.write_text(
            json.dumps(self.encoder, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        lines = ["#version: gpt1-bpe"] + [f"{a} {b}" for a, b in self.bpe_ranks.keys()]
        merges_path.write_text("\n".join(lines), encoding="utf-8")

    @classmethod
    def from_files(
        cls,
        vocab_path: str | Path,
        merges_path: str | Path,
    ) -> "BPEEncoder":
        vocab_path = Path(vocab_path)
        merges_path = Path(merges_path)
        encoder = json.loads(vocab_path.read_text(encoding="utf-8"))
        lines = merges_path.read_text(encoding="utf-8").splitlines()
        merges = [
            tuple(line.split())  # type: ignore[misc]
            for line in lines
            if line and not line.startswith("#")
        ]
        return cls(encoder, merges)  # type: ignore[arg-type]

    # Internal helpers
    def _tokenize_to_words(self, text: str) -> list[str]:
        """Split *text* into raw words (before BPE), lower-cased."""
        return [
            m.group().lower()
            for m in _WORD_RE.finditer(text)
            if m.group().strip()  # skip pure whitespace
        ]

    def _apply_bpe(self, word: str) -> list[str]:
        """Return the BPE sub-word sequence for a single (lower-cased) word."""
        if word in self._cache:
            return self._cache[word]

        # Initial representation: every character is its own symbol;
        # the last character gets the end-of-word marker appended.
        if len(word) == 1:
            symbols = [word + END_OF_WORD]
        else:
            symbols = list(word[:-1]) + [word[-1] + END_OF_WORD]

        # Iteratively merge the highest-priority adjacent pair.
        while len(symbols) > 1:
            pairs = _get_pairs(symbols)
            # Find the pair with the lowest rank (= earliest merge rule)
            best = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if best not in self.bpe_ranks:
                break  # no more applicable merges
            symbols = _merge_symbols(symbols, best)

        self._cache[word] = symbols
        return symbols


# ==========================================
# BPETrainer  (train BPE from a text corpus)
# ==========================================


class BPETrainer:
    """
    Learn BPE merge rules from a corpus and produce a BPEEncoder.

    Parameters
    ----------
    num_merges : int
        Number of merge operations (GPT-1 uses 40,000).
    min_frequency : int
        Ignore words that appear fewer than this many times.
    """

    def __init__(
        self,
        num_merges: int = 40_000,
        min_frequency: int = 2,
    ) -> None:
        self.num_merges = num_merges
        self.min_frequency = min_frequency

    def train(self, texts: list[str]) -> BPEEncoder:
        """
        Train BPE on *texts* (list of strings) and return a BPEEncoder.
        """
        print(f"[BPETrainer] Counting word frequencies ...")
        word_freq = self._count_words(texts)
        # Convert words -> tuple of BPE symbols
        vocab: dict[tuple[str, ...], int] = {
            self._word_to_symbols(word): freq
            for word, freq in word_freq.items()
            if freq >= self.min_frequency
        }

        merges: list[tuple[str, str]] = []
        print(f"[BPETrainer] Running {self.num_merges} merges ...")
        for step in range(self.num_merges):
            pair_counts = self._count_pairs(vocab)
            if not pair_counts:
                print(f"[BPETrainer] No more pairs at step {step}. Stopping.")
                break
            best_pair = max(pair_counts, key=pair_counts.__getitem__)
            merges.append(best_pair)
            vocab = self._merge_vocab(vocab, best_pair)
            if (step + 1) % 5_000 == 0:
                print(f"[BPETrainer]   step {step + 1}/{self.num_merges}")

        encoder = self._build_encoder(vocab, merges)
        print(
            f"[BPETrainer] Done. Vocab size = {len(encoder)}  "
            f"(merges = {len(merges)})"
        )
        return BPEEncoder(encoder, merges)

    def _count_words(self, texts: list[str]) -> collections.Counter:
        counter: collections.Counter = collections.Counter()
        for text in texts:
            for m in _WORD_RE.finditer(text.lower()):
                w = m.group()
                if w.strip():
                    counter[w] += 1
        return counter

    @staticmethod
    def _word_to_symbols(word: str) -> tuple[str, ...]:
        if len(word) == 1:
            return (word + END_OF_WORD,)
        return tuple(word[:-1]) + (word[-1] + END_OF_WORD,)

    @staticmethod
    def _count_pairs(vocab: dict[tuple[str, ...], int]) -> collections.Counter:
        pairs: collections.Counter = collections.Counter()
        for symbols, freq in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq
        return pairs

    @staticmethod
    def _merge_vocab(
        vocab: dict[tuple[str, ...], int],
        pair: tuple[str, str],
    ) -> dict[tuple[str, ...], int]:
        new_vocab: dict[tuple[str, ...], int] = {}
        bigram = pair[0] + pair[1]
        for symbols, freq in vocab.items():
            new_symbols = _merge_symbols(list(symbols), pair)
            new_vocab[tuple(new_symbols)] = freq
        return new_vocab

    @staticmethod
    def _build_encoder(
        vocab: dict[tuple[str, ...], int],
        merges: list[tuple[str, str]],
    ) -> dict[str, int]:
        # Collect all unique sub-word tokens that appear in the final vocab
        token_set: set[str] = set()
        for symbols in vocab:
            token_set.update(symbols)
        # Add special tokens first (gives them low ids)
        tokens = list(_SPECIAL_TOKENS)
        # Then sorted regular tokens for determinism
        tokens += sorted(token_set - set(_SPECIAL_TOKENS))
        return {tok: idx for idx, tok in enumerate(tokens)}


# ==========================================
# Utility functions
# ==========================================


def _get_pairs(symbols: list[str]) -> set[tuple[str, str]]:
    return {(a, b) for a, b in zip(symbols, symbols[1:])}


def _merge_symbols(
    symbols: list[str],
    pair: tuple[str, str],
) -> list[str]:
    """Return *symbols* with all occurrences of *pair* merged."""
    result: list[str] = []
    i = 0
    merged = pair[0] + pair[1]
    while i < len(symbols):
        if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
            result.append(merged)
            i += 2
        else:
            result.append(symbols[i])
            i += 1
    return result
