#
# prepare_datasets.py
#
# Generates pre-training and fine-tuning data for GPT-1 from the SNLI dataset.
#
# Outputs
# -------
#   ./data/pretrain/snli_pretrain.txt   <- Text corpus for pre-training
#   ./bpe_vocab.json                    <- BPE tokenizer vocabulary
#   ./bpe_merges.txt                    <- BPE merge rules (40,000 merges)
#
# For fine-tuning, the original SNLI .jsonl files are used directly (no conversion needed).
# They are read directly by load_snli_jsonl() in snli_dataset.py.
#
# Usage
# -----
#   python prepare_datasets.py
#
#   snli_dir  : Extracted SNLI 1.0 directory (./data/snli_1.0)
#               Must contain snli_1.0_train.jsonl / dev.jsonl / test.jsonl
#   out_dir   : Root directory for outputs
#               - ./data/pretrain/snli_pretrain.txt
#               - ./bpe_vocab.json
#               - ./bpe_merges.txt
#
#   Copyright (c) 2026, Hironobu Suzuki @ interdb.jp

import os
import sys
import json
import time
import collections
import zipfile
import urllib.request

# Add text_processing/ to import path
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "text_processing"))

from text_normalizer import fix_text  # Alternative to ftfy
from bpe_encoder import BPEEncoder, BPETrainer  # BPE Tokenizer


# ============================================================
# Settings
# ============================================================

# SNLI splits to include in the pre-training corpus.
# Usually uses all sentences from train + dev + test.
PRETRAIN_SPLITS = ["train", "dev", "test"]

# BPE Configuration
BPE_NUM_MERGES = 40_000  # Paper Section 4.1: 40,000 merges
BPE_MIN_FREQ = 2  # Exclude words with less than 2 occurrences from BPE training

# File name templates
SNLI_FILENAME = "snli_1.0_{split}.jsonl"
PRETRAIN_TEXT = "./data/pretrain/snli_pretrain.txt"
VOCAB_FILE = "./bpe_vocab.json"
MERGES_FILE = "./bpe_merges.txt"
SNLI_URL = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"
SNLI_ZIP = "snli_1.0.zip"
SNLI_DIR = "./data/snli_1.0"
DATA_DIR = "./data"

# ========================================
# configuration
# ========================================

SNLI_CONFIG = {
    "pretrain_txt": PRETRAIN_TEXT,
    "vocab_file": VOCAB_FILE,
    "merges_file": MERGES_FILE,
    "snli_dir": SNLI_DIR,
}


# ========================================
# Utils
# ========================================


def check_required_files():
    required_files = [
        SNLI_CONFIG["pretrain_txt"],
        SNLI_CONFIG["vocab_file"],
        SNLI_CONFIG["merges_file"],
    ]
    _check_files = True
    for file_path in required_files:
        if not os.path.isfile(file_path):
            _check_files = False
            print("Error:\t{} not found.".format(file_path))

    return _check_files


# ============================================================
# Step 0: Download dataset
# ============================================================


def download_snli_data() -> None:
    """
    Ensure ~/data directory exists and download/extract SNLI dataset if needed.
    """
    # (1) Create ./data directory if it does not exist
    """
    #data_dir = "./data"
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    # (2) Check if ./data/snli_1.0 exists; if not, download and unzip
    """
    #snli_dir = os.path.join(DATA_DIR, "snli_1.0")
    """

    if not os.path.exists(snli_dir):
        # Make sure ./data exists before downloading into it
        os.makedirs(DATA_DIR, exist_ok=True)

        zip_path = os.path.join(DATA_DIR, SNLI_ZIP)

        print(f"Downloading SNLI dataset from {SNLI_URL} ...")
        urllib.request.urlretrieve(SNLI_URL, zip_path)
        print("Download complete.")

        print(f"Extracting {zip_path} to {DATA_DIR} ...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(DATA_DIR)
        print("Extraction complete.")

        # Optionally remove the zip file after extraction
        os.remove(zip_path)
    else:
        print(f"{snli_dir} already exists. Skipping download.")


# ============================================================
# Step 1: Extract text from SNLI jsonl
# ============================================================


def extract_sentences(jsonl_path: str) -> list[str]:
    """
    Extracts sentence1 (premise) and sentence2 (hypothesis) from an SNLI .jsonl file,
    normalizes them using fix_text(), and returns them.

    Lines with gold_label == "-" (unlabeled) are still used as text
    since pre-training does not require labels.

    Returns
    -------
    list[str]  A list of sentences (includes both sentence1 and sentence2)
    """
    sentences = []
    seen: set[str] = set()  # For deduplication

    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            for key in ("sentence1", "sentence2"):
                raw = obj.get(key, "").strip()
                if not raw:
                    continue
                clean = fix_text(raw)
                if clean and clean not in seen:
                    seen.add(clean)
                    sentences.append(clean)

    return sentences


def build_pretrain_corpus(snli_dir: str, out_path: str, splits=None) -> list[str]:
    """
    Extracts all sentences from the specified SNLI splits and combines them into one file.

    Parameters
    ----------
    snli_dir : Directory containing SNLI .jsonl files
    out_path : Output text file path
    splits   : List of split names to use (default: PRETRAIN_SPLITS)

    Returns
    -------
    all_sentences : list[str]  All extracted sentences (reused for BPE training)
    """
    if splits is None:
        splits = PRETRAIN_SPLITS

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    all_sentences: list[str] = []
    for split in splits:
        jsonl_path = os.path.join(snli_dir, SNLI_FILENAME.format(split=split))
        if not os.path.exists(jsonl_path):
            print(f"  [skip] {jsonl_path} not found")
            continue

        sents = extract_sentences(jsonl_path)
        all_sentences.extend(sents)
        print(f"  {split:>5s}: {len(sents):>8,} sentences")

    # Write to text file (one sentence per line)
    with open(out_path, "w", encoding="utf-8") as fh:
        for sent in all_sentences:
            fh.write(sent + "\n")

    size_kb = os.path.getsize(out_path) / 1024
    print(
        f"\n  Total  : {len(all_sentences):>8,} sentences -> {out_path} ({size_kb:.1f} KB)"
    )
    return all_sentences


# ============================================================
# Step 2: Train and save the BPE encoder
# ============================================================


def train_and_save_bpe(
    sentences: list[str],
    vocab_path: str,
    merges_path: str,
    num_merges: int = BPE_NUM_MERGES,
    min_frequency: int = BPE_MIN_FREQ,
) -> BPEEncoder:
    """
    Trains BPE using the given sentences and saves the vocab and merges files.

    Parameters
    ----------
    sentences    : List of normalized text
    vocab_path   : Path to save the JSON vocabulary file
    merges_path  : Path to save the merge rules file
    num_merges   : Number of BPE merges (GPT-1 = 40,000)
    min_frequency: Minimum frequency required to be included in the vocabulary

    Returns
    -------
    BPEEncoder  Trained encoder instance
    """
    print(f"  Training BPE (num_merges={num_merges:,}, min_freq={min_frequency}) ...")
    t0 = time.time()

    trainer = BPETrainer(num_merges=num_merges, min_frequency=min_frequency)
    enc = trainer.train(sentences)

    enc.save(vocab_path, merges_path)
    elapsed = time.time() - t0
    print(f"  Vocab size   : {enc.vocab_size:,}")
    print(f"  Saved to     : {vocab_path}")
    print(f"  Saved to     : {merges_path}")
    print(f"  Training time: {elapsed:.1f} s")
    return enc


# ============================================================
# Step 3: Display vocabulary and corpus statistics
# ============================================================


def show_corpus_stats(sentences: list[str], enc: BPEEncoder) -> None:
    """
    Displays basic statistics of the corpus and the BPE encoder.
    Useful for designing pre-training batches (e.g., token counts, average sequence length).
    """
    print("\n  -- Corpus Statistics --")

    # Estimate total token count (sampling the first 5,000 sentences)
    sample = sentences[:5_000]
    token_counts = [len(enc.encode(s)) for s in sample]
    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
    total_tokens_est = int(avg_tokens * len(sentences))

    print(f"  Total sentences       : {len(sentences):>10,}")
    print(f"  Avg tokens/sentence   : {avg_tokens:>10.1f}")
    print(f"  Total tokens (est.)   : {total_tokens_est:>10,}")

    # Number of chunks at seq_len=512
    seq_len = 512
    n_chunks = total_tokens_est // seq_len
    n_batches = n_chunks // 64  # batch_size=64
    print(f"\n  Chunks at seq_len={seq_len} : {n_chunks:>8,}")
    print(f"  Batches/epoch at batch_size=64: {n_batches:>8,}")
    print(f"  Total steps for 100 epochs    : {n_batches * 100:>8,}")

    # Verify special tokens
    print("\n  -- Special Tokens --")
    for tok in ["<pad>", "<unk>", "<s>", "</s>", "$"]:
        idx = enc.encoder.get(tok, "N/A")
        print(f"  {tok:>8s} -> id {idx}")


# ============================================================
# Step 4: Verify fine-tuning data
# ============================================================


def check_finetune_data(snli_dir: str, enc: BPEEncoder) -> None:
    """
    Verifies the existence of fine-tuning SNLI data and shows a sample encoding.
    Actual data conversion is handled by SNLIDataset.prepare() in snli_dataset.py.
    """
    print("\n  -- Fine-tuning Data Verification --")
    for split in ["train", "dev", "test"]:
        jsonl_path = os.path.join(snli_dir, SNLI_FILENAME.format(split=split))
        if os.path.exists(jsonl_path):
            # Count lines
            with open(jsonl_path, encoding="utf-8") as fh:
                n_lines = sum(1 for _ in fh)
            print(f"  {split:>5s}: {jsonl_path} ({n_lines:,} lines) [OK]")
        else:
            print(f"  {split:>5s}: {jsonl_path} <- Not Found [Missing]")

    # Sample encoding (first pair)
    sample_jsonl = os.path.join(snli_dir, SNLI_FILENAME.format(split="train"))
    if os.path.exists(sample_jsonl):
        with open(sample_jsonl, encoding="utf-8") as fh:
            obj = json.loads(fh.readline())

        premise = fix_text(obj.get("sentence1", ""))
        hypothesis = fix_text(obj.get("sentence2", ""))
        ids = enc.encode_with_special(premise, hypothesis)
        print(f"\n  Sample Encoding:")
        print(f"    premise   : {premise[:60]}...")
        print(f"    hypothesis: {hypothesis[:60]}...")
        print(f"    ids (first 10): {ids[:10]}")
        print(f"    ids length    : {len(ids)}")
        print(f"    gold_label    : {obj.get('gold_label', '?')}")


# ============================================================
# Main
# ============================================================


def main():

    print("=" * 60)
    print("GPT-1 Dataset Preparation Script")
    print("=" * 60)

    """
    "./data/pretrain/snli_pretrain.txt",
    "./bpe_merges.txt",
    "./bpe_vocab.json",
    """
    """
    required_files = [
        SNLI_CONFIG["pretrain_txt"],
        SNLI_CONFIG["vocab_file"],
        SNLI_CONFIG["merges_file"],
    ]
    _files_check = True
    for file_path in required_files:
        if not os.path.isfile(file_path):
            _files_check = False

    """
    if check_required_files():
        print("Preparation is already complete.")
        sys.exit(0)

    # Step 0: Download dataset
    download_snli_data()

    # Command Line Arguments
    SNLI_DIR = "./data/snli_1.0"

    pretrain_txt = SNLI_CONFIG["pretrain_txt"]
    vocab_path = SNLI_CONFIG["vocab_file"]
    merges_path = SNLI_CONFIG["merges_file"]

    print(f"\n  SNLI Directory   : {SNLI_DIR}")
    print(f"  Pretrain Corpus  : {pretrain_txt}")
    print(f"  BPE Vocab        : {vocab_path}")
    print(f"  BPE Merges       : {merges_path}")

    # Step 1: Text Extraction
    print("\n[1/3] Extracting pre-training text from SNLI ...")
    sentences = build_pretrain_corpus(SNLI_DIR, pretrain_txt, splits=PRETRAIN_SPLITS)

    if not sentences:
        print("Error: No sentences extracted. Please check snli_dir.")
        sys.exit(1)

    # Step 2: BPE Training
    if os.path.exists(vocab_path) and os.path.exists(merges_path):
        print(f"\n[2/3] BPE files already exist. Skipping training.")
        print(f"      To retrain, delete {vocab_path} and {merges_path}.")
        enc = BPEEncoder.from_files(vocab_path, merges_path)
        print(f"      Loaded vocab size: {enc.vocab_size:,}")
    else:
        print(f"\n[2/3] Training BPE Encoder ...")
        enc = train_and_save_bpe(sentences, vocab_path, merges_path)

    # Step 3: Statistics
    print("\n[3/3] Checking statistics ...")
    show_corpus_stats(sentences, enc)

    # Fine-tuning Data Verification
    check_finetune_data(SNLI_DIR, enc)

    # Completion Message
    print("\n" + "=" * 60)
    print("Setup complete.\n")
    print("=" * 60)


if __name__ == "__main__":
    main()
