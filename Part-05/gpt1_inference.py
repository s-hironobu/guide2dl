#
# gpt1_inference.py - GPT-1 NLI Inference
#
#   Copyright (c) 2026, Hironobu Suzuki @ interdb.jp

import os
import sys
import re
import argparse
import logging
import tensorflow as tf
import numpy as np

if len(sys.argv) <= 1:
    print(
        "Syntax Error:\n\tpython gpt1_inference.py [checkpoint-pretrain-dir] {file.tsv}"
    )
    sys.exit(1)

os.environ.setdefault("TF_METAL_DEVICE_ENABLE", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "text_processing"))

from gpt1_model import (
    GPT1Model,
    GPT1ForNLI,
    GPT1_CONFIG,
    make_checkpoint_path,
    get_device,
)
from prepare_datasets import (
    SNLI_CONFIG,
    check_required_files,
)
from bpe_encoder import BPEEncoder
from text_normalizer import fix_text

logging.getLogger("tensorflow").setLevel(logging.ERROR)

DEVICE = get_device()

if check_required_files() == False:
    print("Execute prepare_datasets.py.")
    sys.exit(1)

# ============================================================
# Constants
# ============================================================

LABEL_NAMES = {0: "entailment", 1: "neutral", 2: "contradiction"}
LABEL_EMOJI = {0: "✅", 1: "➖", 2: "❌"}

DEFAULT_MAX_SEQ = GPT1_CONFIG["max_seq_len"]  # 256

# ============================================================
# Load model
# ============================================================


def load_model(
    ckpt_dir: str,
) -> tuple[GPT1ForNLI, BPEEncoder]:
    """
    Load GPT1ForNLI from a checkpoint and return it along with the BPEEncoder.

    Automatically selects the latest checkpoint in the checkpoint directory.
    To use a specific epoch, pass a path that includes 'ckpt-N' to ckpt_dir
    (passed directly to tf.train.Checkpoint.restore()).

    Returns
    -------
    (nli_model, encoder)
    """

    vocab_path = SNLI_CONFIG["vocab_file"]
    merges_path = SNLI_CONFIG["merges_file"]
    max_seq_len = DEFAULT_MAX_SEQ

    # -- BPE encoder
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"vocab not found: {vocab_path}")
    if not os.path.exists(merges_path):
        raise FileNotFoundError(f"merges not found: {merges_path}")

    enc = BPEEncoder.from_files(vocab_path, merges_path)
    print(
        f"[BPE]  vocab_size={enc.vocab_size:,}  "
        f"vocab={vocab_path}  merges={merges_path}"
    )

    # Build model
    with tf.device(DEVICE):
        gpt1 = GPT1Model(
            num_layers=GPT1_CONFIG["num_layers"],
            d_model=GPT1_CONFIG["d_model"],
            num_heads=GPT1_CONFIG["num_heads"],
            d_ffn=GPT1_CONFIG["d_ffn"],
            vocab_size=enc.vocab_size,
            max_seq_len=max_seq_len,
            dropout_rate=0.0,  # Disable dropout during inference
            name="gpt1",
        )
        nli_model = GPT1ForNLI(gpt1, num_labels=3, name="gpt1_nli")

        # Initialize weights (requires build before restore)
        _dummy = tf.zeros((1, max_seq_len), dtype=tf.int32)
        nli_model(_dummy, training=False)

    # Restore checkpoint
    # Check if ckpt_dir points to a specific step like "path/to/dir/ckpt-N"
    _specific_ckpt = None
    if re.search(r"/ckpt-\d+$", ckpt_dir):
        # When a path including "ckpt-N" is directly specified
        _specific_ckpt = ckpt_dir
        _ckpt_base_dir = os.path.dirname(ckpt_dir)
    else:
        _ckpt_base_dir = ckpt_dir

    ckpt = tf.train.Checkpoint(model=nli_model)
    manager = tf.train.CheckpointManager(ckpt, _ckpt_base_dir, max_to_keep=None)

    restore_path = _specific_ckpt or manager.latest_checkpoint
    if restore_path is None:
        raise FileNotFoundError(
            f"Checkpoint not found: {_ckpt_base_dir}\n"
            f"  run gpt1_finetune.py to generate checkpoints."
        )

    ckpt.restore(restore_path).expect_partial()

    # Display which epoch checkpoint was restored
    _epoch_num = re.search(r"-(\d+)$", restore_path)
    epoch_str = f"epoch {_epoch_num.group(1)}" if _epoch_num else restore_path
    print(f"[Checkpoint] Restored {epoch_str} <- {restore_path}")
    print(f"[Device]      {DEVICE}\n")

    return nli_model, enc


# ============================================================
# Inference for a Single Pair
# ============================================================


def predict(
    nli_model: GPT1ForNLI,
    enc: BPEEncoder,
    premise: str,
    hypothesis: str,
    max_seq_len: int = DEFAULT_MAX_SEQ,
) -> dict:
    """
    Predicts the NLI label for a given (premise, hypothesis) pair.

    Processing Flow:
        1. Unicode normalization using fix_text()
        2. Convert to GPT-1 NLI format using encode_with_special()
               <s> premise_tokens $ hypothesis_tokens </s>
        3. Truncate / pad to max_seq_len
        4. GPT1ForNLI.call() -> clf_logits with shape (1, 3)
        5. Convert to confidence scores via softmax

    Returns
    -------
    dict:
        label       : str   "entailment" / "neutral" / "contradiction"
        label_id    : int   0 / 1 / 2
        confidence  : float Probability of the top-predicted class (0.0-1.0)
        probs       : dict  {label_name: probability} for all 3 classes
        input_len   : int   Actual number of tokens used
    """

    # -- Text Normalization
    premise_clean = fix_text(premise.strip())
    hypothesis_clean = fix_text(hypothesis.strip())

    # -- BPE Encoding (NLI Format)
    # <s> premise $ hypothesis </s>
    ids = enc.encode_with_special(premise_clean, hypothesis_clean)

    # Truncate if the sequence is too long (always keep the trailing </s>)
    if len(ids) > max_seq_len:
        ids = ids[: max_seq_len - 1] + [ids[-1]]  # Keep the final </s>

    input_len = len(ids)

    # Pad to max_seq_len (pad_id = 0)
    pad_len = max_seq_len - input_len
    ids_padded = ids + [0] * pad_len

    # -- Inference
    input_tensor = tf.constant([ids_padded], dtype=tf.int32)  # (1, max_seq_len)

    with tf.device(DEVICE):
        clf_logits, _, _ = nli_model(input_tensor, training=False)
        # clf_logits: (1, 3)

    probs = tf.nn.softmax(clf_logits, axis=-1).numpy()[0]  # (3,)
    label_id = int(np.argmax(probs))

    return {
        "label": LABEL_NAMES[label_id],
        "label_id": label_id,
        "confidence": float(probs[label_id]),
        "probs": {LABEL_NAMES[i]: float(probs[i]) for i in range(3)},
        "input_len": input_len,
    }


# ============================================================
# print result
# ============================================================


def print_result(premise: str, hypothesis: str, result: dict) -> None:
    emoji = LABEL_EMOJI[result["label_id"]]
    p = result["probs"]
    print(f"  Premise      : {premise}")
    print(f"  Hypothesis   : {hypothesis}")
    print(
        f"  Prediction   : {emoji}  {result['label']:>15s}  "
        f"(confidence: {result['confidence']*100:.1f}%)"
    )
    print(
        f"  Scores       : "
        f"entailment={p['entailment']*100:.1f}%  "
        f"neutral={p['neutral']*100:.1f}%  "
        f"contradiction={p['contradiction']*100:.1f}%"
    )
    print(f"  Token length : {result['input_len']}")
    print()


# ============================================================
# Batch File Inference
# ============================================================


def predict_from_file(
    nli_model: GPT1ForNLI,
    enc: BPEEncoder,
    file_path: str,
) -> None:
    """
    Process a TSV file (premise<TAB>hypothesis) line by line and print results.

    File Format:
        # Comment lines are ignored
        A dog is running.\tA dog is moving.
        The sky is blue.\tThe sky is green.
    """
    max_seq_len = DEFAULT_MAX_SEQ

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    print(f"[File] Processing {file_path} ...\n")
    n_ok = n_skip = 0

    with open(file_path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) < 2:
                print(f"  [Line {lineno}] Skipped (not tab-separated): {line!r}")
                n_skip += 1
                continue

            premise, hypothesis = parts[0], parts[1]
            result = predict(nli_model, enc, premise, hypothesis, max_seq_len)
            print(f"[Pair {lineno}]")
            print_result(premise, hypothesis, result)
            n_ok += 1

    print(f"Done: {n_ok} pairs processed / {n_skip} lines skipped")


# ============================================================
# Interactive mode
# ============================================================


def interactive_mode(
    nli_model: GPT1ForNLI,
    enc: BPEEncoder,
) -> None:

    max_seq_len = DEFAULT_MAX_SEQ

    """
    Run inference repeatedly with user-provided premise and hypothesis.
    Exit on empty input or when 'q', 'quit', or 'exit' is entered.
    """
    print("=" * 60)
    print("GPT-1 NLI Inference - Interactive Mode")
    print("  Enter 'q' or leave empty to exit")
    print("=" * 60)
    print()

    while True:
        # -- Get Premise
        try:
            premise = input("Premise    > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not premise or premise.lower() in ("q", "quit", "exit"):
            print("Exiting")
            break

        # -- Get Hypothesis
        try:
            hypothesis = input("Hypothesis > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not hypothesis or hypothesis.lower() in ("q", "quit", "exit"):
            print("Exiting")
            break

        # -- Run Inference & Display Results
        print()
        result = predict(nli_model, enc, premise, hypothesis, max_seq_len)
        print_result(premise, hypothesis, result)


if __name__ == "__main__":

    nli_model, enc = load_model(ckpt_dir=sys.argv[1])

    if len(sys.argv) > 2:
        predict_from_file(nli_model, enc, sys.argv[2])
    else:
        interactive_mode(nli_model, enc)
