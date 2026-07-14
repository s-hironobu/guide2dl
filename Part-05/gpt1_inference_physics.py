#
# gpt1_inference_physics.py  -  GPT-1
#
#   Copyright (c) 2026, Hironobu Suzuki @ interdb.jp

import os
import sys
import re
import argparse
import logging
import numpy as np

if len(sys.argv) <= 1:
    print(
        "Syntax Error:\n\tpython gpt1_inference_physics.py [checkpoint-pretrain-dir] {file.tsv}"
    )
    sys.exit(1)

os.environ.setdefault("TF_METAL_DEVICE_ENABLE", "1")
import tensorflow as tf

logging.getLogger("tensorflow").setLevel(logging.ERROR)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "text_processing"))

from gpt1_model import (
    GPT1Model,
    GPT1ForNLI,
    GPT1_CONFIG,
    get_device,
)
from prepare_datasets import (
    SNLI_CONFIG,
    check_required_files,
)
from gpt1_physics_qa_generator import PHYSICS_LABELS, ID_TO_LABEL, NUM_PHYSICS_LABELS
from bpe_encoder import BPEEncoder
from text_normalizer import fix_text

DEVICE = get_device()

if check_required_files() == False:
    print("Execute prepare_datasets.py.")
    sys.exit(1)

# ============================================================
# Constants
# ============================================================

DEFAULT_MAX_SEQ = 64

LABEL_EMOJI = {
    "fast": "🏃",
    "slow": "🐢",
    "falls": "⬇️",
    "stays": "🧍",
    "slides": "🧊",
    "no_slide": "🚶",
}


# ============================================================
# Load model
# ============================================================


def load_model(
    ckpt_dir: str,
) -> tuple[GPT1ForNLI, BPEEncoder]:

    vocab_path = SNLI_CONFIG["vocab_file"]
    merges_path = SNLI_CONFIG["merges_file"]
    max_seq_len = DEFAULT_MAX_SEQ

    enc = BPEEncoder.from_files(vocab_path, merges_path)
    print(f"[BPE]  vocab_size={enc.vocab_size:,}")

    MODEL_MAX_SEQ = GPT1_CONFIG["max_seq_len"]
    with tf.device(DEVICE):
        gpt1 = GPT1Model(
            num_layers=GPT1_CONFIG["num_layers"],
            d_model=GPT1_CONFIG["d_model"],
            num_heads=GPT1_CONFIG["num_heads"],
            d_ffn=GPT1_CONFIG["d_ffn"],
            vocab_size=enc.vocab_size,
            max_seq_len=MODEL_MAX_SEQ,
            dropout_rate=0.0,
            name="gpt1",
        )
        model = GPT1ForNLI(gpt1, num_labels=NUM_PHYSICS_LABELS, name="gpt1_physics")
        _dummy = tf.zeros((1, max_seq_len), dtype=tf.int32)
        model(_dummy, training=False)

    _specific = None
    if re.search(r"/ckpt-\d+$", ckpt_dir):
        _specific = ckpt_dir
        _ckpt_base = os.path.dirname(ckpt_dir)
    else:
        _ckpt_base = ckpt_dir

    ckpt = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(ckpt, _ckpt_base, max_to_keep=None)
    restore_path = _specific or manager.latest_checkpoint

    if restore_path is None:
        raise FileNotFoundError(
            f"Checkpoint not found: {_ckpt_base}\n"
            f"  Run gpt1_finetune_physics.py to generate it."
        )
    ckpt.restore(restore_path).expect_partial()

    m = re.search(r"-(\d+)$", restore_path)
    print(
        f"[Checkpoint]  Restored epoch {m.group(1) if m else '?'} from {restore_path}"
    )
    print(f"[Device]      {DEVICE}\n")

    return model, enc


# ============================================================
# Inference for a Single Question
# ============================================================


def predict(
    model: GPT1ForNLI,
    enc: BPEEncoder,
    scenario: str,
    question: str,
    max_seq_len: int = DEFAULT_MAX_SEQ,
) -> dict:
    """
    Run inference on a (scenario, question) pair using the same steps as GPT1ForNLI.
    Uses encode_with_special() to encode the input into the format: <s> scenario $ question </s>
    """
    scenario_clean = fix_text(scenario.strip())
    question_clean = fix_text(question.strip())

    ids = enc.encode_with_special(scenario_clean, question_clean)
    if len(ids) > max_seq_len:
        ids = ids[: max_seq_len - 1] + [ids[-1]]
    input_len = len(ids)
    ids_padded = ids + [0] * (max_seq_len - input_len)

    input_tensor = tf.constant([ids_padded], dtype=tf.int32)
    with tf.device(DEVICE):
        clf_logits, _, _ = model(input_tensor, training=False)

    probs = tf.nn.softmax(clf_logits, axis=-1).numpy()[0]
    label_id = int(np.argmax(probs))

    return {
        "label": ID_TO_LABEL[label_id],
        "label_id": label_id,
        "confidence": float(probs[label_id]),
        "probs": {ID_TO_LABEL[i]: float(probs[i]) for i in range(NUM_PHYSICS_LABELS)},
        "input_len": input_len,
    }


# ============================================================
# Print result
# ============================================================


def print_result(scenario: str, question: str, result: dict) -> None:
    emoji = LABEL_EMOJI.get(result["label"], "❓")
    p = result["probs"]
    print(f"  Scenario   : {scenario}")
    print(f"  Question   : {question}")
    print(
        f"  Prediction : {emoji}  {result['label']:>10s}  (confidence: {result['confidence']*100:.1f}%)"
    )
    scores = "  ".join(f"{k}={v*100:.1f}%" for k, v in p.items())
    print(f"  Scores     : {scores}")
    print(f"  Token length: {result['input_len']}")
    print()


def predict_from_file(
    nli_model: GPT1ForNLI,
    enc: BPEEncoder,
    file_path: str,
) -> None:
    """
    Process a TSV file (premise<TAB>hypothesis) line by line and display the results.

    File format:
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
                print(
                    f"  [Line {lineno}] Skipped because it is not tab-separated: {line!r}"
                )
                n_skip += 1
                continue

            scenario, question = parts[0], parts[1]
            result = predict(nli_model, enc, scenario, question, max_seq_len)
            print(f"[Pair {lineno}]")
            print_result(scenario, question, result)
            n_ok += 1

    print(f"Completed: {n_ok} pairs processed / {n_skip} lines skipped")


# ============================================================
# Interactive mode
# ============================================================


def interactive_mode(
    nli_model: GPT1ForNLI,
    enc: BPEEncoder,
) -> None:

    max_seq_len = DEFAULT_MAX_SEQ

    print("=" * 64)
    print(
        "GPT-1 Physics Concepts QA Inference - Dialogue Mode (Type 'q' or press Enter to quit)"
    )
    print(f"  Answer Vocabulary: {', '.join(PHYSICS_LABELS)}")
    print("=" * 64)
    print()

    while True:
        try:
            scenario = input("Scenario > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not scenario or scenario.lower() in ("q", "quit", "exit"):
            print("Exiting.")
            break

        try:
            question = input("Question > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not question or question.lower() in ("q", "quit", "exit"):
            print("Exiting.")
            break

        print()
        result = predict(nli_model, enc, scenario, question, max_seq_len)
        print_result(scenario, question, result)


if __name__ == "__main__":

    nli_model, enc = load_model(ckpt_dir=sys.argv[1])

    if len(sys.argv) > 2:
        predict_from_file(nli_model, enc, sys.argv[2])
    else:
        interactive_mode(nli_model, enc)
