"""
snli_dataset.py
---------------
SNLI dataset loader and pre-processor for GPT-1 NLI fine-tuning.

Workflow
--------
1.  Load raw SNLI jsonl files (train / dev / test).
2.  Normalise each sentence with TextNormalizer.
3.  Encode each (premise, hypothesis) pair with BPEEncoder using the GPT-1
    NLI input format:  <s> premise $ hypothesis </s>
4.  Return tf.data.Dataset or raw Python lists ready for training.

SNLI label mapping
------------------
  "entailment"    -> 0
  "neutral"       -> 1
  "contradiction" -> 2
  "-"             -> skipped  (gold label not available for this example)
"""

import json
from pathlib import Path
from typing import Optional

from text_normalizer import fix_text
from bpe_encoder import BPEEncoder

# ==========================================
# Constants
# ==========================================

LABEL_MAP: dict[str, int] = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
}

NUM_LABELS = 3

# ==========================================
# Raw record loader
# ==========================================


def load_snli_jsonl(path: str | Path) -> list[dict]:
    """
    Load a SNLI .jsonl file and return a list of dicts with keys:
        premise, hypothesis, label (int), raw_label (str)
    Examples with gold label "-" are silently skipped.
    """
    records: list[dict] = []
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            raw_label = obj.get("gold_label", "-")
            if raw_label not in LABEL_MAP:
                continue  # skip uncertain examples
            premise = fix_text(obj["sentence1"])
            hypothesis = fix_text(obj["sentence2"])
            records.append(
                {
                    "premise": premise,
                    "hypothesis": hypothesis,
                    "label": LABEL_MAP[raw_label],
                    "raw_label": raw_label,
                }
            )
    return records


# ==========================================
# Encoded dataset builder
# ==========================================


class SNLIDataset:
    """
    Pre-processes SNLI splits and produces integer-encoded sequences.

    Parameters
    ----------
    encoder : BPEEncoder
        Trained BPE encoder (vocab + merge rules).
    max_seq_len : int
        Maximum token sequence length. Longer sequences are truncated.
    """

    def __init__(self, encoder: BPEEncoder, max_seq_len: int = 512) -> None:
        self.encoder = encoder
        self.max_seq_len = max_seq_len

    def prepare(
        self,
        records: list[dict],
        split_name: str = "train",
    ) -> dict[str, list]:
        """
        Encode all records and return a dict with:
            input_ids : list[list[int]]   - padded/truncated token id sequences
            labels    : list[int]         - SNLI label (0/1/2)
            lengths   : list[int]         - actual (unpadded) sequence length
        """
        pad_id = self.encoder.encoder.get("<pad>", 0)
        all_ids: list[list[int]] = []
        all_labels: list[int] = []
        all_lengths: list[int] = []

        skipped = 0
        for rec in records:
            ids = self.encoder.encode_with_special(rec["premise"], rec["hypothesis"])
            # Truncate to max_seq_len
            if len(ids) > self.max_seq_len:
                ids = ids[: self.max_seq_len]
                skipped += 1

            length = len(ids)
            # Pad to max_seq_len
            ids = ids + [pad_id] * (self.max_seq_len - length)

            all_ids.append(ids)
            all_labels.append(rec["label"])
            all_lengths.append(length)

        print(
            f"[SNLIDataset] {split_name}: {len(records)} examples loaded, "
            f"{skipped} truncated to {self.max_seq_len} tokens."
        )
        return {
            "input_ids": all_ids,
            "labels": all_labels,
            "lengths": all_lengths,
        }

    def to_tf_dataset(
        self,
        data: dict[str, list],
        batch_size: int = 32,
        shuffle: bool = False,
        shuffle_buffer: int = 10_000,
    ):
        """
        Convert the dict returned by `prepare` into a tf.data.Dataset.
        Requires TensorFlow to be installed.
        """
        import tensorflow as tf  # lazy import - not needed for data prep

        dataset = tf.data.Dataset.from_tensor_slices(
            {
                "input_ids": data["input_ids"],
                "labels": data["labels"],
                "lengths": data["lengths"],
            }
        )
        if shuffle:
            dataset = dataset.shuffle(shuffle_buffer, seed=42)
        dataset = dataset.batch(batch_size, drop_remainder=False)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset


# ==========================================
# Convenience function
# ==========================================


def build_snli_datasets(
    train_path: str | Path,
    dev_path: str | Path,
    test_path: str | Path,
    encoder: BPEEncoder,
    max_seq_len: int = 512,
    batch_size: int = 32,
    as_tf: bool = False,
) -> dict[str, object]:
    """
    High-level helper that loads all three splits and returns a dict:
        {
            "train": prepared_train_data_or_tf_dataset,
            "dev":   prepared_dev_data_or_tf_dataset,
            "test":  prepared_test_data_or_tf_dataset,
        }
    Set `as_tf=True` to get tf.data.Dataset objects instead of raw dicts.
    """
    builder = SNLIDataset(encoder, max_seq_len=max_seq_len)

    splits: dict[str, object] = {}
    for name, path in [("train", train_path), ("dev", dev_path), ("test", test_path)]:
        records = load_snli_jsonl(path)
        data = builder.prepare(records, split_name=name)
        if as_tf:
            splits[name] = builder.to_tf_dataset(
                data,
                batch_size=batch_size,
                shuffle=(name == "train"),
            )
        else:
            splits[name] = data

    return splits
