#
# GPT-1 Supervised Fine-tuning - Natural Language Inference (SNLI)
#
#   Copyright (c) 2026, Hironobu Suzuki @ interdb.jp

import os
import sys
import time
import logging
import numpy as np
import tensorflow as tf

if len(sys.argv) <= 2:
    print("Syntax Error:\n\tpython gpt1_finetune.py [n_epochs] [pretrain_ckpt_dir]")
    sys.exit(1)

os.environ.setdefault("TF_METAL_DEVICE_ENABLE", "1")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from gpt1_model import (
    GPT1Model,
    GPT1ForNLI,
    GPT1_CONFIG,
    GPT1LRSchedule,
    build_optimizer,
    build_checkpoint_manager,
    make_checkpoint_path,
    get_device,
)

from prepare_datasets import (
    SNLI_CONFIG,
    check_required_files,
)

logging.getLogger("tensorflow").setLevel(logging.ERROR)

DEVICE = get_device()

if check_required_files() == False:
    print("Run prepare_datasets.py.")
    sys.exit(1)

# ============================================================
# Configuration
# ============================================================

CHECKPOINT = True

FINETUNE_CONFIG = {
    "n_epochs": 3,
    "batch_size": 32,
    "max_seq_len": GPT1_CONFIG["max_seq_len"],
    "lr": 6.25e-5,
    "lm_coef": 0.5,
    "grad_clip": 1.0,
    "warmup_steps": 500,
    "log_every": 100,  # print every N batches
    "checkpoint_dir": "./checkpoints",
}

# SNLI label mapping (matches snli_dataset.py)
LABEL_NAMES = {0: "entailment", 1: "neutral", 2: "contradiction"}


# ============================================================
# Loss functions
# ============================================================

_loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction="none"
)


def clf_loss_fn(labels, logits):
    """
    Classification cross-entropy  (L2 in Eq. 5).

    Parameters
    ----------
    labels : (B,)    integer class indices  {0=entailment, 1=neutral, 2=contradiction}
    logits : (B, 3)  raw classification logits

    Returns
    -------
    Scalar mean cross-entropy over the batch.
    """
    loss = tf.keras.losses.sparse_categorical_crossentropy(
        labels, logits, from_logits=True
    )  # (B,)
    return tf.reduce_mean(loss)


def lm_loss_fn(targets, logits):
    """
    Masked language-model cross-entropy  (L1 in Eq. 5).

    Identical to gpt1_pretrain.lm_loss_fn - shared auxiliary objective.

    Parameters
    ----------
    targets : (B, T)     next-token ids;  0 = pad -> excluded from loss
    logits  : (B, T, V)  raw predicted logits

    Returns
    -------
    Scalar: sum(loss * mask) / sum(mask)
    """
    loss = _loss_object(targets, logits)  # (B, T)
    mask = tf.cast(tf.not_equal(targets, 0), tf.float32)
    loss = loss * mask
    return tf.reduce_sum(loss) / (tf.reduce_sum(mask) + 1e-9)


# ============================================================
# Dataset builder
# ============================================================


def build_nli_tf_dataset(
    data_dict: dict,
    batch_size: int,
    shuffle: bool = False,
    shuffle_buffer: int = 10_000,
    seed: int = 42,
) -> tf.data.Dataset:
    """
    Convert a dict produced by SNLIDataset.prepare() into a tf.data.Dataset.

    Parameters
    ----------
    data_dict : dict with keys
        'input_ids' : list[list[int]]  shape (N, max_seq_len), dtype int32
        'labels'    : list[int]        shape (N,),              dtype int32
        'lengths'   : list[int]        (not used here, for reference only)

    Returns
    -------
    tf.data.Dataset  yielding (input_ids, labels) tuples of shape
        (batch_size, max_seq_len) and (batch_size,)  dtype int32
    """
    input_ids = tf.constant(data_dict["input_ids"], dtype=tf.int32)
    labels = tf.constant(data_dict["labels"], dtype=tf.int32)

    ds = tf.data.Dataset.from_tensor_slices((input_ids, labels))
    if shuffle:
        ds = ds.shuffle(shuffle_buffer, seed=seed)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds


# ============================================================
# Metrics
# ============================================================

train_loss = tf.keras.metrics.Mean(name="train_loss")
train_clf_loss = tf.keras.metrics.Mean(name="train_clf_loss")
train_lm_loss = tf.keras.metrics.Mean(name="train_lm_loss")
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")

val_loss = tf.keras.metrics.Mean(name="val_loss")
val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="val_accuracy")


# ============================================================
# @tf.function training / evaluation steps
#     Factory pattern -> one compiled function per model instance.
# ============================================================

_finetune_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),  # input_ids (B, T)
    tf.TensorSpec(shape=(None,), dtype=tf.int32),  # labels    (B,)
]


def make_finetune_step(model: GPT1ForNLI, optimizer, lm_coef: float, grad_clip: float):
    """
    Return a @tf.function-compiled fine-tuning step for *model*.

    The returned function implements the joint objective from Eq. 5:
        L3 = L2(clf) + \lambda \dot L1(lm)

    It updates:
        train_loss     <- L3 (joint)
        train_clf_loss <- L2 (classification only)
        train_lm_loss  <- L1 (auxiliary LM only)
        train_accuracy <- NLI classification accuracy
    """

    @tf.function(input_signature=_finetune_step_signature)
    def finetune_step(input_ids, labels):
        """
        One gradient-update step on a (premise, hypothesis, label) batch.

        input_ids : (B, T)  NLI-format sequences  <s> premise $ hypothesis </s>
        labels    : (B,)    integer class  {0=entailment, 1=neutral, 2=contradiction}

        Forward (GPT-1 Section 3.2 & 3.3):
          clf_logits  = Wy \dot h_n^{</s>}                   -> L2 (classification)
          lm_logits   = softmax(h_n \dot We^T)  at all steps  -> L1 (auxiliary LM)
          L3 = L2 + \lambda \dot L1
        """
        with tf.device(DEVICE):
            with tf.GradientTape() as tape:
                clf_logits, lm_logits, _ = model(input_ids, training=True)
                # clf_logits : (B, 3)
                # lm_logits  : (B, T, V)

                # L2: NLI classification loss
                c_loss = clf_loss_fn(labels, clf_logits)

                # L1: auxiliary LM loss on the NLI sequence
                # Position j predicts token j+1 within the full sequence.
                lm_targets = input_ids[:, 1:]  # (B, T-1)  next tokens
                lm_pred = lm_logits[:, :-1, :]  # (B, T-1, V)  shifted logits
                l_loss = lm_loss_fn(lm_targets, lm_pred)

                # L3: joint objective (Eq. 5)
                total = c_loss + lm_coef * l_loss

            gradients = tape.gradient(total, model.trainable_variables)
            gradients, _ = tf.clip_by_global_norm(gradients, grad_clip)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(total)
        train_clf_loss(c_loss)
        train_lm_loss(l_loss)
        train_accuracy.update_state(labels, clf_logits)

    return finetune_step


def make_eval_step(model: GPT1ForNLI):
    """
    Return a @tf.function-compiled evaluation step (no gradient computation).

    Updates val_loss and val_accuracy with classification cross-entropy only.
    The auxiliary LM objective is not included in validation (paper Eq. 5 is
    a training objective; evaluation uses L2 alone).
    """

    @tf.function(input_signature=_finetune_step_signature)
    def eval_step(input_ids, labels):
        clf_logits, _, _ = model(input_ids, training=False)
        c_loss = clf_loss_fn(labels, clf_logits)
        val_loss(c_loss)
        val_accuracy.update_state(labels, clf_logits)

    return eval_step


# ============================================================
# Training loop
# ============================================================


def finetune(
    model: GPT1ForNLI,
    optimizer,
    train_dataset: tf.data.Dataset,
    dev_dataset: tf.data.Dataset,
    n_epochs: int,
    ckpt_manager=None,
    log_every: int = 100,
    lm_coef: float = 0.5,
    grad_clip: float = 1.0,
    start_epoch: int = 1,
) -> None:
    """
    Main fine-tuning loop with per-epoch validation.

    After every epoch:
      (1) Evaluate on dev split (classification loss + accuracy only).
      (2) Save checkpoint (if CHECKPOINT == True).
      (3) Print epoch summary.

    The best validation accuracy is tracked and reported at the end.
    """
    ft_step = make_finetune_step(model, optimizer, lm_coef, grad_clip)
    ev_step = make_eval_step(model)
    best_val_acc = 0.0

    for epoch in range(start_epoch, n_epochs + 1):
        start = time.time()

        # Reset per-epoch metrics
        train_loss.reset_state()
        train_clf_loss.reset_state()
        train_lm_loss.reset_state()
        train_accuracy.reset_state()
        val_loss.reset_state()
        val_accuracy.reset_state()

        # Training
        for batch, (input_ids, labels) in enumerate(train_dataset):
            ft_step(input_ids, labels)

            if batch % log_every == 0:
                print(
                    "Epoch {} Batch {:>5d}  "
                    "Loss {:.4f}  ClfLoss {:.4f}  LMLoss {:.4f}  "
                    "Accuracy {:.4f}".format(
                        epoch,
                        batch,
                        train_loss.result(),
                        train_clf_loss.result(),
                        train_lm_loss.result(),
                        train_accuracy.result(),
                    )
                )

        # Validation
        for input_ids, labels in dev_dataset:
            ev_step(input_ids, labels)

        # Checkpoint
        if CHECKPOINT and ckpt_manager is not None:
            ckpt_save_path = ckpt_manager.save()
            print("Saving checkpoint for epoch {} at {}".format(epoch, ckpt_save_path))

        # Epoch summary
        elapsed = time.time() - start
        print(
            "Epoch {}/{}  "
            "TrainLoss {:.4f}  TrainAcc {:.4f}  "
            "ValLoss {:.4f}  ValAcc {:.4f}".format(
                epoch,
                n_epochs,
                train_loss.result(),
                train_accuracy.result(),
                val_loss.result(),
                val_accuracy.result(),
            )
        )
        print("Time taken for 1 epoch: {:.4f} secs\n".format(elapsed))

        if float(val_accuracy.result()) > best_val_acc:
            best_val_acc = float(val_accuracy.result())

    print("Fine-tuning complete.  Best Val Accuracy: {:.4f}".format(best_val_acc))


# ============================================================
# Test-set evaluation helper
# ============================================================


def evaluate_test(model: GPT1ForNLI, test_dataset: tf.data.Dataset) -> dict:
    """
    Run inference on *test_dataset* and return per-class accuracy + overall accuracy.

    Returns
    -------
    dict with keys 'overall', 'entailment', 'neutral', 'contradiction'
    """
    ev_step = make_eval_step(model)
    val_loss.reset_state()
    val_accuracy.reset_state()

    # Per-class counters for detailed breakdown
    class_correct = {0: 0, 1: 0, 2: 0}
    class_total = {0: 0, 1: 0, 2: 0}

    for input_ids, labels in test_dataset:
        ev_step(input_ids, labels)

        # Detailed per-class stats (eager mode - outside @tf.function)
        clf_logits, _, _ = model(input_ids, training=False)
        preds = tf.argmax(clf_logits, axis=-1)
        for true_lbl, pred_lbl in zip(labels.numpy(), preds.numpy()):
            class_total[int(true_lbl)] += 1
            class_correct[int(true_lbl)] += int(true_lbl == pred_lbl)

    results = {
        "overall": float(val_accuracy.result()),
        "test_loss": float(val_loss.result()),
    }
    for cls_id, name in LABEL_NAMES.items():
        n = class_total[cls_id]
        results[name] = class_correct[cls_id] / n if n > 0 else 0.0

    return results


if __name__ == "__main__":
    _tp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "text_processing")
    sys.path.insert(0, _tp)
    from bpe_encoder import BPEEncoder  # noqa: E402
    from snli_dataset import SNLIDataset, load_snli_jsonl  # noqa: E402
    import re

    _n_epochs = int(sys.argv[1]) if len(sys.argv) > 1 else FINETUNE_CONFIG["n_epochs"]
    _pretrain_ckpt = sys.argv[2] if len(sys.argv) > 2 else None

    _snli_dir = SNLI_CONFIG["snli_dir"]
    _vocab_path = SNLI_CONFIG["vocab_file"]
    _merges_path = SNLI_CONFIG["merges_file"]

    # BPE encoder
    enc = BPEEncoder.from_files(_vocab_path, _merges_path)
    print("BPE encoder loaded: vocab_size={:,}".format(enc.vocab_size))

    # SNLI dataset
    snli_builder = SNLIDataset(enc, max_seq_len=FINETUNE_CONFIG["max_seq_len"])

    print("Loading SNLI splits ...")
    train_records = load_snli_jsonl(os.path.join(_snli_dir, "snli_1.0_train.jsonl"))
    dev_records = load_snli_jsonl(os.path.join(_snli_dir, "snli_1.0_dev.jsonl"))
    test_records = load_snli_jsonl(os.path.join(_snli_dir, "snli_1.0_test.jsonl"))

    train_data = snli_builder.prepare(train_records, "train")
    dev_data = snli_builder.prepare(dev_records, "dev")
    test_data = snli_builder.prepare(test_records, "test")

    train_dataset = build_nli_tf_dataset(
        train_data, FINETUNE_CONFIG["batch_size"], shuffle=True
    )
    dev_dataset = build_nli_tf_dataset(dev_data, FINETUNE_CONFIG["batch_size"])
    test_dataset = build_nli_tf_dataset(test_data, FINETUNE_CONFIG["batch_size"])

    _n_train_steps = sum(1 for _ in train_dataset) * _n_epochs
    print(
        "Train batches/epoch: {:,}  Total steps: {:,}".format(
            sum(1 for _ in train_dataset), _n_train_steps
        )
    )

    # Model
    with tf.device(DEVICE):
        gpt1 = GPT1Model(
            num_layers=GPT1_CONFIG["num_layers"],
            d_model=GPT1_CONFIG["d_model"],
            num_heads=GPT1_CONFIG["num_heads"],
            d_ffn=GPT1_CONFIG["d_ffn"],
            vocab_size=enc.vocab_size,
            max_seq_len=GPT1_CONFIG["max_seq_len"],
            dropout_rate=GPT1_CONFIG["dropout_rate"],
            name="gpt1",
        )
        nli_model = GPT1ForNLI(gpt1, num_labels=3, name="gpt1_nli")

    # Load pre-trained backbone weights (if checkpoint provided)
    # Build weights first so restore has something to write into.
    with tf.device(DEVICE):
        _dummy = tf.zeros((1, FINETUNE_CONFIG["max_seq_len"]), dtype=tf.int32)
        nli_model(_dummy, training=False)

    if _pretrain_ckpt and os.path.isdir(_pretrain_ckpt):
        _pt_ckpt = tf.train.Checkpoint(model=nli_model.gpt1)
        _pt_mgr = tf.train.CheckpointManager(_pt_ckpt, _pretrain_ckpt, max_to_keep=1)
        if _pt_mgr.latest_checkpoint:
            _pt_ckpt.restore(_pt_mgr.latest_checkpoint).expect_partial()
            print(
                "Pre-trained weights loaded from {}".format(_pt_mgr.latest_checkpoint)
            )
        else:
            print("No pre-training checkpoint found - initialising from scratch.")
    else:
        print("No pre-training checkpoint provided - initialising from scratch.")

    # Optimizer
    ft_lr = GPT1LRSchedule(
        max_lr=FINETUNE_CONFIG["lr"],
        warmup_steps=FINETUNE_CONFIG["warmup_steps"],
        total_steps=max(_n_train_steps, FINETUNE_CONFIG["warmup_steps"] + 1),
    )
    optimizer = build_optimizer(ft_lr, weight_decay=GPT1_CONFIG["l2_weight"])

    # Fine-tuning checkpoint
    ckpt_path = make_checkpoint_path(
        FINETUNE_CONFIG["checkpoint_dir"],
        "finetune-nli",
        GPT1_CONFIG["num_layers"],
        GPT1_CONFIG["d_model"],
        GPT1_CONFIG["d_ffn"],
        GPT1_CONFIG["num_heads"],
    )
    ckpt, ckpt_manager = build_checkpoint_manager(nli_model, optimizer, ckpt_path)

    start_epoch = 1
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print("Latest checkpoint restored!!")

        start_epoch += int(
            re.search(r"-(\d+)$", ckpt_manager.latest_checkpoint).group(1)
        )

    # Run fine-tuning
    finetune(
        model=nli_model,
        optimizer=optimizer,
        train_dataset=train_dataset,
        dev_dataset=dev_dataset,
        n_epochs=_n_epochs,
        ckpt_manager=ckpt_manager,
        log_every=FINETUNE_CONFIG["log_every"],
        lm_coef=FINETUNE_CONFIG["lm_coef"],
        grad_clip=FINETUNE_CONFIG["grad_clip"],
        start_epoch=start_epoch,
    )

    # Test-set evaluation
    print("\n-- Test Set Evaluation")
    results = evaluate_test(nli_model, test_dataset)
    print("  Overall Accuracy : {:.4f}".format(results["overall"]))
    print("  Test Loss        : {:.4f}".format(results["test_loss"]))
    for name in ("entailment", "neutral", "contradiction"):
        print("  {:>15s}  : {:.4f}".format(name, results[name]))
