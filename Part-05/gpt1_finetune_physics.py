#
# gpt1_finetune_physics.py  -  GPT-1 physics QA fine-tuning
#
# Usage:
#   python gpt1_finetune_physics.py [n_epochs] [pretrain_ckpt_dir]
#
#   Copyright (c) 2026, Hironobu Suzuki @ interdb.jp

import os
import sys
import time
import logging

if len(sys.argv) != 3:
    print(
        "Syntax Error:\n\tpython gpt1_finetune_physics.py [n_epochs] [pretrain_ckpt_dir]"
    )
    sys.exit(1)

os.environ.setdefault("TF_METAL_DEVICE_ENABLE", "1")
import tensorflow as tf

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "text_processing"))

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

from gpt1_finetune import (
    make_finetune_step,
    make_eval_step,
    build_nli_tf_dataset,
    train_loss,
    train_clf_loss,
    train_lm_loss,
    train_accuracy,
    val_loss,
    val_accuracy,
)
from gpt1_physics_qa_generator import (
    generate_dataset,
    PhysicsQADataset,
    PHYSICS_LABELS,
    NUM_PHYSICS_LABELS,
)
from bpe_encoder import BPEEncoder

logging.getLogger("tensorflow").setLevel(logging.ERROR)

CHECKPOINT = True

PHYSICS_FINETUNE_CONFIG = {
    "n_epochs": 8,
    "batch_size": 32,
    "max_seq_len": 64,  # 64 is sufficient because scenario texts are short.
    "lr": 6.25e-5,
    "lm_coef": 0.5,
    "grad_clip": 1.0,
    "warmup_steps": 100,
    "log_every": 50,
    "checkpoint_dir": "./checkpoints",
    "early_stopping_patience": 3,
}

DEVICE = get_device()


if check_required_files() == False:
    print("Execute prepare_datasets.py.")
    sys.exit(1)


# ============================================================
# Training loop with early stopping and best model saving.
# ============================================================


def finetune_physics(
    model,
    optimizer,
    train_ds,
    val_ds,
    n_epochs,
    ckpt_manager=None,
    log_every=20,
    lm_coef=0.5,
    grad_clip=1.0,
    best_ckpt_manager=None,
    early_stopping_patience=0,
    start_epoch: int = 1,
) -> float:
    ft_step = make_finetune_step(model, optimizer, lm_coef, grad_clip)
    ev_step = make_eval_step(model)
    best_val_acc = 0.0
    epochs_without_improvement = 0

    for epoch in range(start_epoch, n_epochs + 1):
        t0 = time.time()
        train_loss.reset_state()
        train_clf_loss.reset_state()
        train_lm_loss.reset_state()
        train_accuracy.reset_state()
        val_loss.reset_state()
        val_accuracy.reset_state()

        for batch, (ids, lbls) in enumerate(train_ds):
            ft_step(ids, lbls)
            if batch % log_every == 0:
                print(
                    f"Epoch {epoch} Batch {batch:>4d}  "
                    f"Loss {train_loss.result():.4f}  ClfLoss {train_clf_loss.result():.4f}  "
                    f"LMLoss {train_lm_loss.result():.4f}  Acc {train_accuracy.result():.4f}"
                )

        for ids, lbls in val_ds:
            ev_step(ids, lbls)

        if CHECKPOINT and ckpt_manager is not None:
            save_path = ckpt_manager.save()
            print(f"Saving checkpoint for epoch {epoch} at {save_path}")

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch}/{n_epochs}  "
            f"TrainLoss {train_loss.result():.4f}  TrainAcc {train_accuracy.result():.4f}  "
            f"ValLoss {val_loss.result():.4f}  ValAcc {val_accuracy.result():.4f}"
        )
        print(f"Time taken for 1 epoch: {elapsed:.1f}s")

        current_val_acc = float(val_accuracy.result())
        if current_val_acc > best_val_acc:
            best_val_acc = current_val_acc
            epochs_without_improvement = 0
            if best_ckpt_manager is not None:
                best_path = best_ckpt_manager.save()
                print(f"Val Accuracy improved -> Saved the best model: {best_path}")
        else:
            epochs_without_improvement += 1
            print(
                f"  No improvement in Val Accuracy ({epochs_without_improvement}/{early_stopping_patience})"
            )
            print()

        if (
            early_stopping_patience > 0
            and epochs_without_improvement >= early_stopping_patience
        ):
            print(
                f"Early stopping: Terminated because val\\_accuracy did not improve for {early_stopping_patience} consecutive epochs."
            )
            break

    print(f"Fine-tuning complete.  Best Val Accuracy: {best_val_acc:.4f}")
    return best_val_acc


if __name__ == "__main__":

    import re

    _n_epochs = (
        int(sys.argv[1]) if len(sys.argv) > 1 else PHYSICS_FINETUNE_CONFIG["n_epochs"]
    )
    _pretrain_ckpt = sys.argv[2] if len(sys.argv) > 2 else None
    _vocab_path = SNLI_CONFIG["vocab_file"]
    _merges_path = SNLI_CONFIG["merges_file"]

    # BPE encoder (Reuse the exact one from SNLI pretraining)
    enc = BPEEncoder.from_files(_vocab_path, _merges_path)
    print(f"[BPE] vocab_size={enc.vocab_size:,}")

    # Data generation (Synthetic from templates, no download needed)
    train_records, val_records = generate_dataset(seed=42)

    MAX_SEQ = PHYSICS_FINETUNE_CONFIG["max_seq_len"]
    builder = PhysicsQADataset(enc, max_seq_len=MAX_SEQ)
    train_data = builder.prepare(train_records, "train")
    val_data = builder.prepare(val_records, "validation")

    BATCH = PHYSICS_FINETUNE_CONFIG["batch_size"]
    train_ds = build_nli_tf_dataset(train_data, BATCH, shuffle=True)
    val_ds = build_nli_tf_dataset(val_data, BATCH)

    _n_train_steps = sum(1 for _ in train_ds) * _n_epochs
    print(
        f"[Dataset] train batches/epoch={sum(1 for _ in train_ds)}  total_steps={_n_train_steps}"
    )

    # Model definition (Keep position_embedding max_seq_len identical to pretraining)
    MODEL_MAX_SEQ = GPT1_CONFIG["max_seq_len"]
    with tf.device(DEVICE):
        gpt1 = GPT1Model(
            num_layers=GPT1_CONFIG["num_layers"],
            d_model=GPT1_CONFIG["d_model"],
            num_heads=GPT1_CONFIG["num_heads"],
            d_ffn=GPT1_CONFIG["d_ffn"],
            vocab_size=enc.vocab_size,
            max_seq_len=MODEL_MAX_SEQ,
            dropout_rate=GPT1_CONFIG["dropout_rate"],
            name="gpt1",
        )
        # Reuse GPT1ForNLI as-is with num_labels=6
        physics_model = GPT1ForNLI(
            gpt1, num_labels=NUM_PHYSICS_LABELS, name="gpt1_physics"
        )
        _dummy = tf.zeros((1, MAX_SEQ), dtype=tf.int32)
        physics_model(_dummy, training=False)

    # Load pretrained weights (SNLI pretraining)
    if _pretrain_ckpt and os.path.isdir(_pretrain_ckpt):
        _pt = tf.train.Checkpoint(model=physics_model.gpt1)
        _pm = tf.train.CheckpointManager(_pt, _pretrain_ckpt, max_to_keep=1)
        if _pm.latest_checkpoint:
            _pt.restore(_pm.latest_checkpoint).expect_partial()
            print(f"[Pretrain] Loaded weights: {_pm.latest_checkpoint}")
        else:
            print("[Pretrain] No checkpoint found -> Random initialization")
    else:
        print("[Pretrain] Checkpoint not specified -> Random initialization")

    # Optimizer
    ft_lr = GPT1LRSchedule(
        max_lr=PHYSICS_FINETUNE_CONFIG["lr"],
        warmup_steps=PHYSICS_FINETUNE_CONFIG["warmup_steps"],
        total_steps=max(_n_train_steps, PHYSICS_FINETUNE_CONFIG["warmup_steps"] + 1),
    )
    optimizer = build_optimizer(ft_lr, weight_decay=GPT1_CONFIG["l2_weight"])

    # Checkpoint
    ckpt_path = make_checkpoint_path(
        PHYSICS_FINETUNE_CONFIG["checkpoint_dir"],
        "finetune-physics",
        GPT1_CONFIG["num_layers"],
        GPT1_CONFIG["d_model"],
        GPT1_CONFIG["d_ffn"],
        GPT1_CONFIG["num_heads"],
    )
    ckpt, ckpt_mgr = build_checkpoint_manager(physics_model, optimizer, ckpt_path)

    start_epoch = 1
    if ckpt_mgr.latest_checkpoint:
        ckpt.restore(ckpt_mgr.latest_checkpoint).expect_partial()
        print(f"[Checkpoint] Restored: {ckpt_mgr.latest_checkpoint}")

        start_epoch += int(
            re.search(r"-(\d+)$", ckpt_mgr.latest_checkpoint).group(1)
        )

    best_ckpt_path = ckpt_path + "-best"
    best_ckpt, best_ckpt_mgr = build_checkpoint_manager(
        physics_model, optimizer, best_ckpt_path, max_to_keep=1
    )

    # Fine-tuning
    finetune_physics(
        model=physics_model,
        optimizer=optimizer,
        train_ds=train_ds,
        val_ds=val_ds,
        n_epochs=_n_epochs,
        ckpt_manager=ckpt_mgr,
        log_every=PHYSICS_FINETUNE_CONFIG["log_every"],
        lm_coef=PHYSICS_FINETUNE_CONFIG["lm_coef"],
        grad_clip=PHYSICS_FINETUNE_CONFIG["grad_clip"],
        best_ckpt_manager=best_ckpt_mgr,
        early_stopping_patience=PHYSICS_FINETUNE_CONFIG["early_stopping_patience"],
        start_epoch=start_epoch,
    )

    if best_ckpt_mgr.latest_checkpoint:
        best_ckpt.restore(best_ckpt_mgr.latest_checkpoint).expect_partial()
        print(f"[Best Checkpoint] Loaded: {best_ckpt_mgr.latest_checkpoint}")

    # Separate evaluation for unseen tokens, unseen combinations, and compound scenarios
    #  (Verifies both generalization beyond memorization and actual reading comprehension of questions)
    print(
        "\n-- Evaluation Breakdown: Unseen Combos / Unseen Tokens / Question Dependency --"
    )
    from gpt1_physics_qa_generator import generate_dataset_with_breakdown

    _, combo_records, entity_records, compound_records = (
        generate_dataset_with_breakdown(seed=42)
    )

    combo_data = builder.prepare(combo_records, "unseen_combo")
    entity_data = builder.prepare(entity_records, "unseen_entity")
    compound_data = builder.prepare(compound_records, "compound")
    combo_ds = build_nli_tf_dataset(combo_data, BATCH)
    entity_ds = build_nli_tf_dataset(entity_data, BATCH)
    compound_ds = build_nli_tf_dataset(compound_data, BATCH)

    ev_step = make_eval_step(physics_model)

    val_loss.reset_state()
    val_accuracy.reset_state()
    for ids, lbls in combo_ds:
        ev_step(ids, lbls)
    print(
        f"  Unseen combinations (Known tokens $X_{{i}}$, new patterns)   Accuracy : {val_accuracy.result():.4f}"
    )

    val_loss.reset_state()
    val_accuracy.reset_state()
    for ids, lbls in entity_ds:
        ev_step(ids, lbls)
    print(
        f"  Unseen tokens (Out-of-vocabulary words $\\lambda$)      Accuracy : {val_accuracy.result():.4f}"
    )

    val_loss.reset_state()
    val_accuracy.reset_state()
    for ids, lbls in compound_ds:
        ev_step(ids, lbls)
    print(
        f"  Question dependency (Same context, answer varies by $Q$) Accuracy : {val_accuracy.result():.4f}"
    )
    print(
        "  ^ High accuracy here proves the model is actively reading the question before answering."
    )

    print("\nPhysics QA fine-tuning completed.")
    print("Use gpt1_inference_physics.py for inference.")
