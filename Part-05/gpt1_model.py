#
# GPT-1 Core Model - TensorFlow Implementation
#
# Paper: "Improving Language Understanding by Generative Pre-Training"
#        Radford et al., OpenAI, 2018
#        https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
#
#   Copyright (c) 2026, Hironobu Suzuki @ interdb.jp

import math
import numpy as np
import tensorflow as tf


# ========================================
# Hyper-parameter configuration
# ========================================

GPT1_CONFIG = {
    # Architecture
    "num_layers": 12,  # number of Transformer blocks
    "d_model": 512,  # hidden / embedding dimension
    "num_heads": 8,  # attention heads  (depth = 512/8 = 64)
    "d_ffn": 2048,  # FFN inner dimension  (4 \times d_model)
    "vocab_size": 40478,  # BPE 40 000 merges + 478 special tokens
    "max_seq_len": 128,  # context window
    # Regularisation
    "dropout_rate": 0.1,  # residual / embedding / attention dropout
    "l2_weight": 0.01,  # AdamW-style L2 on non-bias/gain weights
    # Fine-tuning (NLI)
    "num_labels": 3,  # entailment / neutral / contradiction
    "lm_coef": 0.5,  # \lambda - auxiliary LM loss coefficient (Eq. 5)
}


# ========================================
# Define functions
# ========================================

import platform, subprocess


def get_device():
    _gpus = tf.config.list_physical_devices("GPU")
    DEVICE = None
    if _gpus:
        for _gpu in _gpus:
            tf.config.experimental.set_memory_growth(_gpu, True)
        DEVICE = "/GPU:0"
        print(f"[Device] Metal GPU is available -> {_gpus[0].name}")
    else:
        DEVICE = "/CPU:0"
        print("[Device] GPU not found -> Use CPU.")

    return DEVICE


def _is_m1_mac():
    """Detect Apple Silicon"""
    if platform.system() != "Darwin":
        return False
    if platform.machine() not in ("arm64", "aarch64"):
        return False
    try:
        model = (
            subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"])
            .decode()
            .strip()
        )
        return "Apple" in model
    except subprocess.CalledProcessError:
        return False


def build_optimizer(lr_schedule, weight_decay: float = GPT1_CONFIG["l2_weight"]):
    """
    Build the AdamW optimizer used in GPT-1.

    AdamW applies weight decay only to weight matrices (not biases/gains),
    matching the paper's "modified L2 regularisation".

    Falls back to legacy Adam on M1/M2 Macs (same pattern as reference).
    """
    try:
        # TF >= 2.11 exposes AdamW directly
        if _is_m1_mac():
            optimizer = tf.keras.optimizers.legacy.AdamW(
                learning_rate=lr_schedule,
                weight_decay=weight_decay,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-8,
            )
        else:
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=lr_schedule,
                weight_decay=weight_decay,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-8,
            )
    except AttributeError:
        # TF < 2.11 fallback: plain Adam (weight decay applied manually in train loop)
        if _is_m1_mac():
            optimizer = tf.keras.optimizers.legacy.Adam(
                learning_rate=lr_schedule,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-8,
            )
        else:
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=lr_schedule,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-8,
            )
    return optimizer


#
# GELU activation
#
def gelu(x):
    """
    Gaussian Error Linear Unit
    """
    cdf = 0.5 * (1.0 + tf.math.erf(x / tf.cast(tf.math.sqrt(2.0), x.dtype)))
    return x * cdf


#
# Mask utilities
#
def create_padding_mask(seq):
    """
    Returns 1.0 for pad positions (token-id == 0), 0.0 elsewhere.
    Output shape: (batch, 1, 1, seq_len)  - broadcast-ready for MHA.
    """
    mask = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]  # (B, 1, 1, T)


def create_causal_mask(seq_len):
    """
    Upper-triangular mask that blocks attention to future tokens.
    Value 1.0 = position is masked out (will receive -1e9 before softmax).
    Output shape: (seq_len, seq_len)
    """
    # band_part(..., -1, 0) keeps lower triangle (including diagonal)
    mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    return mask  # (T, T)


def create_gpt1_mask(input_ids):
    """
    Combine causal mask and padding mask into a single attention mask.

    GPT-1 is Decoder-only, so every forward pass needs:
      - causal mask  (no attending to future tokens)
      - padding mask (no attending to <pad> tokens)

    The two masks are merged with tf.maximum so that any position
    flagged by either mask is blocked.

    Returns shape: (batch, 1, seq_len, seq_len)
    """
    seq_len = tf.shape(input_ids)[1]
    pad_mask = create_padding_mask(input_ids)  # (B, 1, 1,  T)
    causal_mask = create_causal_mask(seq_len)  # (T, T)
    # broadcast: (B,1,1,T) and (T,T) -> (B,1,T,T)
    combined = tf.maximum(pad_mask, causal_mask)
    return combined  # (B, 1, T, T)


#
#  Scaled dot-product attention
#
def scaled_dot_product_attention(q, k, v, mask):
    """
    Args
    ----
    q, k, v : (..., seq_len, depth)
    mask    : broadcastable to (..., seq_len_q, seq_len_k), or None

    Returns
    -------
    output          : (..., seq_len_q, depth)
    attention_weights : (..., seq_len_q, seq_len_k)
    """
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., Tq, Tk)

    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += mask * -1e9  # mask -> - \infinity

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)

    return output, attention_weights


#
# Position-wise Feed-Forward Network  (FFN with GELU)
#
def point_wise_feed_forward_network(d_model, d_ffn):
    """
    Two-layer FFN used inside every Transformer block.
    GPT-1 uses GELU (not ReLU as in the original Transformer).
    """
    _init = tf.keras.initializers.TruncatedNormal(stddev=0.02)
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(d_ffn, activation=gelu, kernel_initializer=_init),
            tf.keras.layers.Dense(d_model, kernel_initializer=_init),
        ],
        name="ffn",
    )


# ========================================
# Define Classes
# ========================================


#
# Multi-Head (Self-)Attention
#
class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Multi-head self-attention.

    Parameter order for call() matches the reference Transformer-tf.py:
        call(v, k, q, mask)

    Weight initialisation follows GPT-1: TruncatedNormal(stddev=0.02).
    """

    # Mask is passed as an explicit argument - suppress Keras implicit propagation.
    supports_masking = True

    def __init__(self, d_model, num_heads, **kwargs):
        super().__init__(**kwargs)
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        _init = tf.keras.initializers.TruncatedNormal(stddev=0.02)

        self.Wq = tf.keras.layers.Dense(
            d_model, use_bias=True, kernel_initializer=_init
        )
        self.Wk = tf.keras.layers.Dense(
            d_model, use_bias=True, kernel_initializer=_init
        )
        self.Wv = tf.keras.layers.Dense(
            d_model, use_bias=True, kernel_initializer=_init
        )
        self.dense = tf.keras.layers.Dense(
            d_model, use_bias=True, kernel_initializer=_init
        )

    def _split_heads(self, x, batch_size):
        """(B, T, d_model) -> (B, num_heads, T, depth)"""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.Wq(q)  # (B, T, d_model)
        k = self.Wk(k)
        v = self.Wv(v)

        q = self._split_heads(q, batch_size)  # (B, H, T, depth)
        k = self._split_heads(k, batch_size)
        v = self._split_heads(v, batch_size)

        # Scaled dot-product attention per head
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask
        )
        # (B, H, T, depth)

        # Concatenate heads
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        # (B, T, H, depth)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        # (B, T, d_model)

        output = self.dense(concat_attention)  # (B, T, d_model)
        return output, attention_weights

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_model": self.d_model, "num_heads": self.num_heads})
        return cfg


#
# GPT-1 Transformer Block
#
class GPT1Block(tf.keras.layers.Layer):
    """
    Single GPT-1 Transformer block.
    """

    # GPT-1 manages its own explicit mask argument; tell Keras not to
    # propagate the Embedding layer's implicit mask through this layer.
    supports_masking = True

    def __init__(self, d_model, num_heads, d_ffn, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)

        self.mha = MultiHeadAttention(d_model, num_heads, name="mha")
        self.ffn = point_wise_feed_forward_network(d_model, d_ffn)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

        # Store for get_config
        self._d_model = d_model
        self._num_heads = num_heads
        self._d_ffn = d_ffn
        self._dropout_rate = dropout_rate

    def call(self, x, training, mask):
        """
        Args
        ----
        x        : (B, T, d_model)
        training : bool
        mask     : (B, 1, T, T)  combined causal + padding mask

        Returns
        -------
        out2           : (B, T, d_model)
        attention_weights : (B, num_heads, T, T)
        """
        # Masked self-attention  (v=k=q=x)
        attn_output, attn_weights = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # Post-norm

        # Position-wise FFN
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # Post-norm

        return out2, attn_weights

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "d_model": self._d_model,
                "num_heads": self._num_heads,
                "d_ffn": self._d_ffn,
                "dropout_rate": self._dropout_rate,
            }
        )
        return cfg


#
# GPT-1 Core Model  (Unsupervised Pre-training)
#
class GPT1Model(tf.keras.Model):
    """
    GPT-1 language model for unsupervised pre-training.
    """

    def __init__(
        self,
        num_layers: int = GPT1_CONFIG["num_layers"],
        d_model: int = GPT1_CONFIG["d_model"],
        num_heads: int = GPT1_CONFIG["num_heads"],
        d_ffn: int = GPT1_CONFIG["d_ffn"],
        vocab_size: int = GPT1_CONFIG["vocab_size"],
        max_seq_len: int = GPT1_CONFIG["max_seq_len"],
        dropout_rate: float = GPT1_CONFIG["dropout_rate"],
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._num_layers = num_layers
        self._d_model = d_model
        self._num_heads = num_heads
        self._d_ffn = d_ffn
        self._vocab_size = vocab_size
        self._max_seq_len = max_seq_len
        self._dropout_rate = dropout_rate

        _init = tf.keras.initializers.TruncatedNormal(stddev=0.02)

        # Token embedding  We   (B, T) -> (B, T, d_model)
        self.token_embedding = tf.keras.layers.Embedding(
            vocab_size,
            d_model,
            embeddings_initializer=_init,
            name="token_embedding",
        )

        # Learned positional embedding  Wp
        # Paper: "We trained ... with a learned position embedding"
        # Shape: (max_seq_len, d_model);  indexed by position [0, T)
        self.position_embedding = tf.keras.layers.Embedding(
            max_seq_len,
            d_model,
            embeddings_initializer=_init,
            name="position_embedding",
        )

        # Embedding dropout
        self.emb_dropout = tf.keras.layers.Dropout(dropout_rate, name="emb_dropout")

        # Transformer blocks
        self.blocks = [
            GPT1Block(d_model, num_heads, d_ffn, dropout_rate, name=f"block_{i+1}")
            for i in range(num_layers)
        ]

        # Final layer normalisation
        # Not in the original 2018 paper but used in nearly all GPT-1
        # reference implementations; improves training stability.
        self.ln_f = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="ln_f")

    def call(self, input_ids, training=False, mask=None):
        """
        Forward pass.

        Args
        ----
        input_ids : (B, T)  integer token ids
        training  : bool
        mask      : (B, 1, T, T) combined causal+padding mask.
                    If None it is created automatically inside this call.

        Returns
        -------
        hidden_states    : (B, T, d_model)   final block activations
        attention_weights : dict  {f"block_{i+1}": tensor(B, H, T, T)}
        """
        if mask is None:
            mask = create_gpt1_mask(input_ids)

        seq_len = tf.shape(input_ids)[1]

        # h_a = token_embedding + position_embedding
        positions = tf.range(seq_len)  # [0, 1, ..., T-1]
        x = self.token_embedding(input_ids)  # (B, T, d_model)
        x = x + self.position_embedding(positions)  # broadcast over B
        x = self.emb_dropout(x, training=training)

        # h_l = transformer_block(h_l-1)
        attention_weights = {}
        for block in self.blocks:
            x, attn_w = block(x, training=training, mask=mask)
            attention_weights[block.name] = attn_w

        x = self.ln_f(x)  # (B, T, d_model)
        return x, attention_weights

    def compute_lm_logits(self, hidden_states):
        """
        Language model output head with tied token-embedding weights.

        P(u) = softmax(h_n \dot We^T)          (paper Eq. 1)

        Args
        ----
        hidden_states : (B, T, d_model)

        Returns
        -------
        logits : (B, T, vocab_size)   (raw, before softmax)
        """
        We = self.token_embedding.embeddings  # (vocab_size, d_model)
        logits = tf.matmul(hidden_states, tf.transpose(We))
        return logits  # (B, T, vocab_size)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "num_layers": self._num_layers,
                "d_model": self._d_model,
                "num_heads": self._num_heads,
                "d_ffn": self._d_ffn,
                "vocab_size": self._vocab_size,
                "max_seq_len": self._max_seq_len,
                "dropout_rate": self._dropout_rate,
            }
        )
        return cfg


#
# GPT-1 for NLI Fine-tuning
#
class GPT1ForNLI(tf.keras.Model):
    """
    GPT-1 with a linear classification head for Natural Language Inference.

    Input format (paper Section 3.3, Textual Entailment):
        <s>  premise tokens  $  hypothesis tokens  </s>

    Classification head (paper Section 3.2):
        P(y | x^1, ..., x^n) = softmax(h_n^m \dot Wy)
        where h_n^m is the hidden state of the *last token* (</s>) after the
        final Transformer block.

    Joint fine-tuning objective (paper Eq. 5):
        L_3(C) = L_2(C) + \lambda \dot L_1(C)
        - L_2 : cross-entropy classification loss
        - L_1 : auxiliary language-model loss  (\lambda = 0.5 by default)
        Both losses are computed inside the training routine (not here),
        but this model returns both clf_logits and lm_logits.
    """

    def __init__(
        self,
        gpt1: GPT1Model,
        num_labels: int = GPT1_CONFIG["num_labels"],
        clf_dropout_rate: float = GPT1_CONFIG["dropout_rate"],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gpt1 = gpt1
        self._num_labels = num_labels

        _init = tf.keras.initializers.TruncatedNormal(stddev=0.02)

        self.clf_dropout = tf.keras.layers.Dropout(clf_dropout_rate, name="clf_dropout")
        self.classifier = tf.keras.layers.Dense(
            num_labels,
            kernel_initializer=_init,
            name="classifier",
        )

    def _gather_last_real_token(self, hidden, ids):
        """
        Extract the hidden state at the last actual token position, avoiding the impact of padding.

        Since pad_id is always 0 and padding only appears at the end of the sequence
        (e.g., <s> ... </s> <pad> <pad> ...), the count of non-zero tokens directly represents
        the actual sequence length. Using a fixed index of -1 would read the hidden state of <pad>
        for sequences shorter than max_seq_len. This causes the input signal to the classification
        head to degrade depending on the sequence length.

        Parameters
        ----------
        hidden : (B, T, d_model)
        ids    : (B, T)  int32, 0 = <pad>

        Returns
        -------
        (B, d_model)  Hidden state of the actual last token (typically </s>) for each sample.
        """
        lengths = tf.reduce_sum(tf.cast(tf.not_equal(ids, 0), tf.int32), axis=1)  # (B,)
        last_idx = tf.maximum(lengths - 1, 0)  # (B,)
        batch_idx = tf.range(tf.shape(ids)[0])
        gather_idx = tf.stack([batch_idx, last_idx], axis=1)  # (B, 2)
        return tf.gather_nd(hidden, gather_idx)  # (B, d_model)

    def call(self, input_ids, training=False):
        """
        Args
        ----
        input_ids : (B, T)  token ids in GPT-1 NLI format

        Returns
        -------
        clf_logits   : (B, num_labels)       classification logits
        lm_logits    : (B, T, vocab_size)    LM logits for auxiliary objective
        attn_weights : dict of per-block attention tensors
        """
        # GPT-1 forward pass
        hidden_states, attn_weights = self.gpt1(input_ids, training=training)
        # hidden_states: (B, T, d_model)

        # Classification head
        # Use the hidden state at the last actual token position (</s>)
        # (Dynamic index considering padding, rather than a fixed index of -1)
        last_hidden = self._gather_last_real_token(
            hidden_states, input_ids
        )  # (B, d_model)
        last_hidden = self.clf_dropout(last_hidden, training=training)
        clf_logits = self.classifier(last_hidden)  # (B, num_labels)

        # Auxiliary LM head (tied weights)
        lm_logits = self.gpt1.compute_lm_logits(hidden_states)
        # (B, T, vocab_size)

        return clf_logits, lm_logits, attn_weights

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "num_labels": self._num_labels,
                "clf_dropout_rate": self.clf_dropout.rate,
            }
        )
        return cfg


#
# Learning-rate Schedule
#
class GPT1LRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Linear warm-up followed by cosine annealing.

    Replaces the rsqrt schedule (CustomSchedule) in Transformer-tf.py,
    which was designed for the original Transformer architecture.
    GPT-1 uses a simpler linear-warmup + cosine-decay schedule.
    """

    def __init__(
        self,
        max_lr: float = 2.5e-4,
        warmup_steps: int = 500,
        total_steps: int = 1_000_000,
    ):
        super().__init__()
        self.max_lr = float(max_lr)
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = tf.cast(self.warmup_steps, tf.float32)
        total = tf.cast(self.total_steps, tf.float32)
        max_lr = tf.cast(self.max_lr, tf.float32)

        # Linear warm-up
        warmup_lr = max_lr * (step / warmup)

        # Cosine annealing
        progress = (step - warmup) / tf.maximum(total - warmup, 1.0)
        # clamp progress to [0, 1] to avoid negative LR after total_steps
        progress = tf.clip_by_value(progress, 0.0, 1.0)
        cosine_lr = max_lr * 0.5 * (1.0 + tf.math.cos(math.pi * progress))

        return tf.where(step < warmup, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "max_lr": self.max_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
        }


# ============================================================
# Checkpoint utilities
# ============================================================


def build_checkpoint_manager(
    model, optimizer, checkpoint_dir: str, max_to_keep: int = 2
):
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, checkpoint_dir, max_to_keep=max_to_keep
    )
    return ckpt, ckpt_manager


def make_checkpoint_path(
    base_dir: str,
    task: str,
    num_layers: int = GPT1_CONFIG["num_layers"],
    d_model: int = GPT1_CONFIG["d_model"],
    d_ffn: int = GPT1_CONFIG["d_ffn"],
    num_heads: int = GPT1_CONFIG["num_heads"],
) -> str:
    return (
        f"{base_dir}/gpt1-{task}"
        f"-layers-{num_layers}"
        f"-d_model-{d_model}"
        f"-d_ffn-{d_ffn}"
        f"-heads-{num_heads}"
    )
