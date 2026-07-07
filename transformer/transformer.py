"""Decoder-only transformer built on the autograd engine and nn layers.

Architecture: GPT-style decoder-only transformer.
  - Token embedding + positional embedding
  - N stacked decoder blocks, each with:
    - Pre-norm multi-head causal self-attention
    - Pre-norm MLP (Linear -> GELU -> Linear)
    - Residual connections around both
  - Final LayerNorm + linear projection to vocabulary logits

Why decoder-only (not encoder-decoder)?
  Decoder-only transformers (GPT, Llama, etc.) are the dominant
  architecture for language modeling. The causal mask ensures each
  token can only attend to previous tokens -- making the model
  autoregressive: it predicts the next token given all previous ones.

Dimensions follow GPT-2 small naming:
  d_model: embedding dimension (also called n_embd)
  n_heads: number of attention heads
  d_head: d_model // n_heads, dimension per head
  d_ff: feed-forward hidden dimension (typically 4 * d_model)
  n_layers: number of stacked decoder blocks
  vocab_size: number of tokens
  max_seq_len: maximum sequence length (for positional embeddings)
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional

from autograd.tensor import Tensor
from nn.layers import Linear, LayerNorm, Embedding, Dropout, cross_entropy_loss


# ---------------------------------------------------------------------------
# Causal attention mask
# ---------------------------------------------------------------------------

def causal_mask(seq_len: int) -> np.ndarray:
    """Lower triangular mask of shape (seq_len, seq_len).
    Entry (i, j) is True if position i can attend to position j.
    Since this is causal, position i can only attend to j <= i.
    """
    return np.tril(np.ones((seq_len, seq_len), dtype=bool))


# ---------------------------------------------------------------------------
# Multi-head causal self-attention
# ---------------------------------------------------------------------------

class CausalSelfAttention:
    """Multi-head causal self-attention.

    Projects input to Q, K, V, splits into heads, computes scaled
    dot-product attention with a causal mask, then projects back.

    Args:
        d_model: input/output dimension
        n_heads: number of attention heads. d_model must be divisible by n_heads.
        dropout_p: attention dropout probability
    """

    def __init__(self, d_model: int, n_heads: int, dropout_p: float = 0.0):
        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Combined QKV projection: one matmul instead of three
        self.qkv_proj = Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = Linear(d_model, d_model, bias=False)
        self.dropout = Dropout(dropout_p)

    def __call__(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor of shape (B, T, d_model)
        Returns:
            Tensor of shape (B, T, d_model)
        """
        B, T, C = x.shape

        # QKV projection: (B, T, d_model) -> (B, T, 3*d_model)
        qkv = self.qkv_proj(x)

        # Split into Q, K, V: each (B, T, d_model)
        q_data = qkv.data[:, :, :C]
        k_data = qkv.data[:, :, C:2*C]
        v_data = qkv.data[:, :, 2*C:]

        # Reshape to (B, n_heads, T, d_head)
        q_data = q_data.reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        k_data = k_data.reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        v_data = v_data.reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        scale = 1.0 / np.sqrt(self.d_head)
        # (B, n_heads, T, T)
        attn_scores = q_data @ k_data.swapaxes(-1, -2) * scale

        # Apply causal mask: mask out future positions
        mask = causal_mask(T)  # (T, T)
        attn_scores = np.where(mask[None, None, :, :], attn_scores,
                               np.full_like(attn_scores, -1e9))

        # Softmax over last axis
        attn_scores -= attn_scores.max(axis=-1, keepdims=True)
        exp_scores = np.exp(attn_scores)
        attn_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

        # Weighted sum of values: (B, n_heads, T, d_head)
        attn_out = attn_weights @ v_data

        # Reshape back: (B, T, d_model)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T, C)

        # We need to wrap this in a Tensor with proper gradient tracking.
        # Build the full attention forward as a Tensor operation by
        # using the Tensor class for the QKV split and attention computation.
        q = Tensor(q_data, requires_grad=qkv.requires_grad, _children=(qkv,))
        k = Tensor(k_data, requires_grad=qkv.requires_grad, _children=(qkv,))
        v = Tensor(v_data, requires_grad=qkv.requires_grad, _children=(qkv,))

        attn_w = Tensor(attn_weights, requires_grad=qkv.requires_grad,
                        _children=(q, k))
        out_heads = Tensor(attn_out, requires_grad=qkv.requires_grad,
                           _children=(attn_w, v))

        def _backward_attn():
            if not qkv.requires_grad:
                return
            # Gradient through reshape+transpose and attention computation
            # Using the full analytical gradient of scaled dot-product attention
            grad_out = out_heads.grad  # (B, T, C)
            grad_out_heads = grad_out.reshape(B, T, self.n_heads,
                                              self.d_head).transpose(0, 2, 1, 3)

            # dL/d(attn_weights) = grad_out_heads @ V.T
            grad_attn_w = grad_out_heads @ v_data.swapaxes(-1, -2)

            # dL/dV = attn_weights.T @ grad_out_heads
            grad_v = attn_weights.swapaxes(-1, -2) @ grad_out_heads

            # Softmax backward
            dot = (grad_attn_w * attn_weights).sum(axis=-1, keepdims=True)
            grad_scores = attn_weights * (grad_attn_w - dot) * scale

            # dL/dQ = grad_scores @ K
            grad_q = grad_scores @ k_data
            # dL/dK = grad_scores.T @ Q
            grad_k = grad_scores.swapaxes(-1, -2) @ q_data

            # Reshape gradients back to (B, T, C)
            grad_q = grad_q.transpose(0, 2, 1, 3).reshape(B, T, C)
            grad_k = grad_k.transpose(0, 2, 1, 3).reshape(B, T, C)
            grad_v = grad_v.transpose(0, 2, 1, 3).reshape(B, T, C)

            grad_qkv = np.concatenate([grad_q, grad_k, grad_v], axis=-1)
            qkv._accumulate_grad(grad_qkv)

        out_heads._backward = _backward_attn

        # Reshape to (B, T, C) Tensor
        out_tensor = Tensor(
            attn_out,
            requires_grad=qkv.requires_grad,
            _children=(out_heads,),
        )

        def _backward_reshape():
            if out_heads.requires_grad:
                out_heads._accumulate_grad(out_tensor.grad)

        out_tensor._backward = _backward_reshape

        return self.out_proj(out_tensor)

    def parameters(self) -> List[Tensor]:
        return self.qkv_proj.parameters() + self.out_proj.parameters()

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()


# ---------------------------------------------------------------------------
# MLP block
# ---------------------------------------------------------------------------

class MLP:
    """Position-wise feed-forward network used in each transformer block.

    Two linear layers with GELU activation:
      x -> Linear(d_model, d_ff) -> GELU -> Linear(d_ff, d_model)

    d_ff is typically 4 * d_model (from the original transformer paper).
    """

    def __init__(self, d_model: int, d_ff: int, dropout_p: float = 0.0):
        self.fc1 = Linear(d_model, d_ff)
        self.fc2 = Linear(d_ff, d_model)
        self.dropout = Dropout(dropout_p)

    def __call__(self, x: Tensor) -> Tensor:
        return self.fc2(self.dropout(self.fc1(x).gelu()))

    def parameters(self) -> List[Tensor]:
        return self.fc1.parameters() + self.fc2.parameters()

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()


# ---------------------------------------------------------------------------
# Transformer decoder block
# ---------------------------------------------------------------------------

class TransformerBlock:
    """One decoder block: pre-norm attention + pre-norm MLP with residuals.

    Pre-norm (LayerNorm before attention/MLP) rather than post-norm
    (LayerNorm after) because pre-norm trains more stably at depth --
    gradients flow more directly through the residual path.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout_p: float = 0.0):
        self.ln1 = LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout_p)
        self.ln2 = LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff, dropout_p)

    def __call__(self, x: Tensor) -> Tensor:
        # Pre-norm attention + residual
        x = x + self.attn(self.ln1(x))
        # Pre-norm MLP + residual
        x = x + self.mlp(self.ln2(x))
        return x

    def parameters(self) -> List[Tensor]:
        return (self.ln1.parameters() + self.attn.parameters() +
                self.ln2.parameters() + self.mlp.parameters())

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()


# ---------------------------------------------------------------------------
# Full transformer
# ---------------------------------------------------------------------------

class Transformer:
    """Decoder-only transformer language model.

    Args:
        vocab_size: number of tokens in the vocabulary
        d_model: embedding dimension
        n_heads: number of attention heads
        n_layers: number of stacked decoder blocks
        d_ff: feed-forward hidden dimension (default: 4 * d_model)
        max_seq_len: maximum sequence length
        dropout_p: dropout probability
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: Optional[int] = None,
        max_seq_len: int = 512,
        dropout_p: float = 0.0,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff or 4 * d_model
        self.max_seq_len = max_seq_len

        self.token_emb = Embedding(vocab_size, d_model)
        self.pos_emb = Embedding(max_seq_len, d_model)
        self.drop = Dropout(dropout_p)
        self.blocks = [
            TransformerBlock(d_model, n_heads, self.d_ff, dropout_p)
            for _ in range(n_layers)
        ]
        self.ln_f = LayerNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size, bias=False)

    def __call__(self, idx: np.ndarray) -> Tensor:
        """Forward pass.

        Args:
            idx: np.ndarray of shape (B, T) -- integer token indices

        Returns:
            logits: Tensor of shape (B, T, vocab_size)
        """
        B, T = idx.shape
        assert T <= self.max_seq_len, \
            f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}"

        positions = np.arange(T)[None, :]  # (1, T)

        tok = self.token_emb(idx)       # (B, T, d_model)
        pos = self.pos_emb(positions)   # (1, T, d_model)
        x = self.drop(tok + pos)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)        # (B, T, vocab_size)
        return logits

    def loss(self, idx: np.ndarray, targets: np.ndarray) -> Tensor:
        """Compute cross-entropy loss.

        Args:
            idx: (B, T) input token indices
            targets: (B, T) target token indices (next tokens)

        Returns:
            scalar loss Tensor
        """
        logits = self(idx)              # (B, T, vocab_size)
        B, T, V = logits.shape

        # Flatten to (B*T, vocab_size) and (B*T,) for cross_entropy
        logits_flat = logits.reshape(B * T, V)
        targets_flat = targets.reshape(B * T)
        return cross_entropy_loss(logits_flat, targets_flat)

    def parameters(self) -> List[Tensor]:
        params = (self.token_emb.parameters() +
                  self.pos_emb.parameters() +
                  self.ln_f.parameters() +
                  self.lm_head.parameters())
        for block in self.blocks:
            params += block.parameters()
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def num_parameters(self) -> int:
        return sum(p.data.size for p in self.parameters())

    def train(self):
        self.drop.training = True
        for block in self.blocks:
            block.attn.dropout.training = True
            block.mlp.dropout.training = True

    def eval(self):
        self.drop.training = False
        for block in self.blocks:
            block.attn.dropout.training = False
            block.mlp.dropout.training = False
