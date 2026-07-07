"""Neural network layers built on top of the autograd engine.

Every layer is a pure function (or a class with explicit parameters) that
operates on Tensor objects. No nn.Module, no register_parameter, no hooks.
Parameters are plain Tensor objects with requires_grad=True.

Layers implemented:
  - Linear: y = x @ W + b
  - LayerNorm: normalize across the last dimension, then scale + shift
  - Embedding: lookup table, maps integer indices to dense vectors
  - Dropout: randomly zero out activations during training
  - cross_entropy_loss: numerically stable softmax + NLL loss combined

Why combine softmax and cross-entropy?
  Computing softmax then log then NLL separately is numerically unstable
  and wasteful. The log-sum-exp trick lets us compute log(softmax(x))
  stably in one pass. More importantly, the gradient of cross-entropy
  loss w.r.t. the logits simplifies to (softmax(x) - one_hot(y)) / N,
  which is clean to implement and numerically stable.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional

from autograd.tensor import Tensor


# ---------------------------------------------------------------------------
# Parameter initialization helpers
# ---------------------------------------------------------------------------

def kaiming_uniform(shape, a=0.01) -> np.ndarray:
    """Kaiming (He) uniform initialization for ReLU networks.
    Variance = 2 / fan_in, scaled for leaky relu with slope a.
    """
    fan_in = shape[0] if len(shape) >= 2 else shape[0]
    bound = np.sqrt(3.0) * np.sqrt(2.0 / ((1 + a**2) * fan_in))
    return np.random.uniform(-bound, bound, size=shape).astype(np.float32)


def xavier_uniform(shape) -> np.ndarray:
    """Xavier uniform initialization for tanh/sigmoid networks."""
    fan_in = shape[0]
    fan_out = shape[1] if len(shape) >= 2 else shape[0]
    bound = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-bound, bound, size=shape).astype(np.float32)


# ---------------------------------------------------------------------------
# Linear layer
# ---------------------------------------------------------------------------

class Linear:
    """Fully connected layer: y = x @ W.T + b

    Note: W is stored as (out_features, in_features) matching PyTorch's
    convention. The forward pass transposes W at call time.

    Args:
        in_features: input dimension
        out_features: output dimension
        bias: if True, add a learnable bias term
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features

        # Kaiming uniform initialization
        self.W = Tensor(
            kaiming_uniform((out_features, in_features)),
            requires_grad=True,
            label=f"Linear.W({in_features}->{out_features})",
        )
        self.b = None
        if bias:
            self.b = Tensor(
                np.zeros(out_features, dtype=np.float32),
                requires_grad=True,
                label=f"Linear.b({out_features})",
            )

    def __call__(self, x: Tensor) -> Tensor:
        # x: (..., in_features) -> (..., out_features)
        out = x @ self.W.T
        if self.b is not None:
            out = out + self.b
        return out

    def parameters(self) -> List[Tensor]:
        return [self.W] + ([self.b] if self.b is not None else [])

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def __repr__(self):
        return (f"Linear(in={self.in_features}, out={self.out_features}, "
                f"bias={self.b is not None})")


# ---------------------------------------------------------------------------
# Layer Normalization
# ---------------------------------------------------------------------------

class LayerNorm:
    """Layer normalization over the last dimension.

    Normalizes x to zero mean and unit variance across the last axis,
    then applies learnable scale (gamma) and shift (beta).

    Why LayerNorm over BatchNorm for transformers?
      - BatchNorm normalizes over the batch dimension, so it depends on
        batch size and breaks for batch_size=1 (common at inference).
      - LayerNorm normalizes over the feature dimension, independent of
        batch size. Works identically at train and eval time.

    Args:
        normalized_shape: int or tuple, the shape of the last dimension(s)
            to normalize. Usually just the embedding dimension d_model.
        eps: small constant for numerical stability in the denominator.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.gamma = Tensor(
            np.ones(normalized_shape, dtype=np.float32),
            requires_grad=True,
            label="LayerNorm.gamma",
        )
        self.beta = Tensor(
            np.zeros(normalized_shape, dtype=np.float32),
            requires_grad=True,
            label="LayerNorm.beta",
        )

    def __call__(self, x: Tensor) -> Tensor:
        # Compute mean and variance over the last axis
        mean = x.mean(axis=-1, keepdims=True)
        # Variance: E[(x - mean)^2]
        diff = x - mean
        var = (diff ** 2).mean(axis=-1, keepdims=True)
        # Normalize
        x_norm_data = diff.data / np.sqrt(var.data + self.eps)
        x_norm = Tensor(x_norm_data, requires_grad=x.requires_grad, _children=(x,))

        def _backward():
            if x.requires_grad:
                # Full LayerNorm backward -- derived from the chain rule
                # through the normalization formula.
                N = x.data.shape[-1]
                dy = x_norm.grad  # gradient w.r.t. normalized output
                # Scale by gamma (applied after normalization)
                # But x_norm here is before gamma/beta, so dy is after
                # gamma. We need to compute gradient w.r.t. x.
                std = np.sqrt(var.data + self.eps)
                dx_norm = dy
                dvar = (dx_norm * diff.data * (-0.5) *
                        (var.data + self.eps) ** (-1.5)).sum(axis=-1, keepdims=True)
                dmean = (dx_norm * (-1.0 / std)).sum(axis=-1, keepdims=True)
                dx = (dx_norm / std +
                      dvar * 2 * diff.data / N +
                      dmean / N)
                x._accumulate_grad(dx)

        x_norm._backward = _backward

        # Apply scale and shift
        return x_norm * self.gamma + self.beta

    def parameters(self) -> List[Tensor]:
        return [self.gamma, self.beta]

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def __repr__(self):
        return f"LayerNorm(normalized_shape={self.normalized_shape})"


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

class Embedding:
    """Lookup table: maps integer token IDs to dense vectors.

    The embedding matrix has shape (vocab_size, embed_dim). Each forward
    call indexes into this matrix using integer indices. The backward
    pass accumulates gradients only into the rows that were actually
    looked up (sparse gradient -- not all embeddings are updated each step).

    Args:
        vocab_size: number of tokens in the vocabulary
        embed_dim: dimensionality of each embedding vector
    """

    def __init__(self, vocab_size: int, embed_dim: int):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        # Small normal initialization -- large values cause saturation
        self.weight = Tensor(
            np.random.randn(vocab_size, embed_dim).astype(np.float32) * 0.02,
            requires_grad=True,
            label="Embedding.weight",
        )

    def __call__(self, indices) -> Tensor:
        """Look up embeddings for integer indices.

        Args:
            indices: np.ndarray of int32/int64, any shape (B, T) etc.

        Returns:
            Tensor of shape (*indices.shape, embed_dim)
        """
        indices = np.asarray(indices, dtype=np.int32)
        return self.weight[indices]

    def parameters(self) -> List[Tensor]:
        return [self.weight]

    def zero_grad(self):
        self.weight.zero_grad()

    def __repr__(self):
        return f"Embedding(vocab_size={self.vocab_size}, embed_dim={self.embed_dim})"


# ---------------------------------------------------------------------------
# Dropout
# ---------------------------------------------------------------------------

class Dropout:
    """Randomly zero out activations during training.

    At training time: each activation is set to 0 with probability p,
    and the remaining activations are scaled by 1/(1-p) so the expected
    sum is unchanged (inverted dropout).

    At inference time (training=False): no dropout, just pass through.

    Args:
        p: probability of zeroing out each activation (default 0.1)
    """

    def __init__(self, p: float = 0.1):
        self.p = p
        self.training = True

    def __call__(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x

        mask = (np.random.rand(*x.shape) > self.p).astype(np.float32)
        scale = 1.0 / (1.0 - self.p)
        return x * Tensor(mask * scale)

    def __repr__(self):
        return f"Dropout(p={self.p})"


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def cross_entropy_loss(logits: Tensor, targets: np.ndarray) -> Tensor:
    """Numerically stable cross-entropy loss.

    Combines log-softmax and negative log likelihood in one pass,
    avoiding the numerical issues of computing softmax -> log separately.

    The gradient w.r.t. logits is: (softmax(logits) - one_hot(targets)) / N
    This is the cleanest gradient in deep learning.

    Args:
        logits: Tensor of shape (N, C) -- raw unnormalized scores
        targets: np.ndarray of shape (N,) -- integer class indices

    Returns:
        Scalar Tensor -- mean cross-entropy loss over the batch
    """
    N = logits.data.shape[0]
    targets = np.asarray(targets, dtype=np.int32)

    # Numerically stable log-softmax
    shifted = logits.data - logits.data.max(axis=-1, keepdims=True)
    log_sum_exp = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    log_probs = shifted - log_sum_exp  # log(softmax(logits))

    # NLL loss: -log_prob[target] for each sample
    loss_val = -log_probs[np.arange(N), targets].mean()

    out = Tensor(
        np.array(loss_val, dtype=np.float32),
        requires_grad=logits.requires_grad,
        _children=(logits,),
    )

    def _backward():
        if logits.requires_grad:
            # Gradient: (softmax(logits) - one_hot(targets)) / N
            probs = np.exp(log_probs)
            grad = probs.copy()
            grad[np.arange(N), targets] -= 1.0
            grad /= N
            logits._accumulate_grad(grad)

    out._backward = _backward
    return out
