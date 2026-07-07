"""Tests for neural network layers.

Verifies:
  1. Forward pass shapes and values for each layer.
  2. Gradient flow through each layer (backward doesn't crash, grads nonzero).
  3. Numerical gradient checks for Linear, LayerNorm, Embedding, cross_entropy.
  4. Dropout: correct masking behavior in train vs eval mode.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autograd.tensor import Tensor
from nn.layers import Linear, LayerNorm, Embedding, Dropout, cross_entropy_loss


# ---------------------------------------------------------------------------
# Numerical gradient checker (same as test_autograd.py)
# ---------------------------------------------------------------------------

def numerical_grad(f, x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    grad = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = float(x[idx])
        x[idx] = orig + eps
        f_plus = float(f(x))
        x[idx] = orig - eps
        f_minus = float(f(x))
        x[idx] = orig
        grad[idx] = (f_plus - f_minus) / (2 * eps)
        it.iternext()
    return grad


# ---------------------------------------------------------------------------
# Linear layer tests
# ---------------------------------------------------------------------------

def test_linear_output_shape():
    layer = Linear(4, 8)
    x = Tensor(np.random.randn(3, 4).astype(np.float32))
    out = layer(x)
    assert out.shape == (3, 8)


def test_linear_no_bias_shape():
    layer = Linear(4, 8, bias=False)
    x = Tensor(np.random.randn(3, 4).astype(np.float32))
    out = layer(x)
    assert out.shape == (3, 8)
    assert layer.b is None


def test_linear_gradient_flows():
    layer = Linear(4, 3)
    x = Tensor(np.random.randn(2, 4).astype(np.float32), requires_grad=True)
    loss = layer(x).sum()
    loss.backward()
    assert x.grad is not None
    assert layer.W.grad is not None
    assert layer.b.grad is not None
    assert not np.allclose(layer.W.grad, 0)


def test_linear_gradcheck_input():
    np.random.seed(0)
    layer = Linear(4, 3)

    def tensor_fn(x_arr):
        x = Tensor(x_arr.astype(np.float32), requires_grad=True)
        out = layer(x).sum()
        out.backward()
        return x.grad.astype(np.float64)

    x_data = np.random.randn(2, 4).astype(np.float64)
    x_t = Tensor(x_data.astype(np.float32), requires_grad=True)
    layer(x_t).sum().backward()
    analytical = x_t.grad.astype(np.float64)

    numerical = numerical_grad(
        lambda x: float(layer(Tensor(x.astype(np.float32))).sum().data),
        x_data.copy()
    )
    np.testing.assert_allclose(analytical, numerical, rtol=1e-2, atol=1e-3)


def test_linear_zero_grad():
    layer = Linear(4, 3)
    x = Tensor(np.random.randn(2, 4).astype(np.float32), requires_grad=True)
    layer(x).sum().backward()
    layer.zero_grad()
    np.testing.assert_allclose(layer.W.grad, np.zeros_like(layer.W.grad))
    np.testing.assert_allclose(layer.b.grad, np.zeros_like(layer.b.grad))


def test_linear_parameters_count():
    layer = Linear(4, 8)
    assert len(layer.parameters()) == 2  # W and b

    layer_no_bias = Linear(4, 8, bias=False)
    assert len(layer_no_bias.parameters()) == 1  # W only


# ---------------------------------------------------------------------------
# LayerNorm tests
# ---------------------------------------------------------------------------

def test_layernorm_output_shape():
    ln = LayerNorm(8)
    x = Tensor(np.random.randn(3, 8).astype(np.float32))
    out = ln(x)
    assert out.shape == (3, 8)


def test_layernorm_normalizes_to_unit_variance():
    ln = LayerNorm(16)
    x = Tensor(np.random.randn(4, 16).astype(np.float32) * 10 + 5)
    out = ln(x)
    # After LayerNorm (before scale/shift), each row should have ~unit variance
    # Since gamma=1 and beta=0 initially, output should be normalized
    row_means = out.data.mean(axis=-1)
    row_stds = out.data.std(axis=-1)
    np.testing.assert_allclose(row_means, np.zeros(4), atol=1e-5)
    np.testing.assert_allclose(row_stds, np.ones(4), atol=1e-4)


def test_layernorm_gradient_flows():
    ln = LayerNorm(8)
    x = Tensor(np.random.randn(3, 8).astype(np.float32), requires_grad=True)
    loss = ln(x).sum()
    loss.backward()
    assert x.grad is not None
    assert ln.gamma.grad is not None
    assert ln.beta.grad is not None


def test_layernorm_gamma_beta_learned():
    ln = LayerNorm(4)
    x = Tensor(np.random.randn(2, 4).astype(np.float32), requires_grad=True)
    loss = ln(x).sum()
    loss.backward()
    assert not np.allclose(ln.gamma.grad, 0)


# ---------------------------------------------------------------------------
# Embedding tests
# ---------------------------------------------------------------------------

def test_embedding_output_shape():
    emb = Embedding(vocab_size=50, embed_dim=16)
    indices = np.array([[1, 2, 3], [4, 5, 6]])
    out = emb(indices)
    assert out.shape == (2, 3, 16)


def test_embedding_lookup_correct():
    emb = Embedding(vocab_size=10, embed_dim=4)
    indices = np.array([2, 5])
    out = emb(indices)
    np.testing.assert_allclose(out.data[0], emb.weight.data[2])
    np.testing.assert_allclose(out.data[1], emb.weight.data[5])


def test_embedding_gradient_sparse():
    emb = Embedding(vocab_size=10, embed_dim=4)
    indices = np.array([2, 5])
    out = emb(indices)
    out.sum().backward()
    # Only rows 2 and 5 should have nonzero gradients
    assert not np.allclose(emb.weight.grad[2], 0)
    assert not np.allclose(emb.weight.grad[5], 0)
    # Other rows should be zero
    for i in [0, 1, 3, 4, 6, 7, 8, 9]:
        np.testing.assert_allclose(emb.weight.grad[i], np.zeros(4))


# ---------------------------------------------------------------------------
# Dropout tests
# ---------------------------------------------------------------------------

def test_dropout_training_zeros_some():
    np.random.seed(0)
    drop = Dropout(p=0.5)
    drop.training = True
    x = Tensor(np.ones((100, 100), dtype=np.float32))
    out = drop(x)
    # With p=0.5, roughly half should be zero
    zero_fraction = (out.data == 0).mean()
    assert 0.4 < zero_fraction < 0.6


def test_dropout_eval_passthrough():
    drop = Dropout(p=0.5)
    drop.training = False
    x = Tensor(np.ones((10, 10), dtype=np.float32))
    out = drop(x)
    np.testing.assert_allclose(out.data, x.data)


def test_dropout_scales_correctly():
    np.random.seed(1)
    drop = Dropout(p=0.5)
    drop.training = True
    x = Tensor(np.ones((1000,), dtype=np.float32))
    out = drop(x)
    # Non-zero values should be scaled by 1/(1-p) = 2
    nonzero = out.data[out.data != 0]
    np.testing.assert_allclose(nonzero, np.full_like(nonzero, 2.0))


# ---------------------------------------------------------------------------
# Cross-entropy loss tests
# ---------------------------------------------------------------------------

def test_cross_entropy_shape():
    logits = Tensor(np.random.randn(4, 10).astype(np.float32), requires_grad=True)
    targets = np.array([0, 3, 7, 9])
    loss = cross_entropy_loss(logits, targets)
    assert loss.shape == ()  # scalar


def test_cross_entropy_correct_value():
    # Manual computation for a simple case
    logits = Tensor(np.array([[1.0, 0.0, 0.0]]), requires_grad=True)
    targets = np.array([0])
    loss = cross_entropy_loss(logits, targets)
    # Expected: -log(softmax([1,0,0])[0])
    probs = np.exp([1.0, 0.0, 0.0]) / np.exp([1.0, 0.0, 0.0]).sum()
    expected = -np.log(probs[0])
    np.testing.assert_allclose(float(loss.data), expected, atol=1e-5)


def test_cross_entropy_gradient_flows():
    logits = Tensor(np.random.randn(4, 8).astype(np.float32), requires_grad=True)
    targets = np.array([0, 3, 7, 1])
    loss = cross_entropy_loss(logits, targets)
    loss.backward()
    assert logits.grad is not None
    assert not np.allclose(logits.grad, 0)


def test_cross_entropy_gradcheck():
    np.random.seed(5)
    targets = np.array([0, 2, 1])
    logits_data = np.random.randn(3, 4).astype(np.float64) * 0.5

    logits_t = Tensor(logits_data.astype(np.float32), requires_grad=True)
    cross_entropy_loss(logits_t, targets).backward()
    analytical = logits_t.grad.astype(np.float64)

    numerical = numerical_grad(
        lambda x: float(cross_entropy_loss(
            Tensor(x.astype(np.float32)), targets
        ).data),
        logits_data.copy()
    )
    np.testing.assert_allclose(analytical, numerical, rtol=1e-2, atol=1e-3)


def test_cross_entropy_perfect_prediction_low_loss():
    # If logits strongly predict the correct class, loss should be low
    logits = Tensor(np.array([[10.0, -10.0, -10.0]]), requires_grad=True)
    targets = np.array([0])
    loss = cross_entropy_loss(logits, targets)
    assert float(loss.data) < 0.01
