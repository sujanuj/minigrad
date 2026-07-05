"""Tests for the autograd engine.

The key test here is numerical gradient checking: for each parameter,
perturb it by a small epsilon, measure the change in output, and compare
to the analytical gradient computed by backward(). If they agree to 1e-4,
the backward pass is correct.

Uses float64 for numerical checking to avoid float32 precision artifacts.
The analytical gradients are computed in float32 (as the engine uses),
then compared to float64 numerical gradients with appropriate tolerance.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autograd.tensor import Tensor


# ---------------------------------------------------------------------------
# Numerical gradient checker
# ---------------------------------------------------------------------------

def numerical_grad(f, x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Compute gradient of scalar f numerically using central differences."""
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


def check_grad(tensor_fn, np_fn, input_shape, rtol=1e-2, atol=1e-3):
    """Run gradient check for a tensor operation."""
    np.random.seed(42)
    x_data = np.random.randn(*input_shape).astype(np.float64) * 0.5

    # Analytical gradient (float32 engine)
    x = Tensor(x_data.copy().astype(np.float32), requires_grad=True)
    out = tensor_fn(x)
    out.backward()
    analytical = x.grad.copy().astype(np.float64)

    # Numerical gradient (float64 for precision)
    numerical = numerical_grad(np_fn, x_data.copy())

    np.testing.assert_allclose(
        analytical, numerical, rtol=rtol, atol=atol,
        err_msg=f"Gradient check failed.\nAnalytical: {analytical}\nNumerical: {numerical}"
    )


# ---------------------------------------------------------------------------
# Basic arithmetic tests
# ---------------------------------------------------------------------------

def test_add_forward():
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([4.0, 5.0, 6.0])
    c = a + b
    np.testing.assert_allclose(c.data, [5.0, 7.0, 9.0])


def test_mul_forward():
    a = Tensor([2.0, 3.0])
    b = Tensor([4.0, 5.0])
    c = a * b
    np.testing.assert_allclose(c.data, [8.0, 15.0])


def test_sub_forward():
    a = Tensor([5.0, 3.0])
    b = Tensor([2.0, 1.0])
    c = a - b
    np.testing.assert_allclose(c.data, [3.0, 2.0])


def test_pow_forward():
    a = Tensor([2.0, 3.0])
    c = a ** 2
    np.testing.assert_allclose(c.data, [4.0, 9.0])


def test_matmul_forward():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[5.0, 6.0], [7.0, 8.0]])
    c = a @ b
    np.testing.assert_allclose(c.data, [[19.0, 22.0], [43.0, 50.0]])


# ---------------------------------------------------------------------------
# Gradient tests
# ---------------------------------------------------------------------------

def test_add_backward():
    a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
    c = (a + b).sum()
    c.backward()
    np.testing.assert_allclose(a.grad, [1.0, 1.0, 1.0])
    np.testing.assert_allclose(b.grad, [1.0, 1.0, 1.0])


def test_mul_backward():
    a = Tensor([2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0], requires_grad=True)
    c = (a * b).sum()
    c.backward()
    np.testing.assert_allclose(a.grad, [4.0, 5.0])
    np.testing.assert_allclose(b.grad, [2.0, 3.0])


def test_matmul_backward():
    np.random.seed(0)
    A = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
    B = Tensor(np.random.randn(4, 2).astype(np.float32), requires_grad=True)
    loss = (A @ B).sum()
    loss.backward()
    expected_A_grad = np.ones((3, 2)) @ B.data.T
    np.testing.assert_allclose(A.grad, expected_A_grad, atol=1e-5)


def test_relu_backward():
    a = Tensor([-1.0, 0.5, 2.0], requires_grad=True)
    b = a.relu().sum()
    b.backward()
    np.testing.assert_allclose(a.grad, [0.0, 1.0, 1.0])


def test_sum_backward():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = a.sum()
    b.backward()
    np.testing.assert_allclose(a.grad, np.ones((2, 2)))


def test_mean_backward():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = a.mean()
    b.backward()
    np.testing.assert_allclose(a.grad, np.full((2, 2), 0.25))


# ---------------------------------------------------------------------------
# Numerical gradient checks
# ---------------------------------------------------------------------------

def test_gradcheck_add():
    check_grad(
        lambda x: (x + Tensor(np.ones_like(x.data))).sum(),
        lambda x: (x + 1).sum(),
        input_shape=(3, 4),
    )


def test_gradcheck_mul():
    np.random.seed(1)
    w = np.random.randn(3, 4).astype(np.float32)
    check_grad(
        lambda x: (x * Tensor(w)).sum(),
        lambda x: (x * w).sum(),
        input_shape=(3, 4),
    )


def test_gradcheck_matmul():
    np.random.seed(2)
    w = np.random.randn(4, 2).astype(np.float32)
    check_grad(
        lambda x: (x @ Tensor(w)).sum(),
        lambda x: (x @ w).sum(),
        input_shape=(3, 4),
    )


def test_gradcheck_relu():
    check_grad(
        lambda x: x.relu().sum(),
        lambda x: np.maximum(0, x).sum(),
        input_shape=(4, 4),
    )


def test_gradcheck_exp():
    check_grad(
        lambda x: x.exp().sum(),
        lambda x: np.exp(x).sum(),
        input_shape=(3, 3),
    )


def test_gradcheck_pow():
    check_grad(
        lambda x: (x ** 2).sum(),
        lambda x: (x ** 2).sum(),
        input_shape=(3, 4),
    )


def test_gradcheck_softmax():
    check_grad(
        lambda x: x.softmax(axis=-1).sum(),
        lambda x: (lambda e: e / e.sum(axis=-1, keepdims=True))(
            np.exp(x - x.max(axis=-1, keepdims=True))
        ).sum(),
        input_shape=(4, 6),
    )


def test_gradcheck_gelu():
    check_grad(
        lambda x: x.gelu().sum(),
        lambda x: (0.5 * x * (1 + np.tanh(
            np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)
        ))).sum(),
        input_shape=(3, 4),
    )


def test_gradcheck_composed():
    np.random.seed(3)
    w = np.random.randn(4, 3).astype(np.float32) * 0.1
    b = np.random.randn(3).astype(np.float32) * 0.1
    check_grad(
        lambda x: (x @ Tensor(w) + Tensor(b)).relu().mean(),
        lambda x: np.maximum(0, x @ w + b).mean(),
        input_shape=(5, 4),
    )


# ---------------------------------------------------------------------------
# Graph structure tests
# ---------------------------------------------------------------------------

def test_backward_accumulates_correctly_for_shared_node():
    a = Tensor([2.0, 3.0], requires_grad=True)
    b = a + a
    c = b.sum()
    c.backward()
    np.testing.assert_allclose(a.grad, [2.0, 2.0])


def test_zero_grad_resets_gradient():
    a = Tensor([1.0, 2.0], requires_grad=True)
    (a * 2).sum().backward()
    assert not np.allclose(a.grad, 0)
    a.zero_grad()
    np.testing.assert_allclose(a.grad, [0.0, 0.0])


def test_no_grad_for_non_requires_grad():
    a = Tensor([1.0, 2.0], requires_grad=False)
    b = Tensor([3.0, 4.0], requires_grad=True)
    c = (a + b).sum()
    c.backward()
    assert a.grad is None
    np.testing.assert_allclose(b.grad, [1.0, 1.0])
