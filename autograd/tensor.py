"""Autograd engine: automatic differentiation via reverse-mode backprop.

This is the core of minigrad. Every operation on a Tensor records:
  1. The output value (the forward pass result)
  2. A backward function that computes how gradients flow back through
     this operation (the backward pass)

When you call loss.backward(), it:
  1. Topologically sorts the computation graph (so each node is
     processed after all nodes that depend on it)
  2. Walks the graph in reverse, calling each backward function to
     accumulate gradients into .grad

Why reverse-mode (backprop) rather than forward-mode?
  For a network with N parameters and 1 scalar loss, reverse-mode
  computes all N gradients in one backward pass. Forward-mode would
  need N passes. Reverse-mode is why neural network training is
  tractable -- it's the key insight behind backprop.

This implementation handles:
  - Matrix operations: matmul, add, sum, mean
  - Element-wise ops: relu, exp, log, multiply, divide, power
  - Shape ops: reshape, transpose
  - Everything needed for a transformer

Gradient correctness is verified in tests/test_autograd.py using
numerical gradient checking: perturb each parameter by epsilon,
measure the change in loss, compare to the analytical gradient.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, List, Optional, Set, Tuple


class Tensor:
    """A multi-dimensional array with automatic gradient tracking.

    Args:
        data: array-like or np.ndarray. The tensor's values.
        requires_grad: if True, gradients will be computed for this
            tensor during backward(). Set True for parameters, False
            for inputs and labels.
        _children: internal -- the Tensor(s) this was computed from.
        _backward: internal -- function that accumulates gradients
            into _children's .grad arrays.
        label: optional name for debugging.
    """

    def __init__(
        self,
        data,
        requires_grad: bool = False,
        _children: Tuple["Tensor", ...] = (),
        _backward: Callable = lambda: None,
        label: str = "",
    ):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._backward = _backward
        self._children = set(_children)
        self.label = label

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def T(self):
        return self.transpose()

    def __repr__(self):
        return (f"Tensor(shape={self.shape}, "
                f"requires_grad={self.requires_grad}"
                f"{', label=' + repr(self.label) if self.label else ''})")

    # ------------------------------------------------------------------
    # Gradient accumulation
    # ------------------------------------------------------------------

    def zero_grad(self):
        if self.grad is not None:
            self.grad = np.zeros_like(self.data)

    def _accumulate_grad(self, grad: np.ndarray):
        """Add grad to self.grad, handling broadcasting.

        When an operation broadcasts (e.g. adding a bias of shape (D,)
        to a matrix of shape (B, D)), the gradient for the bias must be
        summed over the broadcast dimensions. This handles that
        automatically by summing over axes that were broadcast.
        """
        if self.grad is None:
            return
        # Sum over broadcast dimensions
        if grad.shape != self.data.shape:
            # Find axes that were broadcast (present in grad but not in data)
            ndim_diff = grad.ndim - self.data.ndim
            axes = tuple(range(ndim_diff))
            # Also sum over axes where data has size 1 but grad doesn't
            axes += tuple(
                i + ndim_diff
                for i, (sg, sd) in enumerate(zip(self.data.shape, grad.shape[ndim_diff:]))
                if sd == 1 and sg != 1
            )
            grad = grad.sum(axis=axes, keepdims=False)
            # Reshape to match self.data.shape if needed
            grad = grad.reshape(self.data.shape)
        self.grad += grad

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------

    def backward(self):
        """Compute gradients for all tensors in the computation graph.

        Topologically sorts the graph so each node is visited after all
        nodes that use its output. Then walks in reverse, calling each
        node's _backward function to propagate gradients.
        """
        # Start gradient: d(loss)/d(loss) = 1
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        self.grad += np.ones_like(self.data)

        # Topological sort
        topo: List[Tensor] = []
        visited: Set[int] = set()

        def build_topo(node: Tensor):
            if id(node) not in visited:
                visited.add(id(node))
                for child in node._children:
                    build_topo(child)
                topo.append(node)

        build_topo(self)

        # Reverse-mode: walk in reverse topological order
        for node in reversed(topo):
            node._backward()

    # ------------------------------------------------------------------
    # Arithmetic operations
    # ------------------------------------------------------------------

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
        )

        def _backward():
            if self.requires_grad:
                self._accumulate_grad(out.grad)
            if other.requires_grad:
                other._accumulate_grad(out.grad)

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
        )

        def _backward():
            if self.requires_grad:
                self._accumulate_grad(out.grad * other.data)
            if other.requires_grad:
                other._accumulate_grad(out.grad * self.data)

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * other ** -1

    def __rtruediv__(self, other):
        return other * self ** -1

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float))
        out = Tensor(
            self.data ** exponent,
            requires_grad=self.requires_grad,
            _children=(self,),
        )

        def _backward():
            if self.requires_grad:
                self._accumulate_grad(
                    out.grad * exponent * (self.data ** (exponent - 1))
                )

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Matrix operations
    # ------------------------------------------------------------------

    def matmul(self, other):
        """Matrix multiplication. Gradient derivation:
        If out = A @ B, then:
          dL/dA = dL/dout @ B.T
          dL/dB = A.T @ dL/dout
        This is the most important gradient in a transformer.
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
        )

        def _backward():
            if self.requires_grad:
                # Handle batched matmul: (..., m, k) @ (..., k, n) -> (..., m, n)
                # dL/dA = dL/dout @ B.T
                grad_a = out.grad @ other.data.swapaxes(-1, -2)
                self._accumulate_grad(grad_a)
            if other.requires_grad:
                # dL/dB = A.T @ dL/dout
                grad_b = self.data.swapaxes(-1, -2) @ out.grad
                other._accumulate_grad(grad_b)

        out._backward = _backward
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    def sum(self, axis=None, keepdims=False):
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _children=(self,),
        )

        def _backward():
            if self.requires_grad:
                grad = out.grad
                if not keepdims and axis is not None:
                    # Restore the summed-over dimension for broadcasting
                    grad = np.expand_dims(grad, axis=axis)
                self._accumulate_grad(np.broadcast_to(grad, self.data.shape).copy())

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) / n

    def max(self, axis=None, keepdims=False):
        out = Tensor(
            self.data.max(axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _children=(self,),
        )

        def _backward():
            if self.requires_grad:
                grad = out.grad
                if not keepdims and axis is not None:
                    grad = np.expand_dims(grad, axis=axis)
                mask = (self.data == np.max(self.data, axis=axis, keepdims=True))
                self._accumulate_grad(grad * mask)

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Shape operations
    # ------------------------------------------------------------------

    def reshape(self, *shape):
        out = Tensor(
            self.data.reshape(*shape),
            requires_grad=self.requires_grad,
            _children=(self,),
        )

        def _backward():
            if self.requires_grad:
                self._accumulate_grad(out.grad.reshape(self.data.shape))

        out._backward = _backward
        return out

    def transpose(self, axes=None):
        out = Tensor(
            self.data.transpose(axes),
            requires_grad=self.requires_grad,
            _children=(self,),
        )

        def _backward():
            if self.requires_grad:
                if axes is None:
                    self._accumulate_grad(out.grad.transpose())
                else:
                    inverse = np.argsort(axes)
                    self._accumulate_grad(out.grad.transpose(inverse))

        out._backward = _backward
        return out

    def __getitem__(self, idx):
        out = Tensor(
            self.data[idx],
            requires_grad=self.requires_grad,
            _children=(self,),
        )

        def _backward():
            if self.requires_grad:
                grad = np.zeros_like(self.data)
                np.add.at(grad, idx, out.grad)
                self._accumulate_grad(grad)

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Activation functions
    # ------------------------------------------------------------------

    def relu(self):
        out = Tensor(
            np.maximum(0, self.data),
            requires_grad=self.requires_grad,
            _children=(self,),
        )

        def _backward():
            if self.requires_grad:
                self._accumulate_grad(out.grad * (self.data > 0))

        out._backward = _backward
        return out

    def exp(self):
        out = Tensor(
            np.exp(self.data),
            requires_grad=self.requires_grad,
            _children=(self,),
        )

        def _backward():
            if self.requires_grad:
                self._accumulate_grad(out.grad * out.data)

        out._backward = _backward
        return out

    def log(self):
        out = Tensor(
            np.log(self.data + 1e-8),  # small epsilon for numerical stability
            requires_grad=self.requires_grad,
            _children=(self,),
        )

        def _backward():
            if self.requires_grad:
                self._accumulate_grad(out.grad / (self.data + 1e-8))

        out._backward = _backward
        return out

    def softmax(self, axis=-1):
        """Numerically stable softmax + its gradient.

        Subtracts max before exp to prevent overflow (the standard trick).
        The gradient of softmax cross-entropy is computed more efficiently
        in the loss function directly, but this standalone softmax is
        correct for any use case.
        """
        shifted = self.data - self.data.max(axis=axis, keepdims=True)
        exp_x = np.exp(shifted)
        s = exp_x / exp_x.sum(axis=axis, keepdims=True)

        out = Tensor(
            s,
            requires_grad=self.requires_grad,
            _children=(self,),
        )

        def _backward():
            if self.requires_grad:
                # Jacobian of softmax: diag(s) - s @ s.T
                # For efficiency, use the identity:
                # dL/dx_i = s_i * (dL/ds_i - sum_j(dL/ds_j * s_j))
                dot = (out.grad * s).sum(axis=axis, keepdims=True)
                self._accumulate_grad(s * (out.grad - dot))

        out._backward = _backward
        return out

    def gelu(self):
        """GELU activation used in transformers.
        Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        """
        x = self.data
        c = np.sqrt(2.0 / np.pi)
        inner = c * (x + 0.044715 * x ** 3)
        tanh_inner = np.tanh(inner)
        gelu_val = 0.5 * x * (1 + tanh_inner)

        out = Tensor(
            gelu_val,
            requires_grad=self.requires_grad,
            _children=(self,),
        )

        def _backward():
            if self.requires_grad:
                sech2 = 1 - tanh_inner ** 2
                dgelu_dx = (
                    0.5 * (1 + tanh_inner)
                    + 0.5 * x * sech2 * c * (1 + 3 * 0.044715 * x ** 2)
                )
                self._accumulate_grad(out.grad * dgelu_dx)

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @staticmethod
    def zeros(*shape, requires_grad=False):
        return Tensor(np.zeros(shape), requires_grad=requires_grad)

    @staticmethod
    def ones(*shape, requires_grad=False):
        return Tensor(np.ones(shape), requires_grad=requires_grad)

    @staticmethod
    def randn(*shape, requires_grad=False):
        return Tensor(np.random.randn(*shape).astype(np.float32),
                      requires_grad=requires_grad)

    @staticmethod
    def arange(n, requires_grad=False):
        return Tensor(np.arange(n).astype(np.float32), requires_grad=requires_grad)
