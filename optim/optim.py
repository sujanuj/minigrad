"""AdamW optimizer and training loop.

AdamW (Adam with decoupled weight decay) is the standard optimizer for
transformer training. It maintains per-parameter first and second moment
estimates and uses them to adapt the learning rate for each parameter.

Why AdamW over vanilla Adam?
  Adam adds L2 regularization to the gradient, which interacts with the
  adaptive learning rate and doesn't actually regularize the weights the
  way you'd expect. AdamW decouples weight decay from the gradient update,
  applying it directly to the weights. This is what GPT-2, Llama, and
  most modern transformers use.

Why AdamW over SGD?
  SGD with momentum works for CNNs with careful learning rate schedules,
  but transformers are notoriously sensitive to learning rate and benefit
  from the per-parameter adaptive rates that Adam provides. The second
  moment estimate v_t tracks the variance of gradients -- parameters with
  high-variance gradients (like rare token embeddings) get smaller updates,
  while parameters with low-variance gradients get larger updates.

Update rule:
  m_t = beta1 * m_{t-1} + (1 - beta1) * g_t          # first moment
  v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2        # second moment
  m_hat = m_t / (1 - beta1^t)                          # bias correction
  v_hat = v_t / (1 - beta2^t)                          # bias correction
  theta_t = theta_{t-1} * (1 - lr * weight_decay)     # weight decay
           - lr * m_hat / (sqrt(v_hat) + eps)          # gradient step
"""

from __future__ import annotations

import time
import numpy as np
from typing import Callable, List, Optional

from autograd.tensor import Tensor


# ---------------------------------------------------------------------------
# AdamW optimizer
# ---------------------------------------------------------------------------

class AdamW:
    """AdamW optimizer with decoupled weight decay.

    Args:
        params: list of Tensor objects with requires_grad=True
        lr: learning rate (default 3e-4, a common starting point for transformers)
        beta1: exponential decay for first moment (default 0.9)
        beta2: exponential decay for second moment (default 0.95, slightly
               lower than the common 0.999 -- better for transformers where
               gradients can be noisy)
        eps: small constant for numerical stability
        weight_decay: L2 regularization strength (default 0.1)
    """

    def __init__(
        self,
        params: List[Tensor],
        lr: float = 3e-4,
        beta1: float = 0.9,
        beta2: float = 0.95,
        eps: float = 1e-8,
        weight_decay: float = 0.1,
    ):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay

        self.t = 0  # step counter (for bias correction)

        # Per-parameter moment estimates, initialized to zero
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]

    def step(self):
        """Update all parameters using their current gradients."""
        self.t += 1
        # Bias correction factors
        bc1 = 1.0 - self.beta1 ** self.t
        bc2 = 1.0 - self.beta2 ** self.t

        for i, p in enumerate(self.params):
            if p.grad is None or np.allclose(p.grad, 0):
                continue

            g = p.grad.astype(np.float32)

            # Update moment estimates
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g ** 2

            # Bias-corrected estimates
            m_hat = self.m[i] / bc1
            v_hat = self.v[i] / bc2

            # Decoupled weight decay (applied directly to weights, not gradient)
            p.data *= (1.0 - self.lr * self.weight_decay)

            # Gradient step
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        """Zero all parameter gradients."""
        for p in self.params:
            p.zero_grad()


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------

def cosine_lr_schedule(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """Cosine learning rate schedule with linear warmup.

    During warmup: lr increases linearly from 0 to max_lr.
    After warmup: lr decays via cosine annealing from max_lr to min_lr.

    This is the schedule used by GPT-2 and most modern transformers.
    """
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step > max_steps:
        return min_lr
    # Cosine decay
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * progress))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

class Trainer:
    """Training loop for the transformer language model.

    Args:
        model: Transformer instance
        optimizer: AdamW instance
        batch_fn: callable() -> (idx, targets) -- returns a batch of
            token sequences. idx and targets are np.ndarray of shape (B, T).
        eval_fn: optional callable() -> float -- returns validation loss.
            Called every eval_interval steps.
        max_steps: total training steps
        eval_interval: how often to evaluate and log
        grad_clip: gradient clipping norm (prevents gradient explosion)
        lr_schedule: optional callable(step) -> float for lr scheduling
    """

    def __init__(
        self,
        model,
        optimizer: AdamW,
        batch_fn: Callable,
        eval_fn: Optional[Callable] = None,
        max_steps: int = 1000,
        eval_interval: int = 100,
        grad_clip: float = 1.0,
        lr_schedule: Optional[Callable] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.batch_fn = batch_fn
        self.eval_fn = eval_fn
        self.max_steps = max_steps
        self.eval_interval = eval_interval
        self.grad_clip = grad_clip
        self.lr_schedule = lr_schedule

        self.train_losses: List[float] = []
        self.eval_losses: List[float] = []
        self.steps: List[int] = []

    def _clip_gradients(self):
        """Global gradient norm clipping.

        Computes the global norm of all gradients and scales them down
        if it exceeds grad_clip. This prevents gradient explosion which
        is a common failure mode in transformer training.
        """
        total_norm = 0.0
        for p in self.optimizer.params:
            if p.grad is not None:
                total_norm += np.sum(p.grad ** 2)
        total_norm = np.sqrt(total_norm)

        if total_norm > self.grad_clip:
            scale = self.grad_clip / (total_norm + 1e-8)
            for p in self.optimizer.params:
                if p.grad is not None:
                    p.grad *= scale

        return float(total_norm)

    def train(self, verbose: bool = True) -> dict:
        """Run the training loop.

        Returns:
            dict with train_losses, eval_losses, steps, total_time_s
        """
        self.model.train()
        t0 = time.time()

        for step in range(1, self.max_steps + 1):
            # Update learning rate
            if self.lr_schedule is not None:
                self.optimizer.lr = self.lr_schedule(step)

            # Forward pass
            idx, targets = self.batch_fn()
            self.optimizer.zero_grad()
            loss = self.model.loss(idx, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            grad_norm = self._clip_gradients()

            # Parameter update
            self.optimizer.step()

            loss_val = float(loss.data)
            self.train_losses.append(loss_val)
            self.steps.append(step)

            if verbose and (step % self.eval_interval == 0 or step == 1):
                elapsed = time.time() - t0
                eval_str = ""
                if self.eval_fn is not None:
                    eval_loss = self.eval_fn()
                    self.eval_losses.append(eval_loss)
                    eval_str = f" | val_loss={eval_loss:.4f}"
                print(f"step {step:4d}/{self.max_steps} | "
                      f"loss={loss_val:.4f}{eval_str} | "
                      f"grad_norm={grad_norm:.3f} | "
                      f"lr={self.optimizer.lr:.2e} | "
                      f"elapsed={elapsed:.1f}s")

        total_time = time.time() - t0
        return {
            "train_losses": self.train_losses,
            "eval_losses": self.eval_losses,
            "steps": self.steps,
            "total_time_s": total_time,
        }
