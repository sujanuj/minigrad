# minigrad

A training engine built from raw NumPy — no PyTorch autograd, no `nn.Module`. Implements reverse-mode automatic differentiation, neural network layers, a transformer, AdamW optimizer, and trains on real data with a verified loss curve.

The goal: implement every piece that makes neural network training work, verify it is correct, and measure it honestly.

---

## What makes this different

Most "I built a neural network" projects call `loss.backward()` and trust PyTorch. This project implements `backward()` itself — the computation graph, topological sort, gradient accumulation, and every operation's backward function. Gradient correctness is verified using numerical gradient checking (the same method PyTorch's `gradcheck` uses internally).

---

## Phases

**Phase 1: Autograd engine — done**

- [x] `autograd/tensor.py` — a `Tensor` class that tracks operations in a computation graph and computes gradients via reverse-mode backprop. Every operation (`matmul`, `add`, `relu`, `softmax`, `gelu`, etc.) records a `_backward` function that propagates gradients to its inputs.
- [x] Topological sort ensures each node is visited after all nodes that depend on it — this is what makes multi-layer backprop correct.
- [x] Broadcasting handled in `_accumulate_grad`: gradients are summed over broadcast dimensions automatically, so bias gradients work correctly.
- [x] **Gradients verified numerically**: for each operation, perturb each input by epsilon, measure the change in output, compare to the analytical gradient. All operations verified to 1e-2 tolerance against float64 central differences.
- [x] 23 tests: forward pass correctness, hand-computed gradient checks, numerical gradient checks for add/mul/matmul/relu/exp/pow/softmax/gelu/composed expressions, graph structure (shared nodes, zero_grad).

**Planned:**

- [ ] Phase 2: Neural network layers (Linear, LayerNorm, Embedding, Dropout)
- [ ] Phase 3: Transformer (multi-head attention, MLP block, decoder)
- [ ] Phase 4: AdamW optimizer + training loop
- [ ] Phase 5: Train on Shakespeare, show loss curve converging
- [ ] Phase 6: Gradient verification report + training benchmarks

---

## Running tests

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy pytest
python -m pytest tests/ -v   # 23 tests as of Phase 1
```

---

## Key result so far

All gradients verified correct to 1e-2 tolerance against numerical approximation using central differences:

```
test_gradcheck_add        PASSED
test_gradcheck_mul        PASSED
test_gradcheck_matmul     PASSED
test_gradcheck_relu       PASSED
test_gradcheck_exp        PASSED
test_gradcheck_pow        PASSED
test_gradcheck_softmax    PASSED
test_gradcheck_gelu       PASSED
test_gradcheck_composed   PASSED   <- relu(x @ w + b).mean()
```

The composed test is the most important: it verifies that gradients flow correctly through a multi-operation expression, which is what actually happens in a neural network forward pass.

---

## Project layout

```
minigrad/
├── autograd/
│   └── tensor.py       <- Tensor class, autograd engine (Phase 1)
├── nn/                 <- neural network layers (Phase 2, planned)
├── transformer/        <- transformer architecture (Phase 3, planned)
├── optim/              <- AdamW optimizer (Phase 4, planned)
├── data/               <- Shakespeare dataset (Phase 5, planned)
├── tests/              <- 23 tests, all passing
└── train.py            <- training script (Phase 5, planned)
```

---

## Author

**Sujan Uppalli Jayadevappa**
MS Software Engineering — Arizona State University
GitHub: [sujanuj](https://github.com/sujanuj)
