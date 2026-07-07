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
- [x] Topological sort ensures each node is visited after all nodes that depend on it.
- [x] Broadcasting handled in `_accumulate_grad`: gradients are summed over broadcast dimensions automatically.
- [x] **Gradients verified numerically**: all operations verified to 1e-2 tolerance against float64 central differences.
- [x] 23 tests: forward pass, hand-computed gradients, numerical gradient checks for add/mul/matmul/relu/exp/pow/softmax/gelu/composed expressions.

**Phase 2: Neural network layers — done**

- [x] `nn/layers.py` — four layers built on the autograd engine:
  - `Linear(in, out)` — `y = x @ W.T + b`, Kaiming uniform initialization
  - `LayerNorm(d)` — normalize across last dimension, learnable scale/shift
  - `Embedding(vocab, dim)` — lookup table with sparse gradient (only looked-up rows updated)
  - `Dropout(p)` — inverted dropout, no-op at eval time
- [x] `cross_entropy_loss` — numerically stable log-softmax + NLL in one pass. Gradient is `(softmax(logits) - one_hot(targets)) / N`.
- [x] All layers gradient-checked: Linear input gradcheck, cross_entropy gradcheck.
- [x] 21 tests: shape, values, gradient flow, dropout behavior, loss correctness.

**Planned:**

- [ ] Phase 3: Transformer (multi-head attention, MLP block, decoder)
- [ ] Phase 4: AdamW optimizer + training loop
- [ ] Phase 5: Train on Shakespeare, show loss curve converging
- [ ] Phase 6: Gradient verification report + training benchmarks

---

## Key results so far

**Phase 1 — gradient check results:**
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

**Phase 2 — layer verification:**
```
test_layernorm_normalizes_to_unit_variance   PASSED
test_embedding_gradient_sparse               PASSED  <- only looked-up rows updated
test_cross_entropy_gradcheck                 PASSED
test_cross_entropy_perfect_prediction_low_loss PASSED
```

---

## Running tests

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy pytest
python -m pytest tests/ -v   # 44 tests as of Phase 2
```

---

## Project layout

```
minigrad/
├── autograd/
│   └── tensor.py       <- Tensor class, autograd engine (Phase 1)
├── nn/
│   └── layers.py       <- Linear, LayerNorm, Embedding, Dropout, cross_entropy (Phase 2)
├── transformer/        <- transformer architecture (Phase 3, planned)
├── optim/              <- AdamW optimizer (Phase 4, planned)
├── data/               <- Shakespeare dataset (Phase 5, planned)
├── tests/              <- 44 tests, all passing
└── train.py            <- training script (Phase 5, planned)
```

---

## Author

**Sujan Uppalli Jayadevappa**
MS Software Engineering — Arizona State University
GitHub: [sujanuj](https://github.com/sujanuj)
