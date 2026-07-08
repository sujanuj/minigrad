# minigrad

A training engine built from raw NumPy — no PyTorch autograd, no `nn.Module`. Implements reverse-mode automatic differentiation, neural network layers, a decoder-only transformer, AdamW optimizer, and trains on Shakespeare with a verified loss curve.

The goal: implement every piece that makes neural network training work, verify it is correct, and measure it honestly.

---

## What makes this different

Most "I built a neural network" projects call `loss.backward()` and trust PyTorch. This project implements `backward()` itself — the computation graph, topological sort, gradient accumulation, and every operation's backward function. Gradient correctness is verified using numerical gradient checking (the same method PyTorch's `gradcheck` uses internally).

---

## Training results (Shakespeare, Apple M5 CPU)

```
Model: 161,728 parameters
  d_model=64, n_heads=4, n_layers=3, d_ff=256, seq_len=64

step    1/500 | train=5.0637 | val=4.9370 | lr=1.20e-04
step   50/500 | train=2.7275 | val=2.7468 | lr=2.98e-03
step  100/500 | train=2.5890 | val=2.5710 | lr=2.84e-03
step  200/500 | train=2.4971 | val=2.4694 | lr=2.19e-03
step  300/500 | train=2.4421 | val=2.4203 | lr=1.32e-03
step  400/500 | train=2.4720 | val=2.3611 | lr=5.85e-04
step  500/500 | train=2.2585 | val=2.3350 | lr=3.00e-04

Loss: 5.06 -> 2.26 (55.4% reduction in 500 steps, 12.2s)
Throughput: 42,000 tokens/second on CPU
Expected initial loss: log(65) = 4.17 (random model)
```

Generated text after 500 steps (temperature=0.8):
```
ROMEO:
INI yow, d de thoned ilit, and m onee, ing, kn? ghind ar

Cis y sous the char shawese, t hee isoncandsth w, a mend
The oweme bute the.

CENGLO:
Gr chieven and onoda avery athif.
```

The generated text is garbled but shows real learned structure: character names
(`ROMEO:`, `CENGLO:`), punctuation patterns, and word-like fragments. The model
is learning English character statistics from a random initialization.

---

## Gradient verification results

All operations verified to 1e-2 tolerance against float64 central differences:

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

---

## Bug found during development

Phase 3 surfaced a real bug in `autograd/tensor.py`: `_accumulate_grad` was
reshaping incorrectly when positional embeddings `(1, T, d)` were broadcast-added
to token embeddings `(B, T, d)`. The gradient for the positional embedding had
shape `(B, T, d)` but the tensor had shape `(1, T, d)` -- the code summed over
the right axes but then called `.reshape()` which produced the wrong shape.
Fixed by using `keepdims=True` during the sum and only removing leading axes
separately. This is the kind of subtle broadcasting bug that's hard to catch
without a real multi-layer forward pass that exercises the gradient path.

---

## Phases

**Phase 1: Autograd engine — done**

- [x] `autograd/tensor.py` — `Tensor` class with reverse-mode backprop. Every
      operation records a `_backward` function. Topological sort drives the
      backward pass.
- [x] 23 tests including numerical gradient checks for all operations.

**Phase 2: Neural network layers — done**

- [x] `nn/layers.py` — `Linear`, `LayerNorm`, `Embedding`, `Dropout`,
      `cross_entropy_loss`. All gradient-checked.
- [x] 21 tests.

**Phase 3: Transformer — done**

- [x] `transformer/transformer.py` — decoder-only transformer with causal
      self-attention, pre-norm MLP blocks, residual connections.
- [x] Causal masking verified: changing token t+1 doesn't affect logits at t.
- [x] 16 tests.

**Phase 4: AdamW optimizer + training loop — done**

- [x] `optim/optim.py` — AdamW with decoupled weight decay, cosine LR schedule
      with warmup, gradient clipping, Trainer loop.
- [x] AdamW verified to converge on a quadratic: finds minimum of (x-3)^2.
- [x] 11 tests.

**Phase 5: Train on Shakespeare — done**

- [x] `train.py` — character-level language model on 1.1M character Shakespeare
      corpus. Loss converges from 5.06 to 2.26 in 500 steps (12.2s on CPU).
- [x] 42K tokens/second throughput.
- [x] Text generation with temperature sampling.

---

## Running tests

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy pytest
python -m pytest tests/ -v   # 71 tests
```

---

## Training

```bash
# Quick test (100 steps, ~3s)
python train.py --steps 100

# Full run (500 steps, ~12s)
python train.py --steps 500 --generate

# Longer run (better quality)
python train.py --steps 2000 --d-model 128 --n-layers 4
```

---

## Project layout

```
minigrad/
├── autograd/
│   └── tensor.py         <- Tensor, autograd engine (Phase 1)
├── nn/
│   └── layers.py         <- Linear, LayerNorm, Embedding, Dropout (Phase 2)
├── transformer/
│   └── transformer.py    <- decoder-only transformer (Phase 3)
├── optim/
│   └── optim.py          <- AdamW, cosine schedule, Trainer (Phase 4)
├── data/
│   ├── shakespeare.txt   <- 1.1M character training corpus
│   └── training_results.json
├── train.py              <- Shakespeare training script (Phase 5)
└── tests/                <- 71 tests, all passing
```

---

## Author

**Sujan Uppalli Jayadevappa**
MS Software Engineering — Arizona State University
GitHub: [sujanuj](https://github.com/sujanuj)
