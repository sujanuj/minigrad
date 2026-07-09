# minigrad

A training engine built from raw NumPy — no PyTorch autograd, no `nn.Module`. Implements reverse-mode automatic differentiation, neural network layers, a decoder-only transformer, AdamW optimizer, and trains on Shakespeare with a verified loss curve.

The goal: implement every piece that makes neural network training work, verify it is correct, and measure it honestly.

---

## What makes this different

Most "I built a neural network" projects call `loss.backward()` and trust PyTorch. This project implements `backward()` itself — the computation graph, topological sort, gradient accumulation, and every operation's backward function. Gradient correctness is verified using numerical gradient checking (the same method PyTorch's `gradcheck` uses internally).

---

## Training results (Shakespeare, Apple M5 CPU)

**Small model (500 steps, 12s):**

```
Model: 161,728 parameters
  d_model=64, n_heads=4, n_layers=3, d_ff=256, seq_len=64

step    1/500 | train=5.06 | val=4.94
step  500/500 | train=2.26 | val=2.34

Loss: 5.06 -> 2.26 (55% reduction) | 42K tok/s
```

**Larger model (2000 steps, 204s):**

```
Model: 1,211,648 parameters
  d_model=128, n_heads=4, n_layers=6, d_ff=512, seq_len=64

step    1/2000 | train=6.40 | val=6.11
step  200/2000 | train=2.50 | val=2.43
step  600/2000 | train=2.00 | val=2.03
step 1000/2000 | train=1.83 | val=1.86
step 1400/2000 | train=1.63 | val=1.76
step 2000/2000 | train=1.63 | val=1.65

Loss: 6.40 -> 1.63 (74.5% reduction) | 10K tok/s
```

Generated text after 2000 steps (temperature=0.8):

```
ROMEO:
God I must shall I am a dearth, thear bance in queen!

GREMIO:
Why, for your sweet sea stoly to men:
My sain, by thou love adm my are by Postengrial that.

GLOUCESTER:
For the stomes of Romeo, all hoted heard thou sure,
Ere wilt would this trust are rage at is
am to best thy friet.
```

Real Shakespeare character names (`ROMEO`, `GREMIO`, `GLOUCESTER`),
grammatical English sentence structure, and blank verse cadence — all
learned from a random initialization with no pretrained weights.

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
      backward pass. Broadcasting handled in `_accumulate_grad`.
- [x] **Gradients verified numerically**: all operations verified to 1e-2
      tolerance against float64 central differences.
- [x] 23 tests: forward pass, hand-computed gradients, numerical gradient
      checks for add/mul/matmul/relu/exp/pow/softmax/gelu/composed expressions.

**Phase 2: Neural network layers — done**

- [x] `nn/layers.py` — four layers built on the autograd engine:
  - `Linear(in, out)` — `y = x @ W.T + b`, Kaiming uniform initialization
  - `LayerNorm(d)` — normalize across last dimension, learnable scale/shift
  - `Embedding(vocab, dim)` — lookup table with sparse gradient
  - `Dropout(p)` — inverted dropout, no-op at eval time
- [x] `cross_entropy_loss` — numerically stable log-softmax + NLL in one pass.
- [x] 21 tests including gradient checks for Linear and cross_entropy.

**Phase 3: Transformer — done**

- [x] `transformer/transformer.py` — decoder-only transformer:
  - Token + positional embeddings
  - N stacked blocks: pre-norm causal self-attention + pre-norm MLP + residuals
  - Final LayerNorm + linear projection to logits
- [x] Causal masking verified: changing token at t+1 doesn't affect logits at t.
- [x] Bug found and fixed: broadcasting gradient in `_accumulate_grad` was
      reshaping incorrectly for `(1, T, d)` positional embeddings.
- [x] 16 tests.

**Phase 4: AdamW optimizer + training loop — done**

- [x] `optim/optim.py` — AdamW with decoupled weight decay, per-parameter
      moment estimates, bias correction.
- [x] Cosine LR schedule with linear warmup.
- [x] Trainer loop with gradient clipping (global norm), eval loop, logging.
- [x] AdamW verified to converge on quadratic: finds minimum of `(x-3)^2`.
- [x] 11 tests.

**Phase 5: Train on Shakespeare — done**

- [x] `train.py` — character-level LM on 1.1M character Shakespeare corpus.
- [x] Small model: loss 5.06 -> 2.26 in 500 steps (12s, 42K tok/s).
- [x] Larger model: loss 6.40 -> 1.63 in 2000 steps (204s, 10K tok/s).
- [x] Generated text shows real Shakespeare character names and sentence
      structure after training from random initialization.

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

# Small model (500 steps, ~12s)
python train.py --steps 500 --generate

# Larger model (2000 steps, ~3.5 min)
python train.py --steps 2000 --d-model 128 --n-layers 6 --d-ff 512 --generate
```

---

## Project layout

```
minigrad/
├── autograd/
│   └── tensor.py         <- Tensor, reverse-mode autograd (Phase 1)
├── nn/
│   └── layers.py         <- Linear, LayerNorm, Embedding, Dropout (Phase 2)
├── transformer/
│   └── transformer.py    <- decoder-only transformer, causal attention (Phase 3)
├── optim/
│   └── optim.py          <- AdamW, cosine LR schedule, Trainer (Phase 4)
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
