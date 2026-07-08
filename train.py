"""Train a character-level transformer on Shakespeare.

This is the payoff of Phases 1-4: train the transformer we built from
scratch on real data and show the loss curve converging.

Character-level language model: the vocabulary is the set of unique
characters in Shakespeare (~65 characters). The model predicts the next
character given the previous seq_len characters. After training, it can
generate Shakespeare-like text.

Model config (small, trains in ~5 minutes on CPU):
  vocab_size: ~65 (unique chars in Shakespeare)
  d_model: 64
  n_heads: 4
  n_layers: 3
  d_ff: 256
  seq_len: 64
  batch_size: 16

Expected results on Apple M5 CPU:
  Initial loss: ~4.17 (= log(65), random model)
  After 500 steps: ~2.0-2.3
  After 1000 steps: ~1.8-2.0
  Training time: ~8-12 minutes for 500 steps

Run:
  python train.py              # full training run
  python train.py --steps 100  # quick test
  python train.py --generate   # generate text after training
"""

import argparse
import json
import time
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Data loading and tokenization
# ---------------------------------------------------------------------------

def load_shakespeare(path: str = "data/shakespeare.txt"):
    text = Path(path).read_text(encoding="utf-8")
    chars = sorted(set(text))
    vocab_size = len(chars)
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for i, c in enumerate(chars)}
    data = np.array([char_to_idx[c] for c in text], dtype=np.int32)
    return data, vocab_size, char_to_idx, idx_to_char


def get_batch(data: np.ndarray, batch_size: int, seq_len: int):
    """Sample a random batch of (input, target) sequences."""
    ix = np.random.randint(0, len(data) - seq_len - 1, size=batch_size)
    x = np.stack([data[i:i+seq_len] for i in ix])
    y = np.stack([data[i+1:i+seq_len+1] for i in ix])
    return x, y


# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------

def generate(model, idx_to_char, start_text: str, char_to_idx: dict,
             max_new_tokens: int = 200, temperature: float = 0.8) -> str:
    """Generate text from the trained model using top-p sampling."""
    model.eval()

    # Encode start text
    context = np.array([[char_to_idx.get(c, 0) for c in start_text]], dtype=np.int32)

    generated = list(start_text)
    seq_len = model.max_seq_len

    for _ in range(max_new_tokens):
        # Crop context to max_seq_len
        ctx = context[:, -seq_len:]
        logits = model(ctx)

        # Take logits for the last position
        last_logits = logits.data[0, -1, :]  # (vocab_size,)

        # Apply temperature
        last_logits = last_logits / temperature

        # Softmax
        last_logits -= last_logits.max()
        probs = np.exp(last_logits)
        probs /= probs.sum()

        # Sample
        next_idx = np.random.choice(len(probs), p=probs)
        generated.append(idx_to_char[next_idx])

        context = np.concatenate(
            [context, np.array([[next_idx]], dtype=np.int32)], axis=1
        )

    model.train()
    return "".join(generated)


# ---------------------------------------------------------------------------
# Main training script
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Load data
    print("Loading Shakespeare...")
    data, vocab_size, char_to_idx, idx_to_char = load_shakespeare()
    n = len(data)
    split = int(0.9 * n)
    train_data = data[:split]
    val_data = data[split:]
    print(f"  {n:,} characters, vocab_size={vocab_size}")
    print(f"  train: {len(train_data):,} chars, val: {len(val_data):,} chars")

    # Build model
    from transformer.transformer import Transformer
    from optim.optim import AdamW, cosine_lr_schedule

    model = Transformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=args.seq_len,
        dropout_p=0.1,
    )
    n_params = model.num_parameters()
    print(f"\nModel: {n_params:,} parameters")
    print(f"  d_model={args.d_model}, n_heads={args.n_heads}, "
          f"n_layers={args.n_layers}, d_ff={args.d_ff}")

    # Optimizer
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    lr_schedule = lambda step: cosine_lr_schedule(
        step,
        warmup_steps=max(10, args.steps // 20),
        max_steps=args.steps,
        max_lr=args.lr,
        min_lr=args.lr / 10,
    )

    # Training loop
    print(f"\nTraining for {args.steps} steps...")
    print(f"  batch_size={args.batch_size}, seq_len={args.seq_len}, lr={args.lr}")
    print(f"  Expected initial loss: {np.log(vocab_size):.2f} (random model)")
    print()

    train_losses = []
    val_losses = []
    log_steps = []
    t0 = time.time()
    tokens_per_step = args.batch_size * args.seq_len

    for step in range(1, args.steps + 1):
        opt.lr = lr_schedule(step)

        x, y = get_batch(train_data, args.batch_size, args.seq_len)
        opt.zero_grad()
        model.train()
        loss = model.loss(x, y)
        loss.backward()

        # Gradient clipping
        total_norm = 0.0
        for p in opt.params:
            if p.grad is not None:
                total_norm += np.sum(p.grad ** 2)
        total_norm = np.sqrt(total_norm)
        if total_norm > 1.0:
            scale = 1.0 / (total_norm + 1e-8)
            for p in opt.params:
                if p.grad is not None:
                    p.grad *= scale

        opt.step()
        train_loss = float(loss.data)

        if step % args.eval_interval == 0 or step == 1:
            # Estimate val loss
            model.eval()
            val_loss_sum = 0.0
            n_val_batches = 5
            for _ in range(n_val_batches):
                xv, yv = get_batch(val_data, args.batch_size, args.seq_len)
                vl = model.loss(xv, yv)
                val_loss_sum += float(vl.data)
            val_loss = val_loss_sum / n_val_batches

            elapsed = time.time() - t0
            tokens_seen = step * tokens_per_step
            tok_per_sec = tokens_seen / elapsed

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            log_steps.append(step)

            print(f"step {step:4d}/{args.steps} | "
                  f"train={train_loss:.4f} | val={val_loss:.4f} | "
                  f"lr={opt.lr:.2e} | "
                  f"tok/s={tok_per_sec:.0f} | "
                  f"elapsed={elapsed:.1f}s")

    total_time = time.time() - t0
    print(f"\nTraining complete in {total_time:.1f}s")
    print(f"Final train loss: {train_losses[-1]:.4f}")
    print(f"Final val loss:   {val_losses[-1]:.4f}")
    print(f"Loss reduction:   {train_losses[0]:.4f} -> {train_losses[-1]:.4f} "
          f"({(train_losses[0]-train_losses[-1])/train_losses[0]*100:.1f}% reduction)")

    # Save results
    results = {
        "steps": log_steps,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "total_time_s": total_time,
        "config": {
            "vocab_size": vocab_size,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "d_ff": args.d_ff,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "n_params": n_params,
        }
    }
    Path("data/training_results.json").write_text(json.dumps(results, indent=2))
    print("Results saved to data/training_results.json")

    if args.generate:
        print("\n" + "="*60)
        print("Generated text (temperature=0.8):")
        print("="*60)
        sample = generate(model, idx_to_char, "ROMEO:", char_to_idx,
                         max_new_tokens=300)
        print(sample)


if __name__ == "__main__":
    main()
