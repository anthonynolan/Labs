# GPT-2 from scratch (in tinygrad)

You're going to implement GPT-2 from scratch in [tinygrad](https://github.com/tinygrad/tinygrad). The skeleton you edit has the scaffolding — config dataclass, weight initialisation, training loop, optimiser, dataset loader, causal mask, scheduler — and the model itself is missing: token + positional embeddings, attention, FFN, the decoder block, and the forward pass. By the end of the session you'll have a working transformer that you can train on TinyShakespeare and use to generate Shakespeare-flavoured text.

## Pick your difficulty

There are three skeleton files. Pick one and edit it. They all expose the same public API to `train.py`, so the runnable command is the same regardless of difficulty.

| File | What's filled in for you | What you implement |
|---|---|---|
| [skeleton_easy.py](skeleton_easy.py) | Config defaults, init, causal mask, optimiser, scheduler, training loop, dataset, every `__init__` | Just the algorithm bodies: embedding `__call__`s, attention, FFN, decoder block `__call__`, forward pass, loss, generate |
| [skeleton_medium.py](skeleton_medium.py) | `_init_weights`, dataset, `train` outer loop | Everything in easy, **plus** the `GPT2Config` dataclass, `make_causal_mask`, every `__init__` (structural assembly), `CosineScheduleWithWarmup`, `make_optimizer`, and `train_step` |
| [skeleton_hard.py](skeleton_hard.py) | Just `from tinygrad import Tensor`. Have fun. | Everything. |

Default is easy. Run with `--difficulty medium` or `--difficulty hard` (see Setup below) to switch.

## Repo layout

| File | What it is |
|---|---|
| `skeleton_{easy,medium,hard}.py` | **What you edit.** See the table above. Each TODO maps to a step in the roadmap below. |
| [train.py](train.py) | **What you run.** Loads TinyShakespeare, tokenises it with the GPT-2 BPE, trains the model from your chosen skeleton, then samples some text. Picks the skeleton via `--difficulty`. |
| [tinyshakespeare.txt](tinyshakespeare.txt) | The training corpus. ~1.1MB of Shakespeare's plays as a single text file. Already in the repo, don't worry about it. |
| [pyproject.toml](pyproject.toml) | Project + dependency declaration for `uv`. |

`main.py` (the full reference implementation) is **not in the repo by design.** If you go looking for it, you won't find it — that's intentional. Neil may share it after the session.

## Setup

Two paths depending on where you are.

### At the venue (RunPod)

The pod template already has CUDA, Python, and the heavy deps. You just need:

```bash
git clone <repo-url> gpt2-lab && cd gpt2-lab
uv sync
uv run python train.py                        # default: easy
# or
uv run python train.py --difficulty medium
uv run python train.py --difficulty hard
```

That last command will fail until you've implemented enough of your chosen skeleton to make a forward pass go through. That's expected.

### On your own laptop (the fallback path)

If you want to keep tinkering after the session, or you're prepping ahead of time. You need Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone <repo-url> gpt2-lab && cd gpt2-lab
uv sync
uv run python train.py
```

**Apple Silicon caveat (handled for you).** On macOS, tinygrad defaults to the Metal backend, which can hit a shader-buffer limit (31 buffers per kernel) on the kernels this model fuses. The skeleton has a platform-guarded fallback at the top of the file:

```python
if platform.system() == "Darwin":
    os.environ["DEV"] = "CLANG"   # CPU backend on Mac
```

So macOS uses CPU and Linux/CUDA (RunPod) uses GPU automatically. CPU is plenty fast for the toy smoke tests, but full training on TinyShakespeare will be slow on a laptop — RunPod is the path that produces results in 10 minutes.

If you'd like to try Metal anyway (it may be fine on recent tinygrad), comment that block out and run; if you see "shader buffers" or "fuser" errors, put it back.

**First-run kernel compile.** Tinygrad JIT-compiles a CUDA/Metal/Clang kernel for each unique tensor op pattern the first time it sees one. Expect the first forward pass to hang for 10–60 seconds while it does this. **It's not stuck.** Subsequent forward passes reuse the compiled kernels and are fast. The same applies on the first backward pass and the first call to `greedy_generate`. You'll see roughly three "long pauses" during a training+sampling run, then everything cooks.

## Implementation roadmap

Implement in this order. Each step has a checkpoint — a way to verify it works before moving on. The smoke test at the bottom of your skeleton runs a tiny config end-to-end; you can run `uv run python skeleton_easy.py` (or `_medium.py`, etc.) after each step to see what still breaks.

The roadmap below is described in terms of the easy skeleton. Medium and hard expand the same six steps with extra implementation work — the docstring at the top of each file lists exactly what's stubbed for that difficulty.

### Step 1 — Token + positional embeddings

Implement `TokenEmbedding.__call__` and `LearnedPositionalEmbedding.__call__`.

**You'll know it works when:** with a fresh `GPT2(cfg)` and tokens of shape `(B, T)`, `model.token_embed(tokens).shape == (B, T, d_model)` and the positional embedding adds something of the same shape.

### Step 2 — Scaled dot-product attention + multi-head attention

Implement `scaled_dot_product_attention` and `MultiHeadAttention.__call__`.

**You'll know it works when:** the causal mask actually masks. Build two batches that share a prefix but differ in the suffix; pass them through the attention with the causal mask; the activations over the shared prefix should match. If they don't, your mask is wrong or applied in the wrong place.

### Step 3 — Position-wise feed-forward + decoder block

Implement `PositionwiseFeedForward.__call__` and `DecoderBlock.__call__`. Remember pre-norm, not post-norm.

**You'll know it works when:** a single-layer model trains on TinyShakespeare and the loss drops over the first ~50 steps. If it sits at ~10.8 forever, the residual or the norm is in the wrong place.

### Step 4 — Stack blocks + final layer norm + tied unembedding

Implement `GPT2.forward` (and `GPT2.__call__`). The output projection is tied to the input embedding — use `token_embed.weight.transpose()`, don't introduce a new `Linear`.

**You'll know it works when:** `model(tokens, causal_mask).shape == (B, T, vocab_size)`. The smoke test at the bottom of your skeleton checks this.

### Step 5 — Loss + train

Implement `lm_loss`. Then run `uv run python train.py`.

**You'll know it works when:** the loss falls from ~10.8 (uniform random over 50,257 tokens is `ln(50257) ≈ 10.83`) to ~4.x within a few hundred steps. If it stays flat, the loss masking is probably wrong or your gradients aren't flowing.

### Step 6 — Generate

Implement `greedy_generate`. `train.py` calls it after training and prints the result.

**You'll know it works when:** the output is recognisably English-shaped — you'll see real words, the rhythm of Shakespeare's character names ("KING", "GLOUCESTER:"), and roughly sensible punctuation. It won't be coherent. That's fine.

## Hyperparameters

Use these. They train in ~10 minutes on a 4090 and produce recognisable Shakespeare:

```
d_model    = 384
n_heads    = 6
n_layers   = 6
d_ff       = 1536       # 4 * d_model
max_seq_len = 256
batch_size = 32
n_steps    = 2000
```

These are the defaults in `GPT2Config` and `train.py`. **Don't change them until you have a working baseline.** The full GPT-2 124M config (d_model=768, n_layers=12) will not converge to anything useful in 10 minutes and will mostly just confuse you about whether your implementation is correct.

## Dataset

`tinyshakespeare.txt` is the concatenation of Shakespeare's plays — about 1.1MB of plain text, ~300k tokens after BPE. It's already in the repo. The tokeniser is OpenAI's GPT-2 BPE via [tiktoken](https://github.com/openai/tiktoken):

```python
import tiktoken
enc = tiktoken.get_encoding("gpt2")
```

`train.py` handles loading and tokenisation; you don't need to touch it.

## Stretch goals

If you finish early, in rough order of usefulness:

- **KV cache for generation.** Right now `greedy_generate` re-runs the whole forward pass each step — O(T²) total. Cache the K and V tensors per layer so each new token costs O(T).
- **Gradient clipping.** Add `||grad||_2 < 1.0` clipping before `optimizer.step()`. Stabilises long training runs.
- **Decay / no-decay parameter groups.** AdamW is currently applied uniformly — the standard refinement is to exclude biases and LayerNorm gains from weight decay. Splits cleanly with `get_parameters`.
- **Try TinyStories.** Swap the dataset for [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) and see how outputs change with a different distribution.
- **Scale up to GPT-2 124M.** `d_model=768, n_heads=12, n_layers=12, max_seq_len=1024`. Will need a longer training run.

## Stuck?

Three or four things that catch people most often:

- **`tinygrad.codegen` errors mentioning shader buffers on macOS.** You're on Metal. The platform guard at the top of the skeleton should be activating the CPU backend — make sure you haven't deleted or moved the `if platform.system() == "Darwin": os.environ["DEV"] = "CLANG"` block, and that it sits *before* the tinygrad import.
- **Loss is NaN.** Lower the learning rate (try 1e-4 instead of 2.5e-4) and/or add gradient clipping. Often happens if the softmax sees `-inf` everywhere because the mask was built wrong.
- **GPU isn't being used (on RunPod).** Check `Tensor.ones(10).realize().device` — should print `CUDA`. If it says `CLANG` or `CPU`, either the platform guard isn't behaving as expected (`platform.system()` returning something unexpected) or someone explicitly exported `CLANG=1` in the shell. Unset it: `unset CLANG`.
- **First forward pass hangs for ~30s.** That's the JIT kernel compile, not a bug. See the warning above the roadmap. Wait it out.
- **Loss starts at ~10.8 and stays there.** Your model is producing uniform-random logits. Most common cause: forward pass not actually wired through (e.g. embedding output not flowing into blocks, or `ln_f` skipped, or unembedding weight not tied correctly).

If none of those, ask Neil.
