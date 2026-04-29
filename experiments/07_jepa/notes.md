# Experiment 07 — T-JEPA (Educational, Hybrid)

## Why this is a hybrid, not pure JEPA
JEPA (Joint Embedding Predictive Architecture) is fundamentally a
**representation learning** framework — it predicts in EMBEDDING space, not
token space. There is no direct way to extract `p(token | context)` from a
JEPA model, which is what BPB requires.

To get BPB out, we attach a small auxiliary **decoder head** (`Linear(D, vocab_size)`)
trained jointly with the JEPA objective. The decoder treats the context
encoder's output as features and predicts the original tokens. This makes the
model BPB-scorable, at the cost of contaminating "pure JEPA-ness" — the
decoder's CE gradient flows back into the encoder.

If you want **pure JEPA**, throw away the decoder head, train with only the
JEPA loss, and evaluate via downstream tasks (linear probe, nearest neighbor,
etc.) rather than BPB.

## What is JEPA conceptually

### The problem with generative pre-training (per LeCun)
Cross-entropy on tokens forces the model to learn distributional shape over the
discrete output. For images especially, this means matching every pixel — much
of which is noise. The model spends capacity learning textures, not semantics.

JEPA's bet: predict **embeddings of held-out parts** instead of raw values.
Embeddings collapse irrelevant variation (texture, syntactic alternates) and
focus on semantic content. The model learns "what" without learning the
spelling.

### Architecture
Three modules:
1. **Context encoder** `f_theta` (trained): processes a partially-visible
   sequence (some spans masked).
2. **Target encoder** `f_theta_bar` (EMA, no grads): processes the full sequence.
3. **Predictor** `g_phi` (trained): takes context encoder output + position
   info for masked spans, outputs predictions in embedding space.

### Loss
For each masked position p in span:
```
L = || g_phi(context_emb, p) - stopgrad(f_theta_bar(full_seq))[p] ||^2
```

That's it. No softmax, no cross-entropy, no token-level supervision. Just L2
between predicted and actual embeddings.

### Why doesn't this collapse?
A naive setup where both encoders are the same trainable network would have a
trivial solution: both produce constant outputs, predictor outputs the same
constant, loss = 0. JEPA prevents this with **EMA target encoder + stopgrad**:
- Target encoder is a slow EMA copy of context encoder.
- No gradients flow into target encoder (stopgrad).
- The asymmetry (different rates of change, no shared gradient) breaks the
  trivial solution. This is the BYOL trick (Bootstrap Your Own Latent).

### Empirical evidence (from I-JEPA, V-JEPA papers)
- Beats masked-pixel-prediction (MAE) on linear probe accuracy at the same
  param count and pretrain compute.
- Embeddings transfer to downstream tasks better than generative pretraining.
- For text specifically there's less published work; Meta's data2vec uses a
  similar idea but for self-supervised speech and was extended to text.

## Training mechanics
- Sample 4 random spans of mean length 32 within a 1024-token sequence (~12%
  masking). Conservative — too aggressive masking can starve the predictor.
- Encode FULL sequence with both context encoder and target encoder.
- Zero out the masked positions in context_encoder's output before passing to
  the predictor.
- Target encoder updated via EMA: `θ_bar ← 0.996 * θ_bar + 0.004 * θ`.

## Expected BPB
- The decoder head sees BIDIRECTIONAL context (the encoder is bidirectional),
  so its BPB is NOT a valid AR likelihood. It will be lower (better) than a
  causal model could achieve, because it peeks at right-context.
- Realistically, ~1.4-1.8 BPB at this scale and training budget. Won't beat AR
  on this benchmark.
- The interesting metric isn't the BPB; it's **how the JEPA loss decreases**
  during training, which tells you whether the embedding-space prediction is
  actually working.

## What to look for during training
- `jepa_loss` should decrease steadily (it's L2 in embedding space).
- `dec_loss` (the decoder CE) should also decrease but more slowly — the
  decoder is a small linear, and the encoder is shaped primarily by the JEPA
  loss.
- If `jepa_loss` decreases to near zero very fast, you have collapse — the
  EMA decay is probably too fast (try `EMA_DECAY=0.999`) or the masking is
  too easy (try more spans / longer spans).

## Caveats
- **No DDP wrapping**: this script runs single-process for clarity. Multi-rank
  would need DDP wrapping the trainable submodules and careful EMA broadcast.
- **No quantization, no GPTQ, no TTT**: this is pretrain-only.
- **Bidirectional eval**: the decoder uses non-causal context. Don't compare
  this BPB to AR baselines.

## Why this is interesting research-wise (not for parameter-golf)
The interesting question for JEPA is "does embedding-space prediction generalize
better than token-space prediction"? You'd answer this with downstream evals
(linear probe accuracy on classification, retrieval quality), not BPB.

For parameter-golf specifically, JEPA is the wrong objective. The benchmark
rewards exactly the thing JEPA tries to avoid (token-level prediction). So
this experiment is fundamentally a "hello-world for JEPA" — useful for
understanding the architecture, not for winning.

## Run
```bash
source experiments/07_jepa/config.env
RUN_ID=07_tjepa_seed42 SEED=42 \
  torchrun --standalone --nproc_per_node=1 experiments/07_jepa/tw_eval.py
```
(Use `--nproc_per_node=1` for the demo — multi-rank needs DDP code added.)

## Further reading
- LeCun, "A Path Towards Autonomous Machine Intelligence" (2022). The paper
  that popularized JEPA. Section 5 introduces the predictive architectures.
- Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding
  Predictive Architecture" (I-JEPA, 2023). The first concrete JEPA training
  paper, on images.
- Bardes et al., "V-JEPA: Latent Video Prediction for Visual Representation
  Learning" (2024). V-JEPA's mask scheduling tricks transfer to text.
- Grill et al., "Bootstrap Your Own Latent" (BYOL, 2020). The EMA-target
  trick that prevents JEPA-style collapse, predates JEPA's framing.
- Caron et al., "DINO" (2021). Another EMA-target approach, slightly different.
