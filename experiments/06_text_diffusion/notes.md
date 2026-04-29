# Experiment 06 — Discrete (Masked) Text Diffusion (Educational)

## What it is
A self-contained MDLM (Masked Discrete Language Model, Sahoo et al. 2024)
implementation. Uses the same FineWeb sp8192 data and BPB scoring conventions
as the AR experiments, but the model is **non-autoregressive** and trained
with a denoising objective.

This is **not** integrated into bigbag's stack — the architecture is too
different. ~360 lines, fully readable.

## Why is text diffusion interesting?
Autoregressive generation has fundamental tradeoffs:
- O(N) inference even with KV-cache
- Errors compound left-to-right
- Hard to revise earlier tokens given later context

Diffusion offers:
- Constant-step inference (adjustable quality/speed tradeoff)
- Bidirectional context — every token sees the whole sequence
- Iterative refinement — early errors can be corrected

Whether it BEATS AR for compression-as-BPB is an empirical question.
Recent work (MDLM, SEDD) shows it gets close at the same param count, but
typically ~5-10% behind AR on perplexity at small scales. As of late 2025
no diffusion model has beaten AR on a competitive language benchmark, but
the gap is narrowing.

## The math, in detail

### Forward (corruption) process
Define a special MASK token id (we use `vocab_size`, one beyond the BPE vocab).
At time `t in [0, 1]`, each token is independently replaced with MASK with
probability `t`:

  q(x_t | x) = product over i of [ (1 - t) * delta(x_t,i = x_i)  +  t * delta(x_t,i = MASK) ]

This is the simplest possible discrete diffusion process — an "absorbing state"
diffusion where MASK is the absorbing state. At `t=0` we have the clean
sequence; at `t=1` everything is MASK.

### Reverse (denoising) process
A neural network `p_theta(x_i | x_t, t)` predicts each masked token's identity
given the partially-masked sequence and the noise level. Training maximizes the
ELBO of `log p_theta(x)`.

### The MDLM ELBO (eq. 6 in Sahoo et al.)
With linear schedule `alpha_t = 1 - t` (probability of being unchanged), the
negative ELBO simplifies to:

  -log p_theta(x) <= E_{t ~ U(0,1)} [ (1/t) * E_{x_t | x} [ sum over masked i: -log p_theta(x_i | x_t) ] ]

The `1/t` weight comes from `-d/dt(log alpha_t) = 1/t`. It diverges at `t=0`
so we sample `t ~ U(epsilon, 1)` for small epsilon.

### Why the 1/t weight?
At small t, very few tokens are masked, so each masked-token loss is rare but
high-information (the model has near-perfect context). At large t, most tokens
are masked, so each loss is frequent but low-information. The 1/t weight
upweights the rare-but-informative losses to give an unbiased ELBO estimate.

### Eval: estimating BPB
The same ELBO gives an upper bound on `-log p_theta(x)` per token. We estimate
it by Monte Carlo:
  1. Pick T noise levels `t_1, ..., t_T` (we use a uniform grid on (eps, 1)).
  2. For each `t_k`, sample `M` masking realizations and compute the weighted
     per-position cross-entropy.
  3. Average over (k, m) to get per-position negative ELBO contribution.
  4. Convert to BPB using the byte LUT (same as AR experiments).

The MDLM trick: at each grid point, the expected mask probability is `t_k`,
the weight is `1/t_k`, so the expected weight per position is 1. With T*M
samples, every position has roughly T*M expected weight. Hence the divisor.

## Architecture
- Bidirectional transformer (causal masking REMOVED — diffusion needs full context)
- 8 layers x 384 dim x 6 heads (smaller than the AR baseline because diffusion
  requires more compute per step at eval time and we want a comparable runtime)
- Time conditioning: a small MLP `R^1 -> R^d` projects scalar t and adds to
  every token's embedding (the simplest possible time conditioning)
- Output head: `Linear(d, vocab_size)` — predicts only over real tokens, never
  predicts MASK

## What's intentionally simpler than the AR setup
- No GPTQ quantization (16-bit weights ship)
- No layer looping
- No parallel residuals
- No TTT
- No EMA
- AdamW instead of Muon
- Plain bf16 autocast
- Single dataset shard subset

This is purely educational. Without these tricks, BPB will be 1.3+, not 1.08.
The point is to see the diffusion mechanism work end-to-end, not to win.

## Expected behavior
- Training loss starts ~9 (uniform over vocab) and falls as the model learns.
- At convergence, single-step diffusion BPB will be ~1.3-1.5 with this
  small/short config. Larger model + more iterations + multi-step sampling
  closes the gap to AR by another 0.1-0.2.
- The Monte Carlo NLL is noisy — increase `EVAL_T_SAMPLES` or `EVAL_MC_PER_T`
  to tighten the estimate at cost of eval time.

## Run
```bash
source experiments/06_text_diffusion/config.env
RUN_ID=06_diffusion_seed42 SEED=42 \
  torchrun --standalone --nproc_per_node=3 experiments/06_text_diffusion/tw_eval.py
```

## Suggested experiments to deepen understanding
1. **Plot loss vs t**: log per-t loss in `mdlm_loss` (separate buckets) and see
   the U-shape — middle t is hardest, very small/large t is easiest.
2. **Multi-step sampling**: not implemented here, but easy to add. Standard:
   start from all-MASK, iteratively denoise from t=1 to t=0 in K steps. More
   steps = better samples.
3. **Compare absorbing-state to uniform diffusion**: SEDD's alternative is to
   replace tokens with random vocab ids (not MASK). Code change: replace
   MASK in `corrupt_mdlm` with `torch.randint(0, vocab_size, ...)`. Slightly
   worse for small models, slightly better for large ones empirically.

## Further reading
- Sahoo et al., "Simple and Effective Masked Diffusion Language Models" (2024).
  This implementation closely follows their MDLM (Section 4).
- Lou et al., "Discrete Diffusion Language Modeling by Estimating the Ratios
  of the Data Distribution" (SEDD, 2024). More general framework, includes
  uniform diffusion. The score-entropy loss is more flexible but more complex.
- Austin et al., "Structured Denoising Diffusion Models in Discrete State-Spaces"
  (D3PM, 2021). Original discrete diffusion paper. MDLM is a special case.
