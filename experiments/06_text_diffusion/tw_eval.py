"""Discrete (masked) text diffusion language model — educational implementation.

This is a self-contained reference implementation of MDLM (Sahoo et al. 2024,
"Simple and Effective Masked Diffusion Language Models"). It does NOT share
the bigbag stack — the goal is clarity, not record performance.

What's the same as the AR setup:
  - Same FineWeb sp8192 data
  - Same loss reduction (cross-entropy on tokens)
  - Same ~25M-param transformer body

What's different:
  - Attention is BIDIRECTIONAL (no causal mask) — the model sees the whole
    sequence at once, which is required for diffusion since the corruption
    is non-causal.
  - The training objective is the MDLM ELBO, not next-token prediction.
  - Evaluation is diffusion NLL estimation via Monte Carlo over noise levels.

The math, in 8 lines:

  Let x = (x_1, ..., x_L) be a sequence of token ids.
  Forward (corruption) process: at time t in [0, 1], replace each x_i with
    a special MASK token independently with probability t. So at t=0 the
    sequence is unchanged; at t=1 it's all MASK.
  Reverse process: a model p_theta(x_i | x_t) predicts each masked token
    given the partially-masked sequence x_t.
  Training loss (MDLM ELBO, see Sahoo et al. eq. 6):
    L = E_{t ~ U(0,1)} E_{x ~ data} E_{x_t ~ q(.|x,t)} [
          (alpha'_t / (1 - alpha_t)) * sum_i 1{x_t,i = MASK} * -log p_theta(x_i | x_t)
        ]
  With the linear schedule alpha_t = 1 - t:
    alpha'_t = -1, (1 - alpha_t) = t, so weight w(t) = 1/t. Diverges at t=0
    so we sample t ~ U(epsilon, 1) for some small epsilon.
  At eval time, the same expectation gives a NLL upper bound; we estimate
    it by Monte Carlo over t and over masking realizations.

For deeper reading: the SEDD paper (Lou et al. 2024) generalizes this with a
non-mask absorbing state and a more flexible "score entropy" loss; MDLM is a
clean special case.
"""

import math
import os
import time
import glob

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class H:
    data_dir = os.environ.get('DATA_DIR', './data/')
    seed = int(os.environ.get('SEED', 1337))
    iterations = int(os.environ.get('ITERATIONS', 4500))
    batch_seqs = int(os.environ.get('BATCH_SEQS', 16))
    seq_len = int(os.environ.get('SEQ_LEN', 1024))
    log_every = int(os.environ.get('LOG_EVERY', 100))
    eval_t_samples = int(os.environ.get('EVAL_T_SAMPLES', 32))
    eval_mc_per_t = int(os.environ.get('EVAL_MC_PER_T', 4))
    eval_max_seqs = int(os.environ.get('EVAL_MAX_SEQS', 256))
    eps_t = float(os.environ.get('EPS_T', 1e-3))
    vocab_size = int(os.environ.get('VOCAB_SIZE', 8192))
    num_layers = int(os.environ.get('NUM_LAYERS', 8))
    model_dim = int(os.environ.get('MODEL_DIM', 384))
    num_heads = int(os.environ.get('NUM_HEADS', 6))
    mlp_mult = int(os.environ.get('MLP_MULT', 4))
    lr = float(os.environ.get('LR', 3e-4))
    wd = float(os.environ.get('WD', 0.05))
    warmup_steps = int(os.environ.get('WARMUP_STEPS', 200))
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    is_main = rank == 0


def log(msg):
    if H.is_main:
        print(msg, flush=True)


def load_shard(file):
    header = np.fromfile(file, dtype='<i4', count=256)
    n_tokens = int(header[2])
    return torch.from_numpy(np.fromfile(file, dtype='<u2', count=n_tokens, offset=256 * 4).astype(np.int64))


def load_train_tokens():
    files = sorted(glob.glob(os.path.join(H.data_dir, 'datasets', f'fineweb10B_sp{H.vocab_size}', 'fineweb_train_*.bin')))
    return torch.cat([load_shard(f) for f in files[:4]])


def load_val_tokens():
    files = sorted(glob.glob(os.path.join(H.data_dir, 'datasets', f'fineweb10B_sp{H.vocab_size}', 'fineweb_val_*.bin')))
    return torch.cat([load_shard(f) for f in files])


def load_byte_lut():
    sp = spm.SentencePieceProcessor(model_file=os.path.join(H.data_dir, 'tokenizers', f'fineweb_{H.vocab_size}_bpe.model'))
    base = np.zeros((H.vocab_size,), dtype=np.int16)
    has_lead_space = np.zeros((H.vocab_size,), dtype=np.bool_)
    is_boundary = np.ones((H.vocab_size,), dtype=np.bool_)
    for tid in range(int(sp.vocab_size())):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            continue
        is_boundary[tid] = False
        if sp.is_byte(tid):
            base[tid] = 1
            continue
        piece = sp.id_to_piece(tid)
        if piece.startswith('▁'):
            has_lead_space[tid] = True
            piece = piece[1:]
        base[tid] = len(piece.encode('utf-8'))
    return torch.tensor(base, dtype=torch.int16), torch.tensor(has_lead_space, dtype=torch.bool), torch.tensor(is_boundary, dtype=torch.bool)


# Bidirectional attention (no causal mask) — diffusion needs whole-sequence context
class BidirectionalAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, S, D = x.shape
        qkv = self.qkv(x).view(B, S, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_mult):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)
        self.attn = BidirectionalAttention(dim, n_heads)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_mult * dim, bias=False), nn.GELU(),
                                  nn.Linear(mlp_mult * dim, dim, bias=False))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class DiffusionLM(nn.Module):
    """Bidirectional transformer that predicts original tokens from corrupted ones.

    Vocab is extended by 1: token id `vocab_size` is the special MASK symbol.
    The model output is logits over the original vocabulary (size = vocab_size).
    Time `t` is conditioned via a small MLP added to the input embeddings.
    """

    def __init__(self, h=H):
        super().__init__()
        self.mask_token_id = h.vocab_size
        self.tok_emb = nn.Embedding(h.vocab_size + 1, h.model_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, h.seq_len, h.model_dim))
        self.t_proj = nn.Sequential(nn.Linear(1, h.model_dim), nn.SiLU(), nn.Linear(h.model_dim, h.model_dim))
        self.blocks = nn.ModuleList([TransformerBlock(h.model_dim, h.num_heads, h.mlp_mult) for _ in range(h.num_layers)])
        self.norm = nn.RMSNorm(h.model_dim)
        self.head = nn.Linear(h.model_dim, h.vocab_size, bias=False)
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb, std=0.02)
        nn.init.zeros_(self.head.weight)

    def forward(self, x_t, t):
        B, S = x_t.shape
        h = self.tok_emb(x_t) + self.pos_emb[:, :S]
        h = h + self.t_proj(t.view(B, 1, 1).expand(-1, S, -1).to(h.dtype))
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.norm(h))


def corrupt_mdlm(x, t, mask_id):
    """Forward (corruption) process: independently mask each token with prob t."""
    rand = torch.rand_like(x, dtype=torch.float32)
    mask = rand < t.view(-1, 1).expand_as(rand)
    x_t = torch.where(mask, torch.full_like(x, mask_id), x)
    return x_t, mask


def mdlm_loss(model, x, eps_t):
    """MDLM ELBO loss with linear schedule alpha_t = 1 - t.

    L = E_t E_{x_t | x} [ (1/t) * sum_i 1{masked_i} * -log p_theta(x_i | x_t) ]

    The 1/t weight is the importance weight that turns per-position cross-entropy
    at noise level t into a contribution to the global negative ELBO. Sample
    t ~ U(eps_t, 1) (eps_t avoids the 1/t blowup at t=0).
    """
    B, S = x.shape
    t = torch.rand(B, device=x.device) * (1.0 - eps_t) + eps_t
    x_t, mask = corrupt_mdlm(x, t, model.mask_token_id)
    logits = model(x_t, t)
    nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), x.reshape(-1), reduction='none').reshape(B, S)
    masked_count = mask.sum(dim=1).clamp_min(1).float()
    per_example_loss = (nll * mask.float()).sum(dim=1) / masked_count
    weighted = per_example_loss / t
    return weighted.mean()


@torch.no_grad()
def diffusion_nll_bpb(model, val_tokens, byte_lut_tuple, h):
    """Estimate NLL bound via Monte Carlo over t and corruption realizations."""
    base_lut, lead_lut, bound_lut = byte_lut_tuple
    base_lut = base_lut.to(val_tokens.device)
    lead_lut = lead_lut.to(val_tokens.device)
    bound_lut = bound_lut.to(val_tokens.device)
    model.eval()
    S = h.seq_len
    n_seqs = min(h.eval_max_seqs, (val_tokens.numel() - 1) // S)
    if n_seqs == 0:
        return float('inf'), float('inf')
    starts = torch.arange(n_seqs) * S
    batch = torch.stack([val_tokens[s:s + S] for s in starts]).to(val_tokens.device)
    t_grid = torch.linspace(h.eps_t, 1.0 - 1e-4, h.eval_t_samples, device=val_tokens.device)
    accum_nats = torch.zeros(n_seqs, S, device=val_tokens.device, dtype=torch.float64)
    for ti, t in enumerate(t_grid):
        for mc in range(h.eval_mc_per_t):
            t_batch = torch.full((n_seqs,), t.item(), device=batch.device)
            x_t, mask = corrupt_mdlm(batch, t_batch, model.mask_token_id)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(x_t, t_batch)
            nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch.reshape(-1), reduction='none').reshape(n_seqs, S)
            w = (1.0 / t.item())
            accum_nats += (nll * mask.float() * w).double()
    n_samples = len(t_grid) * h.eval_mc_per_t
    nats = accum_nats / n_samples
    tgt = batch
    prev = torch.cat([batch[:, :1] * 0, batch[:, :-1]], dim=1)
    bytes_per_pos = base_lut[tgt].double() + (lead_lut[tgt] & ~bound_lut[prev]).double()
    total_nats = nats.sum().item()
    total_bytes = bytes_per_pos.sum().item()
    val_loss = total_nats / nats.numel()
    val_bpb = total_nats / total_bytes / math.log(2)
    return val_loss, val_bpb


def main():
    distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    if distributed:
        dist.init_process_group(backend='nccl')
    device = torch.device('cuda', H.local_rank)
    torch.cuda.set_device(device)
    torch.manual_seed(H.seed + H.rank)

    model = DiffusionLM().to(device)
    if distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[H.local_rank])
    n_params = sum(p.numel() for p in model.parameters())
    log(f'diffusion_lm params: {n_params}')

    train_tokens = load_train_tokens().to(device)
    val_tokens = load_val_tokens().to(device)
    byte_lut = load_byte_lut()
    log(f'train_tokens: {train_tokens.numel()}  val_tokens: {val_tokens.numel()}')

    opt = torch.optim.AdamW(model.parameters(), lr=H.lr, weight_decay=H.wd, betas=(0.9, 0.95))

    n_train = (train_tokens.numel() - 1) // H.seq_len
    rng = np.random.default_rng(H.seed + H.rank)

    def next_batch():
        idx = rng.integers(0, n_train, size=H.batch_seqs)
        return torch.stack([train_tokens[i * H.seq_len: (i + 1) * H.seq_len] for i in idx])

    log('Starting MDLM training...')
    t0 = time.perf_counter()
    for step in range(1, H.iterations + 1):
        x = next_batch()
        lr_scale = min(1.0, step / max(H.warmup_steps, 1))
        for pg in opt.param_groups:
            pg['lr'] = H.lr * lr_scale
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss = mdlm_loss(model.module if distributed else model, x, H.eps_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step <= 5 or step % H.log_every == 0:
            elapsed = time.perf_counter() - t0
            log(f'{step}/{H.iterations} loss={loss.item():.4f} lr={H.lr*lr_scale:.2e} elapsed={elapsed:.1f}s')
    log('Training done. Estimating diffusion NLL...')
    if H.is_main:
        val_loss, val_bpb = diffusion_nll_bpb(model.module if distributed else model, val_tokens, byte_lut, H)
        log(f'final diffusion val_loss={val_loss:.4f}  val_bpb={val_bpb:.4f}')
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
