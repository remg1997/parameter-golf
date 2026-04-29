"""T-JEPA — Text Joint Embedding Predictive Architecture (Educational).

Pure JEPA does NOT directly produce a probability distribution over tokens, so
"val_bpb" cannot be computed without an auxiliary decoder. This script
implements T-JEPA pre-training (representation learning) AND attaches a tiny
decoder head that's trained jointly to make the embeddings BPB-scorable.

This is essentially a hybrid: JEPA as a representation learner + a thin head
on top. It's the simplest way to convert JEPA's embeddings into bits-per-byte.

Pure JEPA is interesting because:
  - It learns representations without ever generating data.
  - It's the framework LeCun has been advocating as a path beyond purely
    generative pre-training.
  - It (claims to) capture "abstract" features that token-level CE loss
    can't, by predicting in embedding space rather than token space.

The math:
  - Context encoder f_theta and target encoder f_theta_bar (EMA copy of f_theta).
  - Sample a context (visible) sequence c_x and target spans t_1, ..., t_K.
  - Compute s_x = f_theta(c_x): context embedding.
  - Compute s_y = stopgrad(f_theta_bar(full_x))[mask_positions]: target embeddings.
  - Predictor g_phi takes (s_x, mask_position_info) and outputs predictions.
  - Loss: L2 between g_phi(s_x, p_k) and s_y[p_k] for each masked position p_k.
  - To prevent collapse: target encoder uses EMA, never receives gradients.

Bibliography:
  - LeCun, "A Path Towards Autonomous Machine Intelligence" (2022) — JEPA framing.
  - Assran et al., "I-JEPA" (2023) — image masked-position prediction.
  - Bardes et al., "V-JEPA" (2024) — video version.
  - Text JEPA: less canonical; we're implementing the natural adaptation.
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
    vocab_size = int(os.environ.get('VOCAB_SIZE', 8192))
    num_layers = int(os.environ.get('NUM_LAYERS', 6))
    pred_layers = int(os.environ.get('PRED_LAYERS', 2))
    model_dim = int(os.environ.get('MODEL_DIM', 384))
    num_heads = int(os.environ.get('NUM_HEADS', 6))
    mlp_mult = int(os.environ.get('MLP_MULT', 4))
    n_mask_spans = int(os.environ.get('N_MASK_SPANS', 4))
    span_len_mean = int(os.environ.get('SPAN_LEN_MEAN', 32))
    ema_decay = float(os.environ.get('EMA_DECAY', 0.996))
    decoder_loss_weight = float(os.environ.get('DECODER_LOSS_WEIGHT', 1.0))
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


# Bidirectional transformer block (encoders see whole sequence — JEPA reasons about
# representations, not next-token, so causal masking would hurt context aggregation).
class Block(nn.Module):
    def __init__(self, dim, n_heads, mlp_mult):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)
        self.head_dim = dim // n_heads
        self.n_heads = n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_mult * dim, bias=False), nn.GELU(),
                                  nn.Linear(mlp_mult * dim, dim, bias=False))

    def forward(self, x):
        h = self.norm1(x)
        B, S, D = h.shape
        qkv = self.qkv(h).view(B, S, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        x = x + self.proj(out)
        x = x + self.mlp(self.norm2(x))
        return x


class TextEncoder(nn.Module):
    def __init__(self, h=H):
        super().__init__()
        self.tok_emb = nn.Embedding(h.vocab_size, h.model_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, h.seq_len, h.model_dim))
        self.blocks = nn.ModuleList([Block(h.model_dim, h.num_heads, h.mlp_mult) for _ in range(h.num_layers)])
        self.norm = nn.RMSNorm(h.model_dim)
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb, std=0.02)

    def forward(self, x):
        B, S = x.shape
        h = self.tok_emb(x) + self.pos_emb[:, :S]
        for blk in self.blocks:
            h = blk(h)
        return self.norm(h)


class Predictor(nn.Module):
    """Predicts target-encoder embeddings at masked positions, given context embeddings."""

    def __init__(self, h=H):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, h.model_dim))
        self.pos_emb = nn.Parameter(torch.zeros(1, h.seq_len, h.model_dim))
        self.blocks = nn.ModuleList([Block(h.model_dim, h.num_heads, h.mlp_mult) for _ in range(h.pred_layers)])
        self.norm = nn.RMSNorm(h.model_dim)
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.pos_emb, std=0.02)

    def forward(self, ctx_emb, mask_pos):
        B, S, D = ctx_emb.shape
        x = torch.where(mask_pos.unsqueeze(-1), self.mask_token.expand(B, S, D), ctx_emb)
        x = x + self.pos_emb[:, :S]
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class TJEPA(nn.Module):
    """T-JEPA: context encoder + target encoder (EMA) + predictor + tiny decoder head.

    The decoder is a single Linear from embedding -> vocab. It's trained
    jointly with masked-CE so we can compute val_bpb. Strictly a hybrid, not
    pure-JEPA — see notes.md for the reasoning.
    """

    def __init__(self, h=H):
        super().__init__()
        self.context_encoder = TextEncoder(h)
        self.target_encoder = TextEncoder(h)
        self.predictor = Predictor(h)
        self.decoder = nn.Linear(h.model_dim, h.vocab_size, bias=False)
        self._init_target_from_context()
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        nn.init.zeros_(self.decoder.weight)

    @torch.no_grad()
    def _init_target_from_context(self):
        for p_t, p_c in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            p_t.data.copy_(p_c.data)

    @torch.no_grad()
    def update_target(self, decay):
        for p_t, p_c in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            p_t.data.mul_(decay).add_(p_c.data, alpha=1 - decay)


def sample_mask(B, S, n_spans, span_len_mean, device):
    """Per-example, sample n_spans contiguous spans of average length span_len_mean."""
    mask = torch.zeros(B, S, dtype=torch.bool, device=device)
    for b in range(B):
        for _ in range(n_spans):
            sl = max(2, int(np.random.exponential(span_len_mean)))
            start = np.random.randint(0, max(1, S - sl))
            mask[b, start:start + sl] = True
    return mask


def jepa_step(model, x, h):
    """One JEPA training step.

    Forward:
      mask = sample_mask(...)
      ctx_emb_full = context_encoder(x)              # encode full seq
      ctx_emb_in = ctx_emb_full * (~mask)            # zero masked positions
      with no_grad: tgt_emb = target_encoder(x)
      pred = predictor(ctx_emb_in, mask)
    Loss:
      jepa_loss = ‖pred - tgt_emb‖^2 [mask].mean()    # L2 in embedding space
      decoder_loss = CE(decoder(ctx_emb_full), x).mean()
      loss = jepa_loss + decoder_loss_weight * decoder_loss
    """
    B, S = x.shape
    mask = sample_mask(B, S, h.n_mask_spans, h.span_len_mean, x.device)
    ctx_emb_full = model.context_encoder(x)
    ctx_emb_in = torch.where(mask.unsqueeze(-1), torch.zeros_like(ctx_emb_full), ctx_emb_full)
    with torch.no_grad():
        tgt_emb = model.target_encoder(x)
    pred = model.predictor(ctx_emb_in, mask)
    diff = pred - tgt_emb
    jepa_loss = (diff.pow(2).sum(dim=-1) * mask.float()).sum() / mask.sum().clamp_min(1)
    logits = model.decoder(ctx_emb_full)
    decoder_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), x.reshape(-1))
    return jepa_loss, decoder_loss


@torch.no_grad()
def val_bpb_via_decoder(model, val_tokens, byte_lut_tuple, h):
    """Compute BPB via context_encoder + decoder.

    Note: this uses BIDIRECTIONAL context (not AR), so it's not strictly a
    valid log-likelihood — the model peeks at right-context. But the same
    score conventions apply; treat it as a reasonable proxy.
    """
    base_lut, lead_lut, bound_lut = byte_lut_tuple
    base_lut = base_lut.to(val_tokens.device)
    lead_lut = lead_lut.to(val_tokens.device)
    bound_lut = bound_lut.to(val_tokens.device)
    model.eval()
    S = h.seq_len
    n_seqs = (val_tokens.numel() - 1) // S
    n_seqs = min(n_seqs, 256)
    if n_seqs == 0:
        return float('inf'), float('inf')
    starts = torch.arange(n_seqs) * S
    batch = torch.stack([val_tokens[s:s + S] for s in starts]).to(val_tokens.device)
    total_nll = 0.0
    total_bytes = 0.0
    n_pos = 0
    for i in range(0, n_seqs, 8):
        b = batch[i:i + 8]
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            ctx = model.context_encoder(b)
            logits = model.decoder(ctx)
        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), b.reshape(-1), reduction='none').reshape(b.shape)
        prev = torch.cat([b[:, :1] * 0, b[:, :-1]], dim=1)
        bpos = base_lut[b].double() + (lead_lut[b] & ~bound_lut[prev]).double()
        total_nll += nll.double().sum().item()
        total_bytes += bpos.sum().item()
        n_pos += nll.numel()
    val_loss = total_nll / n_pos
    val_bpb = total_nll / total_bytes / math.log(2)
    return val_loss, val_bpb


def main():
    distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    if distributed:
        dist.init_process_group(backend='nccl')
    device = torch.device('cuda', H.local_rank)
    torch.cuda.set_device(device)
    torch.manual_seed(H.seed + H.rank)

    model = TJEPA().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f'tjepa trainable params: {n_params}')

    train_tokens = load_train_tokens().to(device)
    val_tokens = load_val_tokens().to(device)
    byte_lut = load_byte_lut()
    log(f'train_tokens: {train_tokens.numel()}  val_tokens: {val_tokens.numel()}')

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=H.lr, weight_decay=H.wd, betas=(0.9, 0.95))

    n_train = (train_tokens.numel() - 1) // H.seq_len
    rng = np.random.default_rng(H.seed + H.rank)

    def next_batch():
        idx = rng.integers(0, n_train, size=H.batch_seqs)
        return torch.stack([train_tokens[i * H.seq_len: (i + 1) * H.seq_len] for i in idx])

    log('Starting T-JEPA training...')
    t0 = time.perf_counter()
    for step in range(1, H.iterations + 1):
        x = next_batch()
        lr_scale = min(1.0, step / max(H.warmup_steps, 1))
        for pg in opt.param_groups:
            pg['lr'] = H.lr * lr_scale
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            jepa_l, dec_l = jepa_step(model, x, H)
            loss = jepa_l + H.decoder_loss_weight * dec_l
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        model.update_target(H.ema_decay)
        if step <= 5 or step % H.log_every == 0:
            elapsed = time.perf_counter() - t0
            log(f'{step}/{H.iterations} jepa_loss={jepa_l.item():.4f} dec_loss={dec_l.item():.4f} elapsed={elapsed:.1f}s')

    log('Training done. Computing val BPB via the auxiliary decoder...')
    if H.is_main:
        val_loss, val_bpb = val_bpb_via_decoder(model, val_tokens, byte_lut, H)
        log(f'final tjepa decoder val_loss={val_loss:.4f}  val_bpb={val_bpb:.4f}')
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
