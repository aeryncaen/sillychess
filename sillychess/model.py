"""Transformer for chess move prediction.

Two embedding modes:
- plain: token lookup from UCI vocabulary (--uci-plain)
- composite: CompositeSANEmbedding per-feature descriptors, flattened to 1D

Both modes share the same transformer backbone (TransformerBlock stack).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

class CompositeSANEmbedding(nn.Module):
    """Single embedding table with per-feature offsets, flattened to 1D.

    Each feature looks up ``rows_per_feature * w_dim`` values, then all
    features are concatenated and flattened to produce a 1D token embedding
    of size ``n_features * rows_per_feature * w_dim``.
    """

    def __init__(self, feature_sizes, rows_per_feature, w_dim):
        super().__init__()
        self.rows_per_feature = rows_per_feature
        self.w_dim = w_dim
        self.feature_names = list(feature_sizes.keys())
        self.n_features = len(self.feature_names)

        sizes = [feature_sizes[name] for name in self.feature_names]
        offsets = torch.zeros(len(sizes), dtype=torch.long)
        for i in range(1, len(sizes)):
            offsets[i] = offsets[i - 1] + sizes[i - 1]
        self.register_buffer('offsets', offsets)
        self.embed = nn.Embedding(sum(sizes), rows_per_feature * w_dim)

    def forward(self, feature_ids):
        if isinstance(feature_ids, dict):
            ids = feature_ids["features"]                      # (B, T, NF) pre-stacked
        else:
            ids = feature_ids
        ids = ids + self.offsets                                # (B, T, NF)
        e = self.embed(ids)                                    # (B, T, NF, rpf*W)
        B, T = e.shape[:2]
        return e.view(B, T, -1)                                # (B, T, NF*rpf*W)


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

def _apply_rotary(x, cos, sin):
    """Apply rotary embeddings. x: (..., D), cos/sin: broadcastable to x."""
    d = x.shape[-1]
    x1 = x[..., :d // 2]
    x2 = x[..., d // 2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


# ---------------------------------------------------------------------------
# CausalLerp
# ---------------------------------------------------------------------------

class CausalLerp(nn.Module):
    """Trapezoidal one-step accumulation on first 50% of dims.

    Mamba-3 §3.1, Proposition 1 (generalized trapezoidal discretization):

        out[t] = α[t] * B[t-1] * x[t-1]  +  (1 - α[t]) * B[t] * x[t]

    where α[t] = sigmoid(proj(x[t])) is the data-dependent trapezoidal
    parameter (analogous to 1-λ in the paper: α=1 is pure Euler on the
    previous step, α=0 is pure current step).

    B[t] is a content-dependent projection of x[t] into the lerp subspace.

    Remaining dims pass through unchanged.
    Initialized near-identity (α ≈ 0.12 so output ≈ current token).
    """

    def __init__(self, d_model, init_bias=-2.0):
        super().__init__()
        self.lerp_dim = d_model // 2
        # trapezoidal mixing parameter α (data-dependent)
        self.alpha_proj = nn.Linear(d_model, self.lerp_dim, bias=True)
        nn.init.zeros_(self.alpha_proj.weight)
        nn.init.constant_(self.alpha_proj.bias, init_bias)
        # content-dependent input projection B
        self.b_proj = nn.Linear(d_model, self.lerp_dim, bias=False)

    def forward(self, x):
        """x: (B, T, D) -> same shape."""
        # content-dependent projections
        alpha = torch.sigmoid(self.alpha_proj(x))    # (B, T, lerp_dim)
        bx = self.b_proj(x)                          # (B, T, lerp_dim)

        # shift: bx_prev[t] = bx[t-1], zero at t=0
        bx_prev = F.pad(bx[:, :-1], (0, 0, 1, 0))   # (B, T, lerp_dim)

        # trapezoidal one-step: convex combination of both endpoints
        x_lerp = alpha * bx_prev + (1 - alpha) * bx

        return torch.cat([x_lerp, x[..., self.lerp_dim:]], dim=-1)


# ---------------------------------------------------------------------------
# Data-dependent RoPE on Q/K
# ---------------------------------------------------------------------------

class DDRoPE(nn.Module):
    """Data-dependent rotary embeddings on a portion of Q/K head dims.

    Mamba-3 §3.2, Proposition 3: complex-valued SSM is equivalent to
    data-dependent RoPE applied to B,C (= K,Q in the SSD/attention dual).

    Projects input to per-head, per-timestep angle deltas, cumsums for
    cumulative phase.  Returns cos/sin for application to the first
    dd_pairs pairs of each head in Q and K.

    Args:
        d_model:          Full model dim (input to angle projection).
        n_head:           Number of attention heads.
        dd_pairs_per_head: Number of rotation pairs per head.
    """

    def __init__(self, d_model, n_head, dd_pairs_per_head):
        super().__init__()
        self.n_head = n_head
        self.dd_pairs_per_head = dd_pairs_per_head
        self.dd_dim_per_head = dd_pairs_per_head * 2
        total_pairs = n_head * dd_pairs_per_head
        self.dd_proj = nn.Linear(d_model, total_pairs, bias=True)
        nn.init.zeros_(self.dd_proj.weight)
        nn.init.zeros_(self.dd_proj.bias)

    def forward(self, x):
        """x: (B, T, D) -> (dd_cos, dd_sin) each (B, T, n_head, dd_pairs)."""
        B, T, _ = x.shape
        deltas = self.dd_proj(x)                                    # (B, T, total_pairs)
        deltas = deltas.view(B, T, self.n_head, self.dd_pairs_per_head)
        angles = deltas.cumsum(dim=1)                               # cumulative phase
        return angles.cos(), angles.sin()


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Pre-norm transformer block with optional CausalLerp, feature attention,
    and data-dependent RoPE.

    Base: causal SDPA + SwiGLU MLP.
    --lerp: CausalLerp before attention.
    --feat-attn: replaces SwiGLU gate with softmax attention over n_head features.
    --dd-rope: data-dependent RoPE on half of dims.
    """

    def __init__(self, d_model, n_head, dropout=0.1, use_lerp=False, use_feat_attn=False,
                 use_dd_rope=False, n_features=None):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.use_lerp = use_lerp
        self.use_feat_attn = use_feat_attn
        self.use_dd_rope = use_dd_rope

        if use_lerp:
            self.causal_lerp = CausalLerp(d_model)

        if use_dd_rope:
            # ~25% of each head's dims get data-dependent rotation
            dd_pairs = max(self.d_head // 8, 1)
            self.dd_rope = DDRoPE(d_model, n_head, dd_pairs)
            self.dd_pairs = dd_pairs

        self.attn_norm = nn.RMSNorm(d_model)
        self.w_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_norm = nn.RMSNorm(self.d_head)
        self.k_norm = nn.RMSNorm(self.d_head)
        self.q_post_bias = nn.Parameter(torch.ones(n_head, self.d_head))
        self.k_post_bias = nn.Parameter(torch.ones(n_head, self.d_head))
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)

        ffn_dim = ((d_model * 8 // 3 + 7) // 8) * 8
        self.mlp_norm = nn.RMSNorm(d_model)

        if use_feat_attn:
            assert n_features is not None, "n_features required for feat_attn"
            self.n_features = n_features
            self.feat_head_dim = ffn_dim // n_features
            self.feat_q = nn.Linear(d_model, ffn_dim, bias=False)
            self.feat_k = nn.Linear(d_model, ffn_dim, bias=False)
            self.feat_v = nn.Linear(d_model, ffn_dim, bias=False)
            self.feat_down = nn.Linear(ffn_dim, d_model, bias=False)
            self.feat_qk_norm = nn.RMSNorm(self.feat_head_dim)
        else:
            self.gate_proj = nn.Linear(d_model, ffn_dim, bias=False)
            self.up_proj = nn.Linear(d_model, ffn_dim, bias=False)
            self.down_proj = nn.Linear(ffn_dim, d_model, bias=False)
        self.mlp_drop = nn.Dropout(dropout)

    def forward(self, x, rope_cos, rope_sin):
        B, T, D = x.shape
        nh, dh = self.n_head, self.d_head

        # Pre-attention: trapezoidal one-step (CausalLerp)
        if self.use_lerp:
            x = self.causal_lerp(x)

        # Attention with RoPE (fixed + optional data-dependent on Q/K)
        h = self.attn_norm(x)
        qkv = self.w_qkv(h).reshape(B, T, 3, nh, dh)
        q, k, v = qkv.unbind(dim=2)               # (B, T, nh, dh)
        q = self.q_norm(q) * self.q_post_bias
        k = self.k_norm(k) * self.k_post_bias

        if self.use_dd_rope:
            dd = self.dd_pairs * 2  # dims covered by dd-rope per head
            dd_cos, dd_sin = self.dd_rope(x)       # (B, T, nh, dd_pairs)
            # DD-RoPE on first dd dims of each head
            q_dd = _apply_rotary(q[..., :dd], dd_cos, dd_sin)
            k_dd = _apply_rotary(k[..., :dd], dd_cos, dd_sin)
            # Fixed RoPE on remaining dims (slice freqs to match)
            fix_pairs = (dh - dd) // 2
            q_fix = _apply_rotary(q[..., dd:], rope_cos[..., :fix_pairs], rope_sin[..., :fix_pairs])
            k_fix = _apply_rotary(k[..., dd:], rope_cos[..., :fix_pairs], rope_sin[..., :fix_pairs])
            q = torch.cat([q_dd, q_fix], dim=-1)
            k = torch.cat([k_dd, k_fix], dim=-1)
        else:
            q = _apply_rotary(q, rope_cos, rope_sin)
            k = _apply_rotary(k, rope_cos, rope_sin)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            is_causal=True,
        )
        x = x + self.attn_drop(self.w_o(y.transpose(1, 2).reshape(B, T, D)))

        # MLP
        h = self.mlp_norm(x)
        if self.use_feat_attn:
            nf, fhd = self.n_features, self.feat_head_dim
            q = self.feat_qk_norm(self.feat_q(h).view(B * T, nf, fhd))  # (B*T, 8, 128)
            k = self.feat_qk_norm(self.feat_k(h).view(B * T, nf, fhd))
            v = self.feat_v(h).view(B * T, nf, fhd)
            # attend over nf=8 features, head_dim=ffn_dim//nf=128
            feat_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
            feat_out = feat_out.view(B, T, nf * fhd)
            x = x + self.mlp_drop(self.feat_down(feat_out))
        else:
            x = x + self.mlp_drop(self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h)))
        return x


# ---------------------------------------------------------------------------
# Output head: composite vocab (non-UCI mode)
# ---------------------------------------------------------------------------

class CompositeOutputHead(nn.Module):
    """Plain linear projection to composite vocab.

    Direct (V, d_model) weight matrix — no factored embedding indirection.
    """

    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        """x: (B, T, d_model) -> (B, T, V)"""
        return self.proj(x)

    def chunked_loss(self, x, targets, mask, chunk_tokens=4096):
        """Chunked cross-entropy loss — never materializes full (B*T, V) logits.

        Args:
            x: (B, T, d_model) hidden states from encoder.
            targets: (B, T) composite vocab IDs (may contain -1 for invalid).
            mask: (B, T) bool — True for valid positions.
            chunk_tokens: number of tokens per chunk.

        Returns:
            (loss, n_correct, n_valid) — scalar loss, int counts.
        """
        w = self.proj.weight                                     # (V, d_model)
        B, T, D = x.shape
        x_flat = x.reshape(B * T, D)
        tgt_flat = targets.reshape(B * T).clamp(min=0)
        mask_flat = mask.reshape(B * T)

        total_loss = x.new_zeros(())
        total_correct = 0
        n_valid = int(mask_flat.sum().item())

        for start in range(0, B * T, chunk_tokens):
            end = min(start + chunk_tokens, B * T)
            x_c = x_flat[start:end]
            t_c = tgt_flat[start:end]
            m_c = mask_flat[start:end]
            if not m_c.any():
                continue

            if torch.is_grad_enabled():
                chunk_loss, chunk_correct = grad_checkpoint(
                    self._chunk_ce, x_c, w, t_c, m_c, use_reentrant=False,
                )
            else:
                chunk_loss, chunk_correct = self._chunk_ce(x_c, w, t_c, m_c)
            total_loss = total_loss + chunk_loss
            total_correct += int(chunk_correct.item())

        loss = total_loss / max(n_valid, 1)
        return loss, total_correct, n_valid

    @staticmethod
    def _chunk_ce(x_c, w, t_c, m_c):
        """Compute CE loss for one chunk. Checkpointed during training."""
        logits = F.linear(x_c, w)                              # (chunk, V)
        raw = F.cross_entropy(logits, t_c, reduction="none")   # (chunk,)
        loss = (raw * m_c.float()).sum()
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            correct = ((preds == t_c) & m_c).sum()
        return loss, correct


# ---------------------------------------------------------------------------
# Unified transformer model
# ---------------------------------------------------------------------------

class TransformerModel(nn.Module):
    """Unified transformer for chess move prediction.

    Two embedding modes:
    - "plain": nn.Embedding token lookup, weight-tied output head.
    - "composite": CompositeSANEmbedding (per-feature descriptors, flattened
      to a 1D vector of size n_features * rows_per_feature * w_dim).

    Both share the same TransformerBlock backbone.
    """

    def __init__(
        self,
        # Common
        n_head=1,
        n_layer=6,
        dropout=0.1,
        use_lerp=False,
        use_feat_attn=False,
        use_dd_rope=False,
        # Plain mode
        d_model=None,
        vocab_size=None,
        # Composite mode
        feature_sizes=None,
        w_dim=48,
        rows_per_feature=1,
        uci_vocab_size=None,
        composite_vocab=None,
    ):
        super().__init__()

        if feature_sizes is not None:
            self.embed_mode = "composite"
            self.embed = CompositeSANEmbedding(feature_sizes, rows_per_feature, w_dim)
            n_features = len(feature_sizes)
            d_model = n_features * rows_per_feature * w_dim
            self.uci_mode = uci_vocab_size is not None
            self._w_dim = w_dim
            self._rows_per_feature = rows_per_feature
            self._total_rows = n_features * rows_per_feature
        else:
            self.embed_mode = "plain"
            if d_model is None or vocab_size is None:
                raise ValueError("d_model and vocab_size required for plain mode")
            self.embed = nn.Embedding(vocab_size, d_model)
            self.uci_mode = True

        self.d_model = d_model
        self.drop = nn.Dropout(dropout)

        d_head = d_model // n_head
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        n_feat = len(feature_sizes) if feature_sizes is not None else None
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_head, dropout, use_lerp, use_feat_attn, use_dd_rope,
                             n_features=n_feat)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.RMSNorm(d_model)

        if self.embed_mode == "plain":
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            self.head.weight = self.embed.weight          # weight tying
        elif self.uci_mode:
            self.head = nn.Linear(d_model, uci_vocab_size, bias=False)
        else:
            if composite_vocab is None:
                raise ValueError("composite_vocab required for non-UCI composite mode")
            self.output_head = CompositeOutputHead(len(composite_vocab), d_model)

        self._init_weights(n_layer)

    def _init_weights(self, n_layer):
        if self.embed_mode == "plain":
            nn.init.normal_(self.embed.weight, std=self.d_model ** -0.5)
        residual_scale = (2 * n_layer) ** -0.5
        for layer in self.layers:
            nn.init.normal_(layer.w_o.weight, std=self.d_model ** -0.5 * residual_scale)
            down = getattr(layer, 'down_proj', None) or getattr(layer, 'feat_down', None)
            if down is not None:
                nn.init.normal_(down.weight, std=self.d_model ** -0.5 * residual_scale)

    def encode(self, feature_ids):
        """Run backbone (embed → drop → RoPE → layers → ln_f), return hidden states.

        Args:
            feature_ids: dict with "features" (B, T, 8) or "uci_move" (B, T).

        Returns:
            (B, T, d_model) hidden states.
        """
        if self.embed_mode == "composite":
            x = self.embed(feature_ids)                    # (B, T, d_model)
        else:
            if isinstance(feature_ids, dict):
                x_ids = feature_ids["uci_move"]
            else:
                x_ids = feature_ids
            x = self.embed(x_ids.clamp(min=0))

        B, T = x.shape[:2]
        x = self.drop(x)

        pos = torch.arange(T, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(pos, self.inv_freq)
        rope_cos = freqs.cos()[None, :, None, :]           # (1, T, 1, d_head//2)
        rope_sin = freqs.sin()[None, :, None, :]

        for layer in self.layers:
            x = layer(x, rope_cos, rope_sin)

        return self.ln_f(x)

    def forward(self, feature_ids):
        x = self.encode(feature_ids)

        if self.embed_mode == "composite" and not self.uci_mode:
            return self.output_head(x)                     # (B, T, composite_vocab_size)

        return self.head(x)                                # (B, T, vocab_size)
