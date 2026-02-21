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
# Data-dependent RoPE
# ---------------------------------------------------------------------------

class DDRoPE(nn.Module):
    """Data-dependent RoPE on the second half of the head dim.

    First half gets standard fixed-frequency RoPE (positional anchoring).
    Second half gets DD-RoPE: cumsum of learned projection deltas from the
    input, providing SSM-equivalent state tracking through cumulative phase
    shifts.  No fixed RoPE on the DD half.

    Args:
        d_input:   Input dim (d_model).
        n_heads:   Number of attention heads.
        head_dim:  Dim per head.  Must be divisible by 4.
    """

    def __init__(self, d_input, n_heads, head_dim):
        super().__init__()
        assert head_dim % 4 == 0, f"head_dim must be divisible by 4 for dd-rope, got {head_dim}"
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.dd_pairs = head_dim // 4
        self.dd_dim = self.dd_pairs * 2                # head_dim // 2

        self.dd_proj = nn.Linear(d_input, n_heads * self.dd_pairs, bias=True)
        nn.init.zeros_(self.dd_proj.weight)
        nn.init.zeros_(self.dd_proj.bias)

    def compute_angles(self, x):
        """x: (B, T, D) -> (B, T, n_heads, dd_pairs) cumulative angles."""
        B, T, _ = x.shape
        deltas = self.dd_proj(x)                          # (B, T, n_heads * dd_pairs)
        deltas = deltas.view(B, T, self.n_heads, self.dd_pairs)
        return deltas.cumsum(dim=1)

    def apply_hybrid(self, qk, dd_angles, fixed_cos, fixed_sin):
        """Apply hybrid RoPE: fixed on first half, DD on second half.

        Args:
            qk:         (..., T, n_heads, head_dim)
            dd_angles:  (B, T, n_heads, dd_pairs)
            fixed_cos:  (1, T, 1, head_dim//2)
            fixed_sin:  (1, T, 1, head_dim//2)

        Returns:
            Rotated tensor, same shape as qk.
        """
        fd = self.dd_dim                       # fixed half = dd_dim = head_dim // 2
        fp = fd // 2                           # fixed rotation pairs

        qk_fixed = qk[..., :fd]
        qk_dd = qk[..., fd:]

        qk_fixed = _apply_rotary(qk_fixed, fixed_cos[..., :fp], fixed_sin[..., :fp])
        qk_dd = _apply_rotary(qk_dd, dd_angles.cos(), dd_angles.sin())

        return torch.cat([qk_fixed, qk_dd], dim=-1)


# ---------------------------------------------------------------------------
# CausalLerp
# ---------------------------------------------------------------------------

class CausalLerp(nn.Module):
    """Content-gated causal lerp across all dims.

    x_mixed[t] = (1 - gate) * x[t] + gate * x[t-1]

    Gate is content-dependent.
    Initialized near-identity (bias=-2.0 -> sigmoid ~ 0.12).
    """

    def __init__(self, inner_dim, init_bias=-2.0):
        super().__init__()
        self.inner_dim = inner_dim
        self.gate_proj = nn.Linear(inner_dim, inner_dim, bias=True)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, init_bias)

    def forward(self, x):
        """x: (B, T, D) -> same shape."""
        x_prev = F.pad(x[:, :-1], (0, 0, 1, 0))
        gate = torch.sigmoid(self.gate_proj(x))
        return (1 - gate) * x + gate * x_prev


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
                 use_dd_rope=False):
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
            self.dd_rope = DDRoPE(d_input=d_model, n_heads=n_head, head_dim=self.d_head)

        self.attn_norm = nn.RMSNorm(d_model)
        self.w_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_norm = nn.RMSNorm(self.d_head)
        self.k_norm = nn.RMSNorm(self.d_head)
        self.q_post_bias = nn.Parameter(torch.ones(self.d_head))
        self.k_post_bias = nn.Parameter(torch.ones(self.d_head))
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)

        ffn_dim = ((d_model * 8 // 3 + 7) // 8) * 8
        self.mlp_norm = nn.RMSNorm(d_model)

        if use_feat_attn:
            self.feat_gate = nn.Linear(d_model, ffn_dim, bias=False)
            self.feat_up = nn.Linear(d_model, ffn_dim, bias=False)
            self.feat_down = nn.Linear(ffn_dim, d_model, bias=False)
            self.feat_qk_norm = nn.RMSNorm(ffn_dim // n_head)
        else:
            self.gate_proj = nn.Linear(d_model, ffn_dim, bias=False)
            self.up_proj = nn.Linear(d_model, ffn_dim, bias=False)
            self.down_proj = nn.Linear(ffn_dim, d_model, bias=False)
        self.mlp_drop = nn.Dropout(dropout)

    def forward(self, x, rope_cos, rope_sin):
        B, T, D = x.shape
        nh, dh = self.n_head, self.d_head

        # Optional CausalLerp
        if self.use_lerp:
            x = self.causal_lerp(x)

        # Sequence attention
        h = self.attn_norm(x)
        qkv = self.w_qkv(h).reshape(B, T, 3, nh, dh)
        q, k, v = qkv.unbind(dim=2)               # (B, T, nh, dh)
        q = self.q_norm(q) * self.q_post_bias
        k = self.k_norm(k) * self.k_post_bias
        if self.use_dd_rope:
            dd_angles = self.dd_rope.compute_angles(x)
            q = self.dd_rope.apply_hybrid(q, dd_angles, rope_cos, rope_sin)
            k = self.dd_rope.apply_hybrid(k, dd_angles, rope_cos, rope_sin)
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
            gate = self.feat_gate(h)                       # (B, T, ffn_dim)
            val = self.feat_up(h)                          # (B, T, ffn_dim)
            ffn_dim = gate.shape[-1]
            fdh = ffn_dim // nh
            gate = gate.view(B * T, 1, nh, fdh)
            gate = self.feat_qk_norm(gate)
            val = val.view(B * T, 1, nh, fdh)
            feat_out = F.scaled_dot_product_attention(gate, gate, val, is_causal=False)
            feat_out = feat_out.view(B, T, ffn_dim)
            x = x + self.mlp_drop(self.feat_down(feat_out))
        else:
            x = x + self.mlp_drop(self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h)))
        return x


# ---------------------------------------------------------------------------
# Output head: per-feature classifiers (non-UCI mode)
# ---------------------------------------------------------------------------

class FeatureOutputHead(nn.Module):
    """Per-feature linear classifiers.

    Input: (B, T, total_rows, DD).  Output: dict of {feature_name: (B, T, n_classes)}.
    """

    def __init__(self, feature_sizes, rows_per_feature, desc_dim):
        super().__init__()
        self.feature_names = list(feature_sizes.keys())
        self.rows_per_feature = rows_per_feature
        self.classifiers = nn.ModuleDict()
        for name, n_classes in feature_sizes.items():
            self.classifiers[name] = nn.Linear(rows_per_feature * desc_dim, n_classes)

    def forward(self, x):
        """x: (B, T, total_rows, DD) -> dict of {name: (B, T, n_classes)}"""
        rpf = self.rows_per_feature
        outputs = {}
        for idx, name in enumerate(self.feature_names):
            start = idx * rpf
            chunk = x[:, :, start:start + rpf, :]
            flat = chunk.reshape(x.shape[0], x.shape[1], -1)
            outputs[name] = self.classifiers[name](flat)
        return outputs


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

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_head, dropout, use_lerp, use_feat_attn, use_dd_rope)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.RMSNorm(d_model)

        if self.embed_mode == "plain":
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            self.head.weight = self.embed.weight          # weight tying
        elif self.uci_mode:
            self.head = nn.Linear(d_model, uci_vocab_size, bias=False)
        else:
            self.output_head = FeatureOutputHead(feature_sizes, rows_per_feature, w_dim)

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

    def forward(self, feature_ids):
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

        x = self.ln_f(x)

        if self.embed_mode == "composite" and not self.uci_mode:
            x = x.view(B, T, self._total_rows, self._w_dim)
            return self.output_head(x)                     # dict of {name: (B, T, n_classes)}

        return self.head(x)                                # (B, T, vocab_size)
