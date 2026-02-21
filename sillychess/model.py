"""Two-stage transformer for chess move prediction with 2D token embeddings.

Tokens are 2D: (B, T, n_features, desc_dim).  e.g. 9 features x 48 desc.

Each layer:
    1. 4x up-project desc_dim -> inner_dim  (expand working space)
    2. Sequence attention: SDPA over T, n_features as heads, inner_dim per head.
       Causal mask. Each feature independently attends across time.
    3. Feature-attention MLP (per FEATURE_ATTENTION.md):
       gate_proj -> shared Q=K, up_proj -> V, silu² attention over n_features,
       then down_proj. Cross-feature mixing inside the MLP.
    4. 4x down-project inner_dim -> desc_dim  (compress back)

Output head: per-feature linear classifiers on the desc_dim descriptors.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

class CompositeSANEmbedding(nn.Module):
    def __init__(self, feature_sizes, rows_per_feature, w_dim):
        super().__init__()
        self.rows_per_feature = rows_per_feature
        self.w_dim = w_dim
        self.feature_names = list(feature_sizes.keys())
        self.n_features = len(self.feature_names)
        self.total_rows = self.n_features * rows_per_feature

        # Single embedding table: all features share one table with offsets
        sizes = [feature_sizes[name] for name in self.feature_names]
        offsets = torch.zeros(len(sizes), dtype=torch.long)
        for i in range(1, len(sizes)):
            offsets[i] = offsets[i - 1] + sizes[i - 1]
        self.register_buffer('offsets', offsets)
        self.embed = nn.Embedding(sum(sizes), rows_per_feature * w_dim)

    def forward(self, feature_ids):
        # Stack features: (B, T, NF), add offsets, single lookup, reshape
        ids = torch.stack([feature_ids[n] for n in self.feature_names], dim=-1)
        ids = ids + self.offsets                               # (B, T, NF)
        e = self.embed(ids)                                    # (B, T, NF, rpf*W)
        B, T, NF, _ = e.shape
        return e.view(B, T, NF * self.rows_per_feature, self.w_dim)


# ---------------------------------------------------------------------------
# Attention helpers
# ---------------------------------------------------------------------------

def silu2_attention(q, k, v, is_causal=False):
    """SiLU-squared attention: silu(logits)^2, unnormalized.

    Args:
        q, k, v: (B, H, T, D) standard SDPA layout.
        is_causal: If True, apply causal mask.

    Returns:
        (B, H, T, D) attention output.
    """
    scale = 1.0 / math.sqrt(q.shape[-1])
    logits = (q @ k.transpose(-2, -1)) * scale
    weights = F.silu(logits) ** 2
    if is_causal:
        T = logits.shape[-1]
        causal_mask = torch.tril(torch.ones(T, T, device=logits.device, dtype=logits.dtype))
        weights = weights * causal_mask
    return weights @ v


# ---------------------------------------------------------------------------
# 2D projection helper
# ---------------------------------------------------------------------------

def proj_per_feature(x, w, bias=None):
    """Per-feature projection: x (..., NF, D_in) @ w (NF, D_in, D_out) -> (..., NF, D_out).
    Each feature has its own D_in -> D_out projection. No cross-feature mixing."""
    y = torch.einsum('...nd,nde->...ne', x, w)
    if bias is not None:
        y = y + bias
    return y


# ---------------------------------------------------------------------------
# Layer block
# ---------------------------------------------------------------------------

class CausalLerp(nn.Module):
    """Content-gated causal lerp on the last half of descriptor dims.

    x_mixed[t] = (1 - gate) * x[t] + gate * x[t-1]

    Applied only to the last half of the descriptor dimension; the first
    half passes through unchanged.  Gate is content-dependent per-feature.
    Initialized near-identity (bias=-2.0 → sigmoid ≈ 0.12).
    """

    def __init__(self, inner_dim, init_bias=-2.0):
        super().__init__()
        self.half_dim = inner_dim // 2
        self.gate_proj = nn.Linear(inner_dim, self.half_dim, bias=True)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, init_bias)

    def forward(self, x):
        """x: (B, T, ..., D) -> same shape. Works for 3D (B,T,D) or 4D (B,T,NF,D)."""
        hd = self.half_dim
        x_static = x[..., :hd]
        x_cur = x[..., hd:]

        # Shift x_cur one step back along dim 1: pad t=0 with zeros
        # Build pad spec: 0 on last dim, then 0 on any middle dims, then (1,0) on time
        n_trailing = x.ndim - 2  # dims after the time dim
        pad = (0, 0) * n_trailing + (1, 0)
        x_prev = F.pad(x_cur[:, :-1], pad)

        gate = torch.sigmoid(self.gate_proj(x))
        x_mixed = (1 - gate) * x_cur + gate * x_prev
        return torch.cat([x_static, x_mixed], dim=-1)


class ChessBlock(nn.Module):
    """One transformer layer operating on 2D tokens (B, T, NF, DD).

    Architecture:
        x = causal_lerp(x)                               # SSM-style temporal mix on half DD
        x = x + seq_attn(norm1(x))                      # causal over T, NF heads, RoPE
        x = x + feat_attn_mlp(norm2(x))                 # softmax feat attn replacing SwiGLU gate
    """

    def __init__(self, n_features, desc_dim, dropout=0.1, use_lerp=False, use_feat_attn=False):
        super().__init__()
        NF = n_features
        DD = desc_dim
        self.use_lerp = use_lerp
        self.use_feat_attn = use_feat_attn
        # SwiGLU-style 8/3 expansion on descriptor dim, snapped to multiple of 8
        FFN_DD = ((DD * 8 // 3 + 7) // 8) * 8

        # --- Causal lerp: SSM-style temporal mixing on half of DD ---
        if use_lerp:
            self.causal_lerp = CausalLerp(DD)

        # --- Sequence attention (per-feature QKV on descriptor dim) ---
        init_scale = DD ** -0.5
        self.seq_norm = nn.RMSNorm(DD)
        self.w_q = nn.Parameter(torch.randn(NF, DD, DD) * init_scale)
        self.w_k = nn.Parameter(torch.randn(NF, DD, DD) * init_scale)
        self.w_v = nn.Parameter(torch.randn(NF, DD, DD) * init_scale)
        self.w_o = nn.Parameter(torch.randn(NF, DD, DD) * init_scale)
        self.k_bias = nn.Parameter(torch.zeros(NF, DD))
        self.v_bias = nn.Parameter(torch.zeros(NF, DD))
        # QK norm + post-norm multiplicative bias (Mamba-3 style, init ones)
        self.q_norm = nn.RMSNorm(DD)
        self.k_norm = nn.RMSNorm(DD)
        self.q_post_bias = nn.Parameter(torch.ones(DD))
        self.k_post_bias = nn.Parameter(torch.ones(DD))
        # Post-attention norm before residual add
        self.post_attn_norm = nn.RMSNorm(DD)
        self.attn_drop = nn.Dropout(dropout)

        # --- MLP ---
        init_scale_ffn = FFN_DD ** -0.5
        self.feat_norm = nn.RMSNorm(DD)
        if use_feat_attn:
            # Feature-attention MLP: gate_proj -> Q=K, up_proj -> V,
            # softmax attn over NF, down_proj back
            self.feat_gate = nn.Parameter(torch.randn(NF, DD, FFN_DD) * init_scale)
            self.feat_up = nn.Parameter(torch.randn(NF, DD, FFN_DD) * init_scale)
            self.feat_down = nn.Parameter(torch.randn(NF, FFN_DD, DD) * init_scale_ffn)
            # QK norm on feature attention gate (shared Q=K, so one norm)
            self.feat_qk_norm = nn.RMSNorm(FFN_DD)
            self.feat_qk_bias = nn.Parameter(torch.ones(FFN_DD))
        else:
            # Plain per-feature SwiGLU MLP (no cross-feature mixing)
            self.mlp_gate = nn.Parameter(torch.randn(NF, DD, FFN_DD) * init_scale)
            self.mlp_up = nn.Parameter(torch.randn(NF, DD, FFN_DD) * init_scale)
            self.mlp_down = nn.Parameter(torch.randn(NF, FFN_DD, DD) * init_scale_ffn)
        # Post-MLP norm before residual add
        self.post_mlp_norm = nn.RMSNorm(DD)
        self.feat_drop = nn.Dropout(dropout)

        self.n_features = NF
        self.desc_dim = DD
        self.ffn_desc_dim = FFN_DD

    def forward(self, x, rope_cos=None, rope_sin=None):
        """x: (B, T, NF, DD) -> (B, T, NF, DD)"""
        NF = self.n_features
        DD = self.desc_dim
        FD = self.ffn_desc_dim
        b, t = x.shape[:2]

        # --- Causal lerp: temporal mixing on last half of DD ---
        if self.use_lerp:
            x = self.causal_lerp(x)

        # --- Sequence attention ---
        h = self.seq_norm(x)
        q = proj_per_feature(h, self.w_q)                     # (B, T, NF, DD)
        k = proj_per_feature(h, self.w_k, self.k_bias)
        v = proj_per_feature(h, self.w_v, self.v_bias)

        # QK norm + post-norm bias
        q = self.q_norm(q) * self.q_post_bias
        k = self.k_norm(k) * self.k_post_bias

        # RoPE on descriptors (after QK norm)
        if rope_cos is not None:
            q = _apply_rotary(q, rope_cos, rope_sin)
            k = _apply_rotary(k, rope_cos, rope_sin)

        # SDPA: NF as heads, attend over T, each head has dim DD
        q = q.permute(0, 2, 1, 3)                            # (B, NF, T, DD)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.permute(0, 2, 1, 3).contiguous()               # (B, T, NF, DD)
        y = proj_per_feature(y, self.w_o)
        x = x + self.attn_drop(self.post_attn_norm(y))

        # --- MLP ---
        h = self.feat_norm(x)
        if self.use_feat_attn:
            # Feature-attention MLP
            gate = proj_per_feature(h, self.feat_gate)            # (B, T, NF, FD) — Q=K
            val = proj_per_feature(h, self.feat_up)               # (B, T, NF, FD) — V

            # QK norm on gate (shared Q=K)
            gate = self.feat_qk_norm(gate) * self.feat_qk_bias

            # Softmax attention over NF features at expanded FD: (B*T, 1, NF, FD)
            gate = gate.reshape(b * t, 1, NF, FD)
            val = val.reshape(b * t, 1, NF, FD)
            feat_out = F.scaled_dot_product_attention(gate, gate, val, is_causal=False)
            feat_out = feat_out.view(b, t, NF, FD)
            feat_out = proj_per_feature(feat_out, self.feat_down) # (B, T, NF, DD)
        else:
            # Plain per-feature SwiGLU MLP
            gate = proj_per_feature(h, self.mlp_gate)             # (B, T, NF, FD)
            up = proj_per_feature(h, self.mlp_up)                 # (B, T, NF, FD)
            feat_out = proj_per_feature(F.silu(gate) * up, self.mlp_down)  # (B, T, NF, DD)
        x = x + self.feat_drop(self.post_mlp_norm(feat_out))

        return x


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
# Output head: per-feature classifiers
# ---------------------------------------------------------------------------

class FeatureOutputHead(nn.Module):
    """Per-feature linear classifiers.  Each original feature owns
    ``rows_per_feature`` rows; those rows are concatenated before the classifier.

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
            chunk = x[:, :, start:start + rpf, :]              # (B, T, rpf, DD)
            flat = chunk.reshape(x.shape[0], x.shape[1], -1)   # (B, T, rpf*DD)
            outputs[name] = self.classifiers[name](flat)
        return outputs





# ---------------------------------------------------------------------------
# Plain 1D transformer (--uci-plain mode)
# ---------------------------------------------------------------------------

class PlainBlock(nn.Module):
    """Pre-norm transformer block with optional CausalLerp and feature attention.

    Base: causal SDPA + SwiGLU MLP.
    --lerp: adds CausalLerp before attention (SSM-style temporal mixing).
    --feat-attn: replaces SwiGLU gate with softmax attention over n_head
                 "features" (same mechanism as ChessBlock's feature attention).
    """

    def __init__(self, d_model, n_head, dropout=0.1, use_lerp=False, use_feat_attn=False):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.use_lerp = use_lerp
        self.use_feat_attn = use_feat_attn

        if use_lerp:
            self.causal_lerp = CausalLerp(d_model)

        self.attn_norm = nn.RMSNorm(d_model)
        self.w_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)

        ffn_dim = ((d_model * 8 // 3 + 7) // 8) * 8
        self.mlp_norm = nn.RMSNorm(d_model)

        if use_feat_attn:
            # Feature-attention MLP: gate_proj -> shared Q=K, up_proj -> V,
            # softmax attention over n_head features, then down_proj.
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
            # Feature attention: reshape d_model as (nh, dh), attend over nh
            gate = self.feat_gate(h)                       # (B, T, ffn_dim)
            val = self.feat_up(h)                          # (B, T, ffn_dim)
            ffn_dim = gate.shape[-1]
            fdh = ffn_dim // nh
            # Reshape to (B*T, 1, nh, fdh) for attention over features
            gate = gate.view(B * T, 1, nh, fdh)
            gate = self.feat_qk_norm(gate)
            val = val.view(B * T, 1, nh, fdh)
            feat_out = F.scaled_dot_product_attention(gate, gate, val, is_causal=False)
            feat_out = feat_out.view(B, T, ffn_dim)
            x = x + self.mlp_drop(self.feat_down(feat_out))
        else:
            x = x + self.mlp_drop(self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h)))
        return x


class PlainTransformerModel(nn.Module):
    """Standard 1D autoregressive transformer: UCI token in, UCI token out.

    No 2D feature decomposition, no feature attention — just a baseline
    GPT-style model over the 1968-token UCI vocabulary.
    Embedding weights are tied with the output head.
    """

    def __init__(self, d_model, n_head, n_layer, vocab_size, dropout=0.1,
                 use_lerp=False, use_feat_attn=False):
        super().__init__()
        self.uci_mode = True

        self.embed = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)

        d_head = d_model // n_head
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        self.layers = nn.ModuleList([
            PlainBlock(d_model, n_head, dropout, use_lerp=use_lerp, use_feat_attn=use_feat_attn)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight          # weight tying

        self._init_weights(n_layer)

    def _init_weights(self, n_layer):
        # Embedding: std = 1/sqrt(d_model) so initial logits have std ≈ 1
        d_model = self.embed.embedding_dim
        nn.init.normal_(self.embed.weight, std=d_model ** -0.5)
        # Residual output projections (w_o, down_proj/feat_down): scale by 1/sqrt(2*n_layer)
        # to keep residual stream from growing with depth
        residual_scale = (2 * n_layer) ** -0.5
        for layer in self.layers:
            nn.init.normal_(layer.w_o.weight, std=d_model ** -0.5 * residual_scale)
            down = getattr(layer, 'down_proj', None) or getattr(layer, 'feat_down', None)
            if down is not None:
                nn.init.normal_(down.weight, std=d_model ** -0.5 * residual_scale)

    def forward(self, feature_ids):
        if isinstance(feature_ids, dict):
            x_ids = feature_ids["uci_move"]
        else:
            x_ids = feature_ids
        B, T = x_ids.shape

        x = self.drop(self.embed(x_ids.clamp(min=0)))

        pos = torch.arange(T, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(pos, self.inv_freq)
        rope_cos = freqs.cos()[None, :, None, :]      # (1, T, 1, d_head//2)
        rope_sin = freqs.sin()[None, :, None, :]

        for layer in self.layers:
            x = layer(x, rope_cos, rope_sin)

        return self.head(self.ln_f(x))                 # (B, T, vocab_size)


# ---------------------------------------------------------------------------
# 2D feature-attention model
# ---------------------------------------------------------------------------

class TwoStageTransformerModel(nn.Module):
    def __init__(
        self,
        w_dim=48,
        rows_per_feature=4,
        n_layer=6,
        dropout=0.1,
        feature_sizes=None,
        uci_vocab_size=None,
        use_lerp=False,
        use_feat_attn=False,
        # Legacy compat (ignored)
        h_rows=None, n_head=None, n_intra_layer=None, n_inter_layer=None,
        feature_decoder_layers=None, ml_decoder_queries=None,
        ml_decoder_learnable_queries=None, inner_mult=None,
        feat_attn_activation=None,
    ):
        super().__init__()
        if feature_sizes is None:
            raise ValueError("feature_sizes are required")
        # Support legacy n_inter_layer arg
        if n_layer == 6 and n_inter_layer is not None:
            n_layer = n_inter_layer

        n_orig_features = len(feature_sizes)
        total_rows = n_orig_features * rows_per_feature

        self.w_dim = w_dim
        self.rows_per_feature = rows_per_feature
        self.total_rows = total_rows
        self.uci_mode = uci_vocab_size is not None

        self.embed = CompositeSANEmbedding(
            feature_sizes=feature_sizes,
            rows_per_feature=rows_per_feature,
            w_dim=w_dim,
        )
        self.drop = nn.Dropout(dropout)

        # RoPE frequencies (over w_dim, the descriptor dimension)
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, w_dim, 2).float() / w_dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        # Layers
        self.layers = nn.ModuleList([
            ChessBlock(
                n_features=total_rows,
                desc_dim=w_dim,
                dropout=dropout,
                use_lerp=use_lerp,
                use_feat_attn=use_feat_attn,
            )
            for _ in range(n_layer)
        ])

        self.ln_f = nn.RMSNorm(w_dim)

        if self.uci_mode:
            self.uci_head = nn.Linear(w_dim, uci_vocab_size, bias=False)
        else:
            self.output_head = FeatureOutputHead(feature_sizes, rows_per_feature, w_dim)

    def forward(self, feature_ids):
        first = next(iter(feature_ids))
        batch_size, seq_len = feature_ids[first].shape

        x = self.embed(feature_ids)           # (B, T, total_rows, DD)
        x = self.drop(x)

        # Precompute RoPE for w_dim
        pos = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(pos, self.inv_freq)           # (T, DD//2)
        rope_cos = freqs.cos()[None, :, None, :]           # (1, T, 1, DD//2)
        rope_sin = freqs.sin()[None, :, None, :]

        for layer in self.layers:
            x = layer(x, rope_cos, rope_sin)

        if self.uci_mode:
            pooled = x.mean(dim=2)            # (B, T, DD)
            pooled = self.ln_f(pooled)        # norm AFTER pool so std ≈ 1
            return self.uci_head(pooled)      # (B, T, vocab_size)

        x = self.ln_f(x)
        return self.output_head(x)            # dict of {name: (B, T, n_classes)}
