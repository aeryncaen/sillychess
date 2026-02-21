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
    """Content-gated causal lerp on first 50% of dims.

    x_lerp[t] = (1 - gate) * x[t] + gate * x[t-1]

    Remaining dims pass through unchanged.  Gate reads the full d_model
    but only gates the first half.
    Initialized near-identity (bias=-2.0 -> sigmoid ~ 0.12).
    """

    def __init__(self, d_model, init_bias=-2.0):
        super().__init__()
        self.lerp_dim = d_model // 2
        self.gate_proj = nn.Linear(d_model, self.lerp_dim, bias=True)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, init_bias)

    def forward(self, x):
        """x: (B, T, D) -> same shape."""
        x_lerp = x[..., :self.lerp_dim]
        x_pass = x[..., self.lerp_dim:]
        x_prev = F.pad(x_lerp[:, :-1], (0, 0, 1, 0))
        gate = torch.sigmoid(self.gate_proj(x))
        x_lerp = (1 - gate) * x_lerp + gate * x_prev
        return torch.cat([x_lerp, x_pass], dim=-1)


# ---------------------------------------------------------------------------
# Data-dependent RoPE (standalone, NOT inside attention)
# ---------------------------------------------------------------------------

class DDRoPE(nn.Module):
    """Data-dependent cumulative rotation on first dd_dim dims.

    Standalone state-tracking mechanism on the residual stream, inspired
    by Mamba-3's complex-valued SSM (§3.2, Proposition 3).  Projects input
    to per-timestep angle deltas, cumsums for cumulative phase, applies
    rotation.  NOT applied to Q/K inside attention.

    dd_dim is 50% of the CausalLerp dims = 25% of d_model.

    Args:
        d_input: Full model dim (reads all dims to compute angles).
        dd_dim:  Number of dims to rotate (must be even).
    """

    def __init__(self, d_input, dd_dim):
        super().__init__()
        assert dd_dim % 2 == 0, f"dd_dim must be even, got {dd_dim}"
        self.dd_dim = dd_dim
        self.dd_pairs = dd_dim // 2
        self.dd_proj = nn.Linear(d_input, self.dd_pairs, bias=True)
        nn.init.zeros_(self.dd_proj.weight)
        nn.init.zeros_(self.dd_proj.bias)

    def forward(self, x):
        """x: (B, T, D). Applies cumulative rotation to first dd_dim dims."""
        deltas = self.dd_proj(x)               # (B, T, dd_pairs)
        angles = deltas.cumsum(dim=1)          # cumulative phase
        x_dd = x[..., :self.dd_dim]
        x_dd = _apply_rotary(x_dd, angles.cos(), angles.sin())
        return torch.cat([x_dd, x[..., self.dd_dim:]], dim=-1)


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
            # 50% of lerp dims = 25% of d_model, rounded down to even
            dd_dim = (d_model // 4) & ~1
            assert dd_dim > 0, f"d_model too small for dd-rope: {d_model}"
            self.dd_rope = DDRoPE(d_input=d_model, dd_dim=dd_dim)

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

        # Pre-attention SSM: CausalLerp (50% dims) + DD-RoPE (25% dims)
        if self.use_lerp:
            x = self.causal_lerp(x)
        if self.use_dd_rope:
            x = self.dd_rope(x)

        # Sequence attention (standard fixed RoPE on Q/K)
        h = self.attn_norm(x)
        qkv = self.w_qkv(h).reshape(B, T, 3, nh, dh)
        q, k, v = qkv.unbind(dim=2)               # (B, T, nh, dh)
        q = self.q_norm(q) * self.q_post_bias
        k = self.k_norm(k) * self.k_post_bias
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
# Output head: composite vocab (non-UCI mode)
# ---------------------------------------------------------------------------

class CompositeOutputHead(nn.Module):
    """Single softmax over composite vocab, weights tied to input embedding.

    For each of the V vocab entries (a tuple of 8 feature IDs), the weight
    vector is built by looking up each feature ID in the shared embedding
    table (with per-feature offsets) and concatenating — the exact same
    operation as CompositeSANEmbedding.

    Computed fresh each forward pass: one gather of V*8 embeddings + reshape
    + normalize.  No Python loops.
    """

    def __init__(self, vocab_tuples, embed_weight, offsets):
        super().__init__()
        # vocab_tuples: (V, 8) long tensor of feature IDs
        # Pre-add offsets → absolute indices into the shared embedding table
        abs_ids = vocab_tuples + offsets.unsqueeze(0)          # (V, 8)
        self.register_buffer('abs_ids', abs_ids)
        self.embed_weight = embed_weight                       # reference to nn.Embedding.weight

    def _weight(self):
        """Build (V, d_model) weight matrix from shared embedding table."""
        w = self.embed_weight[self.abs_ids]                    # (V, 8, rpf*w_dim)
        w = w.reshape(w.shape[0], -1)                          # (V, d_model)
        return F.normalize(w, dim=-1)

    def forward(self, x):
        """x: (B, T, d_model) -> (B, T, V)"""
        return F.linear(x, self._weight())                     # (B, T, V)

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
        w = self._weight()                                     # (V, d_model)
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
            if composite_vocab is None:
                raise ValueError("composite_vocab required for non-UCI composite mode")
            self.output_head = CompositeOutputHead(
                composite_vocab, self.embed.embed.weight, self.embed.offsets,
            )

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
