"""Transformer model for the diffusion language model."""

import math
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor


@dataclass
class DiffusionTransformerConfig:
    """Configuration for DiffusionTransformer."""

    vocab_size: int = 30522  # Default BERT vocab size
    d_model: int = 768
    nhead: int = 12
    num_layers: int = 12
    dim_feedforward: int = 3072
    dropout: float = 0.1
    layer_norm_eps: float = 1e-12
    max_position_embeddings: int = 512


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization as used in LLaMA/LLaDA.

    Reference: https://arxiv.org/abs/1910.07467
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        """Initialize RMSNorm.

        Args:
            dim: Feature dimension
            eps: Epsilon for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        """Apply RMS normalization.

        Args:
            x: Input tensor

        Returns:
            Normalized tensor
        """
        # Calculate RMS
        norm = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x_normalized = x / norm

        # Apply scale
        result = self.weight * x_normalized

        return result


class SwiGLU(nn.Module):
    """SwiGLU activation function as described in the paper.

    Reference: https://arxiv.org/abs/2002.05202
    """

    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        """Initialize SwiGLU.

        Args:
            in_features: Input dimension
            hidden_features: Hidden dimension
            out_features: Output dimension
        """
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)
        self.w3 = nn.Linear(hidden_features, out_features)

    def forward(self, x: Tensor) -> Tensor:
        """Apply SwiGLU activation.

        Args:
            x: Input tensor

        Returns:
            Output after SwiGLU activation
        """
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = x1 * F.silu(x2)

        return cast(Tensor, self.w3(hidden))


def rotate_half(x: Tensor) -> Tensor:
    """Rotate half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding as used in LLaMA/LLaDA.

    Based on the paper: https://arxiv.org/abs/2104.09864
    """

    def __init__(
        self, dim: int, max_position_embeddings: int = 2048, base: int = 10000
    ):
        """Initialize rotary position embeddings.

        Args:
            dim: Feature dimension
            max_position_embeddings: Maximum sequence length
            base: Base for the angle calculation
        """
        super().__init__()

        # Create position embeddings
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_position_embeddings
        self.cached_seq_len: int | None = None
        self.cached_cos: Tensor | None = None
        self.cached_sin: Tensor | None = None

    def forward(self, x: Tensor, seq_len: int) -> tuple[Tensor, Tensor]:
        """Calculate cos and sin for rotary embeddings.

        Args:
            x: Input tensor
            seq_len: Sequence length

        Returns:
            Tuple of cos and sin tensors for rotary embedding
        """
        # If we've cached these values for this sequence length, use them
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum length {self.max_seq_len}"
            )

        if self.cached_seq_len != seq_len:
            self.cached_seq_len = seq_len

            # Create position indices
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)

            # Cache cos and sin values
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cached_cos = emb.cos().to(x.device)
            self.cached_sin = emb.sin().to(x.device)

        # At this point, cached_cos and cached_sin are guaranteed to be initialized
        assert self.cached_cos is not None and self.cached_sin is not None
        return self.cached_cos, self.cached_sin

    @staticmethod
    def apply_rotary_pos_emb(
        q: Tensor, k: Tensor, cos: Tensor, sin: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Apply rotary position embeddings to queries and keys.

        Args:
            q: Query tensor of shape [batch, seq, heads, dim]
            k: Key tensor of shape [batch, seq, heads, dim]
            cos: Cosine tensor of shape [seq, dim]
            sin: Sine tensor of shape [seq, dim]

        Returns:
            Rotary embedded query and key tensors
        """
        # Reshape for broadcasting
        cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq, 1, dim]
        sin = sin.unsqueeze(0).unsqueeze(2)  # [1, seq, 1, dim]

        # Apply rotary embeddings
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)

        return q_embed, k_embed


class RotaryMultiheadAttention(nn.Module):
    """Multi-head attention with rotary position embeddings."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        """Initialize rotary multi-head attention.

        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        # Query, key, value projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Rotary position embeddings
        self.rotary_emb = RotaryPositionEmbedding(self.head_dim)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Perform multi-head attention with rotary embeddings.

        Args:
            query: Query tensor of shape [batch, seq_q, dim]
            key: Key tensor of shape [batch, seq_k, dim]
            value: Value tensor of shape [batch, seq_k, dim]
            attn_mask: Optional attention mask
            key_padding_mask: Optional key padding mask

        Returns:
            Output tensor after attention
        """
        batch_size, seq_len, _ = query.shape

        # Project queries, keys, and values
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape for multi-head attention: [batch, seq, heads, head_dim]
        q = q.view(batch_size, seq_len, self.nhead, self.head_dim)
        k = k.view(batch_size, k.size(1), self.nhead, self.head_dim)
        v = v.view(batch_size, v.size(1), self.nhead, self.head_dim)

        # Apply rotary embeddings
        cos, sin = self.rotary_emb(q, seq_len)
        q, k = self.rotary_emb.apply_rotary_pos_emb(q, k, cos, sin)

        # Transpose for attention: [batch, heads, seq, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scale query
        q = q / math.sqrt(self.head_dim)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1))  # [batch, heads, seq_q, seq_k]

        # Apply attention mask if provided
        if attn_mask is not None:
            scores = scores + attn_mask

        # Apply key padding mask if provided
        if key_padding_mask is not None:
            # Convert from [batch, seq] to [batch, 1, 1, seq]
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(key_padding_mask, float("-inf"))

        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)  # [batch, heads, seq_q, head_dim]

        # Reshape back to [batch, seq_q, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)

        # Final projection
        output = self.out_proj(attn_output)

        # Cast to ensure type safety
        return cast(Tensor, output)


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with RMSNorm, Rotary Attention, and SwiGLU."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-6,
    ):
        """Initialize transformer encoder layer.

        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            dim_feedforward: Feedforward dimension
            dropout: Dropout probability
            layer_norm_eps: Epsilon for layer normalization
        """
        super().__init__()

        # Self-attention with rotary embeddings
        self.self_attn = RotaryMultiheadAttention(d_model, nhead, dropout)

        # RMSNorm
        self.norm1 = RMSNorm(d_model, eps=layer_norm_eps)
        self.norm2 = RMSNorm(d_model, eps=layer_norm_eps)

        # SwiGLU
        self.swiglu = SwiGLU(d_model, dim_feedforward, d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        src: Tensor,
        src_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Process one encoder layer.

        Args:
            src: Source tensor of shape [batch, seq, dim]
            src_mask: Optional source mask
            src_key_padding_mask: Optional key padding mask

        Returns:
            Output tensor after processing
        """
        # Apply pre-norm architecture (norm -> sublayer -> residual)
        src_norm = self.norm1(src)
        attn_output = self.self_attn(
            src_norm,
            src_norm,
            src_norm,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )
        src = src + self.dropout(attn_output)

        # Feed-forward network with SwiGLU
        src_norm = self.norm2(src)
        ff_output = self.swiglu(src_norm)
        src = src + self.dropout(ff_output)

        return src


class DiffusionTransformer(nn.Module):
    """Transformer model for diffusion language model matching LLaDA architecture.

    Key components:
    1. RMSNorm instead of LayerNorm
    2. Rotary Position Embeddings instead of sinusoidal
    3. SwiGLU activation instead of GELU
    """

    def __init__(self, config: DiffusionTransformerConfig):
        """Initialize the transformer model.

        Args:
            config: Configuration for the transformer model
        """
        super().__init__()

        # Word embedding
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Create transformer encoder layers
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=config.d_model,
                    nhead=config.nhead,
                    dim_feedforward=config.dim_feedforward,
                    dropout=config.dropout,
                    layer_norm_eps=config.layer_norm_eps,
                )
                for _ in range(config.num_layers)
            ]
        )

        # Final normalization layer
        self.norm = RMSNorm(config.d_model, eps=config.layer_norm_eps)

        # Output layer (prediction head)
        self.output_layer = nn.Linear(config.d_model, config.vocab_size)

        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self) -> None:
        """Initialize parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        """Forward pass.

        Args:
            input_ids: Tensor of token ids of shape [batch_size, seq_len]
            attention_mask: Attention mask of shape [batch_size, seq_len]

        Returns:
            Logits of shape [batch_size, seq_len, vocab_size]
        """
        # Get embeddings
        x = self.embedding(input_ids)

        # Convert attention mask to proper format if provided
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0

        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=key_padding_mask)

        # Final normalization
        x = self.norm(x)

        # Get logits
        logits = self.output_layer(x)

        # Cast to ensure type safety
        return cast(Tensor, logits)
