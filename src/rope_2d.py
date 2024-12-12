# Reference https://github.com/THUDM/CogVideo/blob/main/sat/dit_video_concat.py
from functools import partial
from einops import rearrange, repeat
from functools import reduce
from operator import mul
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import torch


def get_2d_sincos_pos_embed(context_dim, grid_height, grid_width, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, context_dim] or [1+grid_size*grid_size, context_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_height, dtype=np.float32)
    grid_w = np.arange(grid_width, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_height, grid_width])
    pos_embed = get_2d_sincos_pos_embed_from_grid(context_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, context_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(context_dim, grid):
    assert context_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(context_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(context_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(context_dim, pos):
    """
    context_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert context_dim % 2 == 0
    omega = np.arange(context_dim // 2, dtype=np.float64)
    omega /= context_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def default(val, d):
    return val if val is not None else d


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


# def rotate_half(x):
#     # Rotate the last dimension by half
#     x = rearrange(x, '... (d r) -> ... d r', r=2)
#     x1, x2 = x.unbind(dim=-1)
#     return torch.cat((-x2, x1), dim=-1)

class Basic2DPositionEmbeddingMixin(nn.Module):
    def __init__(self, height, width, compressed_num_frames, hidden_size, text_length=0):
        super(Basic2DPositionEmbeddingMixin, self).__init__()
        self.height = height
        self.width = width
        self.spatial_length = height * width
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, int(text_length + self.spatial_length), int(hidden_size)), requires_grad=False
        )

    def position_embedding_forward(self, position_ids, **kwargs):
        return self.pos_embedding

    def reinit(self, parent_model=None):
        del self.transformer.position_embeddings
        pos_embed = get_2d_sincos_pos_embed(self.pos_embedding.shape[-1], self.height, self.width)
        self.pos_embedding.data[:, -self.spatial_length :].copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))


class Rotary2DPositionEmbeddingMixin(nn.Module):
    def __init__(
        self,
        height,
        width,
        hidden_size,
        hidden_size_head,
        text_length,
        theta=10000,
        rot_v=False,
        height_interpolation=1.0,
        width_interpolation=1.0,
        learnable_pos_embed=False,
    ):
        """
        Rotary2DPositionEmbeddingMixin initializes rotary positional embeddings for 2D data.

        Args:
            height (int): Height of the input feature map.
            width (int): Width of the input feature map.
            hidden_size (int): Total hidden size.
            hidden_size_head (int): Hidden size per head.
            text_length (int): Length of the text sequence.
            theta (float, optional): Frequency scaling factor. Defaults to 10000.
            rot_v (bool, optional): Whether to apply rotary embedding to value vectors. Defaults to False.
            height_interpolation (float, optional): Interpolation factor for height. Defaults to 1.0.
            width_interpolation (float, optional): Interpolation factor for width. Defaults to 1.0.
            learnable_pos_embed (bool, optional): Whether to use learnable positional embeddings. Defaults to False.
        """
        super(Rotary2DPositionEmbeddingMixin, self).__init__()
        self.rot_v = rot_v

        # Calculate dimensions for height and width
        dim_h = hidden_size_head // 8 * 3
        dim_w = hidden_size_head // 8 * 3

        # Compute frequencies for height and width
        freqs_h = 1.0 / (theta ** (torch.arange(0, dim_h, 2).float() / dim_h))
        freqs_w = 1.0 / (theta ** (torch.arange(0, dim_w, 2).float() / dim_w))

        # Create grid for height and width
        grid_h = torch.arange(height, dtype=torch.float32) * height_interpolation
        grid_w = torch.arange(width, dtype=torch.float32) * width_interpolation

        # Compute the outer product of grid and frequencies
        freqs_h = torch.einsum("h,f->hf", grid_h, freqs_h)  # Shape: [height, dim_h/2]
        freqs_w = torch.einsum("w,f->wf", grid_w, freqs_w)  # Shape: [width, dim_w/2]

        # Repeat frequencies for sine and cosine
        freqs_h = repeat(freqs_h, "... f -> ... (f 2)", f=1)
        freqs_w = repeat(freqs_w, "... f -> ... (f 2)", f=1)

        # Concatenate height and width frequencies
        freqs = torch.cat((freqs_h[:, None, :], freqs_w[None, :, :]), dim=-1)  # Shape: [height, width, dim_h + dim_w]

        freqs = freqs.contiguous()

        # Register buffers for sine and cosine
        self.register_buffer('freqs_sin', freqs.sin(), persistent=False)
        self.register_buffer('freqs_cos', freqs.cos(), persistent=False)

        self.text_length = text_length

        if learnable_pos_embed:
            num_patches = height * width + text_length
            self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=True)
        else:
            self.pos_embedding = None

    def rotary(self, t, **kwargs):
        """
        Applies rotary embedding to the input tensor.

        Args:
            t (torch.Tensor): Input tensor to apply rotary embedding. Shape: [batch, seq_length, hidden_size]
        
        Returns:
            torch.Tensor: Tensor after applying rotary embedding.
        """
        def reshape_freq(freqs):
            # Extract relevant frequencies based on the current sequence length
            freqs = freqs[: kwargs["rope_H"], : kwargs["rope_W"]].contiguous()  # Shape: [H, W, dim]
            freqs = rearrange(freqs, "h w d -> (h w) d")  # Shape: [H*W, dim]
            freqs = freqs.unsqueeze(0)  # Shape: [1, H*W, dim]
            return freqs

        freqs_cos = reshape_freq(self.freqs_cos).to(t.dtype)  # Shape: [1, H*W, dim]
        freqs_sin = reshape_freq(self.freqs_sin).to(t.dtype)  # Shape: [1, H*W, dim]

        return t * freqs_cos + rotate_half(t) * freqs_sin

    def position_embedding_forward(self, position_ids, **kwargs):
        """
        Returns positional embeddings if learnable.

        Args:
            position_ids (torch.Tensor): Position indices.
        
        Returns:
            torch.Tensor or None: Positional embeddings.
        """
        if self.pos_embedding is not None:
            return self.pos_embedding[:, : self.text_length + kwargs["seq_length"]]
        else:
            return None

    def attention_fn(
        self,
        query_layer,
        key_layer,
        value_layer,
        attention_mask,
        attention_dropout=None,
        log_attention_weights=None,
        scaling_attention_score=True,
        **kwargs,
    ):
        """
        Custom attention function integrating rotary embeddings.

        Args:
            query_layer (torch.Tensor): Query tensor.
            key_layer (torch.Tensor): Key tensor.
            value_layer (torch.Tensor): Value tensor.
            attention_mask (torch.Tensor): Attention mask.
            attention_dropout (nn.Module, optional): Dropout layer. Defaults to None.
            log_attention_weights (bool, optional): Whether to log attention weights. Defaults to None.
            scaling_attention_score (bool, optional): Whether to scale attention scores. Defaults to True.
        
        Returns:
            torch.Tensor: Output of the attention mechanism.
        """
        attention_fn_default = HOOKS_DEFAULT["attention_fn"]  # Ensure HOOKS_DEFAULT is defined

        # Split text and spatial tokens
        text_queries = query_layer[:, :, : kwargs["text_length"]]
        spatial_queries = query_layer[:, :, kwargs["text_length"]:]

        # Apply rotary to spatial queries
        spatial_queries = self.rotary(spatial_queries, **kwargs)

        # Concatenate back
        query_layer = torch.cat((text_queries, spatial_queries), dim=2)

        # Repeat the same for keys and values
        key_layer = torch.cat(
            (
                key_layer[:, :, : kwargs["text_length"]],
                self.rotary(key_layer[:, :, kwargs["text_length"]:], **kwargs),
            ),
            dim=2,
        )

        if self.rot_v:
            value_layer = torch.cat(
                (
                    value_layer[:, :, : kwargs["text_length"]],
                    self.rotary(value_layer[:, :, kwargs["text_length"]:], **kwargs),
                ),
                dim=2,
            )

        return attention_fn_default(
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            attention_dropout=attention_dropout,
            log_attention_weights=log_attention_weights,
            scaling_attention_score=scaling_attention_score,
            **kwargs,
        )

if __name__ == "__main__":

    sample_img_embedd = 512
    grid_height = 8
    grid_width = 8

    rope2d_pos_emb = get_2d_sincos_pos_embed(
        embed_dim = sample_img_embedd,
        grid_height=grid_height,
        grid_width=grid_width,
        cls_token=False,
        extra_tokens=0
    )
    rope2d_pos_emb = torch.from_numpy(rope2d_pos_emb).float()
    print(rope2d_pos_emb.shape)
