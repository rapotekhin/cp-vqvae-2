# Reference https://github.com/vvvm23/vqvae-2/blob/main/vqvae.py
# Reference https://github.com/THUDM/CogVideo/blob/main/sat/sgm/modules/diffusionmodules/model.py#L155

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from math import log2
from typing import Tuple

from attention import CrossAttention
from rope_2d import Rotary2DPositionEmbeddingMixin, Basic2DPositionEmbeddingMixin, get_2d_sincos_pos_embed

def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class ReZero(nn.Module):
    def __init__(self, in_channels: int, res_channels: int):
        super(ReZero, self).__init__()  # Initialize the parent nn.Module
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, res_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(res_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(res_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x) * self.alpha + x

class ResidualStack(nn.Module):
    def __init__(self, in_channels: int, res_channels: int, nb_layers: int):
        super(ResidualStack, self).__init__()  # Initialize the parent nn.Module
        self.stack = nn.Sequential(*[ReZero(in_channels, res_channels) 
                        for _ in range(nb_layers)
                    ])

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.stack(x)

class TextProjection(nn.Module):
    def __init__(self, text_emb_dim: int, context_emb_dim: int):
        super(TextProjection, self).__init__()
        self.projection = nn.Linear(text_emb_dim, context_emb_dim)
        self.activation = nn.ReLU()
        self.norm = nn.LayerNorm(context_emb_dim)

    def forward(self, text_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_emb (torch.Tensor): Shape (batch, text_len, text_emb_dim)
        
        Returns:
            torch.Tensor: Shape (batch, text_len, context_emb_dim)
        """
        projected = self.projection(text_emb)  # Shape: (batch, text_len, context_emb_dim)
        projected = self.activation(projected)
        projected = self.norm(projected)
        return projected

class Encoder(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 hidden_channels: int, 
                 context_dim: int,
                 res_channels: int, 
                 nb_res_layers: int,
                 downscale_factor: int):
        super(Encoder, self).__init__()
        
        # Ensure downscale_factor is a power of 2
        assert log2(downscale_factor).is_integer(), "Downscale must be a power of 2"
        downscale_steps = int(log2(downscale_factor))
        
        # Check for a specific number of downscale_steps
        # For example, handle up to 4 downscale steps
        if downscale_steps > 4:
            raise ValueError("This Encoder implementation supports up to 4 downscale steps.")
        
        # Define each downscale layer explicitly
        # Example for downscale_steps = 3 (downscale_factor = 8)
        c_channel, n_channel = in_channels, hidden_channels // 2
        if downscale_steps >= 1:
            self.layer1 = nn.Sequential(
                nn.Conv2d(c_channel, n_channel, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(n_channel),
                nn.ReLU(inplace=True)
            )
            self.cross_att1 = CrossAttention(n_channel, context_dim=context_dim)
            c_channel, n_channel = n_channel, hidden_channels
        if downscale_steps >= 2:
            self.layer2 = nn.Sequential(
                nn.Conv2d(c_channel, n_channel, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(n_channel),
                nn.ReLU(inplace=True)
            )
            self.cross_att2 = CrossAttention(n_channel, context_dim=context_dim)
            c_channel, n_channel = n_channel, hidden_channels
        if downscale_steps >= 3:
            self.layer3 = nn.Sequential(
                nn.Conv2d(c_channel, n_channel, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(n_channel),
                nn.ReLU(inplace=True)
            )
            self.cross_att3 = CrossAttention(n_channel, context_dim=context_dim)
            c_channel, n_channel = n_channel, hidden_channels
        if downscale_steps == 4:
            self.layer4 = nn.Sequential(
                nn.Conv2d(c_channel, n_channel, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(n_channel),
                nn.ReLU(inplace=True)
            )
            self.cross_att4 = CrossAttention(n_channel, context_dim=context_dim)
            c_channel, n_channel = n_channel, hidden_channels

        # Final convolution and residual stack
        self.final_conv = nn.Sequential(
            nn.Conv2d(c_channel, n_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_channel),
            ResidualStack(n_channel, res_channels, nb_res_layers)
        )
        
    def forward(self, x: torch.FloatTensor, context: torch.FloatTensor = None) -> torch.FloatTensor:
        # Apply each layer conditionally based on downscale_steps
        # This requires knowing how many layers to apply
        # Alternatively, you can use try-except blocks if layers are not defined
        if hasattr(self, 'layer1'):
            x = self.layer1(x)
            b, d, h, w = x.size()
            x = rearrange(x, 'b d h w -> b (h w) d')
            x = self.cross_att1(x, context)
            x = rearrange(x, 'b (h w) d -> b d h w', h=h, w=w)
        if hasattr(self, 'layer2'):
            x = self.layer2(x)
            b, d, h, w = x.size()
            x = rearrange(x, 'b d h w -> b (h w) d')
            x = self.cross_att2(x, context)
            x = rearrange(x, 'b (h w) d -> b d h w', h=h, w=w)
        if hasattr(self, 'layer3'):
            x = self.layer3(x)
            b, d, h, w = x.size()
            x = rearrange(x, 'b d h w -> b (h w) d')
            x = self.cross_att3(x, context)
            x = rearrange(x, 'b (h w) d -> b d h w', h=h, w=w)
        if hasattr(self, 'layer4'):
            x = self.layer4(x)
            b, d, h, w = x.size()
            x = rearrange(x, 'b d h w -> b (h w) d')
            x = self.cross_att4(x, context)
            x = rearrange(x, 'b (h w) d -> b d h w', h=h, w=w)
        # Apply final convolution and residual stack
        x = self.final_conv(x)
        return x

class Decoder(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 hidden_channels: int, 
                 context_dim: int,
                 out_channels: int,
                 res_channels: int, 
                 nb_res_layers: int,
                 upscale_factor: int,
                 max_upscale_steps: int = 4  # Define a maximum number of upscale steps
                ):
        super(Decoder, self).__init__()
        
        # Ensure upscale_factor is a power of 2
        assert log2(upscale_factor).is_integer(), "Upscale factor must be a power of 2"
        upscale_steps = int(log2(upscale_factor))
        
        # Define a maximum number of supported upscale steps
        if upscale_steps > max_upscale_steps:
            raise ValueError(f"This Decoder implementation supports up to {max_upscale_steps} upscale steps.")
        
        # Initial convolution
        self.initial_conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        
        # Residual Stack
        self.residual_stack = ResidualStack(hidden_channels, res_channels, nb_res_layers)
        
        # Define upscaling layers explicitly
        c_channel, n_channel = hidden_channels, hidden_channels // 2
        if upscale_steps >= 1:
            self.cross_att1 = CrossAttention(c_channel, context_dim=context_dim)
            self.layer1 = nn.Sequential(
                nn.ConvTranspose2d(c_channel, n_channel, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(n_channel),
                nn.ReLU(inplace=True)
            )
            c_channel, n_channel = n_channel, out_channels
        if upscale_steps >= 2:
            self.cross_att2 = CrossAttention(c_channel, context_dim=context_dim)
            self.layer2 = nn.Sequential(
                nn.ConvTranspose2d(c_channel, n_channel, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(n_channel),
                nn.ReLU(inplace=True)
            )
            c_channel, n_channel = n_channel, out_channels
        if upscale_steps >= 3:
            self.cross_att3 = CrossAttention(c_channel, context_dim=context_dim)
            self.layer3 = nn.Sequential(
                nn.ConvTranspose2d(c_channel, n_channel, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(n_channel),
                nn.ReLU(inplace=True)
            )
            c_channel, n_channel = n_channel, out_channels
        if upscale_steps == 4:
            self.cross_att4 = CrossAttention(c_channel, context_dim=context_dim)
            self.layer4 = nn.Sequential(
                nn.ConvTranspose2d(c_channel, n_channel, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(n_channel),
                nn.ReLU(inplace=True)
            )
            c_channel, n_channel = n_channel, out_channels
        
        # Final convolution
        self.final_conv = nn.Sequential(
            nn.Conv2d(c_channel, n_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_channel)
            # nn.ReLU(inplace=True)  # Uncomment if activation is needed
        )
    
    def forward(self, x: torch.FloatTensor, context: torch.FloatTensor = None) -> torch.FloatTensor:
        """
        Args:
            x (torch.FloatTensor): Input tensor.
            context (torch.FloatTensor, optional): Context tensor. Default is None.
        
        Returns:
            torch.FloatTensor: Output tensor after decoding.
        """
        # Initial convolution
        x = self.initial_conv(x)
        
        # Apply Residual Stack
        x = self.residual_stack(x)

        # Apply upscaling layers conditionally
        if hasattr(self, 'layer1'):
            b, d, h, w = x.size()
            x = rearrange(x, 'b d h w -> b (h w) d')
            x = self.cross_att1(x, context)
            x = rearrange(x, 'b (h w) d -> b d h w', h=h, w=w)
            x = self.layer1(x)
        if hasattr(self, 'layer2'):
            b, d, h, w = x.size()
            x = rearrange(x, 'b d h w -> b (h w) d')
            x = self.cross_att2(x, context)
            x = rearrange(x, 'b (h w) d -> b d h w', h=h, w=w)
            x = self.layer2(x)
        if hasattr(self, 'layer3'):
            b, d, h, w = x.size()
            x = rearrange(x, 'b d h w -> b (h w) d')
            x = self.cross_att3(x, context)
            x = rearrange(x, 'b (h w) d -> b d h w', h=h, w=w)
            x = self.layer3(x)
        if hasattr(self, 'layer4'):
            b, d, h, w = x.size()
            x = rearrange(x, 'b d h w -> b (h w) d')
            x = self.cross_att4(x, context)
            x = rearrange(x, 'b (h w) d -> b d h w', h=h, w=w)
            x = self.layer4(x)

        # Final convolution
        x = self.final_conv(x)
        
        return x

"""
    Almost directly taken from https://github.com/rosinality/vq-vae-2-pytorch/blob/master/vqvae.py
    No reason to reinvent this rather complex mechanism.

    Essentially handles the "discrete" part of the network, and training through EMA rather than 
    third term in loss function.
"""
class CodeLayer(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, nb_entries: int):
        super(CodeLayer, self).__init__()  # Initialize the parent nn.Module
        self.conv_in = nn.Conv2d(in_channels, embed_dim, 1)

        self.dim = embed_dim
        self.n_embed = nb_entries
        self.decay = 0.99
        self.eps = 1e-5

        embed = torch.randn(embed_dim, nb_entries, dtype=torch.float32)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(nb_entries, dtype=torch.float32))
        self.register_buffer("embed_avg", embed.clone())

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x: torch.FloatTensor) -> Tuple[torch.FloatTensor, float, torch.LongTensor]:
        x = self.conv_in(x.float()).permute(0,2,3,1)
        flatten = x.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*x.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            # TODO: Replace this? Or can we simply comment out?
            # dist_fn.all_reduce(embed_onehot_sum)
            # dist_fn.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - x).pow(2).mean()
        quantize = x + (quantize - x).detach()

        return quantize.permute(0, 3, 1, 2), diff, embed_ind

    def embed_code(self, embed_id: torch.LongTensor) -> torch.FloatTensor:
        return F.embedding(embed_id, self.embed.transpose(0, 1))

class Upscaler(nn.Module):
    def __init__(self,
            embed_dim: int,
            scaling_rates: list[int],
        ):
        super(Upscaler, self).__init__()  # Initialize the parent nn.Module

        self.stages = nn.ModuleList()
        for sr in scaling_rates:
            upscale_steps = int(log2(sr))
            layers = []
            for _ in range(upscale_steps):
                layers.append(nn.ConvTranspose2d(embed_dim, embed_dim, 4, stride=2, padding=1))
                layers.append(nn.BatchNorm2d(embed_dim))
                layers.append(nn.ReLU(inplace=True))
            self.stages.append(nn.Sequential(*layers))

    def forward(self, x: torch.FloatTensor, stage: int) -> torch.FloatTensor:
        return self.stages[stage](x)

"""
    Main VQ-VAE-2 Module, capable of support arbitrary number of levels
    TODO: A lot of this class could do with a refactor. It works, but at what cost?
    TODO: Add disrete code decoding function
"""
class VQVAE(nn.Module):
    def __init__(self,
            in_channels: int                = 3,
            hidden_channels: int            = 128,
            res_channels: int               = 32,
            nb_res_layers: int              = 2,
            nb_levels: int                  = 3,
            embed_dim: int                  = 64,
            nb_entries: int                 = 512,
            text_dim: int                   = 768,
            context_dim: int                = 512,
            scaling_rates: list[int]        = [4, 2]  # [8, 4, 2]
        ):
        super(VQVAE, self).__init__()  # Initialize the parent nn.Module
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.res_channels = res_channels
        self.nb_res_layers = nb_res_layers
        self.nb_levels = nb_levels
        self.embed_dim = embed_dim
        self.nb_entries = nb_entries
        self.text_dim = text_dim
        self.context_dim = context_dim
        self.scaling_rates = scaling_rates

        self.nb_levels = nb_levels
        assert len(scaling_rates) == nb_levels, "Number of scaling rates not equal to number of levels!"

        self.text_emb_projector = TextProjection(text_dim, context_dim)

        self.encoders = nn.ModuleList([Encoder(in_channels, hidden_channels, context_dim, res_channels, nb_res_layers, scaling_rates[0])])
        for i, sr in enumerate(scaling_rates[1:]):
            self.encoders.append(Encoder(hidden_channels, hidden_channels, context_dim, res_channels, nb_res_layers, sr))

        self.codebooks = nn.ModuleList()
        for i in range(nb_levels - 1):
            self.codebooks.append(CodeLayer(hidden_channels+embed_dim, embed_dim, nb_entries))
        self.codebooks.append(CodeLayer(hidden_channels, embed_dim, nb_entries))

        self.decoders = nn.ModuleList([Decoder(embed_dim*nb_levels, hidden_channels, context_dim, in_channels, res_channels, nb_res_layers, scaling_rates[0])])
        for i, sr in enumerate(scaling_rates[1:]):
            self.decoders.append(Decoder(embed_dim*(nb_levels-1-i), hidden_channels, context_dim, embed_dim, res_channels, nb_res_layers, sr))

        self.upscalers = nn.ModuleList()
        for i in range(nb_levels - 1):
            self.upscalers.append(Upscaler(embed_dim, scaling_rates[1:len(scaling_rates) - i][::-1]))

    def get_pos_embeddings(self, grid_height, grid_width):
        pos_embed_all = get_2d_sincos_pos_embed(
            context_dim=self.context_dim,
            grid_height=grid_height,
            grid_width=grid_width,
            cls_token=False,
            extra_tokens=0
        )
        pos_embed_all = torch.from_numpy(pos_embed_all).float()
        pos_embed_all = rearrange(pos_embed_all, "n d -> 1 n d")
        return pos_embed_all

    def forward(self, x, text_emb, pos_embed):
        encoder_outputs = []
        code_outputs = []
        decoder_outputs = []
        upscale_counts = []
        id_outputs = []
        diffs = []

        text_emb = self.text_emb_projector(text_emb)
        context = torch.concat([pos_embed, text_emb], axis=1)

        for enc in self.encoders:
            if len(encoder_outputs):
                encoder_outputs.append(enc(encoder_outputs[-1], context))
            else:
                encoder_outputs.append(enc(x, context))

        for l in range(self.nb_levels-1, -1, -1):
            codebook, decoder = self.codebooks[l], self.decoders[l]

            if len(decoder_outputs): # if we have previous levels to condition on
                code_q, code_d, emb_id = codebook(torch.cat([encoder_outputs[l], decoder_outputs[-1]], axis=1))
            else:
                code_q, code_d, emb_id = codebook(encoder_outputs[l])
            diffs.append(code_d)
            id_outputs.append(emb_id)

            code_outputs = [self.upscalers[i](c, upscale_counts[i]) for i, c in enumerate(code_outputs)]
            upscale_counts = [u+1 for u in upscale_counts]
            decoder_outputs.append(decoder(torch.cat([code_q, *code_outputs], axis=1), context))

            code_outputs.append(code_q)
            upscale_counts.append(0)

        return decoder_outputs[-1], diffs, encoder_outputs, decoder_outputs, id_outputs

    def decode_codes(self, *cs):
        decoder_outputs = []
        code_outputs = []
        upscale_counts = []

        for l in range(self.nb_levels - 1, -1, -1):
            codebook, decoder = self.codebooks[l], self.decoders[l]
            code_q = codebook.embed_code(cs[l]).permute(0, 3, 1, 2)
            code_outputs = [self.upscalers[i](c, upscale_counts[i]) for i, c in enumerate(code_outputs)]
            upscale_counts = [u+1 for u in upscale_counts]
            decoder_outputs.append(decoder(torch.cat([code_q, *code_outputs], axis=1)))

            code_outputs.append(code_q)
            upscale_counts.append(0)

        return decoder_outputs[-1]

if __name__ == '__main__':
    from helpers import get_parameter_count
    device = 'cpu' # torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 1
    nb_levels = 3
    text_length = 10
    text_dim = 768
    embed_dim = 512
    grid_height = 8
    grid_width = 8
    crop_index = 14  # Crop index from 8x8 grid (0, ..., 63)

    net = VQVAE(
        text_dim=text_dim,
        context_dim=embed_dim,
        nb_levels=nb_levels, 
        scaling_rates=[2, 2, 2]
    ).to(device)

    pos_embed_all = net.get_pos_embeddings(grid_height, grid_width)

    print(f"Number of trainable parameters: {get_parameter_count(net)}")

    x = torch.randn(batch_size, 3, 128, 128).to(device)
    text_emb = torch.randn((batch_size, text_length, text_dim)).to(device)
    pos_embed = pos_embed_all[:, crop_index, :].view(batch_size, 1, embed_dim)

    _, diffs, encoder_outputs, decoder_outputs, id_outputs = net(x, text_emb, pos_embed)

    print('\n'.join(str(y.shape) for y in encoder_outputs))
    print()
    print('\n'.join(str(y.shape) for y in decoder_outputs))
    print()
    print('\n'.join(str(y.shape) for y in id_outputs))
    print()
    print('\n'.join(str(y) for y in diffs))