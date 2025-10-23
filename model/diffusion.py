import torch
import torch.nn as nn
from torch.nn.functional import scaled_dot_product_attention
import torch.nn.functional as F

    
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = torch.exp(
            torch.arange(half_dim, device=t.device) *
            -(torch.log(torch.tensor(10000.0)) / (half_dim - 1))
        )
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb

class MultiHeadSelfAttention2d(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        assert channels % num_heads == 0
        self.head_dim = channels // num_heads
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj_out = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_norm = self.norm(x)
        x_flat = x_norm.view(B, C, H * W)  # [B, C, N]
        
        qkv = self.qkv(x_flat)
        q, k, v = qkv.chunk(3, dim=1)

        # reshape to [B, num_heads, N, head_dim]
        def reshape_heads(tensor):
            return tensor.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.transpose(2, 3).reshape(B, C, H * W)  # [B, C, N]

        out = self.proj_out(out)
        out = out.view(B, C, H, W)
        return x + out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, out_channels),
            nn.ReLU()
        )
        self.block1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.block2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x, t_emb):
        h = self.block1(x)
        h = self.norm1(h)
        h = self.relu(h)

        time_emb = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + time_emb

        h = self.block2(h)
        h = self.norm2(h)
        out = self.relu(h + self.shortcut(x))
        return out

class UNet(nn.Module):
    def __init__(
        self,
        in_channels=2,
        out_channels=1,
        base_channels=64,
        num_levels=4,
        time_emb_dim=128,
        attention_heads=8,
        attention_locations=[],
        device: str = None,
    ):
        super().__init__()
        self.device = device
        self.time_embedding = SinusoidalTimeEmbedding(time_emb_dim)
        self.attention_locations = attention_locations
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.downsamples = nn.ModuleList()
        self.enc_blocks = nn.ModuleList()
        self.enc_attn_blocks = nn.ModuleDict()
        self.num_levels = num_levels
        in_chs = in_channels
        for i in range(num_levels):
            out_chs = base_channels * (2 ** i)
            self.enc_blocks.append(ResidualBlock(in_chs, out_chs, time_emb_dim))
            self.downsamples.append(nn.Conv2d(out_chs, out_chs, kernel_size=3, stride=2, padding=1))
            if f"enc{i}" in attention_locations:
                self.enc_attn_blocks[f"enc{i}"] = MultiHeadSelfAttention2d(out_chs, num_heads=attention_heads)
            in_chs = out_chs

        # Bottleneck
        self.bot = ResidualBlock(in_chs, in_chs, time_emb_dim)
        self.bot_attn = MultiHeadSelfAttention2d(in_chs, num_heads=attention_heads) if "bottleneck" in attention_locations else nn.Identity()

        # Decoder
        self.dec_blocks = nn.ModuleList()
        self.dec_attn_blocks = nn.ModuleDict()
        for i in reversed(range(num_levels)):
            in_chs = base_channels * (2 ** (i + 1))
            out_chs = base_channels * (2 ** (i - 1)) if i > 0 else base_channels
            self.dec_blocks.append(ResidualBlock(in_chs, out_chs, time_emb_dim))
            if f"dec{i}" in attention_locations:
                self.dec_attn_blocks[f"dec{i}"] = MultiHeadSelfAttention2d(out_chs, num_heads=attention_heads)

        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        self.to(device)
            
    def forward(self, x, t, condition):
        t_emb = self.time_embedding(t.squeeze(1))
        x = torch.cat([x, condition], dim=1)

        skips = []
        for i, (enc, downsample) in enumerate(zip(self.enc_blocks, self.downsamples)):
            x = enc(x, t_emb)
            if f"enc{i}" in self.attention_locations:
                x = self.enc_attn_blocks[f"enc{i}"](x)
            skips.append(x)
            x = downsample(x)

        x = self.bot(x, t_emb)
        x = self.bot_attn(x)

        for i, (dec, skip) in enumerate(zip(self.dec_blocks, reversed(skips))):
            x = self.upsample(x)
            x = torch.cat([x, skip], dim=1)
            x = dec(x, t_emb)
            layer_num = self.num_levels - i - 1
            if f"dec{layer_num}" in self.attention_locations:
                x = self.dec_attn_blocks[f"dec{layer_num}"](x)

        x = self.out_conv(x)
        return x