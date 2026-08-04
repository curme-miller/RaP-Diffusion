import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.layers import DropPath, to_2tuple, trunc_normal_
try:
    import xformers
    import xformers.ops as xop
    XFORMERS_IS_AVAILBLE = True
except Exception:
    XFORMERS_IS_AVAILBLE = False
from abc import abstractmethod

def linear(*args, **kwargs):
    return nn.Linear(*args, **kwargs)

def conv_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def avg_pool_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

def normalization(channels):
    return GroupNorm32(32, channels)

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]                        # B x half
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, H, W)
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1).contiguous())
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0).to(attn.dtype)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).contiguous().reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=normalization):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        H, W = x_size
        img_mask = torch.zeros((1, 1, H, W))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size).permute(0,2,3,1).contiguous()
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

class PatchEmbed(nn.Module):
    def __init__(
            self,
            in_chans,
            img_size=224,
            patch_size=4,
            embed_dim=96,
            patch_norm=False,
            ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if patch_norm:
            self.norm = normalization(embed_dim)
        else:
            self.norm = nn.Identity()
    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x

class PatchUnEmbed(nn.Module):
    def __init__(self, out_chans, embed_dim=96, patch_norm=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(embed_dim, out_chans, kernel_size=1, stride=1)
        if patch_norm:
            self.norm = normalization(out_chans)
        else:
            self.norm = nn.Identity()
    def forward(self, x):
        x = self.norm(self.proj(x))
        return x

class BasicLayer(nn.Module):
    def __init__(
            self,
            in_chans,
            embed_dim,
            num_heads,
            window_size,
            depth=2,
            img_size=224,
            patch_size=4,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop=0.,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=normalization,
            use_checkpoint=False,
            patch_norm=True,
             ):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.patch_embed = PatchEmbed(
                in_chans=in_chans,
                embed_dim=embed_dim,
                img_size=img_size,
                patch_size=patch_size,
                patch_norm=patch_norm,
                )
        input_resolution = self.patch_embed.patches_resolution
        self.input_resolution = input_resolution
        self.patch_unembed = PatchUnEmbed(
                out_chans=in_chans,
                embed_dim=embed_dim,
                patch_norm=patch_norm,
                )
        self.blocks = nn.ModuleList([
            TransformerBlock(
                        dim=embed_dim,
                        input_resolution=input_resolution,
                        num_heads=num_heads,
                        window_size=window_size,
                        shift_size=0 if (i % 2 == 0) else window_size // 2,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop,
                        attn_drop=attn_drop,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        norm_layer=norm_layer,
                                 )
            for i in range(depth)])
    def forward(self, x):
        x = self.patch_embed(x)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        x = self.patch_unembed(x)
        return x

class Upsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)
    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=1)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)
    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

class ResBlock(TimestepBlock):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm
        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )
        self.updown = up or down
        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
        )
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)
    def forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1, num_head_channels=-1, use_new_attention_order=False):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0
            self.num_heads = channels // num_head_channels
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention(self.num_heads) if use_new_attention_order else QKVAttentionLegacy(self.num_heads)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)

class QKVAttentionLegacy(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        if XFORMERS_IS_AVAILBLE:
            qkv = qkv.reshape(bs, self.n_heads, ch * 3, length).permute(0, 3, 1, 2).contiguous()
            q, k, v = qkv.split(ch, dim=3)
            a = xop.memory_efficient_attention(q, k, v, p=0.0)
            out = a.permute(0, 2, 3, 1).contiguous().reshape(bs, -1, length)
        else:
            q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
            scale = 1 / math.sqrt(math.sqrt(ch))
            weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)
            weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
            a = torch.einsum("bts,bcs->bct", weight, v)
            out = a.reshape(bs, -1, length)
        return out

class QKVAttention(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        if XFORMERS_IS_AVAILBLE:
            qkv = qkv.reshape(bs, self.n_heads, ch * 3, length).permute(0, 3, 1, 2).contiguous()
            q, k, v = qkv.split(ch, dim=3)
            a = xop.memory_efficient_attention(q, k, v, p=0.0)
            out = a.permute(0, 2, 3, 1).contiguous().reshape(bs, -1, length)
        else:
            q, k, v = qkv.chunk(3, dim=1)
            scale = 1 / math.sqrt(math.sqrt(ch))
            weight = torch.einsum(
                "bct,bcs->bts",
                (q * scale).view(bs * self.n_heads, ch, length),
                (k * scale).view(bs * self.n_heads, ch, length),
            )
            weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
            a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
            out = a.reshape(bs, -1, length)
        return out

    def forward(self, H):
        Bn, T, C = H.shape
        Hn = self.norm(H)
        qkv = self.to_qkv(Hn).chunk(3, dim=-1)
        def split_heads(x):
            x = x.view(Bn, T, self.heads, self.dim_head)
            return x.permute(0, 2, 1, 3)
        q, k, v = map(split_heads, qkv)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        y = attn @ v
        y = y.permute(0, 2, 1, 3).contiguous().view(Bn, T, -1)
        y = self.to_out(y)
        f = self.alpha * (y - H)
        if self.use_mlp:
            f = f + self.mlp(Hn)
        tau = F.softplus(self.log_tau) + 1e-3
        return f / tau

def _window_partition(x, window_size):
    B, C, H, W = x.shape
    ws = window_size
    assert H % ws == 0 and W % ws == 0, "error"
    x = x.view(B, C, H // ws, ws, W // ws, ws).permute(0, 2, 4, 3, 5, 1).contiguous()
    windows = x.view(B * (H // ws) * (W // ws), ws * ws, C)
    return windows, (B, H, W)

def _window_reverse(windows, window_size, meta):
    B, H, W = meta
    ws = window_size
    C = windows.shape[-1]
    x = windows.view(B, H // ws, W // ws, ws, ws, C).permute(0, 5, 1, 3, 2, 4).contiguous()
    x = x.view(B, C, H, W)
    return x

class LiquidWindowODE(nn.Module):
    def __init__(self, dim, window_size=8, steps=2, dt=0.5,
                 heads=4, dim_head=32, mlp_ratio=0.0, tau_init=0.1,
                 use_heun=False, dtype=None):
        super().__init__()
        self.ws = window_size
        self.steps = steps
        self.dt = dt
        self.use_heun = use_heun
        self.vf = TokenLiquidVF(dim, heads=heads, dim_head=dim_head,
                                mlp_ratio=mlp_ratio, tau_init=tau_init)
        self._dtype = dtype

    def forward(self, x):
        dtype_in = x.dtype
        if self._dtype is not None:
            x = x.to(self._dtype)
        H, meta = _window_partition(x, self.ws)
        if not self.use_heun:
            for _ in range(self.steps):
                H = H + self.dt * self.vf(H)
        else:
            for _ in range(self.steps):
                k1 = self.vf(H)
                Ht = H + self.dt * k1
                k2 = self.vf(Ht)
                H = H + 0.5 * self.dt * (k1 + k2)
        x = _window_reverse(H, self.ws, meta)
        return x.to(dtype_in)

class LiRaAKODEnet(nn.Module):
    def __init__(
        self,
        image_size=64,
        in_channels=3,
        model_channels=160,
        out_channels=3,
        num_res_blocks=[2, 2, 2, 2],
        attention_resolutions=[64,32,16,8],
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_fp16=False,
        num_heads=8,
        num_head_channels=32,
        use_scale_shift_norm=True,
        resblock_updown=False,
        swin_depth=2,
        swin_embed_dim=192,
        window_size=8,
        mlp_ratio=2.0,
        patch_norm=False,
        cond_x0=True,
        cond_mask=False,
        x0_size=256,

        use_liquid=True,
        liquid_steps=2,
        liquid_dt=0.5,
        liquid_heads=4,
        liquid_dim_head=32,
        liquid_tau_init=0.1,
        liquid_mlp_ratio=0.0,
        liquid_use_heun=False,
    ):
        super().__init__()
        if isinstance(num_res_blocks, int):
            num_res_blocks = [num_res_blocks] * len(channel_mult)
        else:
            assert len(num_res_blocks) == len(channel_mult)
        if num_heads == -1:
            assert swin_embed_dim % num_head_channels == 0 and num_head_channels > 0
        self.num_res_blocks = num_res_blocks

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.cond_x0 = cond_x0
        self.cond_mask = cond_mask

        self.use_liquid = use_liquid
        self.liquid_steps = liquid_steps
        self.liquid_dt = liquid_dt
        self.liquid_heads = liquid_heads
        self.liquid_dim_head = liquid_dim_head
        self.liquid_tau_init = liquid_tau_init
        self.liquid_mlp_ratio = liquid_mlp_ratio
        self.liquid_use_heun = liquid_use_heun

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        if cond_x0 and x0_size == image_size:
            self.feature_extractor = nn.Identity()
            base_chn = 4 if cond_mask else 3
        else:
            feature_extractor = []
            feature_chn = 4 if cond_mask else 3
            base_chn = 16
            for _ in range(int(math.log(x0_size / image_size) / math.log(2))):
                feature_extractor.append(nn.Conv2d(feature_chn, base_chn, 3, 1, 1))
                feature_extractor.append(nn.SiLU())
                feature_extractor.append(Downsample(base_chn, True, out_channels=base_chn * 2))
                base_chn *= 2
                feature_chn = base_chn
            self.feature_extractor = nn.Sequential(*feature_extractor)

        ch = input_ch = int(channel_mult[0] * model_channels)
        in_channels += base_chn
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))])
        input_block_chans = [ch]
        ds = image_size

        for level, mult in enumerate(channel_mult):
            for jj in range(num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions and jj == 0:
                    if self.use_liquid:
                        layers.append(
                            LiquidWindowODE(
                                dim=ch,
                                window_size=window_size,
                                steps=self.liquid_steps,
                                dt=self.liquid_dt,
                                heads=self.liquid_heads,
                                dim_head=self.liquid_dim_head,
                                mlp_ratio=self.liquid_mlp_ratio,
                                tau_init=self.liquid_tau_init,
                                use_heun=self.liquid_use_heun,
                                dtype=self.dtype,
                            )
                        )
                    layers.append(
                        BasicLayer(
                            in_chans=ch,
                            embed_dim=swin_embed_dim,
                            num_heads=num_heads if num_head_channels == -1 else swin_embed_dim // num_head_channels,
                            window_size=window_size,
                            depth=swin_depth,
                            img_size=ds,
                            patch_size=1,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=True,
                            qk_scale=None,
                            drop=dropout,
                            attn_drop=0.0,
                            drop_path=0.0,
                            use_checkpoint=False,
                            norm_layer=normalization,
                            patch_norm=patch_norm,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds //= 2

        middle_layers = [
            ResBlock(ch, time_embed_dim, dropout, dims=dims, use_scale_shift_norm=use_scale_shift_norm)
        ]
        if self.use_liquid:
            middle_layers.append(
                LiquidWindowODE(
                    dim=ch,
                    window_size=window_size,
                    steps=self.liquid_steps,
                    dt=self.liquid_dt,
                    heads=self.liquid_heads,
                    dim_head=self.liquid_dim_head,
                    mlp_ratio=self.liquid_mlp_ratio,
                    tau_init=self.liquid_tau_init,
                    use_heun=self.liquid_use_heun,
                    dtype=self.dtype,
                )
            )
        middle_layers.append(
            BasicLayer(
                in_chans=ch,
                embed_dim=swin_embed_dim,
                num_heads=num_heads if num_head_channels == -1 else swin_embed_dim // num_head_channels,
                window_size=window_size,
                depth=swin_depth,
                img_size=ds,
                patch_size=1,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                qk_scale=None,
                drop=dropout,
                attn_drop=0.0,
                drop_path=0.0,
                use_checkpoint=False,
                norm_layer=normalization,
                patch_norm=patch_norm,
            )
        )
        middle_layers.append(
            ResBlock(ch, time_embed_dim, dropout, dims=dims, use_scale_shift_norm=use_scale_shift_norm)
        )
        self.middle_block = TimestepEmbedSequential(*middle_layers)
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions and i == 0:
                    if self.use_liquid:
                        layers.append(
                            LiquidWindowODE(
                                dim=ch,
                                window_size=window_size,
                                steps=self.liquid_steps,
                                dt=self.liquid_dt,
                                heads=self.liquid_heads,
                                dim_head=self.liquid_dim_head,
                                mlp_ratio=self.liquid_mlp_ratio,
                                tau_init=self.liquid_tau_init,
                                use_heun=self.liquid_use_heun,
                                dtype=self.dtype,
                            )
                        )
                    layers.append(
                        BasicLayer(
                            in_chans=ch,
                            embed_dim=swin_embed_dim,
                            num_heads=num_heads if num_head_channels == -1 else swin_embed_dim // num_head_channels,
                            window_size=window_size,
                            depth=swin_depth,
                            img_size=ds,
                            patch_size=1,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=True,
                            qk_scale=None,
                            drop=dropout,
                            attn_drop=0.0,
                            drop_path=0.0,
                            use_checkpoint=False,
                            norm_layer=normalization,
                            patch_norm=patch_norm,
                        )
                    )
                if level and i == num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds *= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            conv_nd(dims, input_ch, out_channels, 3, padding=1),
        )
    def forward(self, x, timesteps, x0=None, mask=None):
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels)).type(self.dtype)
        if x0 is not None:
            assert self.cond_x0
            if mask is not None:
                assert self.cond_mask
                x0 = torch.cat([x0, mask], dim=1)
            x0 = self.feature_extractor(x0.type(self.dtype))
            x = torch.cat([x, x0], dim=1)
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        out = self.out(h)
        return out