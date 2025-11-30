import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from einops import rearrange
import numbers


def get_relative_position_index(win_h, win_w):
    coords = torch.stack(torch.meshgrid([torch.arange(win_h), torch.arange(win_w)], indexing='ij'))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += win_h - 1
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w - 1
    return relative_coords.sum(-1)

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(UpsampleConvLayer, self).__init__()
        self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2, output_padding=kernel_size//2)

    def forward(self, x):
        out = self.conv2d(x)
        return out

class Du_Att(nn.Module):
    def __init__(self, dim, num_heads, window_size):
        super(Du_Att, self).__init__()
        self.norm1 = LayerNorm(dim)
        self.attn1 = CCA(dim//2, num_heads)
        self.attn2 = TRA(dim//2, num_heads, window_size)

    def forward(self, x):
        x0 = self.norm1(x)
        b, c, h, w = x.shape
        x1 = x0[:, :c // 2, :, :]
        x2 = x0[:, c // 2:, :, :]
        x11 = self.attn1(x1)
        x22 = self.attn2(x2)
        xout = torch.cat((x11, x22), 1)
        x11 = xout + x
        return x11

class TRA(nn.Module):
    def __init__(self, dim, num_head, window_size=4):
        super(TRA, self).__init__()
        self.dim = dim
        self.num_head = num_head
        self.window_size = window_size
        self.window_area = window_size ** 2
        self.qkv = nn.Conv2d(dim, 3 * dim, 1, bias=True)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_head))
        self.register_buffer("relative_position_index", get_relative_position_index(window_size, window_size))
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_head, 1, 1))), requires_grad=True)
        self.proj = nn.Conv2d(dim, dim, 1)

    def _get_rel_pos_bias(self):
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_area, self.window_area, -1)
        return relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)

    def forward(self, x0):
        qkv_0 = self.qkv(x0)
        qkv = rearrange(qkv_0, 'b (l c) h w -> b l c h w', l=self.num_head)
        B, L, C, H, W = qkv.size()
        q, k, v = rearrange(
            qkv,
            'b l c (h wh) (w ww) -> (b h w) l (wh ww) c',
            wh=self.window_size, ww=self.window_size
        ).chunk(3, dim=-1)
        attn0 = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(self.logit_scale, max=math.log(1. / 0.01)).exp()
        attn1 = attn0 * logit_scale + self._get_rel_pos_bias()
        attn = F.softmax(attn1, dim=-1)
        x = attn @ v
        x = rearrange(
            x,
            '(b h w) l (wh ww) c -> b (l c) (h wh) (w ww)',
            h=H // self.window_size, w=W // self.window_size, wh=self.window_size
        )
        x_out = self.proj(x)
        return x_out

class CCA(nn.Module):
    def __init__(self, dim, num_heads, ifBox=True):
        super(CCA, self).__init__()
        self.factor = num_heads
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)

    def Calculate_attn(self, q, k, v):
        b, c = q.shape[:2]
        hw = q.shape[-1] // self.factor
        shape_ori = "b (head c) (factor hw)"
        shape_tar = "b head (c factor) hw"
        q = rearrange(q, f'{shape_ori} -> {shape_tar}', factor=self.factor, hw=hw, head=self.num_heads)
        k = rearrange(k, f'{shape_ori} -> {shape_tar}', factor=self.factor, hw=hw, head=self.num_heads)
        v = rearrange(v, f'{shape_ori} -> {shape_tar}', factor=self.factor, hw=hw, head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = torch.softmax(attn, dim=-1)
        out = (attn @ v)
        out = rearrange(out, f'{shape_tar} -> {shape_ori}', factor=self.factor, hw=hw, head=self.num_heads)
        return out

    def forward(self, x):
        b, c, h, w = x.shape
        x_sort, idx_h = x.sort(-2)
        x_sort, idx_w = x_sort.sort(-1)
        x = x_sort
        qkv = self.qkv_dwconv(self.qkv(x))
        q1, k1, v = qkv.chunk(3, dim=1)
        v, idx = v.view(b, c, -1).sort(dim=-1)
        q1 = torch.gather(q1.view(b, c, -1), dim=2, index=idx)
        k1 = torch.gather(k1.view(b, c, -1), dim=2, index=idx)
        out1 = self.Calculate_attn(q1, k1, v)
        out1 = torch.scatter(out1, 2, idx, out1).view(b, c, h, w)
        out = out1
        out_replace = torch.scatter(out, -1, idx_w, out)
        out_replace = torch.scatter(out_replace, -2, idx_h, out_replace)
        return out_replace

class CTDF(nn.Module):
    def __init__(self, dim, ffn_expansion_factor):
        super(CTDF, self).__init__()
        self.norm1 = LayerNorm(dim)
        self.ffn = ChannelSpatialMixer(dim, ffn_expansion_factor)

    def forward(self, x):
        x1 = self.norm1(x)
        x11 = self.ffn(x1)
        x11 = x11 + x
        return x11

class ChannelSpatialMixer(nn.Module):
    def __init__(self, dim, ffn_expansion_factor):
        super(ChannelSpatialMixer, self).__init__()
        hidden_features = int(dim)
        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, stride=1, padding=0)
        self.dwconv3x3 = nn.Conv2d(hidden_features, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features)
        self.relu3 = nn.PReLU()
        self.dwconv5x5 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=5, stride=1, padding=2, groups=hidden_features)
        self.relu5 = nn.ReLU()
        self.dwconv5x5_1 = nn.Conv2d(hidden_features, hidden_features * 2, kernel_size=5, stride=1, padding=2, groups=hidden_features)
        self.relu5_1 = nn.ReLU()
        self.dwconv3x3_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features)
        self.relu3_1 = nn.PReLU()
        self.AFF = AdaptiveFeatureFusion(dim)

    def forward(self, x):
        x0 = self.project_in(x)
        x1 = self.relu3(self.dwconv3x3(x0))
        x1 = self.relu3_1(self.dwconv5x5(x1))
        x1 = x1 + x
        x2 = self.relu5(self.dwconv5x5_1(x0))
        x2 = self.relu5_1(self.dwconv3x3_1(x2))
        x2 = x2 + x
        x3 = self.AFF(x1, x2)
        return x3

class AdaptiveFeatureFusion(nn.Module):
    def __init__(self, features, G=32, r=2, L=32):
        super(AdaptiveFeatureFusion, self).__init__()
        d = max(int(features / r), L)
        self.features = features
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gap1 = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(features * 2, d, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.fc1 = nn.Conv2d(d, features, kernel_size=1, stride=1)
        self.fc2 = nn.Conv2d(d, features, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=1)
        self.conv3 = nn.Conv2d(features, features, 3, 1, 1)

    def forward(self, x1, x2):
        feats_U = x1 + x2
        feats_U = self.conv3(feats_U)
        feats_S = self.gap(feats_U)
        feats_S1 = self.gap1(feats_U)
        feats_S = torch.cat([feats_S, feats_S1], dim=1)
        feats_Z = self.fc(feats_S)
        attention_vector1 = self.fc1(feats_Z)
        attention_vector2 = self.fc2(feats_Z)
        attention_vectors = torch.cat([attention_vector1, attention_vector2], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        weight1 = attention_vectors[:, :self.features, :, :]
        weight2 = attention_vectors[:, self.features:, :, :]
        feats_V = x1 * weight1 + x2 * weight2
        return feats_V

class DuAF(nn.Module):
    def __init__(self, dim=32, heads=[2, 2, 4], num_blocks=[2, 4, 4], ffn_expansion_factor=2, window_size=[8, 4, 2], LayerNorm_type='WithBias'):
        super(DuAF, self).__init__()
        self.relu = nn.PReLU()
        self.DualAttn1 = nn.Sequential(*[Du_Att(dim=dim, num_heads=heads[0], window_size=window_size[0]) for _ in range(num_blocks[0])])
        self.CTDF1 = nn.Sequential(*[CTDF(dim=dim, ffn_expansion_factor=ffn_expansion_factor) for _ in range(num_blocks[0])])
        self.DualAttn2 = nn.Sequential(*[Du_Att(dim=int(dim * 2 ** 1), num_heads=heads[1], window_size=window_size[1]) for i in range(num_blocks[1])])
        self.CTDF2 = nn.Sequential(*[CTDF(dim=int(dim * 2 ** 1), ffn_expansion_factor=ffn_expansion_factor) for i in range(num_blocks[1])])
        self.DualAttn3 = nn.Sequential(*[Du_Att(dim=int(dim * 2 ** 2), num_heads=heads[2], window_size=window_size[2]) for i in range(num_blocks[2])])
        self.CTDF3 = nn.Sequential(*[CTDF(dim=int(dim * 2 ** 2), ffn_expansion_factor=ffn_expansion_factor) for i in range(num_blocks[2])])
        self.DualAttnUp3 = nn.Sequential(*[Du_Att(dim=int(dim * 2 ** 2), num_heads=heads[2], window_size=window_size[2]) for i in range(num_blocks[2])])
        self.CTDFUp3 = nn.Sequential(*[CTDF(dim=int(dim * 2 ** 2), ffn_expansion_factor=ffn_expansion_factor) for i in range(num_blocks[2])])
        self.DualAttnUp2 = nn.Sequential(*[Du_Att(dim=int(dim * 2 ** 1), num_heads=heads[1], window_size=window_size[1]) for i in range(num_blocks[1])])
        self.CTDFUp2 = nn.Sequential(*[CTDF(dim=int(dim * 2 ** 1), ffn_expansion_factor=ffn_expansion_factor) for i in range(num_blocks[1])])
        self.DualAttnUp1 = nn.Sequential(*[Du_Att(dim=int(dim), num_heads=heads[0], window_size=window_size[0]) for i in range(num_blocks[0])])
        self.CTDFUp1 = nn.Sequential(*[CTDF(dim=int(dim), ffn_expansion_factor=ffn_expansion_factor) for i in range(num_blocks[0])])
        self.Downsample1 = ConvLayer(dim << 0, dim << 1, 3, 2)
        self.Downsample2 = ConvLayer(dim << 1, dim << 2, 3, 2)
        self.Downsample3 = ConvLayer(dim << 2, dim << 2, 3, 1)
        self.FeatureRefine3 = ConvLayer(dim << 2, dim << 2, 3, 1)
        self.Upsample2 = UpsampleConvLayer(dim << 2, dim << 1, kernel_size=3, stride=2)
        self.Upsample1 = UpsampleConvLayer(dim << 1, dim << 0, kernel_size=3, stride=2)
        self.InputConv = nn.Conv2d(3, dim, kernel_size=3, stride=1, padding=1)
        self.OutputConv = nn.Conv2d(dim, 3, kernel_size=3, stride=1, padding=1)
        self.Output44 = nn.Conv2d(dim * 4, 3, kernel_size=3, stride=1, padding=1)
        self.Output22 = nn.Conv2d(dim * 2, 3, kernel_size=3, stride=1, padding=1)
        self.Output11 = nn.Conv2d(dim, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x1 = self.InputConv(x)
        x2_1 = self.DualAttn1(x1)
        x2_2 = self.CTDF1(x2_1)
        x2 = self.Downsample1(x2_2)
        x3_1 = self.DualAttn2(x2)
        x3_2 = self.CTDF2(x3_1)
        x3 = self.Downsample2(x3_2)
        x4_1 = self.DualAttn3(x3)
        x4_2 = self.CTDF3(x4_1)
        x4 = self.Downsample3(x4_2)
        x8 = self.FeatureRefine3(x4)
        x8_1 = self.DualAttnUp3(x8)
        x8_2 = self.CTDFUp3(x8_1)
        out_4 = self.Output44(x8_2)
        x10 = self.Upsample2(x3 + x8_2)
        x10_1 = self.DualAttnUp2(x10)
        x10_2 = self.CTDFUp2(x10_1)
        out_2 = self.Output22(x10_2)
        x11 = self.Upsample1(x2 + x10_2)
        x11_1 = self.DualAttnUp1(x11)
        x11_2 = self.CTDFUp1(x11_1)
        out_1 = self.Output11(x11_2)
        x12 = self.OutputConv(x1 + x11_2)
        return x12, out_1, out_2, out_4