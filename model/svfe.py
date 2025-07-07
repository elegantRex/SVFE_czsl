from itertools import product
from turtle import shape
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import argparse
import clip
from collections import OrderedDict
from clip_modules.model_loader import load
from model.common import *
import numpy as np
import torch.fft
import math
from functools import partial
from timm.models.layers import DropPath, trunc_normal_

class FourierSpectralLayer(nn.Module):
    def __init__(self, dim, h=32, w=17):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        
        B, C, H, W = x.size()       
        x = x.view(B, H * W, C)
        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size
        x = x.view(B, a, b, C)
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')
        x = x.reshape(B, N, C)
        B, N, C = x.size()
        H = 32
        W = 32
        x = x.reshape(B, C, H, W)
        return x



# LVC
class Encoding(nn.Module):
    def __init__(self, in_channels, num_codes):
        super(Encoding, self).__init__()
        self.in_channels, self.num_codes = in_channels, num_codes
        num_codes = 64
        std = 1. / ((num_codes * in_channels) ** 0.5)
        self.codewords = nn.Parameter(
            torch.empty(num_codes, in_channels, dtype=torch.float).uniform_(-std, std), requires_grad=True)
        self.scale = nn.Parameter(torch.empty(num_codes, dtype=torch.float).uniform_(-1, 0), requires_grad=True)

    @staticmethod
    def scaled_l2(x, codewords, scale):
        num_codes, in_channels = codewords.size()
        b = x.size(0)
        expanded_x = x.unsqueeze(2).expand((b, x.size(1), num_codes, in_channels))
        reshaped_codewords = codewords.view((1, 1, num_codes, in_channels))
        reshaped_scale = scale.view((1, 1, num_codes))
        scaled_l2_norm = reshaped_scale * (expanded_x - reshaped_codewords).pow(2).sum(dim=3)
        return scaled_l2_norm

    @staticmethod
    def aggregate(assignment_weights, x, codewords):
        num_codes, in_channels = codewords.size()
        reshaped_codewords = codewords.view((1, 1, num_codes, in_channels))
        b = x.size(0)
        expanded_x = x.unsqueeze(2).expand((b, x.size(1), num_codes, in_channels))

        assignment_weights = assignment_weights.unsqueeze(3)  
        encoded_feat = (assignment_weights * (expanded_x - reshaped_codewords)).sum(1)
        return encoded_feat

    def forward(self, x):
        assert x.dim() == 4 and x.size(1) == self.in_channels
        b, in_channels, w, h = x.size()
        x = x.view(b, self.in_channels, -1).transpose(1, 2).contiguous()
        assignment_weights = F.softmax(self.scaled_l2(x, self.codewords, self.scale), dim=2)
        encoded_feat = self.aggregate(assignment_weights, x, self.codewords)
        return encoded_feat

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, res_conv=False, act_layer=nn.ReLU, groups=1,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), drop_block=None, drop_path=None):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        expansion = 4
        c = out_channels // expansion

        self.conv1 = nn.Conv2d(in_channels, c, kernel_size=1, stride=1, padding=0, bias=False)  # [64, 256, 1, 1]
        self.bn1 = norm_layer(c)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(c, c, kernel_size=3, stride=stride, groups=groups, padding=1, bias=False)
        self.bn2 = norm_layer(c)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(c, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(out_channels)
        self.act3 = act_layer(inplace=True)

        if res_conv:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
            self.residual_bn = norm_layer(out_channels)

        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x, return_x_2=True):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x) 
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x2 = self.act2(x)

        x = self.conv3(x2)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.res_conv:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)

        x += residual
        x = self.act3(x)

        if return_x_2:
            return x, x2
        else:
            return x


class Mean(nn.Module):
    def __init__(self, dim, keep_dim=False):
        super(Mean, self).__init__()
        self.dim = dim
        self.keep_dim = keep_dim

    def forward(self, input):
        return input.mean(self.dim, self.keep_dim)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GroupNorm(nn.GroupNorm):
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)
def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class BaseConv(nn.Module):
    def __init__(
            self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Depthwise(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )
    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)



class LocalVisualCenter(nn.Module):
    def __init__(self, in_channels, out_channels, num_codes, channel_ratio=0.25, base_channel=64):
        super(LocalVisualCenter, self).__init__()
        self.out_channels = out_channels
        self.num_codes = num_codes
        num_codes = 64

        self.conv_1 = ConvBlock(in_channels=in_channels, out_channels=in_channels, res_conv=True, stride=1)

        self.LVC = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            Encoding(in_channels=in_channels, num_codes=num_codes),
            nn.BatchNorm1d(num_codes),
            nn.ReLU(inplace=True),
            Mean(dim=1))
        self.fc = nn.Sequential(nn.Linear(in_channels, in_channels), nn.Sigmoid())

    def forward(self, x):
        x = self.conv_1(x, return_x_2=False)
        en = self.LVC(x)
        gam = self.fc(en)
        b, in_channels, _, _ = x.size()
        y = gam.view(b, in_channels, 1, 1)
        x = F.relu_(x + x * y)
        return x


class LightMultiLayerPerceptron(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu",
                 mlp_ratio=4., drop=0., act_layer=nn.GELU,
                 use_layer_scale=True, layer_scale_init_value=1e-5, drop_path=0.,
                 norm_layer=GroupNorm):  # act_layer=nn.GELU,
        super().__init__()
        self.dw = Depthwise(in_channels, out_channels, ksize=1, stride=1, act="silu")
        self.linear = nn.Linear(out_channels, out_channels)  # learnable position embedding
        self.out_channels = out_channels

        self.norm1 = norm_layer(in_channels)
        self.norm2 = norm_layer(in_channels)

        mlp_hidden_dim = int(in_channels * mlp_ratio)
        self.mlp = Mlp(in_features=in_channels, hidden_features=mlp_hidden_dim, act_layer=nn.GELU,
                       drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()

        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((out_channels)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((out_channels)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.dw(self.norm1(x)))
            x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.dw(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class FeatureFusion(nn.Module):
    def __init__(self, in_channels, out_channels, channel_ratio=4, base_channel=16):
        super().__init__()
        expansion = 2
        ch = out_channels * expansion
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=7, stride=1, padding=3,
                               bias=False) 
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1) 
        
        self.lvc = LocalVisualCenter(in_channels=in_channels, out_channels=out_channels, num_codes=64)  
        self.l_MLP = LightMultiLayerPerceptron(in_channels, out_channels, ksize=1, stride=1, act="silu", act_layer=nn.GELU,
                                   mlp_ratio=4., drop=0.,
                                   use_layer_scale=True, layer_scale_init_value=1e-5, drop_path=0.,
                                   norm_layer=GroupNorm)
        self.cnv1 = nn.Conv2d(ch, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        
        x = x.to(torch.float32)
        x1 = self.maxpool(self.act1(self.bn1(self.conv1(x))))
        x_lvc = self.lvc(x1)
        x_lmlp = self.l_MLP(x1)
        # concat
        x = torch.cat((x_lvc, x_lmlp), dim=1)
        x = self.cnv1(x)
        return x

class Unfold(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()

        self.kernel_size = kernel_size

        weights = torch.eye(kernel_size ** 2)
        weights = weights.reshape(kernel_size ** 2, 1, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)

    def forward(self, x):
        b, c, h, w = x.shape
        
        x = x.to(torch.float32)
        
        x = F.conv2d(x.reshape(b * c, 1, h, w), self.weights, stride=1, padding=self.kernel_size // 2)
        return x.reshape(b, c * 9, h * w)


class Fold(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()

        self.kernel_size = kernel_size

        weights = torch.eye(kernel_size ** 2)
        weights = weights.reshape(kernel_size ** 2, 1, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)

    def forward(self, x):
        b, _, h, w = x.shape
        
        x = x.to(torch.float32)
        x = F.conv_transpose2d(x, self.weights, stride=1, padding=self.kernel_size // 2)
        return x



class Refine(nn.Module):
    def __init__(self, dim, window_size=None, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.window_size = window_size

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        
        x = x.to(torch.float32)
        B, C, H, W = x.shape
        N = H * W
        q, k, v = self.qkv(x).reshape(B, self.num_heads, C // self.num_heads * 3, N).chunk(3,
                                                                                           dim=2) 
        attn = (k.transpose(-1, -2) @ q) * self.scale

        attn = attn.softmax(dim=-2)  # (B, h, N, N)
        attn = self.attn_drop(attn)

        x = (v @ attn).reshape(B, C, H, W)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SuperpixelIntegration(nn.Module):
    def __init__(self, dim, pixel_size, n_iter=1, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):
        super().__init__()

        self.n_iter = n_iter
        self.pixel_size = pixel_size

        self.scale = dim ** - 0.5

        self.unfold = Unfold(3)
        self.fold = Fold(3)

        self.pixel_refine = Refine(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                       attn_drop=attn_drop, proj_drop=proj_drop)

    def forward(self, x):
        B, C, H0, W0 = x.shape
        h, w = self.pixel_size
        pad_l = pad_t = 0
        pad_r = (w - W0 % w) % w
        pad_b = (h - H0 % h) % h
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
        _, _, H, W = x.shape
        hh, ww = H // h, W // w
        pixel_features = F.adaptive_avg_pool2d(x, (hh, ww))  # (B, C, hh, ww)
        pixel_features = x.reshape(B, C, hh, h, ww, w).permute(0, 2, 4, 3, 5, 1).reshape(B, hh * ww, h * w, C)
        with torch.no_grad():
            for idx in range(self.n_iter):
                pixel_features = self.unfold(pixel_features)  # (B, C*9, hh*ww)
                pixel_features = pixel_features.transpose(1, 2).reshape(B, hh * ww, C, 9)
                affinity_matrix = pixel_features.float() @ pixel_features.float() * self.scale  # (B, hh*ww, h*w, 9)
                affinity_matrix = affinity_matrix.softmax(-1)  # (B, hh*ww, h*w, 9)
                affinity_matrix_sum = affinity_matrix.sum(2).transpose(1, 2).reshape(B, 9, hh, ww)
                affinity_matrix_sum = self.fold(affinity_matrix_sum)
                if idx < self.n_iter - 1:
                    pixel_features = pixel_features.transpose(-1, -2) @ affinity_matrix  # (B, hh*ww, C, 9)
                    pixel_features = self.fold(pixel_features.permute(0, 2, 3, 1).reshape(B * C, 9, hh, ww)).reshape(
                        B, C, hh, ww)
                    pixel_features = pixel_features / (affinity_matrix_sum + 1e-12)  # (B, C, hh, ww)
        pixel_features = pixel_features.float()
        pixel_features = pixel_features.float()
        affinity_matrix =  affinity_matrix.float()
        pixel_features = pixel_features.transpose(-1, -2) @ affinity_matrix  # (B, hh*ww, C, 9)
        pixel_features = self.fold(pixel_features.permute(0, 2, 3, 1).reshape(B * C, 9, hh, ww)).reshape(B, C, hh, ww)
        pixel_features = pixel_features / (affinity_matrix_sum.detach() + 1e-12)  # (B, C, hh, ww)
        pixel_features = self.pixel_refine(pixel_features)
        pixel_features = self.unfold(pixel_features)  # (B, C*9, hh*ww)
        pixel_features = pixel_features.transpose(1, 2).reshape(B, hh * ww, C, 9)  # (B, hh*ww, C, 9)
        pixel_features = pixel_features @ affinity_matrix.transpose(-1, -2)  # (B, hh*ww, C, h*w)
        pixel_features = pixel_features.reshape(B, hh, ww, C, h, w).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)
        if pad_r > 0 or pad_b > 0:
            pixel_features = pixel_features[:, :, :H0, :W0]

        return pixel_features

class SVFE(nn.Module):
    def __init__(self, config, attributes, classes, offset):
        super().__init__()
        clip_model, _ = load(config.clip_model, context_length=config.context_length)
        self.clip = clip_model
        self.config = config
        self.attributes = attributes
        self.classes = classes
        self.attr_dropout = nn.Dropout(config.attr_dropout)
        self.token_ids, self.soft_att_obj, ctx_vectors = self.construct_soft_prompt()
        self.offset = offset
        self.enable_pos_emb = True
        dtype = None
        if dtype is None:
            self.dtype = torch.float16
        else:
            self.dtype = dtype
        self.text_encoder = CustomTextEncoder(self.clip, self.dtype)
        for p in self.parameters():
            p.requires_grad=False

        self.soft_att_obj = nn.Parameter(self.soft_att_obj)
        self.soft_prompt = nn.Parameter(ctx_vectors).cuda()
        self.fusion = FusionTextImageBlock(config.width_img, config.width_txt, len(self.attributes), len(self.classes), config.SA_K, context_length=self.config.context_length, fusion=self.config.fusion)
        self.weight = config.res_w
        self.lf = FeatureFusion(257, 257)
        self.sis = SuperpixelIntegration(257, pixel_size=[8,8])
        self.fsl = FourierSpectralLayer(257)


    def construct_soft_prompt(self):
        token_ids = clip.tokenize("a photo of x x",
                              context_length=self.config.context_length).cuda()

        tokenized = torch.cat(
            [
                clip.tokenize(tok, context_length=self.config.context_length)
                for tok in self.attributes + self.classes
            ]
        )
        orig_token_embedding = self.clip.token_embedding(tokenized.cuda())

        # with torch.no_grad():
        soft_att_obj = torch.zeros(
            (len(self.attributes) + len(self.classes), orig_token_embedding.size(-1)),
        )
        for idx, rep in enumerate(orig_token_embedding):
            eos_idx = tokenized[idx].argmax()
            soft_att_obj[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)

        ctx_init = "a photo of "
        n_ctx = len(ctx_init.split())
        prompt = clip.tokenize(ctx_init,
                            context_length=self.config.context_length).cuda()
        with torch.no_grad():
            embedding = self.clip.token_embedding(prompt)
        ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
        return token_ids, soft_att_obj, ctx_vectors


    def construct_token_tensors(self, pair_idx):
        attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]
        class_token_ids = self.token_ids.repeat(len(pair_idx), 1)
        token_tensor = self.clip.token_embedding(
            class_token_ids.cuda()
        ).type(self.clip.dtype)
        soft_att_obj = self.attr_dropout(self.soft_att_obj)
        eos_idx = int(self.token_ids[0].argmax())
        token_tensor[:, eos_idx - 2, :] = soft_att_obj[
            attr_idx
        ].type(self.clip.dtype)
        token_tensor[:, eos_idx - 1, :] = soft_att_obj[
            obj_idx + self.offset
        ].type(self.clip.dtype)

        # adding the correct learnable context
        token_tensor[
            :, 1 : len(self.soft_prompt) + 1, :
        ] = self.soft_prompt.type(self.clip.dtype)
        return token_tensor



    def visual(self, x: torch.Tensor):
        x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        img_feature = self.clip.visual.transformer(x)

        x = img_feature.permute(1, 0, 2)  # LND -> NLD

        x = self.clip.visual.ln_post(x[:, 0, :])
        if self.clip.visual.proj is not None:
            x = x @ self.clip.visual.proj
        return x, img_feature


    def ft_to_logit(self, img, txt):
        img_feature = img.permute(1, 0, 2)  # LND -> NLD

        img_feature = self.clip.visual.ln_post(img_feature[:, 0, :])
        if self.clip.visual.proj is not None:
            img_feature = img_feature @ self.clip.visual.proj
        
        if self.config.fusion in ["BiFusion", "img2txt"]: 
            txt_feature = txt.permute(0, 2, 1, 3)
            txt_feature = self.text_encoder.ln_final(txt_feature)
            txt_tf = (
                txt_feature[
                    :, torch.arange(txt_feature.shape[1]), self.token_ids.argmax(dim=-1)
                ]  # POS of <EOS>
                @ self.text_encoder.text_projection
            )
        else:
            txt_feature = txt.permute(1, 0, 2)
            txt_feature = self.text_encoder.ln_final(txt_feature)
            txt_tf = (
                txt_feature[
                    torch.arange(txt_feature.shape[0]), self.token_ids.argmax(dim=-1)
                ]  # POS of <EOS>
                @ self.text_encoder.text_projection
            )
        return img_feature, txt_tf

    def decompose_logits(self, logits, idx):
        att_idx, obj_idx = idx[:, 0].cpu().numpy(), idx[:, 1].cpu().numpy()
        logits_att = torch.zeros(logits.shape[0], len(self.attributes)).cuda()
        logits_obj = torch.zeros(logits.shape[0], len(self.classes)).cuda()
        for i in range(len(self.attributes)):
            logits_att[:, i] = logits[:, np.where(att_idx==i)[0]].mean(-1)
        for i in range(len(self.classes)):
            logits_obj[:, i] = logits[:, np.where(obj_idx==i)[0]].mean(-1)        
        return logits_att, logits_obj


    def forward(self, batch_img, idx):
        b = batch_img.shape[0]
        l, _ = idx.shape
        batch_img, img_ft = self.visual(batch_img.type(self.clip.dtype))   ## bs * 768
        C, B, N = img_ft.size()
        H, W = 32, 32
        x_new = img_ft.view(C, B, H, W).permute(1, 0, 2, 3)
        x_new = self.sis(x_new)
        x_new = self.fsl(x_new)
        x_new = self.lf(x_new)
        B, C, H, W = x_new.size()
        img_ft = x_new.view(C, B, H*W)
        token_tensors = self.construct_token_tensors(idx)
        text_features, text_ft = self.text_encoder(
            self.token_ids,
            token_tensors,
            enable_pos_emb=self.enable_pos_emb,
        )  
        batch_img_soft_prompt = batch_img / batch_img.norm(dim=-1, keepdim=True)
        text_features_soft_prompt = text_features / text_features.norm(dim=-1, keepdim=True)
        img_ft, text_ft = self.fusion(img_ft.type(torch.float), text_ft.type(torch.float), idx, b)
        img_ft, text_ft = self.ft_to_logit(img_ft.type(self.clip.dtype), text_ft.type(self.clip.dtype))
        batch_img = self.weight * batch_img + (1 - self.weight) * img_ft
        normalized_img = batch_img / batch_img.norm(dim=-1, keepdim=True)
        if self.config.fusion in ["BiFusion", "img2txt"]: 
            text_features = self.weight * text_features.repeat(b, 1, 1) + (1 - self.weight) * text_ft
        else:
            text_features = self.weight * text_features + (1 - self.weight) * text_ft
        idx_text_features = text_features / text_features.norm(
            dim=-1, keepdim=True
        )
        if self.config.fusion in ["BiFusion", "img2txt"]: 
            logits = (
                self.clip.logit_scale.exp()
                * normalized_img.unsqueeze(1)
                @ idx_text_features.permute(0,2,1)
            ).squeeze()     ###     48 * 1262
        else:
            logits = (
                self.clip.logit_scale.exp()
                * normalized_img
                @ idx_text_features.t()
            )   

        logits_soft_prompt = (
            self.clip.logit_scale.exp()
            * batch_img_soft_prompt
            @ text_features_soft_prompt.t()
        )     

        logits_att, logits_obj = self.decompose_logits(logits_soft_prompt, idx)

        return (logits, logits_att, logits_obj, logits_soft_prompt)
