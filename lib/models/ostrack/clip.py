from functools import partial
from collections import OrderedDict
import math
from typing import Callable, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from itertools import repeat
import collections.abc
import logging

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)

class LayerNormFp32(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16 (by casting to float32 and back)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x.to(torch.float32), self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class QuickGELU(nn.Module):
    # NOTE This is slower than nn.GELU or nn.SiLU and uses more GPU memory
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    
class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
    
class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token

    def forward(self, x):
        if not self.training or self.prob == 0.:
            return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        return x
    
class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=True,
            scaled_cosine=False,
            scale_heads=False,
            logit_scale_max=math.log(1. / 0.01),
            attn_drop=0.,
            proj_drop=0.
    ):
        super().__init__()
        self.scaled_cosine = scaled_cosine
        self.scale_heads = scale_heads
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.logit_scale_max = logit_scale_max

        # keeping in_proj in this form (instead of nn.Linear) to match weight scheme of original
        self.in_proj_weight = nn.Parameter(torch.randn((dim * 3, dim)) * self.scale)
        if qkv_bias:
            self.in_proj_bias = nn.Parameter(torch.zeros(dim * 3))
        else:
            self.in_proj_bias = None

        if self.scaled_cosine:
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        else:
            self.logit_scale = None
        self.attn_drop = nn.Dropout(attn_drop)
        if self.scale_heads:
            self.head_scale = nn.Parameter(torch.ones((num_heads, 1, 1)))
        else:
            self.head_scale = None
        self.out_proj = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        L, N, C = x.shape
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        q = q.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
        k = k.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
        v = v.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)

        if self.logit_scale is not None:
            attn = torch.bmm(F.normalize(q, dim=-1), F.normalize(k, dim=-1).transpose(-1, -2))
            logit_scale = torch.clamp(self.logit_scale, max=self.logit_scale_max).exp()
            attn = attn.view(N, self.num_heads, L, L) * logit_scale
            attn = attn.view(-1, L, L)
        else:
            q = q * self.scale
            attn = torch.bmm(q, k.transpose(-1, -2))

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
                new_attn_mask.masked_fill_(attn_mask, float("-inf"))
                attn_mask = new_attn_mask
            attn += attn_mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = torch.bmm(attn, v)
        if self.head_scale is not None:
            x = x.view(N, self.num_heads, L, C) * self.head_scale
            x = x.view(-1, L, C)
        x = x.transpose(0, 1).reshape(L, N, C)
        x = self.out_proj(x)
        x = self.out_drop(x)
        return x


class AttentionalPooler(nn.Module):
    def __init__(
            self,
            d_model: int,
            context_dim: int,
            n_head: int = 8,
            n_queries: int = 256,
            norm_layer: Callable = LayerNorm
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_queries, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_head, kdim=context_dim, vdim=context_dim)
        self.ln_q = norm_layer(d_model)
        self.ln_k = norm_layer(context_dim)

    def forward(self, x: torch.Tensor):
        x = self.ln_k(x).permute(1, 0, 2)  # NLD -> LND
        N = x.shape[1]
        q = self.ln_q(self.query)
        out = self.attn(self._repeat(q, N), x, x, need_weights=False)[0]
        return out.permute(1, 0, 2)  # LND -> NLD

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)


class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            is_cross_attention: bool = False,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        if is_cross_attention:
            self.ln_1_kv = norm_layer(d_model)

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def attention(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x

        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        return self.attn(
            q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask
        )[0]

    def forward(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None

        x = q_x + self.ls_1(self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


class CustomResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            scale_cosine_attn: bool = False,
            scale_heads: bool = False,
            scale_attn: bool = False,
            scale_fc: bool = False,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = Attention(
            d_model, n_head,
            scaled_cosine=scale_cosine_attn,
            scale_heads=scale_heads,
        )
        self.ln_attn = norm_layer(d_model) if scale_attn else nn.Identity()
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ('ln', norm_layer(mlp_width) if scale_fc else nn.Identity()),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.ls_1(self.ln_attn(self.attn(self.ln_1(x), attn_mask=attn_mask)))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Module):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                width, heads, mlp_ratio, ls_init_value=ls_init_value, act_layer=act_layer, norm_layer=norm_layer)
            for _ in range(layers)
        ])

    def get_cast_dtype(self) -> torch.dtype:
        if hasattr(self.resblocks[0].mlp.c_fc, 'int8_original_dtype'):
            return self.resblocks[0].mlp.c_fc.int8_original_dtype
        return self.resblocks[0].mlp.c_fc.weight.dtype

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                x = checkpoint(r, x, None, None, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

class VisionTransformer(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
            self,
            image_size: int,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float,
            ls_init_value: float = None,
            global_average_pool: bool = False,
            attentional_pool: bool = False,
            patch_dropout: float = 0.,
            input_patchnorm: bool = False,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            output_tokens: bool = False,
            add_cls_token: bool = True,
    ):
        super().__init__()
        self.output_tokens = output_tokens
        self.add_cls_token = add_cls_token
        image_height, image_width = self.image_size = to_2tuple(image_size)
        self.patch_size = patch_size
        self.grid_size = (image_height // self.patch_size, image_width // self.patch_size)

        self.logger = logging.getLogger()

        self.embed_dim = width

        # whether to layernorm each patch, as done in dual patchnorm paper - https://arxiv.org/abs/2302.01327v1
        self.input_patchnorm = input_patchnorm

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        # class embeddings and positional embeddings
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width))

        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0. else nn.Identity()

        self.ln_pre = norm_layer(width)
        self.transformer = Transformer(
            width,
            layers,
            heads,
            mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

        # if attentional_pool:
        #     self.attn_pool = AttentionalPooler(output_dim, width, n_head=attn_pooler_heads, n_queries=n_queries)
        #     self.ln_post = norm_layer(output_dim)
        #     self.proj = nn.Parameter(scale * torch.randn(output_dim, output_dim))
        # else:
        #     self.attn_pool = None
        #     self.ln_post = norm_layer(width)
        #     self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        for param in self.parameters():
            param.requires_grad = False

        if unlocked_groups != 0:
            groups = [
                [
                    self.conv1,
                    self.class_embedding,
                    self.positional_embedding,
                    self.ln_pre,
                ],
                *self.transformer.resblocks[:-1],
                [
                    self.transformer.resblocks[-1],
                    self.ln_post,
                ],
                self.proj,
            ]

            def _unlock(x):
                if isinstance(x, Sequence):
                    for g in x:
                        _unlock(g)
                else:
                    if isinstance(x, torch.nn.Parameter):
                        x.requires_grad = True
                    else:
                        for p in x.parameters():
                            p.requires_grad = True

            _unlock(groups[-unlocked_groups:])

    def init_parameters(self):
        # FIXME OpenAI CLIP did not define an init for the VisualTransformer
        # TODO experiment if default PyTorch init, below, or alternate init is best.

        # nn.init.normal_(self.class_embedding, std=self.scale)
        # nn.init.normal_(self.positional_embedding, std=self.scale)
        #
        # proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        # attn_std = self.transformer.width ** -0.5
        # fc_std = (2 * self.transformer.width) ** -0.5
        # for block in self.transformer.resblocks:
        #     nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
        #     nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
        #     nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
        #     nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        #
        # if self.text_projection is not None:
        #     nn.init.normal_(self.text_projection, std=self.scale)
        pass

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def _global_pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.global_average_pool:
            return x.mean(dim=1), x
        else:
            return x[:, 0], x[:, 1:]

    def interpolate_pos_encoding(self, x, w, h):
        """Interpolate pos encoding with input size

        Args:
            x (Tensor): input features
            w (int): width
            h (int): height

        Returns:
            embed: Interpolated position embedding, 1st dim is the cls_pos_embed
        """

        previous_dtype = x.dtype
        npatch = x.shape[1]
        N = self.positional_embedding.shape[0] - 1
        if npatch == N and w == h:
            return self.positional_embedding.unsqueeze(0)
        pos_embed = self.positional_embedding.float()
        # class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[1:, :]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        # w0, h0 = w0 + 0.01, h0 + 0.01
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            size=(w0, h0),
            # scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )

        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed.to(previous_dtype)
    
    def forward_features(self, z, x, text_embed=None):
        
        B, H_x, W_x = x.shape[0], x.shape[2], x.shape[3]
        W_z, H_z = z.size(2), z.size(3)
        assert x.shape[2] % self.patch_size == 0 and x.shape[3] % self.patch_size == 0
        assert z.shape[2] % self.patch_size == 0 and z.shape[3] % self.patch_size == 0
        
        x = self.conv1(x).flatten(2).transpose(1, 2).contiguous()    # BCHW -> BNC
        z = self.conv1(z).flatten(2).transpose(1, 2).contiguous()    # BCHW -> BNC
        
        x = x + self.interpolate_pos_encoding(x, W_x, H_x)
        z = z + self.interpolate_pos_encoding(z, W_z, H_z)
        
        _, L_x, C = x.size()
        L_z = z.size(1)
        add_dim = 0
        if self.add_cls_token:
            cls_embed = self.class_embedding.to(x.dtype).view(1, 1, C).expand(B, -1, -1)
            x = torch.cat([cls_embed, z, x], dim=1)
            add_dim += 1
        else:
            x = torch.cat([z, x], dim=1)
            
        if text_embed is not None:
            x = torch.cat([text_embed.unsqueeze(1), x], dim=1)
            add_dim += 1
            
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # BNC -> NBC
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # NBC -> BNC
        
        if add_dim > 0:
            cls_token, z, x = torch.split(x, [add_dim, L_z, L_x], dim=1)
        else:
            z, x = torch.split(x, [L_z, L_x], dim=1)

        z_2d = z.transpose(1, 2).contiguous().reshape(B, C, W_z // self.patch_size, H_z // self.patch_size)
        x_2d = x.transpose(1, 2).contiguous().reshape(B, C, W_x // self.patch_size, H_x // self.patch_size)
        
        aux_dict = {"attn": None, "z_2d": z_2d}

        return x_2d, aux_dict
    
    def forward(self, z, x, text_embed=None, **kwargs):
        """
        Joint feature extraction and relation modeling for the basic ViT backbone.
        Args:
            z (torch.Tensor): template feature, [B, C, H_z, W_z]
            x (torch.Tensor): search region feature, [B, C, H_x, W_x]

        Returns:
            x (torch.Tensor): merged template and search region feature, [B, L_z+L_x, C]
            attn : None
        """
        x, aux_dict = self.forward_features(z, x, text_embed=text_embed)

        return x, aux_dict

    def forward_online(self, x_t, x_ot, x_s, txt_emb):

        assert x_t.shape[2] % self.patch_size == 0 and x_t.shape[3] % self.patch_size == 0
        assert x_ot.shape[2] % self.patch_size == 0 and x_ot.shape[3] % self.patch_size == 0
        assert x_s.shape[2] % self.patch_size == 0 and x_s.shape[3] % self.patch_size == 0

        W_s, H_s = x_s.size(2), x_s.size(3)
        W_t, H_t = x_t.size(2), x_t.size(3)

        x_t = self.conv1(x_t).flatten(2).transpose(1, 2).contiguous()    # BCHW -> BNC
        x_ot = self.conv1(x_ot).flatten(2).transpose(1, 2).contiguous()  # BCHW -> BNC
        x_s = self.conv1(x_s).flatten(2).transpose(1, 2).contiguous()    # BCHW -> BNC

        B, L_t, C = x_t.size()
        L_s = x_s.size(1)

        x_t = x_t + self.interpolate_pos_encoding(x_t, W_t, H_t)
        x_ot = x_ot + self.interpolate_pos_encoding(x_ot, W_t, H_t)
        x_s = x_s + self.interpolate_pos_encoding(x_s, W_s, H_s)

        cls_embed = self.class_embedding.to(x_t.dtype).view(1, 1, C).expand(B, -1, -1)
        txt_emb = txt_emb.unsqueeze(1)
        x = torch.cat([cls_embed, txt_emb, x_t, x_ot, x_s], dim=1)
        x = self.patch_dropout(x)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # BNC -> NBC
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # NBC -> BNC

        cls_token, txt_token, x_t, x_ot, x_s = torch.split(x, [1, 1, L_t, L_t, L_s], dim=1)

        x_t_2d = x_t.transpose(1, 2).contiguous().reshape(B, C, W_t // self.patch_size, H_t // self.patch_size)
        x_ot_2d = x_ot.transpose(1, 2).contiguous().reshape(B, C, W_t // self.patch_size, H_t // self.patch_size)
        x_s_2d = x_s.transpose(1, 2).contiguous().reshape(B, C, W_s // self.patch_size, H_s // self.patch_size)

        return x_t_2d, x_ot_2d, x_s_2d
    
        
class TextTransformer(nn.Module):
    output_tokens: torch.jit.Final[bool]
    
    def __init__(
            self,
            context_length: int = 77,
            vocab_size: int = 49408,
            width: int = 512,
            heads: int = 8,
            layers: int = 12,
            ls_init_value: float = None,
            output_dim: int = 512,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            embed_cls: bool = False,
            pad_id: int = 0,
            output_tokens: bool = False,
            init_cfg: Optional[dict] = None
    ):
        super().__init__()
        self.num_pos = self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.output_dim = output_dim
        self.heads = heads
        self.pad_id = pad_id
        
        self.output_tokens = output_tokens

        self.init_cfg = init_cfg
        self.text_projection = nn.Parameter(torch.empty(width, output_dim))

        if embed_cls:
            self.cls_emb = nn.Parameter(torch.empty(width))
            self.num_pos += 1
        else:
            self.cls_emb = None

        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(self.num_pos, width))
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        self.ln_final = norm_layer(width)

        self.register_buffer('attn_mask', self.build_attention_mask(), persistent=False)

        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        if self.cls_emb is not None:
            nn.init.normal_(self.cls_emb, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.num_pos, self.num_pos)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def build_cls_mask(self, text, cast_dtype: torch.dtype):
        cls_mask = (text != self.pad_id).unsqueeze(1)
        cls_mask = F.pad(cls_mask, (1, 0, cls_mask.shape[2], 0), value=1.0)
        additive_mask = torch.empty(cls_mask.shape, dtype=cast_dtype, device=cls_mask.device)
        additive_mask.fill_(0)
        additive_mask.masked_fill_(~cls_mask, float("-inf"))
        additive_mask = torch.repeat_interleave(additive_mask, self.heads, 0)
        return additive_mask

    def _repeat(self, t, N: int):
        return t.reshape(1, 1, -1).repeat(N, 1, 1)

    def forward(self, text):
        cast_dtype = self.transformer.get_cast_dtype()
        seq_len = text.shape[1]

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]
        attn_mask = self.attn_mask
        if self.cls_emb is not None:
            seq_len += 1
            x = torch.cat([x, self._repeat(self.cls_emb, x.shape[0])], dim=1)
            cls_mask = self.build_cls_mask(text, cast_dtype)
            attn_mask = attn_mask[None, :seq_len, :seq_len] + cls_mask[:, :seq_len, :seq_len]

        x = x + self.positional_embedding[:seq_len].to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        if self.cls_emb is not None:
            pooled, tokens = x[:, -1], x[:, :-1]
            pooled = self.ln_final(pooled)
        else:
            x = self.ln_final(x)
            pooled, tokens = x[torch.arange(x.shape[0]), text.argmax(dim=-1)], x

        if self.text_projection is not None:
            pooled = pooled @ self.text_projection

        if self.output_tokens:
            return pooled, tokens

        return pooled
    
    def encode_text(self, text, normalize: bool = False):
        
        cast_dtype = self.transformer.get_cast_dtype()
        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        
        return F.normalize(x, dim=-1) if normalize else x
    

# class BaseBackbone(nn.Module):
#     def __init__(self):
#         super().__init__()

#         # for original ViT
#         self.pos_embed = None
#         self.img_size = [224, 224]
#         self.patch_size = 16
#         self.embed_dim = 384

#         self.cat_mode = 'direct'

#         self.pos_embed_z = None
#         self.pos_embed_x = None

#         self.template_segment_pos_embed = None
#         self.search_segment_pos_embed = None

#         self.return_inter = False
#         self.return_stage = [2, 5, 8, 11]

#         self.add_cls_token = False
#         self.add_sep_seg = False

#     def finetune_track(self, cfg, patch_start_index=1):

#         search_size = to_2tuple(cfg.DATA.SEARCH.SIZE)
#         template_size = to_2tuple(cfg.DATA.TEMPLATE.SIZE)
#         new_patch_size = cfg.MODEL.BACKBONE.STRIDE

#         self.cat_mode = cfg.MODEL.BACKBONE.CAT_MODE
#         self.return_inter = cfg.MODEL.RETURN_INTER
#         self.return_stage = cfg.MODEL.RETURN_STAGES
#         self.add_sep_seg = cfg.MODEL.BACKBONE.SEP_SEG

#         # resize patch embedding
#         if new_patch_size != self.patch_size:
#             print('Inconsistent Patch Size With The Pretrained Weights, Interpolate The Weight!')
#             old_patch_embed = {}
#             for name, param in self.patch_embed.named_parameters():
#                 if 'weight' in name:
#                     param = nn.functional.interpolate(param, size=(new_patch_size, new_patch_size),
#                                                       mode='bicubic', align_corners=False)
#                     param = nn.Parameter(param)
#                 old_patch_embed[name] = param
#             self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=new_patch_size, in_chans=3,
#                                           embed_dim=self.embed_dim)
#             self.patch_embed.proj.bias = old_patch_embed['proj.bias']
#             self.patch_embed.proj.weight = old_patch_embed['proj.weight']

#         # for patch embedding
#         patch_pos_embed = self.pos_embed[:, patch_start_index:, :]
#         patch_pos_embed = patch_pos_embed.transpose(1, 2)
#         B, E, Q = patch_pos_embed.shape
#         P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
#         patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

#         # for search region
#         H, W = search_size
#         new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
#         search_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
#                                                            align_corners=False)
#         search_patch_pos_embed = search_patch_pos_embed.flatten(2).transpose(1, 2)

#         # for template region
#         H, W = template_size
#         new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
#         template_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
#                                                              align_corners=False)
#         template_patch_pos_embed = template_patch_pos_embed.flatten(2).transpose(1, 2)

#         self.pos_embed_z = nn.Parameter(template_patch_pos_embed)
#         self.pos_embed_x = nn.Parameter(search_patch_pos_embed)

#         # for cls token (keep it but not used)
#         if self.add_cls_token and patch_start_index > 0:
#             cls_pos_embed = self.pos_embed[:, 0:1, :]
#             self.cls_pos_embed = nn.Parameter(cls_pos_embed)

#         # separate token and segment token
#         if self.add_sep_seg:
#             self.template_segment_pos_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
#             self.template_segment_pos_embed = trunc_normal_(self.template_segment_pos_embed, std=.02)
#             self.search_segment_pos_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
#             self.search_segment_pos_embed = trunc_normal_(self.search_segment_pos_embed, std=.02)

#         # self.cls_token = None
#         # self.pos_embed = None

#         if self.return_inter:
#             for i_layer in self.return_stage:
#                 if i_layer != 11:
#                     norm_layer = partial(nn.LayerNorm, eps=1e-6)
#                     layer = norm_layer(self.embed_dim)
#                     layer_name = f'norm{i_layer}'
#                     self.add_module(layer_name, layer)

#     def forward_features(self, z, x):
#         B, H, W = x.shape[0], x.shape[2], x.shape[3]

#         x = self.patch_embed(x)
#         z = self.patch_embed(z)

#         if self.add_cls_token:
#             cls_tokens = self.cls_token.expand(B, -1, -1)
#             cls_tokens = cls_tokens + self.cls_pos_embed

#         z += self.pos_embed_z
#         x += self.pos_embed_x

#         if self.add_sep_seg:
#             x += self.search_segment_pos_embed
#             z += self.template_segment_pos_embed

#         x = combine_tokens(z, x, mode=self.cat_mode)
#         if self.add_cls_token:
#             x = torch.cat([cls_tokens, x], dim=1)

#         x = self.pos_drop(x)

#         for i, blk in enumerate(self.blocks):
#             x = blk(x)

#         lens_z = self.pos_embed_z.shape[1]
#         lens_x = self.pos_embed_x.shape[1]
#         x = recover_tokens(x, lens_z, lens_x, mode=self.cat_mode)

#         aux_dict = {"attn": None}
#         return self.norm(x), aux_dict

#     def forward(self, z, x, **kwargs):
#         """
#         Joint feature extraction and relation modeling for the basic ViT backbone.
#         Args:
#             z (torch.Tensor): template feature, [B, C, H_z, W_z]
#             x (torch.Tensor): search region feature, [B, C, H_x, W_x]

#         Returns:
#             x (torch.Tensor): merged template and search region feature, [B, L_z+L_x, C]
#             attn : None
#         """
#         x, aux_dict = self.forward_features(z, x,)

#         return x, aux_dict
