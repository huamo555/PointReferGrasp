# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from model.attention import MultiheadAttention

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self,
                 in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
    

class PWCA(nn.Module):

    def __init__(self, embed_dims, num_heads=4, attn_drop=0., drop=0., ffn_ratio=4.):
        super().__init__()
        self.norm_g = nn.LayerNorm(embed_dims)
        self.norm_t = nn.LayerNorm(embed_dims)

        # use your project's MultiheadAttention (keeps API consistent)
        self.attn = MultiheadAttention(
            embed_dims,
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=attn_drop,
            proj_drop=drop,
            q_project=True
        )

        hidden = int(embed_dims * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dims),
            nn.Linear(embed_dims, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dims),
            nn.Dropout(drop)
        )
        self.dropout = nn.Dropout(drop)

        # init like other modules if needed
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, group_tokens, text_tokens, text_mask=None):

        # normalize
        g = self.norm_g(group_tokens)
        t = self.norm_t(text_tokens)

        # convert mask: your attention likely expects key_padding_mask where True means masked.
        key_padding_mask = None
        if text_mask is not None:
            # text_mask: True means valid -> key_padding_mask expects True for padded positions
            key_padding_mask = (~text_mask).to(torch.bool)  # [B, L]

        # Q = group tokens, K/V = text tokens (group queries text)
        attn_out = self.attn(g, t, t, key_padding_mask=key_padding_mask)  # returns [B, N_g, C]

        # residual + dropout
        out = group_tokens + self.dropout(attn_out)

        # FFN (with residual)
        out = out + self.ffn(out)
        return out
        
class FullAttnCatBlock(nn.Module):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 ffn_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 key_is_query=False,
                 value_is_key=False,
                 q_project=True,
                 with_cp=False,
                 **kwargs):
        super().__init__()
        self.with_cp = with_cp

        self.norm_query = nn.LayerNorm(embed_dims)

        if not key_is_query:
            self.norm_key = nn.LayerNorm(embed_dims)
        else:
            self.norm_key = None
        self.key_is_query = key_is_query

        if not value_is_key:
            self.norm_value = nn.LayerNorm(embed_dims)
        else:
            self.norm_value = None
        self.value_is_key = value_is_key

        self.attn = MultiheadAttention(
            embed_dims,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            q_project=q_project)

        self.ffn = Mlp(in_features=embed_dims, hidden_features=int(embed_dims * ffn_ratio), drop=drop)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.drop = nn.Dropout(drop)
        # self.proj = nn.Linear(embed_dims * 2, embed_dims, bias=True)

    def forward(self, query, key, value, key_padding_mask=None):
        def _inner_forward(query, key, value, key_padding_mask):
            q = self.norm_query(query)
            k = q if self.key_is_query else self.norm_key(key)
            v = k if self.value_is_key else self.norm_value(value)

            x = self.attn(q, k, v, key_padding_mask) + self.drop(query)
            # x = self.proj(x)
            x = self.ffn(self.norm2(x)) + x
            return x

        if self.with_cp:
            return cp.checkpoint(_inner_forward, query, key, value, key_padding_mask)
        else:
            return _inner_forward(query, key, value, key_padding_mask)
        
    
class LightGroupAttnBlock(nn.Module):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 ffn_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 key_is_query=False,
                 value_is_key=False,
                 with_cp=False,
                 lan_dim = 768):
        super().__init__()
        self.drop = nn.Dropout(drop)
        self.with_cp = with_cp

        self.norm_query = nn.GELU()

        if not key_is_query:
            self.norm_key = nn.LayerNorm(embed_dims)
        else:
            self.norm_key = None
        self.key_is_query = key_is_query

        if not value_is_key:
            self.norm_value = nn.LayerNorm(embed_dims)
        else:
            self.norm_value = None
        self.value_is_key = value_is_key

        self.attn = MultiheadAttention(
            embed_dims,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            q_project=True,
            k_project=True,
            v_project=False,
            proj_after_att=False,
            lan_dim = lan_dim)


    def forward(self, query, key, value, q_mask=None):
        def _inner_forward(query, key, value):
            q = self.norm_query(query)
            k = q if self.key_is_query else self.norm_key(key)
            v = k if self.value_is_key else self.norm_value(value)
            x = self.attn(q, k, v, q_mask) + self.drop(q)
            return x

        if self.with_cp:
            return cp.checkpoint(_inner_forward, query, key, value)
        else:
            return _inner_forward(query, key, value)



class GPBlock(nn.Module):
    def __init__(self,
                 embed_dims,
                 num_group_token,
                 depth=1,
                 num_group_heads=4,
                 num_ungroup_heads=4,
                 lan_dim=768,
                 ffn_ratio=4.,
                 qkv_bias=True,
                 group_qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 with_cp=False,
                 group_att_cfg=dict(),
                 fwd_att_cfg=dict(),
                 ungroup_att_cfg=dict(),
                 **kwargs):

        super().__init__()

        self.embed_dims = embed_dims
        self.num_group_token = num_group_token
        self.with_cp = with_cp

        self.drop = nn.Dropout(drop)

        # Group 阶段：保持你原来的 LightGroupAttnBlock
        _group_att_cfg = dict(
            embed_dims=embed_dims,
            num_heads=num_group_heads,
            ffn_ratio=ffn_ratio,
            qkv_bias=qkv_bias,
            qk_scale=group_qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            key_is_query=False,
            value_is_key=True,
            with_cp=with_cp,
            lan_dim = lan_dim)
        _group_att_cfg.update(group_att_cfg)
        self.group_layer = LightGroupAttnBlock(**_group_att_cfg)

        # === 替换点：将 Mixer 换成 PWCA（group_token 与 text_token 的 cross-attn） ===
        pwca_cfg = dict(
            embed_dims=embed_dims,
            num_heads=num_group_heads,
            attn_drop=attn_drop,
            drop=drop,
            ffn_ratio=ffn_ratio
        )
        pwca_cfg.update(fwd_att_cfg)
        self.pwca = PWCA(**pwca_cfg)

        # Ungroup 阶段：保持 FullAttnCatBlock，用来把 group 信息投回点
        _ungroup_att_cfg = dict(
            embed_dims=embed_dims,
            num_heads=num_ungroup_heads,
            ffn_ratio=ffn_ratio,
            qkv_bias=qkv_bias,
            qk_scale=None,
            drop=drop,
            attn_drop=attn_drop,
            key_is_query=False,
            value_is_key=True,
            with_cp=with_cp)
        _ungroup_att_cfg.update(ungroup_att_cfg)
        self.un_group_layer = FullAttnCatBlock(**_ungroup_att_cfg)

    def forward(self, q, x, q_mask=None):

        # Group: compress point -> group tokens
        # Note: your original group_layer used query=q, key=x, value=x
        # Keep same call to preserve behavior
        gt = self.group_layer(query=q, key=x, value=x)  # gt: [B, N_g, C]
        if q_mask is not None:
            # 你原代码对 gt 做了乘以 q_mask.unsqueeze(-1)，保持一致
            # 但这里 q_mask 是 [B, L]，乘到 gt 上意图是把 group token 与 text mask 对齐——原作者这样写过
            # 为安全起见，只在 q_mask 长度等于 gt 的第2维时才乘，否则忽略（通常 q_mask 长度为 L）
            if q_mask.shape[1] == gt.shape[1]:
                gt = gt * q_mask.unsqueeze(-1)
            # 原代码里对 gt 乘 mask 的目的在具体实现中可能不同，保留这一兼容分支

        # PWCA: use group tokens as queries, text tokens as key/value
        gt = self.pwca(gt, q, q_mask)  # [B, N_g, C]
        gt = gt + self.drop(gt)  # optional extra dropout as in original

        # Ungroup: propagate group tokens back to points
        # keep the same interface as before: query=x, key=gt, value=gt
        ungroup_tokens = self.un_group_layer(query=x, key=gt, value=gt, key_padding_mask=q_mask)

        return ungroup_tokens
