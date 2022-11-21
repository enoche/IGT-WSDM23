# coding: utf-8
# @Time   : 2021/06/06
# @Author : enoche
# @Email  : enoche.chow@gmail.com
#
"""
Inductive Graph Transformer
##########################
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum
import pandas as pd

from models.common.general_model import GeneralModel, AbstractDataPreprocessing


class IGT(GeneralModel):
    def __init__(self, config, data):
        super(IGT, self).__init__(config, data)

        # (263:296) accept_county_lon:total_cost
        self.feature_len_seller = data.fea_dim_seller       # s7-s63 (0:56)
        self.feature_len_accept = data.fea_dim_accept       # f16-f136 (57:177)
        self.feature_len_receiver = data.fea_dim_receiver   # r52-r136 (178:262)
        # node ids + date = 5
        self.feature_len_hour = data.feature_len - self.feature_len_seller - \
                                self.feature_len_accept - self.feature_len_receiver - 5
        self.n_sellers = data.graph_info_dict['seller_id']
        self.n_cities = data.graph_info_dict['accept_city']
        self.n_counties = data.graph_info_dict['receiver_county']
        self.n_hours = 24
        # adj-matrix
        self.seller_accept_adj = self._normalize_adj_m(data.adj_matrices[0]).to(self.device)
        self.accept_receiver_adj = self._normalize_adj_m(data.adj_matrices[1]).to(self.device)
        self.receiver_hour_adj = self._normalize_adj_m(data.adj_matrices[2]).to(self.device)

        # full adjs for inference
        self.full_seller_accept_adj = self._normalize_adj_m(data.additional_adj_matrices[0]).to(self.device)
        self.full_accept_receiver_adj = self._normalize_adj_m(data.additional_adj_matrices[1]).to(self.device)
        self.full_receiver_hour_adj = self._normalize_adj_m(data.additional_adj_matrices[2]).to(self.device)

        self.latent_dim = config['embedding_size']
        self.n_layers = config['n_layers']
        # define hidden embeddings
        self.seller_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_sellers, self.latent_dim)))
        self.accept_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_cities, self.latent_dim)))
        self.receiver_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_counties, self.latent_dim)))
        self.hour_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_hours, self.latent_dim)))

        self.vTransformer = ViT(config, None, self.feature_len_accept+self.latent_dim)
        self.gru_seller = nn.GRUCell(self.feature_len_seller, self.latent_dim)
        self.gru_accept = nn.GRUCell(self.feature_len_accept, self.latent_dim)
        self.gru_receiver = nn.GRUCell(self.feature_len_receiver, self.latent_dim)
        self.gru_hour = nn.GRUCell(self.feature_len_hour, self.latent_dim)
        self.loss = nn.L1Loss()

        self.adjs = [self.seller_accept_adj, self.accept_receiver_adj, self.receiver_hour_adj]
        self.in_inference = False

    def _normalize_adj_m(self, adj):
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        col_sum = 1e-7 + torch.sparse.sum(adj.t(), -1).to_dense()
        indices = adj._indices()
        values = adj._values()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        c_inv_sqrt = torch.pow(col_sum, -0.5)
        cols_inv_sqrt = c_inv_sqrt[indices[1]]
        norm_values = values * rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, norm_values, adj.size())

    def forward(self, orders):
        x = torch.tensor(orders[:, :-5]).type(torch.FloatTensor).to(self.device)
        # first: get nodes
        nodes = torch.tensor(orders[:, -5:-1]).type(torch.LongTensor).to(self.device)
        # latent_embeddings = self.forward(x)

        adjs = self.adjs
        embeddings_layers = [[self.seller_embeddings], [self.accept_embeddings],
                             [self.receiver_embeddings], [self.hour_embeddings]]
        for layer_idx in range(self.n_layers):
            # accept city-->seller
            temp_emb_s = torch.sparse.mm(adjs[0], embeddings_layers[1][layer_idx])
            embeddings_layers[0].append(temp_emb_s)
            # seller-->accept city & receiver county-->accept city
            temp_emb_c1 = torch.sparse.mm(adjs[0].t(), embeddings_layers[0][layer_idx])
            temp_emb_c2 = torch.sparse.mm(adjs[1], embeddings_layers[2][layer_idx])
            embeddings_layers[1].append(temp_emb_c1 + temp_emb_c2)
            # accept city-->receiver county & hour-->receiver county
            temp_emb_r1 = torch.sparse.mm(adjs[1].t(), embeddings_layers[1][layer_idx])
            temp_emb_r2 = torch.sparse.mm(adjs[2], embeddings_layers[3][layer_idx])
            embeddings_layers[2].append(temp_emb_r1 + temp_emb_r2)
            # receiver county-->hour
            temp_emb_h = torch.sparse.mm(adjs[2].t(), embeddings_layers[2][layer_idx])
            embeddings_layers[3].append(temp_emb_h)

        latent_embeddings = []
        for i in range(4):
            nodes_embeddings = torch.stack(embeddings_layers[i], dim=1)
            nodes_embeddings = torch.mean(nodes_embeddings, dim=1)
            latent_embeddings.append(nodes_embeddings)

        # for test
        # latent_embeddings = [self.seller_embeddings, self.accept_embeddings, self.receiver_embeddings, self.hour_embeddings]
        latent_s = latent_embeddings[0][nodes[:, 0]]
        latent_a = latent_embeddings[1][nodes[:, 1]]
        latent_r = latent_embeddings[2][nodes[:, 2]]
        latent_h = latent_embeddings[3][nodes[:, 3]]

        # GRU
        s_gru = self.gru_seller(x[:, :self.feature_len_seller], latent_s)
        a_gru = self.gru_accept(x[:, self.feature_len_seller:self.feature_len_seller+self.feature_len_accept], latent_a)
        r_gru = self.gru_receiver(x[:, self.feature_len_seller+self.feature_len_accept:
                                     self.feature_len_seller+self.feature_len_accept+self.feature_len_receiver], latent_r)
        h_gru = self.gru_hour(x[:, self.feature_len_seller+self.feature_len_accept+self.feature_len_receiver:
                                     self.feature_len_seller+self.feature_len_accept+self.feature_len_receiver+
                                     self.feature_len_hour], latent_h)

        s = self.pad_tensor(s_gru, self.feature_len_accept - self.feature_len_seller)
        r = self.pad_tensor(r_gru, self.feature_len_accept - self.feature_len_receiver)
        h = self.pad_tensor(h_gru, self.feature_len_accept - self.feature_len_hour)

        # concat all embeddings
        fixed_node_size = self.feature_len_accept + self.latent_dim
        x = self.insert_t(s, x, self.feature_len_seller)
        x = self.insert_t(a_gru, x, self.feature_len_accept + fixed_node_size)
        x = self.insert_t(r, x, self.feature_len_receiver + 2 * fixed_node_size)
        x = self.insert_t(h, x, self.feature_len_hour + 3 * fixed_node_size)

        # input into transformers
        b, _ = x.shape
        x = torch.reshape(x, (b, -1, fixed_node_size))

        return self.vTransformer(x)

    def calculate_loss(self, orders, y):
        if self.in_inference:
            self.in_inference = False
            self.adjs = [self.seller_accept_adj, self.accept_receiver_adj, self.receiver_hour_adj]
        y_pred = self.forward(orders)
        y = torch.tensor(y).type(torch.FloatTensor).to(self.device)
        return self.loss(torch.squeeze(y_pred), y)

    def predict(self, orders):
        if self.in_inference == False:
            self.in_inference = True
            self.adjs = [self.full_seller_accept_adj, self.full_accept_receiver_adj, self.full_receiver_hour_adj]
        # x = torch.tensor(orders).type(torch.FloatTensor).to(self.device)
        return torch.squeeze(self.forward(orders))

    def pad_tensor(self, t, len):
        result = F.pad(input=t, pad=(0, len, 0, 0), mode='constant', value=0)
        return result

    def insert_t(self, source, target, idx):
        target = torch.cat([target[:, :idx], source, target[:, idx:]], dim=1)
        return target



# classes
class ViT(GeneralModel):
    def __init__(self, config, data, patch_dim):
        super(ViT, self).__init__(config, data)
        # order: seller/src_city/other(post)/dst_city
        self.num_patches = 4        # order length
        self.pool = config['pool']           # order aggregation
        self.patch_dim = patch_dim        # order total embedding
        emb_dropout = config['emb_dropout']
        dropout = config['dropout']
        self.dim = config['dim']              # latent dim
        self.depth = config['depth']          # transformer layers
        self.heads = config['heads']          # transformer heads
        self.dim_head = config['dim_head']
        self.mlp_dim = config['mlp_dim']           # mlp latent dim
        assert self.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Linear(self.patch_dim, self.dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, self.dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(self.dim, self.depth, self.heads, self.dim_head, self.mlp_dim, dropout)
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, 1)      # regression
        )

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
