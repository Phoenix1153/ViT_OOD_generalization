import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Function
from collections import OrderedDict

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super(FeedForward, self).__init__()

        self.mlp1 = nn.Linear(dim, hidden_dim)
        self.gelu = nn.GELU()
        self.mlp2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.mlp1(x)
        x = self.gelu(x)
        x = self.dropout(x)

        x = self.mlp2(x)
        x = self.dropout(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, dim * 3)

        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = self.dropout(out)
        return out


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, mlp_dim, heads, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attention = Attention(hidden_dim, heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.feedforward = FeedForward(hidden_dim, mlp_dim, dropout)

    def forward(self, x):

        x1 = self.norm1(x)
        x1 = self.attention(x1)
        x1 = x1 + x

        x2 = self.norm2(x1)
        x2 = self.feedforward(x2)

        return x1 + x2


class TransformerEncoder(nn.Module):
    def __init__(self, hidden_dim, depth, mlp_dim, heads, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        encoder_layer = OrderedDict()
        for d in range(depth):
            encoder_layer['encoder_{}'.format(d)] = \
                TransformerEncoderLayer(hidden_dim, mlp_dim, heads, dropout)
        self.encoders = nn.Sequential(encoder_layer)
        self.encoder_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.encoders(x)
        x = self.encoder_norm(x)
        return x


class EmbeddingLayer(nn.Module):
    def __init__(self, in_channel, out_channel, img_size, patch_size):
        super(EmbeddingLayer, self).__init__()
        self.embedding1 = nn.Conv2d(in_channel, out_channel, kernel_size=patch_size, stride=patch_size, padding=0)

    def forward(self, x):
        out = self.embedding1(x).flatten(2).transpose(1, 2)
        return out


class VisionTransformer(nn.Module):
    def __init__(self,
                 image_size,
                 patch_size,
                 num_classes,
                 hidden_dim,
                 depth,
                 mlp_dim,
                 heads,
                 in_channel=3,
                 dropout=0.1,
                 aggr='token'):

        super(VisionTransformer, self).__init__()
        num_patches = (image_size // patch_size) ** 2

        self.aggr = aggr
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_classes = num_classes

        self.embedding = EmbeddingLayer(in_channel, hidden_dim, image_size, patch_size)
        if aggr == 'token':
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, hidden_dim))
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        elif aggr == 'gap':
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, hidden_dim))

        self.transformer = TransformerEncoder(hidden_dim, depth, mlp_dim, heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, num_classes)


    def forward(self, img):
        
        x = self.embedding(img)

        if self.aggr == 'token':
            cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        x += self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x)

        if self.aggr == 'token':
            x = x[:, 0]
        elif self.aggr == 'gap':
            x = torch.mean(x, dim=1)
        return self.head(x)


def deit_small_b16_384(num_classes):
    vit = VisionTransformer(
        image_size=384,
        patch_size=16,
        num_classes=num_classes,
        hidden_dim=384,
        depth=12,
        mlp_dim=1536,
        heads=6
    )
    return vit
