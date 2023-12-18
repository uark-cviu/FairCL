from math import sqrt
from functools import partial
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from modules.segformer_offical.mix_transformer import *

# helpers

def exists(val):
    return val is not None

def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth


segformer_type_args = {
    'segformer': {
        'model_name': 'segformer',
        'dims': (64, 128, 320, 512),      # dimensions of each stage
        'heads' : (1, 2, 5, 8),           # heads of each stage
        'ff_expansion' : (8, 8, 4, 4),    # feedforward expansion factor of each stage
        'reduction_ratio' : (8, 4, 2, 1), # reduction ratio of each stage for efficient attention
        'num_layers' : (3, 3, 6, 3),                 # num layers of each stage
        'decoder_dim' : 256,              # decoder dimension
    }
}

segformer_model_dict = {
    'segformer': mit,
}


class SegFormer_Body(nn.Module):
    def __init__(
        self,
        *,
        dims = (32, 64, 160, 256),
        heads = (1, 2, 5, 8),
        ff_expansion = (8, 8, 4, 4),
        reduction_ratio = (8, 4, 2, 1),
        num_layers = 2,
        channels = 3,
        decoder_dim = 256,
        num_classes = 4,
        return_attn = False,
        model_name = 'segformer'
    ):
        super(SegFormer_Body, self).__init__()
        dims, heads, ff_expansion, reduction_ratio, num_layers = map(partial(cast_tuple, depth = 4), (dims, heads, ff_expansion, reduction_ratio, num_layers))
        assert all([*map(lambda t: len(t) == 4, (dims, heads, ff_expansion, reduction_ratio, num_layers))]), 'only four stages are allowed, all keyword arguments must be either a single value or a tuple of 4 values'

        self.mit = segformer_model_dict[model_name](return_attn=return_attn)

    def forward(self, x):
        return self.mit(x)



class SegFormer_Head(nn.Module):
    def __init__(
        self,
        *,
        dims = (32, 64, 160, 256),
        heads = (1, 2, 5, 8),
        ff_expansion = (8, 8, 4, 4),
        reduction_ratio = (8, 4, 2, 1),
        num_layers = 2,
        channels = 3,
        decoder_dim = 256,
        num_classes = 4,
        return_attn = False,
        model_name = 'segformer'
    ):
        super(SegFormer_Head, self).__init__()

        self.to_fused = nn.ModuleList([nn.Sequential(
            nn.Conv2d(dim, decoder_dim, 1),
            nn.Upsample(scale_factor = 2 ** i)
        ) for i, dim in enumerate(dims)])

        self.to_segmentation = nn.Conv2d(4 * decoder_dim, decoder_dim, 1)

    def forward(self, layer_outputs):
        fused = [to_fused(output) for output, to_fused in zip(layer_outputs, self.to_fused)]
        fused = torch.cat(fused, dim = 1)
        return self.to_segmentation(fused).contiguous(), layer_outputs
