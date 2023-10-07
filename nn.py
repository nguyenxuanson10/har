import torch as th
import torch.nn as nn
import numpy as np
from torch.autograd import Function as F
from . import functional

class PoolingLayerProjector(nn.Module):
    def __init__(self, stride=2):
        super().__init__()
        self.stride = stride

    def forward(self, input):
        shape = input.shape
        
        m = th.nn.AvgPool2d(self.stride, stride=self.stride)
        output = m(input)        

        return output

class OrthoLayerProjector(nn.Module):
    def __init__(self, top_eigen=10):
        super().__init__()
        self.top_eigen = top_eigen

    def forward(self, input):
        u, _, _ = th.svd(input)        

        tu = u[...,:self.top_eigen]        

        return tu @ th.transpose(tu, -1, -2)
