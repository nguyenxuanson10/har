import torch as th
import torch.nn as nn
import numpy as np
from torch.autograd import Function as F
from . import functional

def grlog(y, x):
    ytx = y.transpose(-1,-2).matmul(x)                                        
    At = y.transpose(-1,-2).subtract( ytx.matmul(x.transpose(-1,-2)) )            
    Bt = th.linalg.pinv(ytx).matmul(At)
    u, s, v = th.svd(Bt.transpose(-1,-2), some=True)                              

    s = th.diag_embed(th.atan(s)) 

    return u.matmul(s.matmul(v.transpose(-1,-2)))

class PoolingLayer(nn.Module):
    def __init__(self, stride=2):
        super().__init__()
        self.stride = stride

    def forward(self, input):
        shape = input.shape
        
        m = th.nn.AvgPool2d(self.stride, stride=self.stride)
        output = m(input)        

        return output

class OrthoLayer(nn.Module):
    def __init__(self, top_eigen=10):
        super().__init__()
        self.top_eigen = top_eigen

    def forward(self, input):
        u, _, _ = th.svd(input)        

        tu = u[...,:self.top_eigen]        

        return tu @ th.transpose(tu, -1, -2)
