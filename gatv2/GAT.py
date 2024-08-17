
import dgl
import torch
from dgl.nn.pytorch import GATv2Conv
from gatv2.graph import Graph
from torch import nn
import torch.nn.functional as F



class GAT(nn.Module):
    def __init__(self,
                 in_feats=1024,
                 hid_feats=512,
                 out_feats=512,
                 num_heads=2,
                 feat_drop=0.0,
                 attn_drop=0.0,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=True,
                 share_weights=False,
                 ):
        super().__init__()
        self.conv1 = GATv2Conv(in_feats, hid_feats // num_heads, num_heads)
        self.conv2 = GATv2Conv(hid_feats, hid_feats, num_heads)  
        self.projection = nn.Linear(hid_feats, out_feats)
        self.hid_feats = hid_feats
        self.num_heads=num_heads

    def forward(self, graph, inputs):
        
        h = self.conv1(graph, inputs) 
        h = F.relu(h).view(-1, self.hid_feats)  
        h = self.conv2(graph, h)  
        h = torch.sum(F.relu(h), dim=1)
        h=self.projection(F.relu(h))
        return h
