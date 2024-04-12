#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/21 22:03
# @Author  : gao qiang
# @Site    : 
# @File    : module
# @Project : sota_end2end_parser

from torch_geometric.nn.models import GAT


def train():
    kwargs = {"v2": True}
    model = GAT(**kwargs, in_channels=14,hidden_channels=1024, out_channels=1, num_layers=3)
    print(model)

if __name__ == "__main__":
    train()
