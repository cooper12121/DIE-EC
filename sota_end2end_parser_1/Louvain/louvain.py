#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/5 20:04
# @Author  : gao qiang
# @File    : louvain.py
# @Project : sota_end2end_parser

'''
鲁汶算法进行聚类：划分社区
'''
from community import community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx

def construct_graph():
# load the karate club graph
    G = nx.karate_club_graph()
    print(G)
    # compute the best partition
    partition = community_louvain.best_partition(G)
    print(partition)
    # draw the graph
    pos = nx.spring_layout(G)
    # color the nodes according to their partition
    cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
    plt.show()

    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=34,
                           cmap=cmap, node_color=list(partition.values()))
    plt.show()

if __name__ == "__main__":
    construct_graph()