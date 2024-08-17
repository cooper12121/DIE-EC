
import json

import dgl
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from gatv2.GAT import *




class Graph:
    def __init__(self, graph_info_list,edu_list,mention_node_index,edu_node_hidden,node_features=1024, edge_features=1024,):
        self.edu_list = edu_list
        self.graph_info_list =process_graphinfo_list(graph_info_list)
        self.G = None
        self.node_features = node_features
        self.edge_features = edge_features
        self.mention_node_index = mention_node_index
        self.edu_node_hidden = edu_node_hidden
        self.process_graph()
        self.inputs = None


    def build_graph(self):
        try:

            src =[]
            dst = []
            for edge in self.graph_info_list:
                if edge[0][0]==None or edge[0][1]==None:continue

                
                rhetorical="让步关系" # change this to the rhetorical relation you want to focus on
                if len(edge)==3 and edge[1][1]==rhetorical:
                    src.append(edge[0][0]-1)
                    dst.append(edge[0][0]-1)

                    src.append(edge[0][1]-1)
                    dst.append(edge[0][1]-1)

                src.append(edge[0][0]-1)
                dst.append(edge[0][1]-1)
            src = torch.tensor(src,dtype=torch.int32)
            dst = torch.tensor(dst,dtype=torch.int32)
            self.G = dgl.graph((src, dst), idtype=torch.int32)
        except Exception as e:
            print(e)

    def graph_visualization(self,):
        nx_G = self.G.to_networkx()
        pos = nx.kamada_kawai_layout(nx_G)
        nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])
        plt.show()

    def add_node_features(self):
    
        edu2node=[]
        for i in range(len(self.edu_list)):
            if self.edu_list[i]!=None and self.edu_list[i]!=0:
                edu2node.append(i)
        node_nubmers = self.G.number_of_nodes()
        embed = nn.Embedding(node_nubmers, self.node_features)  
        for i in range(self.edu_node_hidden.shape[0]):
            embed.weight.data[edu2node[i]] = self.edu_node_hidden[i]
        self.G.ndata['feat'] = embed.weight 

    def add_edge_features(self):
        edge_numbers = self.G.number_of_edges()
        embed = nn.Embedding(edge_numbers, self.edge_features)
        self.G.edata['feat'] = embed.weight
       

    def process_graph(self):
        self.build_graph()
        self.add_node_features()
def process_graphinfo_list(graph_info_list):
    for i in range(len(graph_info_list)):
        if len(graph_info_list[i])==2:
            graph_info_list[i]=[graph_info_list[i]]
    return graph_info_list

def test(edu_list):
    from transformers import BertModel,BertTokenizer
    model = BertModel.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    encode_ids_list = [tokenizer.cls_token_id]
    encode_edu_index = []
    for edu in edu_list:
        if edu!=None and edu!=0:
            encode=tokenizer.encode(edu,add_special_tokens=False)
            start=len(encode_ids_list)
            encode_ids_list.extend(encode)
            end = len(encode_ids_list)-1
            encode_edu_index.append([start,end])
    print(encode_edu_index)



# if __name__ == "__main__":
    # graph_info_list=[[[1, 2], ["Root", "Elaborate", "N"], ["temp_edu:", "From then on , the members continued their separate careers ."]], [[3, 1], ["S", "span", "Root"], ["temp_edu:", None]], [[4, 3], ["N", "Joint", "S"], ["temp_edu:", "Danny Boy founded an art company ."]], [[3, 4], ["S", "Joint", "N"], ["temp_edu:", "Danny Boy founded an art company ."]], [[5, 3], ["N", "Joint", "S"], ["temp_edu:", None]], [[3, 5], ["S", "Joint", "N"], ["temp_edu:", None]], [[6, 5], ["N", "Topic-Change", "N"], ["temp_edu:", None]], [[5, 6], ["N", "Topic-Change", "N"], ["temp_edu:", None]], [[7, 6], ["N", "Joint", "N"], ["temp_edu:", None]], [[6, 7], ["N", "Joint", "N"], ["temp_edu:", None]], [[7, 8], ["N", "Elaborate", "N"], ["temp_edu:", "DJ Lethal became a member of nu metal band Limp Bizkit ,"]], [[9, 7], ["S", "span", "N"], ["temp_edu:", "who would cover \" Jump Around \" at live concerts , particularly in Limp Bizkit 's early years during the Family Values Tour 1998 ."]], [[10, 6], ["N", "Joint", "N"], ["temp_edu:", None]], [[6, 10], ["N", "Joint", "N"], ["temp_edu:", None]], [[10, 11], ["N", "Elaborate", "N"], ["temp_edu:", "Everlast achieved multi-platinum solo fame in 1998 with his album '' Whitey Ford Sings the Blues ."]], [[12, 10], ["S", "span", "N"], ["temp_edu:", "'' The first single from that album was \" What It 's Like \" ."]], [[13, 5], ["N", "Topic-Change", "N"], ["temp_edu:", None]], [[5, 13], ["N", "Topic-Change", "N"], ["temp_edu:", None]], [[14, 13], ["N", "Joint", "N"], ["temp_edu:", None]], [[13, 14], ["N", "Joint", "N"], ["temp_edu:", None]], [[14, 15], ["N", "Elaborate", "N"], ["temp_edu:", "In 2000 , a feud between Everlast and rapper Eminem coincided with the gold-selling '' Eat at Whitey 's '' ,"]], [[16, 14], ["S", "span", "N"], ["temp_edu:", None]], [[17, 16], ["N", "Joint", "S"], ["temp_edu:", "which included minor hits \" Black Jesus \" and \" Black Coffee \" ,"]], [[16, 17], ["S", "Joint", "N"], ["temp_edu:", "which included minor hits \" Black Jesus \" and \" Black Coffee \" ,"]], [[18, 16], ["N", "Joint", "S"], ["temp_edu:", "and featured a collaboration with Carlos Santana ."]], [[16, 18], ["S", "Joint", "N"], ["temp_edu:", "and featured a collaboration with Carlos Santana ."]], [[19, 13], ["N", "Joint", "N"], ["temp_edu:", None]], [[13, 19], ["N", "Joint", "N"], ["temp_edu:", None]], [[20, 19], ["N", "Joint", "N"], ["temp_edu:", "After the sale of the Tommy Boy Records ' master tapes to Warner Bros ."]], [[19, 20], ["N", "Joint", "N"], ["temp_edu:", "After the sale of the Tommy Boy Records ' master tapes to Warner Bros ."]], [[21, 19], ["N", "Joint", "N"], ["temp_edu:", None]], [[19, 21], ["N", "Joint", "N"], ["temp_edu:", None]], [[22, 21], ["N", "Joint", "N"], ["temp_edu:", "Records , Everlast signed with Island/Def Jam ,"]], [[21, 22], ["N", "Joint", "N"], ["temp_edu:", "Records , Everlast signed with Island/Def Jam ,"]], [[23, 21], ["N", "Joint", "N"], ["temp_edu:", "and released his '' White Trash Beautiful '' LP in 2004"]], [[21, 23], ["N", "Joint", "N"], ["temp_edu:", "and released his '' White Trash Beautiful '' LP in 2004"]], [2, 8], [2, 9], [2, 11], [2, 15], [2, 17], [2, 18], [2, 20], [4, 8], [4, 11], [4, 12], [4, 20], [4, 23], [8, 11], [8, 17], [8, 18], [8, 20], [8, 23], [9, 15], [9, 17], [9, 18], [11, 12], [11, 17], [11, 18], [11, 20], [11, 23], [12, 20], [12, 23], [15, 17], [15, 18], [15, 20], [17, 18], [17, 20], [18, 20], [20, 23]]


    # edu_list = [None, "From then on , the members continued their separate careers .", None, "Danny Boy founded an art company .", None, None, None, "DJ Lethal became a member of nu metal band Limp Bizkit ,", "who would cover \" Jump Around \" at live concerts , particularly in Limp Bizkit 's early years during the Family Values Tour 1998 .", None, "Everlast achieved multi-platinum solo fame in 1998 with his album '' Whitey Ford Sings the Blues .", "'' The first single from that album was \" What It 's Like \" .", None, None, "In 2000 , a feud between Everlast and rapper Eminem coincided with the gold-selling '' Eat at Whitey 's '' ,", None, "which included minor hits \" Black Jesus \" and \" Black Coffee \" ,", "and featured a collaboration with Carlos Santana .", None, "After the sale of the Tommy Boy Records ' master tapes to Warner Bros .", None, "Records , Everlast signed with Island/Def Jam ,", "and released his '' White Trash Beautiful '' LP in 2004"]

    # with open("../RST_example/graph_info_list.json") as f:
    #     result = json.load(f)
    #     print(result)

    # graph_info_list = process_graphinfo_list(graph_info_list)
    # G = Graph(graph_info_list,edu_list)
    # test()

