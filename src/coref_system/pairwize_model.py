import math
import torch
from torch import nn

import dgl
from gatv2.GAT import GAT
from gatv2.graph import Graph


class PairWiseModelKenton(nn.Module):
    def __init__(self, f_in_dim, f_hid_dim, f_out_dim, embed_utils, embed_node,use_cuda,use_arguments=False):
        super(PairWiseModelKenton, self).__init__()
        self.W = nn.Linear(f_hid_dim, f_out_dim)
        # self.pairwize = self.get_sequential(9 * f_in_dim+4, f_hid_dim)
        self.attend = self.get_sequential(embed_utils.get_embed_size(), f_hid_dim)
        self.w_alpha = nn.Linear(f_hid_dim, 1)
        self.embed_utils = embed_utils
        self.use_cuda = use_cuda
        self.use_arguments=use_arguments

        if self.use_arguments:
            self.pairwize = self.get_sequential(9 * f_in_dim+4, f_hid_dim)
        else:
            self.pairwize = self.get_sequential(9* f_in_dim, f_hid_dim)

        self.arguments_pairwize=self.get_sequential(3*f_in_dim,f_hid_dim)

        self.embed_node = embed_node
        self.gat = GAT(in_feats=embed_node.get_embed_size(),out_feats=embed_node.get_embed_size())
        self.device = torch.device("cuda" if torch.cuda.is_available()else "cpu")

    @staticmethod
    def get_sequential(ind, hidd):
        return nn.Sequential(
            nn.Linear(ind, hidd),
            nn.ReLU(),
            nn.Linear(hidd, hidd),
            nn.ReLU(),
            nn.Linear(hidd, hidd),
            nn.ReLU(),
        )

    def forward(self, batch_features, bs):
        embeded_features, gold_labels = self.get_bert_rep(batch_features, bs)
        prediction = self.W(self.pairwize(embeded_features))  #[batch,9,hidden]
        return prediction, gold_labels

    def predict(self, batch_features, bs):
        output, gold_labels = self.__call__(batch_features, bs)
        prediction = torch.sigmoid(output)
        return prediction, gold_labels

    def get_bert_rep(self, batch_features, batch_size=32):
        mentions1, mentions2 = zip(*batch_features)

        if self.use_arguments:
            # (batch_size, embed_utils.get_embed_size())
            hiddens1, first1_tok, last1_tok, ment1_size ,arg0_1_hidden,arg0_1_size,arg1_1_hidden,arg1_1_size,loc_1_hidden,loc_1_size,time_1_hidden,time_1_size= zip(*self.embed_utils.get_mentions_rep(mentions1))
        
            hiddens2, first2_tok, last2_tok, ment2_size,arg0_2_hidden,arg0_2_size,arg1_2_hidden,arg1_2_size,loc_2_hidden,loc_2_size,time_2_hidden,time_2_size = zip(*self.embed_utils.get_mentions_rep(mentions2))
            #[batch,seqlen,hidden]
        
        else:
            hiddens1, first1_tok, last1_tok, ment1_size= zip(*self.embed_utils.get_mentions_rep(mentions1))
            hiddens2, first2_tok, last2_tok, ment2_size= zip(*self.embed_utils.get_mentions_rep(mentions2)) 

        max_ment_span = max([max(ment1_size), max(ment2_size)])
        hiddens1_pad = [torch.nn.functional.pad(hid, [0, 0, 0, max_ment_span - hid.shape[0]]) for hid in hiddens1] 
        hiddens2_pad = [torch.nn.functional.pad(hid, [0, 0, 0, max_ment_span - hid.shape[0]]) for hid in hiddens2]

        


        hiddens1_pad = torch.cat(hiddens1_pad)
        hiddens2_pad = torch.cat(hiddens2_pad)
        first1_tok = torch.cat(first1_tok).reshape(batch_size, -1)
        first2_tok = torch.cat(first2_tok).reshape(batch_size, -1)
        last1_tok = torch.cat(last1_tok).reshape(batch_size, -1)
        last2_tok = torch.cat(last2_tok).reshape(batch_size, -1)

        if self.use_cuda:
    
            hiddens1_pad = hiddens1_pad.to(self.device)
            hiddens2_pad = hiddens2_pad.to(self.device)
            first1_tok = first1_tok.to(self.device)
            first2_tok = first2_tok.to(self.device)
            last1_tok = last1_tok.to(self.device)
            last2_tok = last2_tok.to(self.device)

        attend1 = self.attend(hiddens1_pad)
        attend2 = self.attend(hiddens2_pad)

        att1_w = self.w_alpha(attend1)
        att2_w = self.w_alpha(attend2)#[batch,max_len,1]

        # Clean attention on padded tokens
        att1_w = att1_w.reshape(batch_size, max_ment_span)
        att2_w = att2_w.reshape(batch_size, max_ment_span)
        self.clean_attnd_on_zero(att1_w, ment1_size, att2_w, ment2_size, max_ment_span)

        att1_soft = torch.softmax(att1_w, dim=1) #[batch_size,max_sent_span]
        att2_soft = torch.softmax(att2_w, dim=1)
        hidden1_reshape = hiddens1_pad.reshape(batch_size, max_ment_span, -1)#[batch,max_sent,hidden]
        hidden2_reshape = hiddens2_pad.reshape(batch_size, max_ment_span, -1)
        att1_head = hidden1_reshape * att1_soft.reshape(batch_size, max_ment_span, 1)
        att2_head = hidden2_reshape * att2_soft.reshape(batch_size, max_ment_span, 1) 
        
        
        g1 = torch.cat((first1_tok, last1_tok, torch.sum(att1_head, dim=1)), dim=1)
        g2 = torch.cat((first2_tok, last2_tok, torch.sum(att2_head, dim=1)), dim=1)
        span1_span2 = g1 * g2 
        concat_result = torch.cat((g1, g2, span1_span2), dim=1)

       
        hidden_node_features1,hidden_node_features2 = self.get_node_rep(batch_features,batch_size)#[batch,hidden]
        g1 = torch.cat((first1_tok,last1_tok,torch.sum(att1_head,dim=1),hidden_node_features1),dim=1)#[batch,4*hidden]
        g2 = torch.cat((first2_tok,last2_tok,torch.sum(att2_head,dim=1),hidden_node_features2),dim=1)#[batch,4*hidden]
        span1_span2 = g1 * g2 
        concat_result = torch.cat((g1, g2, span1_span2), dim=1) 
        
     
        ret_golds = torch.tensor(self.get_gold_labels(batch_features))

        if self.use_cuda:

            concat_result = concat_result.to(self.device)
            ret_golds = ret_golds.to(self.device)
           
        return concat_result,ret_golds


    def get_node_rep(self,batch_features,batch_size=32):
        mentions1, mentions2 = zip(*batch_features)
        hiddens_node1,mentions1_node_index=zip(*self.embed_node.get_hidden_rep(mentions1))#(batch,node_num,1024) ([[tensor1,tenfor2,]])
        hiddens_node2,mentions2_node_index=zip(*self.embed_node.get_hidden_rep(mentions2))
       
        
        batched_graph1,graphs1 = self.construct_graph(mentions1,hiddens_node1)
        output1 = self.gat(batched_graph1,batched_graph1.ndata['feat'])
        # graphs = dgl.unbatch(batched_graph1)
        node_features1=[]
        start=0 
        for i,mention in enumerate(mentions1):
            node_index =mention.mention_node_index
            if node_index==None:node_index=1
            node_hidden1 = output1[start+node_index].tolist()
            start+=graphs1[i].num_nodes()
            node_features1.append(node_hidden1)
        hidden_node_features1 = torch.tensor(node_features1,requires_grad=True).view(batch_size,-1).to(self.device)#[batch,hidden]

        batched_graph2,graphs2 = self.construct_graph(mentions2,hiddens_node2)
        output2 = self.gat(batched_graph2,batched_graph2.ndata['feat'])

        node_features2=[]
        start=0 
        for i,mention in enumerate(mentions2):
            node_index =mention.mention_node_index
            if node_index==None:node_index=1
            node_hidden2 = output2[start+node_index].tolist()
            start+=graphs2[i].num_nodes()
            node_features2.append(node_hidden2)
        hidden_node_features2 = torch.tensor(node_features2,requires_grad=True).view(batch_size,-1).to(self.device)


        return hidden_node_features1,hidden_node_features2
        


        
    def construct_graph(self,mentions,hidden_nodes):
        graphs = []
        for i,mention in enumerate(mentions):
            try:

                graph = Graph(mention.graph_info_list,mention.edu_list,mention.mention_node_index,hidden_nodes[i]).G
                g = dgl.add_self_loop(graph).to(self.device)
                graphs.append(g)
            except Exception as e:
                print(e)
        batched_graph = dgl.batch(graphs).to(self.device)
        return batched_graph,graphs

    @staticmethod
    def clean_attnd_on_zero(attend1, ment_size1, attend2, ment_size2, max_mention_span):
        for i, vals in enumerate(list(zip(ment_size1, ment_size2))):
            val1, val2 = vals
            if val1 > max_mention_span or val2 > max_mention_span:
                raise Exception("Mention size exceed maximum!")
           
            attend1_fx = attend1[i:i + 1, 0:val1]
            attend1_fx = torch.nn.functional.pad(attend1_fx, [0, max_mention_span - val1, 0, 0], value=-math.inf)
            attend1[i:i + 1] = attend1_fx

            attend2_fx = attend2[i:i + 1, 0:val2]
            attend2_fx = torch.nn.functional.pad(attend2_fx, [0, max_mention_span - val2, 0, 0], value=-math.inf)
            attend2[i:i + 1] = attend2_fx

    @staticmethod
    def get_gold_labels(batch_features):
        batch_labels = list()
        for mentions1, mentions2 in batch_features:
            gold_label = 1 if mentions1.coref_chain == mentions2.coref_chain else 0
            batch_labels.append(gold_label)
        return batch_labels

    def set_embed_utils(self, embed_utils):
        self.embed_utils = embed_utils
    
    def set_embed_node(self, embed_node):
        self.embed_node = embed_node
