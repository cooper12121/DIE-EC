from __future__ import absolute_import, division, print_function

import argparse

import logging


import torch
import torch.nn as nn

from transformers import RobertaModel#RobertaForSequenceClassification
from transformers import BertTokenizer, BertModel

from torch.nn import CrossEntropyLoss, MSELoss

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

# from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertModel
# import torch.nn as nn

bert_hidden_dim = 1024
pretrain_model_dir = 'roberta-large' #'roberta-large' , 'roberta-large-mnli', 'bert-large-uncased'
# pretrain_model_dir = 'bert-large-uncased' #'roberta-large' , 'roberta-large-mnli', 'bert-large-uncased'
class RobertaForSequenceClassification(nn.Module):
    def __init__(self, tagset_size,args_dict):
        super(RobertaForSequenceClassification, self).__init__()
        self.tagset_size = tagset_size
        self.args_dict = args_dict
        self.encoder = RobertaModel.from_pretrained(pretrain_model_dir)
        # self.encoder = BertModel.from_pretrained(pretrain_model_dir)

        self.hidden_layer_0 = nn.Linear(bert_hidden_dim*3+4, bert_hidden_dim)
        self.hidden_layer_1 = nn.Linear(bert_hidden_dim, bert_hidden_dim)
        self.hidden_layer_2 = nn.Linear(bert_hidden_dim, bert_hidden_dim)
        self.single_hidden2tag = RobertaClassificationHead(bert_hidden_dim, tagset_size)

        self.hidden_layer_0_arg = nn.Linear(bert_hidden_dim * 3, bert_hidden_dim)
        self.hidden_layer_1_arg = nn.Linear(bert_hidden_dim, bert_hidden_dim)
        self.hidden_layer_2_arg = nn.Linear(bert_hidden_dim, bert_hidden_dim)
        self.single_hidden2tag_arg = RobertaClassificationHead(bert_hidden_dim, 1)


    def forward(self, all_input_a_ids, all_input_a_mask, all_segment_a_ids, all_span_a_mask,all_a_arg0_mask, all_a_arg1_mask, all_a_loc_mask, all_a_time_mask,all_cls_a_mask,
                all_input_b_ids, all_input_b_mask, all_segment_b_ids,
                all_span_b_mask, all_b_arg0_mask, all_b_arg1_mask, all_b_loc_mask,
                all_b_time_mask, all_cls_b_mask, all_label_ids):

        outputs_a = self.encoder(all_input_a_ids, all_input_a_mask, None)
        output_a_last_layer_tensor3= outputs_a[0] #(batch_size, sequence_length, hidden_size)`)
        span_a_reps = torch.sum(output_a_last_layer_tensor3*all_span_a_mask.unsqueeze(2), dim=1) #(batch, hidden)

        a_arg0_reps = torch.sum(output_a_last_layer_tensor3 * all_a_arg0_mask.unsqueeze(2), dim=1)  # (batch, hidden)
        a_arg1_reps = torch.sum(output_a_last_layer_tensor3 * all_a_arg1_mask.unsqueeze(2), dim=1)  # (batch, hidden)
        a_loc_reps = torch.sum(output_a_last_layer_tensor3 * all_a_loc_mask.unsqueeze(2), dim=1)  # (batch, hidden)
        a_time_reps = torch.sum(output_a_last_layer_tensor3 * all_a_time_mask.unsqueeze(2), dim=1)  # (batch, hidden)

        outputs_b = self.encoder(all_input_b_ids, all_input_b_mask, None)
        output_b_last_layer_tensor3= outputs_b[0] #(batch_size, sequence_length, hidden_size)`)
        span_b_reps = torch.sum(output_b_last_layer_tensor3*all_span_b_mask.unsqueeze(2), dim=1) #(batch, hidden)

        b_arg0_reps = torch.sum(output_b_last_layer_tensor3 * all_b_arg0_mask.unsqueeze(2), dim=1)  # (batch, hidden)
        b_arg1_reps = torch.sum(output_b_last_layer_tensor3 * all_b_arg1_mask.unsqueeze(2), dim=1)  # (batch, hidden)
        b_loc_reps = torch.sum(output_b_last_layer_tensor3 * all_b_loc_mask.unsqueeze(2), dim=1)  # (batch, hidden)
        b_time_reps = torch.sum(output_b_last_layer_tensor3 * all_b_time_mask.unsqueeze(2), dim=1)  # (batch, hidden)


        event_dict = {# 时间地点ren'wu
            "event1":{
                ""
            },
            "event2":{}
        }
        #先判断span的相似性，再判断参数的相似性（如果span相似性很低就不用判断参数）
        if self.args_dict['classification']:
            arg_0_score = torch.tanh(
                self.hidden_layer_0_arg(torch.cat([a_arg0_reps, b_arg0_reps, a_arg0_reps * b_arg0_reps], dim=1)))
            arg_0_score = torch.tanh(self.hidden_layer_1_arg(arg_0_score))
            arg_0_score = self.single_hidden2tag_arg(torch.tanh(self.hidden_layer_2_arg(arg_0_score)))

            arg_1_score = torch.tanh(
                self.hidden_layer_0_arg(torch.cat([a_arg1_reps, b_arg1_reps, a_arg1_reps * b_arg1_reps], dim=1)))
            arg_1_score = torch.tanh(self.hidden_layer_1_arg(arg_1_score))
            arg_1_score = self.single_hidden2tag_arg(torch.tanh(self.hidden_layer_2_arg(arg_1_score)))

            loc_score = torch.tanh(self.hidden_layer_0_arg(torch.cat([a_loc_reps, b_loc_reps, a_loc_reps * b_loc_reps], dim=1)))
            loc_score = torch.tanh(self.hidden_layer_1_arg(loc_score))
            loc_score = self.single_hidden2tag_arg(torch.tanh(self.hidden_layer_2_arg(loc_score)))

            time_score = torch.tanh(
                self.hidden_layer_0_arg(torch.cat([a_time_reps, b_time_reps, a_time_reps * b_time_reps], dim=1)))
            time_score = torch.tanh(self.hidden_layer_1_arg(time_score))
            time_score = self.single_hidden2tag_arg(torch.tanh(self.hidden_layer_2_arg(time_score)))

            combined_rep = torch.cat([span_a_reps, span_b_reps, span_a_reps * span_b_reps,
                                    arg_0_score, arg_1_score, loc_score, time_score
                                    ], dim=1)

            MLP_input = torch.tanh(self.hidden_layer_0(combined_rep))

            hidden_states_single = torch.tanh(self.hidden_layer_2(torch.tanh(self.hidden_layer_1(MLP_input))))

            logits = self.single_hidden2tag(hidden_states_single)

            loss_fct = CrossEntropyLoss()

            loss = loss_fct(logits.view(-1, self.tagset_size), all_label_ids.view(-1))
            return logits,loss
        elif self.args_dict['classification']:
            a = self.args_dict['A']   
            b = self.args_dict['B']

            mention_score = torch.cosine_similarity(span_a_reps,span_b_reps,dim=1)# 消融，不加权重
            arg_0_score   =torch.cosine_similarity(a_arg0_reps,b_arg1_reps,dim=1)
            arg_1_score   =torch.cosine_similarity(a_arg1_reps,b_arg1_reps,dim=1)
            loc_score     =torch.cosine_similarity(a_loc_reps,b_loc_reps,dim=1)
            time_socre    =torch.cosine_similarity(a_time_reps,b_time_reps,dim=1)

            score = a*mention_score+b*(arg_0_score+arg_1_score+loc_score+time_socre)/4 #串接，拼接

            loss_fct = MSELoss()

            loss = loss_fct(score.view(-1, self.tagset_size), all_label_ids.view(-1))
            return score,loss



class RobertaClassificationHead(nn.Module):
    """wenpeng overwrite it so to accept matrix as input"""

    def __init__(self, bert_hidden_dim, num_labels):
        super(RobertaClassificationHead, self).__init__()
        self.out_proj = nn.Linear(bert_hidden_dim, num_labels)

    def forward(self, features):
        x = features
        x = self.out_proj(x)
        return x
