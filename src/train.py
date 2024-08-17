"""

Usage:
    train.py --tpf=<TrainPosFile> --tnf=<TrainNegFile> --dpf=<DevPosFile> --dnf=<DevNegFile>
                    --te=<TrainEmbed> --de=<DevEmbed> --ten=<TrainEmbedNode> --den=<DevEmbedNode> --mf=<ModelFile> [--bs=<x>] [--lr=<y>] [--ratio=<z>] [--itr=<k>]
                    [--cuda=<b>] [--ft=<b1>] [--wd=<t>] [--hidden=<w>] [--dataset=<d>] [--arguments=<f>] [--rst=<t>] [--lexical_chains=<t>]

Options:
    -h --help       Show this screen.
    --bs=<x>        Batch size [default: 32]
    --lr=<y>        Learning rate [default: 5e-4]
    --ratio=<z>     Ratio of positive:negative, were negative is the controlled list (ratio=-1 => no ratio) [default: -1]
    --itr=<k>       Number of iterations [default: 5]
    --cuda=<y>      True/False - Whether to use cuda device or not [default: True]
    --ft=<b1>       Fine-tune the LM or not [default: False]
    --wd=<t>        Adam optimizer Weight-decay [default: 0.01]
    --hidden=<w>    hidden layers size [default: 150]
    --dataset=<d>   wec/ecb/zh - which dataset to generate for [default: zh]
    --arguments=<f> whether use the information of arguments [default:False]
    --rst=<t>       whether use the rst trees [default:True]
    --lexical_chains=<t> whether use the lexical chains trees [default:True]
"""
from datetime import datetime
import logging

import numpy as np
import random
import torch
import os,sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2]) 
sys.path.append(root_path)

import json
import pickle
from tqdm import tqdm,trange
from docopt import docopt
from src.utils.embed_utils import EmbedFromFile
from src.utils.eval_utils import get_confusion_matrix, get_prec_rec_f1
from src.utils.log_utils import create_logger_with_fh
from src.utils.io_utils import create_and_get_path

from src.dataobjs.dataset import DataSet, Split
from src.coref_system.pairwize_model import PairWiseModelKenton
from transformers import  get_linear_schedule_with_warmup



from src.preprocess_edu_embed import EmbedNodeHidden

logger = logging.getLogger(__name__)


def train_pairwise(pairwize_model, train, validation, batch_size, epochs=5,
                   lr=1e-7, model_out=None, weight_decay=0.01):
    # loss_func = torch.nn.CrossEntropyLoss()
    loss_func = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(pairwize_model.parameters(), lr, weight_decay=weight_decay)
    # optimizer = AdamW(pairwize_model.parameters(), lr)
    dataset_size = len(train)

    num_train_optimization_steps = int(dataset_size/_batch_size) * epochs
    num_warmup_ratio=0.1
    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=num_warmup_ratio*num_train_optimization_steps,num_training_steps=num_train_optimization_steps)

    best_result_so_far = -1
    best_predict,best_label=None,None
    best_model=None

    for epoch in trange(epochs,desc="Epoch"):
        pairwize_model.train()
        end_index = batch_size
        random.shuffle(train)

        cum_loss, count_btch = (0.0, 1)
        for start_index in tqdm(range(0, dataset_size, batch_size),desc="Iteration"):
            if end_index > dataset_size:
                end_index = dataset_size

            optimizer.zero_grad()

            batch_features = train[start_index:end_index].copy()
            bs = end_index - start_index
            prediction, gold_labels = pairwize_model(batch_features, bs)

            loss = loss_func(prediction, gold_labels.reshape(-1, 1).float())
            loss.backward()
            optimizer.step()
            scheduler.step()

            cum_loss += loss.item()
            end_index += batch_size
            count_btch += 1

            # if count_btch % 10000 == 0:
            #     report = "%d: %d: loss: %.10f:" % (epoch + 1, end_index, cum_loss / count_btch)
            #     logger.info(report)

        pairwize_model.eval()
        # accuracy_on_dataset("Train", epoch + 1, pairwize_model, train)
        _, _, _, dev_f1,all_labels, all_predictions = accuracy_on_dataset("Dev", epoch + 1, pairwize_model, validation)
        # accuracy_on_dataset(accum_count_btch / 10000, embed_utils, pairwize_model, test, use_cuda)
        pairwize_model.train()
        
        if best_result_so_far < dev_f1:
            logger.info("Found better model saving")
            # torch.save(pairwize_model, model_out + "iter_" + str(epoch + 1))
            best_model=pairwize_model
            best_result_so_far = dev_f1
            
        logger.info('epoch:{},loss:{}'.format(epoch+1,cum_loss))
    torch.save(best_model, model_out + "_iter_" + str(epoch + 1)+".pickle")
    return best_result_so_far


def accuracy_on_dataset(testset, epoch, pairwize_model, features, batch_size=5000):
    all_labels, all_predictions = run_inference(pairwize_model, features, batch_size=batch_size)
    accuracy = torch.mean((all_labels == all_predictions).float())
    tn, fp, fn, tp = get_confusion_matrix(all_labels, all_predictions)
    precision, recall, f1 = get_prec_rec_f1(tp, fp, fn)

    logger.info("%s: %d: Accuracy: %.10f: precision: %.10f: recall: %.10f: f1: %.10f" % \
                (testset + "-Acc", epoch, accuracy.item(), precision, recall, f1))

    return accuracy, precision, recall, f1,all_labels, all_predictions


def run_inference(pairwize_model, features, round_pred=True, batch_size=10000):
    dataset_size = len(features)
    end_index = batch_size
    labels = list()
    predictions = list()
    for start_index in range(0, dataset_size, batch_size):
        if end_index > dataset_size:
            end_index = dataset_size

        batch_features = features[start_index:end_index].copy()
        batch_size = end_index - start_index
        # batch_predictions, batch_label = pairwize_model.module.predict(batch_features, batch_size)
        batch_predictions, batch_label = pairwize_model.predict(batch_features, batch_size)

        if round_pred:
            batch_predictions = torch.round(batch_predictions.reshape(-1)).long()

        predictions.append(batch_predictions.detach())
        labels.append(batch_label.detach())
        end_index += batch_size

    all_labels = torch.cat(labels).cpu()
    all_predictions = torch.cat(predictions).cpu()

    return all_labels, all_predictions


def init_basic_training_resources():
    torch.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)

    embed_files = [_train_embed, _dev_embed]
    embed_utils = EmbedFromFile(embed_files)

    embed_node_file=[_trainEmbedNode,_devEmbedNode]
    embed_node = EmbedNodeHidden(embed_node_file)

    pairwize_model = PairWiseModelKenton(embed_utils.embed_size, _hidden_size, 1, embed_utils,embed_node, _use_cuda)
    train_dataset = DataSet.get_dataset(_dataset_arg, ratio=_ratio, split=Split.Train)
    dev_dataset = DataSet.get_dataset(_dataset_arg, split=Split.Dev)
    train_feat = train_dataset.load_pos_neg_pickle(_train_pos_file, _train_neg_file)
    validation_feat = dev_dataset.load_pos_neg_pickle(_dev_pos_file, _dev_neg_file)
    # pickle.dump(validation_feat,open(f'../model/predict/dev/{_dataset_arg}_use_argu_{_use_arguments}_dev_best_validation_dataset.pickle','wb'))
    device_ids=[0,1]
    if _use_cuda:
        torch.cuda.manual_seed(1234)
        # pairwize_model.cuda()
        device = torch.device('cuda:{}'.format(device_ids[0]))
        n_gpu = torch.cuda.device_count()
        logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    
    # pairwize_model = torch.nn.DataParallel(pairwize_model,device_ids=device_ids,output_device=device_ids[0])
    pairwize_model.to(device)

    return train_feat, validation_feat, pairwize_model


if __name__ == '__main__':

    os.chdir(sys.path[0])


   
    argv = ['--tpf','../datasets/CDEC-zh/final_processed/pair/train_event_PosPairs_10.pickle',
            '--tnf','../datasets/CDEC-zh/final_processed/pair/train_event_NegPairs_10.pickle',
            '--dpf','../datasets/CDEC-zh/final_processed/pair/dev_event_validated_PosPairs_-1.pickle',
            '--dnf','../datasets/CDEC-zh/final_processed/pair/dev_event_validated_NegPairs_-1.pickle',
            '--te','../datasets/CDEC-zh/final_processed/embed/train_event_roberta_large.pickle',
            '--de','../datasets/CDEC-zh/final_processed/embed/dev_event_validated_roberta_large.pickle',
            '--ten','../datasets/CDEC-zh/final_processed/gat_embed/train_event_roberta_large.pickle',
            '--den','../datasets/CDEC-zh/final_processed/gat_embed/dev_event_validated_roberta_large.pickle',
            '--mf','zh_pairwise_model','--ratio',-1,'--bs',1024,'--lr',1e-5,'--hidden',512,'--arguments',True,'--rst',True,'--lexical_chains',True]

    #adfj
    _arguments = docopt(__doc__, argv=argv, help=True, version=None, options_first=False)
    start_time = datetime.now()
    dt_string = start_time.strftime("%Y-%m-%d_%H-%M")
    print(_arguments)
    _output_folder = create_and_get_path("checkpoints")
    # _output_folder = create_and_get_path("")
    _batch_size = int(_arguments.get("--bs"))
    _learning_rate = float(_arguments.get("--lr"))
    _ratio = int(_arguments.get("--ratio"))
    _iterations = int(_arguments.get("--itr"))
    _use_cuda = True if _arguments.get("--cuda").lower() == "true" else False
    _use_arguments=True if _arguments.get('arguments')==True else False
    _fine_tune = True if _arguments.get("--ft").lower() == "true" else False
    _weight_decay = float(_arguments.get("--wd"))
    _hidden_size = int(_arguments.get("--hidden"))
    _dataset_arg = _arguments.get("--dataset")

    _train_pos_file = _arguments.get("--tpf")
    _train_neg_file = _arguments.get("--tnf")
    _dev_pos_file = _arguments.get("--dpf")
    _dev_neg_file = _arguments.get("--dnf")
    _train_embed = _arguments.get("--te")
    _dev_embed = _arguments.get("--de")

    _use_rst = _arguments.get("--rst")
    _use_lexical_chains = _arguments.get("lexical_chains")

    _trainEmbedNode = _arguments.get("--ten")
    _devEmbedNode = _arguments.get("--den")

    _model_file = _output_folder + "/" + _arguments.get("--mf")+"no_rst"

    log_params_str = dt_string+"_ds_" + _dataset_arg + "_lr_" + str(_learning_rate) + "_bs_" + str(_batch_size) + "_r" + \
                     str(_ratio) + "_itr" + str(_iterations)
    create_logger_with_fh(_output_folder + "/train_89_" + log_params_str)
 
    logger.info(f"use_rst={_use_rst},use_lexical_chains={_use_lexical_chains},use_arguments={_use_arguments},"+"train_set=" + _dataset_arg + ", lr=" + str(_learning_rate) + ", bs=" + str(_batch_size) +
                ", ratio=1:" + str(_ratio) + ", itr=" + str(_iterations) +
                ", hidden_s=" + str(_hidden_size) + ", weight_decay=" + str(_weight_decay))
    logger.info("dev/test ratio:-1")#
    _event_train_feat, _event_validation_feat, _pairwize_model = init_basic_training_resources()

    train_pairwise(_pairwize_model, _event_train_feat, _event_validation_feat, _batch_size,
                   _iterations, _learning_rate, model_out=_model_file, weight_decay=_weight_decay)
