"""
Usage:
    inference.py --tpf=<TestPosFile> --tnf=<testNegFile> --te=<TestEmbed> --mf=<ModelFile> [--cuda=<b>] --arguments=<f> [--dataset=<d>]

Options:
    -h --help       Show this screen.
    --cuda=<y>      True/False - Whether to use cuda device or not [default: True]
    --arguments=<f> whether use the information of arguments [default:False]
    --dataset=<d>   wec/ecb/zh - which dataset to generate for [default: zh]

"""

import logging
import ntpath
import os,sys
import json
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2]) 
sys.path.append(root_path)
import torch
from docopt import docopt

from src.train import accuracy_on_dataset
from src.utils.log_utils import create_logger_with_fh
from src.dataobjs.dataset import EcbDataSet
from src.utils.embed_utils import EmbedFromFile
from src.Metric.calculate_metric import process_cluster

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    os.chdir(sys.path[0])
    argv = ['--tpf','../datasets/CDEC-zh/arguments/pair/test_event_validated_PosPairs_-1.pickle','--tnf','../datasets/CDEC-zh/arguments/pair/test_event_validated_NegPairs_-1.pickle','--te','../datasets/CDEC-zh/arguments/embed/test_event_validated_roberta_large.pickle','--mf','../model/predict/zh_pairwise_model_iter_10.pickle','--arguments',True]
    _arguments = docopt(__doc__, argv=argv, help=True, version=None, options_first=False)
    print(_arguments)
    _model_in = _arguments.get("--mf")
    _dataset_arg = _arguments.get("--dataset")
    _event_test_file_pos = _arguments.get("--tpf")
    _event_test_file_neg = _arguments.get("--tnf")
    
    _use_cuda = True if _arguments.get("--cuda").lower() == "true" else False
    _use_arguments=True if _arguments.get('arguments')==True else False
    if _use_arguments:
        _embed_file = _arguments.get("--te")
    else:
        _embed_file='../datasets/CDEC-zh/embed/test_event_validated_roberta_large.pickle'

    _dataset = EcbDataSet()

    log_param_str = os.path.dirname(_model_in) + "/inference_" + ntpath.basename(_model_in)
    create_logger_with_fh(log_param_str)

    logger.info(f"use_arguments={_use_arguments},Loading the model from-" + _model_in)
    _pairwize_model = torch.load(_model_in)
    _embed_utils = EmbedFromFile([_embed_file])
    _pairwize_model.set_embed_utils(_embed_utils)
    _pairwize_model.eval()

    positive_ = _dataset.load_pair_pickle(_event_test_file_pos)
    negative_ = _dataset.load_pair_pickle(_event_test_file_neg)
    split_feat = _dataset.create_features_from_pos_neg(positive_, negative_)

    _, _, _, dev_f1,all_labels, all_predictions =accuracy_on_dataset("Test", 0, _pairwize_model, split_feat)
    best_predict,best_label=all_predictions,all_labels
    best_predict,best_label=best_predict.tolist(),best_label.tolist()
    
    process_cluster(best_predict,best_label,split_feat,"test")