""" 
将Model加载进来再保存为当前torch的版本
"""
import torch
from transformers import *
from segmenter import *
import os,sys
def transfer():
    model = torch.load('data/seg/models_saved/EN_200.model', map_location=torch.device('cpu'))
    torch.save(model,'data/seg/models_saved/EN_200_3_8.model')
if __name__ == "__main__":
    os.chdir(sys.path[0])
    transfer()