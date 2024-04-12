""" 
将Model加载进来再保存为当前torch的版本
"""
import torch
from transformers import *
def transfer():
    model = torch.load('./xl_model.pth')
    torch.save(model,'./xl_model_3_9.pth')