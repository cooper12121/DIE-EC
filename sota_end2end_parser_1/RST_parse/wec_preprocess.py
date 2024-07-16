#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/25 10:23
# @Author  : gao qiang
# @Site    : 
# @File    : zh_preprocess.py
# @Project : sota_end2end_parser

import json
import os, sys
import traceback

root_path = os.path.abspath(__file__)
root_path = '\\'.join(root_path.split('\\')[:-2])  # windows 用
print(f"root_path:{root_path}")
sys.path.append(root_path)
# os.chdir(root_path)
from os import path
from pipeline import *
from RST_parse.pre_tree import order_tree
from tqdm import tqdm


def readfile(files_path):
    # os.chdir(sys.path[0])
    files = os.listdir(files_path)
    dir_name = os.path.dirname(files_path)
    print(dir_name,files)
    for file in files:
        if 'dev' in file or os.path.isdir(files_path+file): continue
        basename = path.basename(path.splitext(file)[0])
        output_file = dir_name + "/final_processed/" + basename + "_processed.json"
        data = load_json_file(dir_name + "/" + basename + ".json")
        data_processed = process_data(data)
        write_json_file(output_file, data_processed)


def load_json_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        f.close()
        return data


def write_json_file(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
        f.close()


def process_data(data):
    result = []
    for doc in tqdm(data, desc="process_doc"):
        try:
            sentences = []
            text = ""
            for i in doc['mention_context']:
                text += i + " "
                if i == ".":
                    sentences.append(text)
                    text = ""
            if i != ".": sentences.append(text)  # 不以.结尾的情况
            # token_start, token_end = doc['tokens_number'][0], doc['tokens_number'][-1]
            tree = construct_rst(sentences)[0]  
            graph_info_list, edu_list, rel_list, type_list = order_tree(tree)
            if len(graph_info_list)==0:continue
            if len(edu_list) == 1 and isinstance(edu_list[0], tuple):
                edu_list = list(edu_list[0])
            elif len(edu_list) == 1 and isinstance(edu_list[0], list):
                edu_list = edu_list[0]
            doc['graph_info_list'], doc['edu_list'], doc['rel_list'], doc[
                'type_list'] = graph_info_list, edu_list, rel_list, type_list
            doc['mention_node_index'] = calculate_event2edu(edu_list, doc)
            result.append(doc)
        except Exception as e:
            print(e)
            print('\n', '>>>' * 20)
            print(traceback.print_exc())
    return result


def calculate_event2edu(edu_list, doc):
    '''
        计算event_mention 所在的edu
        token_start,token_end
    '''

    notNone_edu_list = []
    edu_index = []
    for i, edu in enumerate(edu_list):
        if edu != None and not isinstance(edu, int):
            notNone_edu_list.append(edu)
            edu_index.append(i)  
    
    index = 0
    str_text = []
    range_list = []
    try:
        for i, edu in enumerate(notNone_edu_list):
            split_txt = edu.split(" ")
            range_list.append((index, len(split_txt) - 1))
            index += len(split_txt)
            for j in split_txt:
                str_text.append(j)
    except Exception as e:
        print(e)
        print('\n', '>>>' * 20)
        print(traceback.print_exc())
    # assert doc['mention_context'][start:end + 1] == str_text[start:end + 1]
    start, end = doc['tokens_number'][0], doc['tokens_number'][-1]
    edu_number = 0  
    for i in range(len(range_list)):
        if range_list[i][0] <= start and end <= range_list[i][1]:
            edu_number = i
            break

    node_index = edu_index[edu_number]
    return node_index


def construct_rst(text: list):
    # os.chdir(sys.path[-1])
    sents_dt = prep_seg(text)
    seg_edus = do_seg(sents_dt)
    # print(seg_edus)
    # # parsing
    trees_ = do_parse(seg_edus)
    # print(trees_)
    return trees_


if __name__ == "__main__":
    # E:\direction\EE\sota_end2end_parser\RST_parse
    readfile("./RST_parse/data_process/ecb+/")

'''

'''
# def test():
#     l = ['In', '1993', ',', 'Lench', 'Mob', 'member', ',', 'J-Dee', ',', 'was', 'sentenced', 'to', 'life', 'imprisonment', 'for', 'attempted', 'murder', ',', 'and', 'Ice', 'Cube', 'did', 'not', 'produce', 'their', 'next', 'album', ',', "''", 'Planet', 'of', 'tha', 'Apes', "''", '.', 'Around', 'this', 'time', 'in', '1993', ',', 'he', 'also', 'worked', 'with', 'Tupac', 'Shakur', 'on', 'his', 'album', "''", 'Strictly', '4', 'My', 'N.I.G.G.A.Z.', ',', "''", 'appearing', 'on', 'the', 'track', '"', 'Last', 'Wordz', '"', 'with', 'Ice-T', '.', 'He', 'also', 'did', 'a', 'song', 'with', 'Dr.', 'Dre', 'for', 'the', 'first', 'time', 'since', 'he', 'left', 'N.W.A', ':', '"', 'Natural', 'Born', 'Killaz', '"', ',', 'for', 'the', "''", 'Murder', 'Was', 'The', 'Case', "''", 'soundtrack']
#     senten = " ".join(i for i in l if i!=".")
#     print(senten)
# test()
