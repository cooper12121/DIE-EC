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
sys.path.append(root_path)
os.chdir(sys.path[-1])

from os import path

from RST_parse.pre_tree import order_tree
from tqdm import tqdm
import time

from pipeline import build_pipeline

pipeline = build_pipeline(schema="topdown", segmenter_name="svm", use_gpu=False)
# text = "由于欧洲当时英法两国和德国呈宣而不战的「假战」状态，故公众的注意力皆放到了冬季战争上，并一面倒地认为苏联为侵略者、同情芬兰，进而使世界各地有不少志愿军前往芬兰加入对苏作战包括法西斯义大利（150人，其中最多的是来自邻国瑞典，整场战争里共有8,760人的参加战斗，提供（装备格斗士式战斗机）、装备波佛斯40公厘的防空砲营来提升芬兰北部图尔库的防御，瑞典志愿军仅300人来自这里，另外还有数量不详的爱沙尼亚,350名芬裔美国人（由指挥）和210名来自各地国家的志愿者于战争结束前来到芬兰助战。"
# # ""
# start = time.time()
# tree = pipeline(text)
# end = time.time()
# print(f"time:{end - start}")


# 188

def test_maxlength(data):
    max_length = 0
    for doc in tqdm(data, desc="doc"):
        text = ""
        for i in doc['mention_context']:
            text += i
        max_length = max_length if max_length > len(text) else len(text)
    print(f"max_length:{max_length}")


def readfile(files_path):
    # os.chdir(sys.path[0])

    files = os.listdir(files_path)
    dir_name = os.path.dirname(files_path)
    for file in files:
        if os.path.isdir(dir_name + "/" + file) or "dev" not in file: continue
        # if file != "Train_Event_gold_mentions.json": continue
        basename = path.basename(path.splitext(file)[0])
        output_file = dir_name + "/process_rst/" + basename + "_processed.json"
        data = load_json_file(dir_name + "/" + basename + ".json")
        # test_maxlength(data)
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
    count = 0
    for doc in tqdm(data, desc="process_doc"):
        count += 1
        # if count == 40:
        #     print(count)
        try:

            text = ""
            for i in doc['mention_context']:
                text += i
            text.strip()
            # 不以.结尾的情况
            # token_start, token_end = doc['tokens_number'][0], doc['tokens_number'][-1]
            tree = pipeline(text)  # 一般只有一个树，有多个时取第一个
            graph_info_list, edu_list, rel_list, type_list = order_tree(tree)
            if len(graph_info_list) == 0: continue
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
        注意中文的mention_context不是按空格划分，故无法通过字的个数判断其在mention_contenxt的位置
        直接采用判断的方式吧
    '''

    notNone_edu_list = []
    edu_index = []
    mention = doc['tokens_str']
    try:
        for i, edu in enumerate(edu_list):
            if edu != None and not isinstance(edu, int):
                notNone_edu_list.append(edu)
                # edu_index.append(i)  # 非0edu对应的node
                if mention in edu:
                    return i
        # 对edu分词判断其范围

        # for j in edu:
        #     str_text.append(j)
    except Exception as e:
        print(e)
        print('\n', '>>>' * 20)
        print(traceback.print_exc())
        return None
    # assert doc['mention_context'][start:end + 1] == str_text[start:end + 1]
    # start, end = doc['tokens_number'][0], doc['tokens_number'][-1]
    # edu_number = 0  # 非0edu的位置
    # for i in range(len(range_list)):
    #     if range_list[i][0] <= start and end <= range_list[i][1]:
    #         edu_number = i
    #         break

    # node_index = edu_index[edu_number]
    # return node_index


if __name__ == "__main__":
    readfile("./CDCR-ZH/")
    pass
'''
问题处理：用.分割句子存在问题  A.A.E.E 某些单词用.连接，这种情况怎么考虑,应该再拼接句子时考虑
'''
# def test():
#     l = ['In', '1993', ',', 'Lench', 'Mob', 'member', ',', 'J-Dee', ',', 'was', 'sentenced', 'to', 'life', 'imprisonment', 'for', 'attempted', 'murder', ',', 'and', 'Ice', 'Cube', 'did', 'not', 'produce', 'their', 'next', 'album', ',', "''", 'Planet', 'of', 'tha', 'Apes', "''", '.', 'Around', 'this', 'time', 'in', '1993', ',', 'he', 'also', 'worked', 'with', 'Tupac', 'Shakur', 'on', 'his', 'album', "''", 'Strictly', '4', 'My', 'N.I.G.G.A.Z.', ',', "''", 'appearing', 'on', 'the', 'track', '"', 'Last', 'Wordz', '"', 'with', 'Ice-T', '.', 'He', 'also', 'did', 'a', 'song', 'with', 'Dr.', 'Dre', 'for', 'the', 'first', 'time', 'since', 'he', 'left', 'N.W.A', ':', '"', 'Natural', 'Born', 'Killaz', '"', ',', 'for', 'the', "''", 'Murder', 'Was', 'The', 'Case', "''", 'soundtrack']
#     senten = " ".join(i for i in l if i!=".")
#     print(senten)
# test()
