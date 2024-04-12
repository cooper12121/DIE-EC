#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/3 09:41
# @Author  : gao qiang
# @File    : wordnet_lexicalChain.py
# @Project : sota_end2end_parser
# -*- coding: utf-8 -*-

import json
import os

import traceback
import nltk
import string
from heapq import nlargest
from nltk.tag import pos_tag
from string import punctuation
from inspect import getsourcefile
from collections import defaultdict
from nltk.tokenize import word_tokenize
from os.path import abspath, join, dirname
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm
from os import path

"""
Create a list with all the relations of each noun 
"""


def relation_list(nouns):
    relation_list = defaultdict(list)

    for k in range(len(nouns)):
        relation = []
        for syn in wordnet.synsets(nouns[k], pos=wordnet.NOUN):
            for l in syn.lemmas():
                relation.append(l.name())
                if l.antonyms():
                    relation.append(l.antonyms()[0].name())
            for l in syn.hyponyms():
                if l.hyponyms():
                    relation.append(l.hyponyms()[0].name().split('.')[0])
            for l in syn.hypernyms():
                if l.hypernyms():
                    relation.append(l.hypernyms()[0].name().split('.')[0])
        relation_list[nouns[k]].append(relation)
    return relation_list


"""
Compute the lexical chain between each noun and their relation and 
apply a threshold of similarity between each word. 
"""


def create_lexical_chain(nouns, relation_list):
    lexical = []
    threshold = 0.5
    for noun in nouns:
        flag = 0
        for j in range(len(lexical)):
            if flag == 0:
                for key in list(lexical[j]):  # key指的是已有的链,list后只保留字典的key，故此处key是链的单词
                    if key == noun and flag == 0:
                        lexical[j][noun] += 1
                        flag = 1
                    elif key in relation_list[noun][0] and flag == 0:  # 判断key或Noun在不在其同义词中：把noun并入对应链
                        syns1 = wordnet.synsets(key, pos=wordnet.NOUN)
                        syns2 = wordnet.synsets(noun, pos=wordnet.NOUN)
                        if syns1[0].wup_similarity(syns2[0]) >= threshold:
                            lexical[j][noun] = 1
                            flag = 1
                    elif noun in relation_list[key][0] and flag == 0:
                        syns1 = wordnet.synsets(key, pos=wordnet.NOUN)
                        syns2 = wordnet.synsets(noun, pos=wordnet.NOUN)
                        if syns1[0].wup_similarity(syns2[0]) >= threshold:
                            lexical[j][noun] = 1
                            flag = 1
        if flag == 0:
            dic_nuevo = {}
            dic_nuevo[noun] = 1
            lexical.append(dic_nuevo)
            flag = 1
    return lexical


"""
Prune the lexical chain deleting the chains that are more weak with 
just a few words. 
"""


def prune(lexical):
    final_chain = []
    while lexical:
        result = lexical.pop()
        if len(result.keys()) == 1:
            for value in result.values():
                if value != 1:
                    final_chain.append(result)
        else:
            final_chain.append(result)
    return final_chain


"""
Class for summarize the text: 
    Input:
        text: The input text that we have read.
        lexical_chain: The final lexical chain with the most important
        n: The number of sentence we want our summary to have. 
    Output:
        summary: the n best sentence.
"""


class Summarizer:

    def __init__(self, threshold_min=0.1, threshold_max=0.9):
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self._stopwords = set(stopwords.words('english') + list(punctuation))

    """ 
      Compute the frequency of each of word taking into account the 
      lexical chain and the frequency of other words in the same chain. 
      Normalize and filter the frequencies. 
    """

    def return_frequencies(self, words, lexical_chain):
        frequencies = defaultdict(int)
        for word in words:
            for w in word:
                if w not in self._stopwords:
                    flag = 0
                    for i in lexical_chain:
                        if w in list(i.keys()):
                            frequencies[w] = sum(list(i.values()))
                            flag = 1
                            break
                    if flag == 0:
                        frequencies[w] += 1
        m = float(max(frequencies.values()))
        for w in list(frequencies.keys()):
            frequencies[w] = frequencies[w] / m
            if frequencies[w] >= self.threshold_max or frequencies[w] <= self.threshold_min:
                del frequencies[w]
        return frequencies

    """
      Compute the final summarize using a heap for the most importante 
      sentence and return the n best sentence. 
    """

    def summarize(self, sentence, lexical_chain, n):
        assert n <= len(sentence)
        word_sentence = [word_tokenize(s.lower()) for s in sentence]
        self.frequencies = self.return_frequencies(word_sentence, lexical_chain)
        ranking = defaultdict(int)
        for i, sent in enumerate(word_sentence):
            for word in sent:
                if word in self.frequencies:
                    ranking[i] += self.frequencies[word]
                    idx = self.rank(ranking, n)
        final_index = sorted(idx)
        return [sentence[j] for j in final_index]

    """
        Create a heap with the best sentence taking into account the 
        frequencie of each word in the sentence and the lexical chain. 
        Return the n best sentence. 
    """

    def rank(self, ranking, n):
        return nlargest(n, ranking, key=ranking.get)


def get_edu_chain(edu_list):
    '''
    得到edu_chain:用编号表示
    是否需要添加距离限制():
        认为词汇链通常存在于在距离较近的基本篇章单元之间，所以在构建词汇链时对距离添加了限制Dist，当两个筅筄筕的距离大于Dist之后，则不再判断筅筄筕之间是否存在词汇链。
    '''

    edu_noun_relation = []
    for edu in edu_list:
        if edu ==None or edu ==0:
            edu_noun_relation.append(None)
            continue
        if isinstance(edu,list):
            print("list_edu")
            edu_new=""
            edu_new+="".join(i for i in edu if i!=None and i!=0)
            edu=edu_new
        nouns, relation = get_chain(edu)
        edu_noun_relation.append([nouns, relation])
    lexical_chains = cal_similarity(edu_noun_relation)
    return lexical_chains

def cal_similarity(edu_noun_relation_list):
    lexical_chains = []
    for i in range(len(edu_noun_relation_list)):
        if edu_noun_relation_list[i]==None or (len(edu_noun_relation_list[i][0]) == 1 and edu_noun_relation_list[i][0][0] == 'none'): continue
        for j in range(i + 1, len(edu_noun_relation_list)):
            if edu_noun_relation_list[j]==None or (len(edu_noun_relation_list[j][0]) == 1 and edu_noun_relation_list[j][0][0] == 'none'): continue
            has_enough_chain = get_lexical(edu_noun_relation_list[i][0], edu_noun_relation_list[i][1],
                                           edu_noun_relation_list[j][0], edu_noun_relation_list[j][1])
            if has_enough_chain:
                lexical_chains.append([i + 1, j + 1])  # 要注意对节点从1开始编号的，故这里的编号也要从1开始,这样最总读入模型处理时同意减一即可
    return lexical_chains


def get_lexical(nouns_1, relations_1, nouns_2, relations_2):
    lexical_chain_num = 0
    # 假定当链数超过一定数量时才认为两个edu之间存在词汇链
    for i in range(len(nouns_1)):
        syns1 = wordnet.synsets(nouns_1[i], pos=wordnet.NOUN)
        if len(syns1) == 0: continue
        for j in range(len(nouns_2)):
            syns2 = wordnet.synsets(nouns_2[j], pos=wordnet.NOUN)
            if len(syns2) == 0: continue
            if syns1[0].wup_similarity(syns2[0]) >= threshold:
                lexical_chain_num += 1
    if lexical_chain_num > min(len(nouns_1), len(nouns_2)) / 2:  # 认为链的数量大于最小长度的一般才存在词汇链
        return True
    return False


def get_chain(input_txt):
    sentence = nltk.sent_tokenize(input_txt)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = [tokenizer.tokenize(w) for w in sentence]
    tagged = [pos_tag(tok) for tok in tokens]
    nouns = [word.lower() for i in range(len(tagged)) for word, pos in tagged[i] if pos in position]

    relation = relation_list(nouns)  # 这里得到nouns中的每个词对应的同义、下义等词
    return nouns, relation
    # final_chain = prune(lexical)
    """
    Print the lexical chain. 
    """
    # for i in range(len(final_chain)):
    #     print("Chain " + str(i + 1) + " : " + str(final_chain[i]))  # 数字表示出现次数


def readfile():
    files_path =os.listdir('./CDCR-ZH/')
    dir_name = os.path.dirname(files_path)

    for file in files_path:
        if os.path.isdir(dir_name + "/" + file): continue
        basename = path.basename(path.splitext(file)[0])
        data = json.load(open(file, 'r', encoding='utf-8'))
        print(f"start len:{len(data)}")
        result = []
        for doc in tqdm(data, desc="process_doc"):
            try:
                if 'mention_node_index' not in doc.keys() or 'graph_info_list' not in doc.keys():
                    # data.remove(doc)
                    continue
                edu_list =  doc['edu_list']
                lexical_chain = get_edu_chain(edu_list)
                if len(lexical_chain)>0:
                    doc['graph_info_list'].extend(lexical_chain)
                result.append(doc)
            except Exception as e:
                # data.remove(doc)
                print(e)
                print('\n', '>>>' * 20)
                print(traceback.print_exc())
                continue
        print(f"end len:{len(result)}")
        write_file(basename,result)


def write_file(filename,data):
    with open(f"../RST_parse/data_process/WEC-Eng/fina_processed/{filename}.json",'w',encoding='utf-8')as f:
        json.dump(data,f,ensure_ascii=False)

def test():
    l = {"a": 1, 'b': 2}
    m = list(l)
    print(m)


if __name__ == "__main__":
    """
    Read the .txt in this folder.
    """
    # in_txt = join(dirname(abspath(getsourcefile(lambda:0))) , "input.txt")
    # with open(in_txt, "r", encoding="utf-8" ) as f:
    #     input_txt = f.read()
    #     f.close()
    # print(input_txt,type(input_txt))
    """
    Return the nouns of the entire text.
    """
    position = ['NN', 'NNS', 'NNP', 'NNPS']
    threshold = 0.5
    # test()
    # get_edu_chain()
    readfile()
# 若用此方法做要做的修改
# 1 指明每条词汇链中词汇代表的edu的指向，得到edu链


#进一步清洗：keyerror:edu_list,graph的错误