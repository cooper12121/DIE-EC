#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/21 17:05
# @Author  : gao qiang
# @Site    : 
# @File    : raw_process.py
# @Project : sota_end2end_parser
# @Software: PyCharm
import string


def load_raw_sentence(raw_file):
    lines = []
    for line in raw_file.readlines():
        lines.append(line)
    word_list = []
    sentence_idx = []
    raw_word_list = []
    word_dict = {}
    i = 0
    for line in lines:
        i = i + 1
        line = line.replace('\n', '')
        word_line = line.split(' ')
        for word in word_line:
            raw_word_list.append(word)
            for c in string.punctuation:
                word = word.replace(c, '')
            if word != '':
                word_list.append(word)
                sentence_idx.append(i)
                key = len(word_list) - 1
                value = len(raw_word_list) - 1
                word_dict[key] = value
    return word_list, sentence_idx, word_dict, raw_word_list
