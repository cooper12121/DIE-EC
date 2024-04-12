#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/24 23:13
# @Author  : gao qiang
# @Site    : 
# @File    : rst_tree
# @Project : sota_end2end_parser

import copy



def get_blank(line):
    count = 0
    while line[count] == " ":
        count += 1
    return count


class RST_tree:
    def __init__(self, type_=None, l_ch=None, r_ch=None, rel=None,temp_edu=None,node_index=None):

        self.type = type_
        self.left_child = l_ch
        self.right_child = r_ch
        self.rel = rel
        self.temp_edu = temp_edu
        self.node_index=node_index
        # self.is_leaf=is_leaf
        # self.edu_index=edu_index

