#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/21 17:15
# @Author  : gao qiang
# @File    : pre_tree.py
# @Project : sota_end2end_parser
# @Software: PyCharm

import pickle
import json
from RST_parse.rst_tree import RST_tree


class tree:
    def __init__(self, temp_edu, rel, node_type):
        self.temp_edu = temp_edu
        self.rel = rel
        self.node_type = node_type


def preorder_traversal(root, parent):
    # 遍历过程已经按方向保存了信息，故不用再后续便利了


    # 为了防止出现非连通图，是否应该把N这条边设置成双向
    global node_index
    if root is not None:
        root = RST_tree(rel=root.rel, type_=root.type, temp_edu=root.temp_edu, l_ch=root.left_child,
                        r_ch=root.right_child)
        # print(f"root_type:{root.type},rel:{root.rel},temp_edu:{root.temp_edu}")
        list_append(root.temp_edu, root.rel, root.type)
        root.node_index = node_index + 1
        node_index += 1
        if root.type != "Root":
            # edu_index=node_index,这样寻找mention所在edu时可直接对应node_index，然后再将edu中的none去除后加入seq送入encoder即可
            # if root.type!="Root":#不行，要保留下所有的节点信息，根是自环
            if root.type == "S" or root.type == "Sz":  # 卫星，由孩子指向父亲，root自环
                src_tuple = (root.node_index, parent.node_index)
                rel_tuple = (root.type, root.rel, parent.type)
            else:  # NN不用考虑,把N看成双向可否？
                if (parent.left_child and parent.left_child.type == "N") and(parent.right_child and parent.right_child.type == "N"):  # 双向
                    src_tuple = (root.node_index, parent.node_index)
                    rel_tuple = (root.type, root.rel, parent.type)
                    if src_tuple[0]!=None and src_tuple[1]!=None:
                        graph_info_list.append([src_tuple, rel_tuple, ("temp_edu:", root.temp_edu)])
                src_tuple = (parent.node_index, root.node_index)#parent的type是针对所有子树而言的，并不是与孩子间的关系，故不用考虑parent与root之间的关系
                rel_tuple = (parent.type, root.rel, root.type)
            if src_tuple[0] != None and src_tuple[1] != None:
                graph_info_list.append(
                    [src_tuple, rel_tuple, ("temp_edu:", root.temp_edu)])

        preorder_traversal(root.left_child, root)
        preorder_traversal(root.right_child, root)


def list_append(edu, rel, node_type):
    edu_list.append(edu)
    rel_list.append(rel)
    type_list.append(node_type)


def read_trees(file_path):
    with open(file_path, 'rb') as f:
        trees = pickle.load(f)
        f.close()
    return trees


def write_list(filepath, file):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(file, f)
        f.close()


def write_iterate(ite, file_path, append_=False):
    if append_:
        with open(file_path, "a") as f:
            for line in ite:
                if line is None:
                    f.write("None")
                else:
                    f.write(line)
                f.write('\n')
            f.close()
    else:
        with open(file_path, "w") as f:
            for line in ite:
                if line is None:
                    f.write("None")
                else:
                    f.write(line)
                f.write('\n')
            f.close()


def write_json(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)


def save_preorder_list(index):
    write_iterate(file_path=f'./RST_example/rel_list_{index}.txt', ite=rel_list)
    write_iterate(file_path=f'./RST_example/edu_list_{index}.txt', ite=edu_list)
    write_iterate(file_path=f'./RST_example/node_type_list_{index}.txt', ite=type_list)
    rel_list.clear()
    edu_list.clear()
    type_list.clear()


def order_tree(tree):
    # tree_path = "../data/e2e/trees.pkl"
    # trees = read_trees(tree_path)
    global node_index, graph_info_list, edu_list, rel_list, type_list
    node_index = 0
    edu_list, rel_list, type_list = [], [], []
    graph_info_list = []
    '''
    rst树的结构应该要用后序遍历？#
    #其次RST关系的左右方向，是的，因为RST树是自底向上构造。由卫星指向核心
    #即对于父节点：左孩子是核心，父指向左孩子；右孩子是卫星，右孩子指向父节点：则最终就有一条右孩子指向左孩子的路径
    # tree=RST_tree(is_leaf=False,rel=tree.rel,type_=tree.type,temp_edu=tree.temp_edu,l_ch=tree.left_child,r_ch=tree.right_child)
    #先先序遍历分配所有，再后续遍历得到图吧'''
    parent = RST_tree(rel=tree.rel, type_=tree.type, temp_edu=tree.temp_edu, l_ch=tree.left_child,
                      r_ch=tree.right_child)
    preorder_traversal(tree, parent)  # postorder_traversal(tree,tree)
    # write_json('./RST_example/graph_info_list.json', graph_info_list)
    return graph_info_list,edu_list,rel_list,type_list