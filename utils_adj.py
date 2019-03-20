#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2018/10/22 4:09
# @Author : {ZM7}
# @File : utils.py
# @Software: PyCharm

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse as sp
import random


def data_session_seq_mask(session_seq, item_tail, node, node_num):
    ##构造history session
    us_lens = [len(u) for s in session_seq for u in s]
    gr_lens = [len(s) for s in session_seq]  #每个session_group的中session的数量
    len_max = max(us_lens)                   #最长的session，每个session包含最大item的数量
    group_max = max(gr_lens)                 #最大的session组,每个用户历史session包含的最多session数量
    max_node = np.max(node_num)              #最大的节点数量
    items = [sub_node.tolist()+(max_node-len(sub_node))*[0] for sub_node in node]

    session = [[sess + item_tail*(len_max-len(sess)) for sess in sub_sess] +
               [item_tail*(len_max)]*(group_max-len(sub_sess)) for sub_sess in session_seq]
    session_alias = [[[np.where(np.array(sub_node_) == s)[0][0] for s in subsub_sess] for subsub_sess in sub_sess]
                     for sub_node_, sub_sess in zip(node, session)]
    # session_mask = [[[1]*len(sess) + item_tail*(len_max-len(sess)) for sess in sub_sess] +
    #            [item_tail*(len_max)]*(group_max-len(sub_sess)) for sub_sess in session_seq]
    session_mask = [[len(sess) for sess in sub_sess]+[1]*(group_max-len(sub_sess)) for sub_sess in session_seq]

    ##构造seq
    # seq_lens = [len(upois) for upois in seq]
    # max_seq = max(seq_lens)
    # seq_seq = [upois + item_tail * (max_seq - le) for upois, le in zip(seq, seq_lens)]
    # seq_alias = [[np.where(np.array(sub_node_) == s)[0][0] for s in sub_seq] for sub_node_, sub_seq in zip(node_, seq_seq)]

    return np.array(session), np.array(session_mask),np.array(session_alias), np.array(gr_lens),\
           group_max, len_max, np.array(items)


class Data(object):
    def __init__(self, data, batch_size, sparse=False, shuffle=True):
        self.length = len(data[0])
        self.batch_size = batch_size
        self.sparse = sparse
        self.shuffle = shuffle
        self.all_session = np.asarray(data[0])
        self.all_seq = np.asarray(data[1])
        self.all_seq_alias = np.asarray(data[2])
        self.all_seq_mask = np.asarray(data[3])
        self.all_seq_max = data[4]
        self.users = np.asarray(data[5])
        self.targets = np.asarray(data[6])
        self.adj_in_row = np.asarray(data[7])
        self.adj_in_col = np.asarray(data[8])
        self.adj_in_values = np.asarray(data[9])
        self.adj_out_row = np.asarray(data[10])
        self.adj_out_col = np.asarray(data[11])
        self.adj_out_values = np.asarray(data[12])
        self.node = np.asarray(data[13])
        self.node_num = np.asarray(data[14])

        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.all_session = self.all_session[shuffled_arg]
            self.all_seq = self.all_seq[shuffled_arg]
            self.all_seq_alias = self.all_seq_alias[shuffled_arg]
            self.all_seq_mask = self.all_seq_mask[shuffled_arg]
            self.users = self.users[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
            self.adj_in_row = self.adj_in_row[shuffled_arg]
            self.adj_in_col = self.adj_in_col[shuffled_arg]
            self.adj_in_values = self.adj_in_values[shuffled_arg]
            self.adj_out_row = self.adj_out_row[shuffled_arg]
            self.adj_out_col = self.adj_out_col[shuffled_arg]
            self.adj_out_values = self.adj_out_values[shuffled_arg]
            self.node = self.node[shuffled_arg]
            self.node_num = self.node_num[shuffled_arg]
        self.slices = self.slice()

    def slice(self):
        n_batch = int(self.length / self.batch_size)
        if self.length % self.batch_size !=0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * self.batch_size), n_batch)
        slices[-1] = np.arange(self.length-self.batch_size, self.length)
        return slices

    def adj_sparse(self, adj_in_row, adj_in_col, adj_in_values, adj_out_row, adj_out_col, adj_out_values):
        batch_size = len(adj_in_row)
        array_in = []
        array_out = []
        value_in = []
        value_out = []
        for i in range(batch_size):
            array_in.append(np.stack([np.ones(len(adj_in_row[i])) * i, adj_in_row[i], adj_in_col[i]]).transpose())
            array_out.append(np.stack([np.ones(len(adj_out_row[i])) * i, adj_out_row[i], adj_out_col[i]]).transpose())
            value_in.append(adj_in_values[i])
            value_out.append(adj_out_values[i])
        return np.vstack(array_in).astype(np.int64), np.hstack(value_in).astype(np.float32), \
               np.vstack(array_out).astype(np.int64), np.hstack(value_out).astype(np.float32)

    def generate_batch(self):
        for i in self.slices:
            session, session_mask, session_alias, session_len, group_max, session_max, items=\
                data_session_seq_mask(self.all_session[i], [0],  self.node[i], self.node_num[i])
            if self.sparse:
                adj_in_indices, adj_in_values, adj_out_indices, adj_out_values\
                    = self.adj_sparse(self.adj_in_row[i], self.adj_in_col[i], self.adj_in_values[i],
                                      self.adj_in_row[i], self.adj_out_col[i], self.adj_out_values[i])

                yield [session, session_mask, session_len, group_max, session_max, session_alias,
                       self.all_seq[i], self.all_seq_mask[i], self.all_seq_max, self.users[i], self.targets[i], items,
                       self.all_seq_alias[i], adj_in_indices, adj_in_values, adj_out_indices, adj_out_values,
                       np.max(self.node_num[i])]

    def generate_ver(self, size=0.1):
        i = random.sample(range(1, self.length), self.batch_size)
        session, session_mask, session_alias, session_len, group_max, session_max, items = \
            data_session_seq_mask(self.all_session[i], [0], self.node[i], self.node_num[i])
        if self.sparse:
            adj_in_indices, adj_in_values, adj_out_indices, adj_out_values \
                = self.adj_sparse(self.adj_in_row[i], self.adj_in_col[i], self.adj_in_values[i],
                                  self.adj_in_row[i], self.adj_out_col[i], self.adj_out_values[i])

            return [session, session_mask, session_len, group_max, session_max, session_alias,
                   self.all_seq[i], self.all_seq_mask[i], self.all_seq_max, self.users[i], self.targets[i], items,
                   self.all_seq_alias[i], adj_in_indices, adj_in_values, adj_out_indices, adj_out_values,
                   np.max(self.node_num[i])]

