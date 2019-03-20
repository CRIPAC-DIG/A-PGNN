#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2018/10/22 10:50
# @Author : {ZM7}
# @File : preprocess.py
# @Software: PyCharm
import networkx as nx
import numpy as np
import pandas as pd
import pickle
import sys as sys
from utils import Data
from sklearn.externals import joblib
from scipy import sparse as sp
from datetime import datetime
import argparse
import datetime
import time
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='dataset name: all_data/sample')
parser.add_argument('--user', type=int, default=50, help='the number of need user')
parser.add_argument('--g raph', default='gcn')
parser.add_argument('--xing', action='store_true')
opt = parser.parse_args()
train_ggnn_path = './datasets/'+opt.dataset+'/train_data_adj.pickle'
test_ggnn_path = './datasets/'+opt.dataset+'/test_data_adj.pickle'
train_gcn_path = './datasets/'+opt.dataset+'/train_data_adj_gcn.pickle'
test_gcn_path = './datasets/'+opt.dataset+'/test_data_adj_gcn.pickle'
##------换一种切割方式----------------
xing_data = './datasets/xing.csv'
train_xing_path = './datasets/' + opt.dataset+ '/train_data_xing.pickle'
test_xing_path = './datasets/' + opt.dataset+ '/test_data_xing.pickle'
#------------------------------------------------
# train_object = './datasets/'+opt.dataset+'/train.pkl'
# test_object = './datasets/'+opt.dataset+'/test.pkl'
original_data = './datasets/sessions.hdf'



train_key = 'train'
test_key = 'test'


def save_pickle(data_object, data_file):
    pickle.dump(data_object, open(data_file, 'wb'))


def save_object(data_object, data_file):
    joblib.dump(data_object, data_file)


def rename_data(train_data, test_data):
    user_al = pd.DataFrame(
        {'user_id': train_data['user_id'].unique(), 'user': np.arange(train_data['user_id'].unique().shape[0]) + 1})
    item_al = pd.DataFrame(
        {'item_id': train_data['item_id'].unique(), 'item': np.arange(train_data['item_id'].unique().shape[0]) + 1})
    train_data = train_data.merge(user_al, on='user_id').merge(item_al, on='item_id').sort_values(
        ['user', 'session_id', 'created_at'])
    test_data = test_data.merge(user_al, on='user_id').merge(item_al, on='item_id').sort_values(
        ['user', 'session_id', 'created_at'])
    return train_data.sort_values(['user', 'created_at']), test_data.sort_values(['user', 'created_at'])

#切分序列将序列切成session，seq，user,target格式
def segment_seq(train_data, test_data, train_path, test_path, graph='ggnn', user_number=None, xing=False):
    train_session = []
    train_seq = []
    train_user = []
    train_tar = []
    train_adj_in_row = []
    train_adj_in_col = []
    train_adj_in_values = []
    train_adj_out_row = []
    train_adj_out_col = []
    train_adj_out_values = []
    train_node_num = []
    train_node = []

    test_session = []
    test_seq = []
    test_user = []
    test_tar = []
    test_adj_in_row = []
    test_adj_in_col = []
    test_adj_in_values = []
    test_adj_out_row = []
    test_adj_out_col = []
    test_adj_out_values = []
    test_node_num = []
    test_node = []
    if not xing:
        train_data, test_data = rename_data(train_data, test_data)
    if user_number:
        user_sample = np.random.choice(test_data['user'].unique(), user_number)
        train_data = train_data[train_data['user'].isin(user_sample)]
        test_data = test_data[test_data['user'].isin(user_sample)]

    def select_train(data):
        session = []
        seq = []
        user = []
        tar = []
        A_in_row = []
        A_in_col = []
        A_in_values = []
        A_out_row = []
        A_out_col = []
        A_out_values = []
        all_node = []
        num_node = []
        all_sess = data['session_id'].unique()
        for i in range(1, len(all_sess)):
            all_seq = data[data['session_id'] == all_sess[i]]['item'].values.tolist()
            sub_sess = [data[data['session_id'] == sess]['item'].values.tolist() for sess in all_sess[0:i]]
            sub_node = np.hstack(sub_sess)
            for j in range(len(all_seq) - 1):
                session.append(sub_sess)
                sub_seq = all_seq[0:j + 1]
                seq.append(sub_seq)
                tar.append(all_seq[j + 1])
                user.append(data['user'].unique()[0])
                node = np.unique(np.hstack([sub_node, sub_seq, [0]]))
                num_node.append(len(node))
                all_node.append(node)
                if graph == 'ggnn':
                    u_A = np.zeros((len(node), len(node)))
                elif graph == 'gcn':
                    u_A = np.eye(len(node))
                for u_input in sub_sess:
                    for k in np.arange(len(u_input)-1):
                        u = np.where(node == u_input[k])[0][0]
                        v = np.where(node == u_input[k + 1])[0][0]
                        u_A[u][v] = 1
                for l in np.arange(len(sub_seq)-1):
                    u = np.where(node == sub_seq[l])[0][0]
                    v = np.where(node == sub_seq[l + 1])[0][0]
                    u_A[u][v] = 1
                u_sum_in = np.sum(u_A, 0)
                u_sum_in[np.where(u_sum_in == 0)] = 1
                u_A_in = np.divide(u_A, u_sum_in)
                u_sum_out = np.sum(u_A, 1)
                u_sum_out[np.where(u_sum_out == 0)] = 1
                u_A_out = np.divide(u_A.transpose(), u_sum_out)
                u_A_in = sp.coo_matrix(u_A_in)
                u_A_out = sp.coo_matrix(u_A_out)
                A_in_row.append(u_A_in.row)
                A_in_col.append(u_A_in.col)
                A_in_values.append(u_A_in.data)
                A_out_row.append(u_A_out.row)
                A_out_col.append(u_A_out.col)
                A_out_values.append(u_A_out.data)

        return [session, seq, user, tar, A_in_row, A_in_col, A_in_values, A_out_row, A_out_col, A_out_values,
                all_node, num_node]

    def select_test(data):
        test_sess = []
        test_seq = []
        test_user = []
        test_tar = []
        A_in_row = []
        A_in_col = []
        A_in_values = []
        A_out_row = []
        A_out_col = []
        A_out_values = []
        num_node = []
        all_node = []
        all_sess = data['session_id'].unique()
        if len(all_sess) == 1:
            return None
        else:
            all_seq = data[data['session_id'] == all_sess[-1]]['item'].values.tolist()
            sub_sess = [data[data['session_id'] == sess]['item'].values.tolist() for sess in all_sess[0:-1]]
            sub_node = np.hstack(sub_sess)
            for i in range(len(all_seq) - 1):
                test_sess.append(sub_sess)
                sub_seq = all_seq[0:i + 1]
                test_seq.append(sub_seq)
                test_tar.append(all_seq[i + 1])
                test_user.append(data['user'].unique()[0])
                node = np.unique(np.hstack([sub_node, sub_seq, [0]]))
                all_node.append(node)
                num_node.append(len(node))
                if graph == 'ggnn':
                    u_A = np.zeros((len(node), len(node)))
                elif graph == 'gcn':
                    u_A = np.eye(len(node))
                for u_input in sub_sess:
                    for j in np.arange(len(u_input)-1):
                        u = np.where(node == u_input[j])[0][0]
                        v = np.where(node == u_input[j + 1])[0][0]
                        u_A[u][v] = 1
                for k in np.arange(len(sub_seq)-1):
                    u = np.where(node == sub_seq[k])[0][0]
                    v = np.where(node == sub_seq[k + 1])[0][0]
                    u_A[u][v] = 1
                u_sum_in = np.sum(u_A, 0)
                u_sum_in[np.where(u_sum_in == 0)] = 1
                u_A_in = np.divide(u_A, u_sum_in)
                u_sum_out = np.sum(u_A, 1)
                u_sum_out[np.where(u_sum_out == 0)] = 1
                u_A_out = np.divide(u_A.transpose(), u_sum_out)
                u_A_in = sp.coo_matrix(u_A_in)
                u_A_out = sp.coo_matrix(u_A_out)
                A_in_row.append(u_A_in.row)
                A_in_col.append(u_A_in.col)
                A_in_values.append(u_A_in.data)
                A_out_row.append(u_A_out.row)
                A_out_col.append(u_A_out.col)
                A_out_values.append(u_A_out.data)
            return [test_sess, test_seq, test_user, test_tar,
                    A_in_row, A_in_col, A_in_values, A_out_row, A_out_col, A_out_values, all_node, num_node]

    for group in train_data.groupby('user').apply(select_train):
        train_session.extend(group[0])
        train_seq.extend(group[1])
        train_user.extend(group[2])
        train_tar.extend(group[3])
        train_adj_in_row.extend(group[4])
        train_adj_in_col.extend(group[5])
        train_adj_in_values.extend(group[6])
        train_adj_out_row.extend(group[7])
        train_adj_out_col.extend(group[8])
        train_adj_out_values.extend(group[9])
        train_node.extend(group[10])
        train_node_num.extend(group[11])

    print('train numbers:%d'%(len(train_session)))
    print('train user number:%d'%(len(np.unique(train_user))))
    test_u = test_data['user'].unique()
    all_data = pd.concat([train_data, test_data])
    if not xing:
        test = all_data[all_data['user'].isin(test_u)].sort_values(['user', 'created_at'])
    else:
        test = all_data[all_data['user'].isin(test_u)].sort_values(['user', 'ts'])
    for group in test.groupby('user').apply(select_test).dropna():
        test_session.extend(group[0])
        test_seq.extend(group[1])
        test_user.extend(group[2])
        test_tar.extend(group[3])
        test_adj_in_row.extend(group[4])
        test_adj_in_col.extend(group[5])
        test_adj_in_values.extend(group[6])
        test_adj_out_row.extend(group[7])
        test_adj_out_col.extend(group[8])
        test_adj_out_values.extend(group[9])
        test_node.extend(group[10])
        test_node_num.extend(group[11])
    print('test numbers:%d'%(len(test_session)))
    print('test user number:%d'%(len(np.unique(test_user))))
    train_seq, train_seq_alias, train_seq_mask, train_seq_max = data_seq_mask(train_seq, train_node, [0])
    test_seq, test_seq_alias, test_seq_mask, test_seq_max = data_seq_mask(test_seq, test_node, [0])
    save_pickle((train_session, train_seq, train_seq_alias, train_seq_mask, train_seq_max, train_user, train_tar,
                 train_adj_in_row, train_adj_in_col, train_adj_in_values,
                 train_adj_out_row, train_adj_out_col, train_adj_out_values,
                 train_node, train_node_num), train_path)
    save_pickle((test_session, test_seq, test_seq_alias, test_seq_mask, test_seq_max, test_user, test_tar,
                 test_adj_in_row, test_adj_in_col, test_adj_in_values,
                 test_adj_out_row, test_adj_out_col, test_adj_out_values,
                 test_node, test_node_num), test_path)

#生成session数据，根据切分好的数据集，得到session，session_mask, session_len, session_max, group_max
def data_session_mask(session_seq, item_tail):
    us_lens = [len(u) for s in session_seq for u in s]
    gr_lens = [len(s) for s in session_seq]  #每个session_group的中session的数量
    len_max = max(us_lens)                   #最长的session，每个session包含最大item的数量
    group_max = max(gr_lens)                 #最大的session组,每个用户历史session包含的最多session数量
    session = [[sess + item_tail*(len_max-len(sess)) for sess in sub_sess] +
               [item_tail*(len_max)]*(group_max-len(sub_sess)) for sub_sess in session_seq]
    # session_mask = [[[1]*len(sess) + item_tail*(len_max-len(sess)) for sess in sub_sess] +
    #            [item_tail*(len_max)]*(group_max-len(sub_sess)) for sub_sess in session_seq]
   # session_mask = [[len(sess) for sess in sub_sess]+[1]*(group_max-len(sub_sess)) for sub_sess in session_seq]
    return np.array(session)#, np.array(session_mask), np.array(gr_lens), len_max, group_max

def data_seq_mask(seq, node, item_tail):
    us_lens = [len(upois) for upois in seq]
    len_max = max(us_lens)            #seq序列中包含最大的item的数量
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(seq, us_lens)]
    seq_alias = [[np.where(np.array(sub_node_) == s)[0][0] for s in sub_seq] for sub_node_, sub_seq in
                 zip(node, us_pois)]
    #us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return us_pois, seq_alias, us_lens, len_max


def xing(data):
    train, test = [], []
    for u in data['user'].unique():
        id = data[data['user']==u]['session_id'].unique()
        train.append(data[(data['user']==u) & (data['session_id'].isin(id[0:-1]))])
        test.append(data[(data['user']==u) & (data['session_id'] == id[-1])])
    return pd.concat(train), pd.concat(test)


if __name__ == '__main__':
    if opt.xing:
        data = pd.read_csv(xing_data)
        train_data, test_data = xing(data)
        train_path = train_xing_path
        test_path = test_xing_path
    else:
        train_data = pd.read_hdf(original_data, train_key)
        test_data = pd.read_hdf(original_data, test_key)
        if opt.graph == 'ggnn':
            train_path = train_ggnn_path
            test_path = test_ggnn_path
        elif opt.graph == 'gcn':
            train_path = train_gcn_path
            test_path = test_gcn_path
    print('start:',datetime.datetime.now())
    if opt.dataset == 'sample':
        segment_seq(train_data, test_data, train_path, test_path, graph=opt.graph, user_number=opt.user, xing=opt.xing)
    else:
        segment_seq(train_data, test_data, train_path, test_path, graph=opt.graph, xing=opt.xing)
    print('end:',datetime.datetime.now())



