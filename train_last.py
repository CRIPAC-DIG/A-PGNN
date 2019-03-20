#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2018/12/4 2:16
# @Author : {ZM7}
# @File : train_last.py
# @Software: PyCharm
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import csv
import numpy as np
import argparse
import datetime
import time
from model_last import Graph, parse_function_, run_epoch, eval_epoch,random_name, random_validation

lastfm_path = './datasets/3_user_sessions.pickle'
parser = argparse.ArgumentParser()
parser.add_argument('--data', default='reddit', help='dataname: xing/reddit/last')
parser.add_argument('--dataset', default='sample', help='dataset name: all_data/sample')
parser.add_argument('--max_session',type=int, default=50)
parser.add_argument('--max_length',type=int, default=20)
parser.add_argument('--buffer_size',type=int, default=10000)
parser.add_argument('--ggnn_drop', type=float, default=0.0)
parser.add_argument('--adj', default='adj', help='adj_all')
parser.add_argument('--last', action='store_true')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--epoch', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=50, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--userSize', type=int, default=50, help='hidden state size')
parser.add_argument('--decay', type=int, default=None, help='learning rate decay after step')
parser.add_argument('--l2', type=float, default=0.0, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--graph', default='ggnn', help='gcn/gat')
parser.add_argument('--mode', default='transformer')
parser.add_argument('--decoder_attention', action='store_true', help='decoder_self_attention')
parser.add_argument('--encoder_attention', action='store_true', help='encoder_self_attention')
parser.add_argument('--user_', action='store_true', help='user_embedding')
parser.add_argument('--history_', action='store_true', help='history_embedding')
parser.add_argument('--behaviour_', action='store_true', help='behaviour_embedding')
parser.add_argument('--pool', type=str, default='max', help='max/mean')
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
opt = parser.parse_args()
if opt.last:
    data_path = './datasets/'+opt.data+'/' + opt.graph + '/'+'tfrecord_'\
                + str(opt.max_session) + '_' + str(opt.max_length)+ '_' + opt.adj + '_last' + '/'+opt.dataset
else:
    data_path = './datasets/'+opt.data+'/' + opt.graph + '/' + 'tfrecord_' \
                + str(opt.max_session) + '_' + str(opt.max_length)+ '_'+ opt.adj + '/' + opt.dataset
if opt.data == 'last':
    n_item = 94285
    n_user = 997
elif opt.data == 'reddit':
    n_item = 27453
    n_user = 18271
elif opt.data == 'xing':
    if opt.max_length == 20:
        n_item = 59122
        n_user = 11479
    elif opt.max_length == 30:
        n_item = 57887
        n_user = 11463


padded_shape = { 'A_in': [None, None],
                'A_out': [None, None],
                'session_alias_shape': [None],
                'session_alias': [None, None],
                'seq_mask': [],
                'session_len':[],
                'tar': [],
                'user': [],
                'session_mask':[None],
                'seq_alias': [None],
                'num_node': [],
                'all_node': [None],
                'A_in_shape': [None],
                'A_out_shape': [None],
                'A_in_row': [None],
                'A_in_col': [None],
                'A_out_row': [None],
                'A_out_col': [None]}
#--------------------------从文件中读取文件名-------------------------------
# train_filenames = tf.train.match_filenames_once(data_path+'/'+'train_user_'+'*'+'.tfrecord')
# test_filenames = tf.train.match_filenames_once(data_path+'/'+'test_user_'+'*'+'.tfrecord')
# train_filenames = tf.train.match_filenames_once(data_path+'/'+'user_*/'+'train_*.tfrecord')
test_filenames = tf.train.match_filenames_once(data_path+'/'+'user_*/'+'test_*.tfrecord')
train_filenames = random_name(data_path+'/'+'user_*/'+'train_*.tfrecord') #打乱训练集数据顺序，生成部分验证集检测是否过拟合
valid_filenames = random_validation(data_path+'/'+'user_*/'+'test_*.tfrecord')
# = random_name(data_path+'/'+'user_*/'+'test_*.tfrecord')
#--------------------------从文件中读取数据---------------------------------
train_dataset = tf.data.TFRecordDataset(train_filenames)
test_dataset = tf.data.TFRecordDataset(test_filenames)
valid_dataset = tf.data.TFRecordDataset(valid_filenames)
#-----------对数据集进行shuffle和padding操作-----------------------
train_dataset = train_dataset.map(parse_function_(opt.max_session)).shuffle(buffer_size=opt.buffer_size)
test_dataset = test_dataset.map(parse_function_(opt.max_session))
valid_dataset = valid_dataset.map(parse_function_(opt.max_session))

train_batch_padding_dataset = train_dataset.padded_batch(opt.batchSize, padded_shapes=padded_shape, drop_remainder=True)
train_iterator = train_batch_padding_dataset.make_initializable_iterator()

test_batch_padding_dataset = test_dataset.padded_batch(opt.batchSize, padded_shapes=padded_shape, drop_remainder=True)
test_iterator = test_batch_padding_dataset.make_initializable_iterator()

valid_batch_padding_dataset = valid_dataset.padded_batch(opt.batchSize, padded_shapes=padded_shape, drop_remainder=True)
valid_iterator = valid_batch_padding_dataset.make_initializable_iterator()
#-----------模型初始化----------------------------
model = Graph(hidden_size=opt.hiddenSize, user_size=opt.userSize, batch_size=opt.batchSize, seq_max=opt.max_length,
              group_max=opt.max_session,
              n_item=n_item, n_user=n_user, lr=opt.lr,
              l2=opt.l2, step=opt.step, decay=opt.decay, ggnn_drop=opt.ggnn_drop, graph=opt.graph, mode=opt.mode,
              decoder_attention=opt.decoder_attention, encoder_attention=opt.encoder_attention,
              behaviour_=opt.behaviour_,  pool=opt.pool)

train_data = train_iterator.get_next()
test_data = test_iterator.get_next()
valid_data = valid_iterator.get_next()

with tf.variable_scope('model', reuse=None):
    train_loss, train_opt = model.forward(train_data['A_in'], train_data['A_out'], train_data['all_node'],
                                          train_data['seq_alias'],train_data['seq_mask'], train_data['session_alias'],
                                          train_data['session_len'], train_data['session_mask'], train_data['tar'],
                                          train_data['user'])

with tf.variable_scope('model', reuse=True):
    test_loss, test_index = model.forward(test_data['A_in'], test_data['A_out'], test_data['all_node'],
                                          test_data['seq_alias'], test_data['seq_mask'], test_data['session_alias'],
                                          test_data['session_len'], test_data['session_mask'], test_data['tar'],
                                          test_data['user'], train=False)
with tf.variable_scope('model', reuse=True):
    valid_loss, valid_index = model.forward(valid_data['A_in'], valid_data['A_out'], valid_data['all_node'],
                                           valid_data['seq_alias'], valid_data['seq_mask'], valid_data['session_alias'],
                                           valid_data['session_len'], valid_data['session_mask'], valid_data['tar'],
                                           valid_data['user'], train=False)
print(opt)
best_result = [0, 0, 0, 0, 0, 0] #hit5,hit10,hit20,mrr5,mrr10,mrr20
best_epoch = [0, 0, 0, 0, 0, 0]
#----------开始训练----------------------------------
with tf.Session() as sess:
    tf.local_variables_initializer().run()
    sess.run(tf.global_variables_initializer())
    step = 0
    date_now = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d')
    if opt.user_:
        csvfile = str(date_now) + opt.graph + '_' + opt.data + '_' +str(opt.max_session)+'_'+str(opt.max_length)+'_d' +\
                  str(opt.hiddenSize) + '_u' + str(opt.userSize) + '_' + opt.mode

        his_csvfile = str(date_now) + opt.graph + '_history_' + opt.data + '_' + str(opt.max_session) + '_' + str(
                      opt.max_length) + '_d' + str(opt.hiddenSize) + '_u' + str(opt.userSize) + '_' + opt.mode
    else:
        csvfile = str(date_now) + opt.graph + '_' + opt.data + '_' + str(opt.max_session) + '_' + str(opt.max_length) + '_d' + str(
                  opt.hiddenSize) + '_' + opt.mode
        his_csvfile = str(date_now) + opt.graph + '_history_'  + opt.data + '_' + str(opt.max_session) + '_' + str(
                      opt.max_length) + '_d' + str(opt.hiddenSize) + '_' + opt.mode
    for epoch in range(opt.epoch):
        sess.run([train_iterator.initializer, test_iterator.initializer])
        print('epoch: ', epoch, '====================================================')
        print('start training: ', datetime.datetime.now())
        step, mean_train_loss = run_epoch(sess, train_loss, train_opt, valid_loss, valid_index, valid_iterator,
                                          valid_data, step, max_length=opt.max_length, max_session=opt.max_session)
        print('start predicting: ', datetime.datetime.now())
        mean_test_loss, hit5, hit10, hit20, mrr5, mrr10, mrr20, len_index, history_index\
            = eval_epoch(sess, test_index, test_loss, test_data, max_length=opt.max_length, max_session=opt.max_session)
        #----select recall or hit-----------------
        if hit5>=best_result[0]:
            best_result[0] = hit5
            best_epoch[0] = epoch
        if hit10>=best_result[1]:
            best_result[1] = hit10
            best_epoch[1] = epoch
        if hit20>=best_result[2]:
            best_result[2] = hit20
            best_epoch[2] = epoch
        #------select mrr------------------
        if mrr5>=best_result[3]:
            best_result[3] = mrr5
            best_epoch[3] = epoch
        if mrr10>=best_result[4]:
            best_result[4] = mrr10
            best_epoch[4] = epoch
        if mrr20>=best_result[5]:
            best_result[5] = mrr20
            best_epoch[5] = epoch

        with open(csvfile, 'a') as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerow(['\nepoch: ' + str(epoch) + '\n'])
            writer.writerows(len_index)

        with open(his_csvfile, 'a') as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerow(['\nepoch: ' + str(epoch) + '\n'])
            writer.writerows(history_index)

        print('train_loss:%.4f\ttest_loss:%.4f\tRecall@5:%.4f\tRecall@10:%.4f\tRecall@20:%.4f\tMRR@5:%.4f'
              '\tMRR10@10:%.4f\tMMR@20:%.4f\tEpoch:%d,%d,%d,%d,%d,%d' %
              (mean_train_loss, mean_test_loss, best_result[0], best_result[1], best_result[2], best_result[3],
               best_result[4], best_result[5], best_epoch[0], best_epoch[1],
               best_epoch[2],best_epoch[3], best_epoch[4], best_epoch[5]))




