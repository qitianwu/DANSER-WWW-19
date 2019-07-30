import os
import time
import pickle
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import csv
import eval
from input import DataInput
from model import Model

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
random.seed(1234)
np.random.seed(1234)
tf.set_random_seed(1234) 

learning_rate = 0.1
keep_prob = 0.5
lambda1 = 0.001
lambda2 = 0.001
trunc_len = 10
train_batch_size = 64
test_batch_size = 64

workdir = '/home/myronwu/DANSER-WWW-19' # change to your workdir
with open(workdir+'/data/dataset.pkl', 'rb') as f:
	train_set = pickle.load(f)
	test_set = pickle.load(f)
with open(workdir+'/data/list.pkl', 'rb') as f:
    u_friend_list = pickle.load(f)
    u_read_list = pickle.load(f)
    uf_read_list = pickle.load(f) 
    i_friend_list = pickle.load(f)
    i_read_list = pickle.load(f)
    if_read_list = pickle.load(f)
    i_link_list = pickle.load(f)
    user_count, item_count = pickle.load(f)

def calc_metric(score_label_u):
	score_label_u = sorted(score_label_u, key=lambda d:d[0], reverse=True)
	precision = np.array([eval.precision_k(score_label_u, k) for k in range(1, 21)])
	ndcg = np.array([eval.ndcg_k(score_label_u, k) for k in range(1, 21)])
	auc = eval.auc(score_label_u)
	mae = eval.mae(score_label_u)
	rmse = eval.rmse(score_label_u)
	return precision, ndcg, auc, mae, rmse

def get_metric(score_label):
	Precision = np.zeros(20)
	NDCG = np.zeros(20)
	AUC = 0.
	score_df = pd.DataFrame(score_label, columns=['uid', 'score', 'label'])
	num = 0
	score_label_all = []
	for uid, hist in score_df.groupby('uid'):
		if hist.shape[0]<10:
			continue
		score = hist['score'].tolist()
		label = hist['label'].tolist()
		score_label_u = []
		for i in range(len(score)):
			score_label_u.append([score[i], label[i]])
			score_label_all.append([score[i], label[i]])
		precision, ndcg, auc, mae, rmse = calc_metric(score_label_u)
		Precision += precision
		NDCG += ndcg
		AUC += auc
		num += 1
	score_label_all = sorted(score_label_all, key=lambda d:d[0], reverse=True)
	GPrecision = np.array([eval.precision_k(score_label_all, k*len(score_label_all)/100) for k in range(1, 21)])
	GAUC = eval.auc(score_label_all)
	MAE = eval.mae(score_label_all)
	RMSE = eval.rmse(score_label_all)
	return Precision / num, NDCG / num, AUC / num, GPrecision, GAUC, MAE, RMSE
		
def _eval(sess, model):
	loss_sum = 0.
	batch = 0
	score_label = []
	for _, datainput, u_readinput, u_friendinput, uf_readinput, u_read_l, u_friend_l, uf_read_linput, \
		i_readinput, i_friendinput, if_readinput, i_linkinput, i_read_l, i_friend_l, if_read_linput in \
	DataInput(test_set, u_read_list, u_friend_list, uf_read_list, i_read_list, i_friend_list, if_read_list, \
		i_link_list, test_batch_size, trunc_len):
		score_, loss = model.eval(sess, datainput, u_readinput, u_friendinput, uf_readinput, u_read_l, \
		u_friend_l, uf_read_linput, i_readinput, i_friendinput, if_readinput, i_linkinput, i_read_l, i_friend_l, if_read_linput, lambda1, lambda2)
		for i in range(len(score_)):
			score_label.append([datainput[1][i], score_[i], datainput[2][i]])
		loss_sum += loss
		batch += 1
	Precision, NDCG, AUC, GPrecision, GAUC, MAE, RMSE = get_metric(score_label) 
	return loss_sum/batch, Precision, NDCG, MAE, RMSE

gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session() as sess:
	model = Model(user_count, item_count)
	model.restore(sess, workdir+'/model/DUAL_GAT.ckpt')

	Test_loss, P, N, MAE, RMSE = _eval(sess, model)
	print('Test_loss: %.4f P@3: %.4f P@5: %.4f P@10: %.4f NDCG@3: %.4f NDCG@5: %.4f NDCG@10: %.4f MAE: %.4f RMSE: %.4f' %
	(Test_loss, P[2], P[4], P[9], N[2], N[4], N[9], MAE, RMSE))

	sys.stdout.flush()
	
