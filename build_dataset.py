import random
import pickle
import numpy as np
import pandas as pd

random.seed(1234)

workdir = '/home/myronwu/epinions' # change to your workdir
click_f = np.loadtxt(workdir+'/ratings_data.txt', dtype = np.int32)
trust_f = np.loadtxt(workdir+'/trust_data.txt', dtype = np.int32)

click_list = []
trust_list = []

u_read_list = []
u_friend_list = []
uf_read_list = []
i_read_list = []
i_friend_list = []
if_read_list = []
i_link_list = []
user_count = 0
item_count = 0

for s in click_f:
	uid = s[0]
	iid = s[1]
	label = s[2]
	if uid > user_count:
		user_count = uid
	if iid > item_count:
		item_count = iid
	click_list.append([uid, iid, label])

pos_list = []
for i in range(len(click_list)):
	pos_list.append((click_list[i][0], click_list[i][1], click_list[i][2]))
random.shuffle(pos_list)
train_set = pos_list[:int(0.8*len(pos_list))]
test_set = pos_list[int(0.8*len(pos_list)):len(pos_list)]
print(len(train_set))

with open(workdir+'/dataset.pkl', 'wb') as f:
	pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)


train_df = pd.DataFrame(train_set, columns = ['uid', 'iid', 'label'])
test_df = pd.DataFrame(test_set, columns = ['uid', 'iid', 'label'])

click_df = pd.DataFrame(click_list, columns = ['uid', 'iid', 'label'])
train_df = train_df.sort_values(axis = 0, ascending = True, by = 'uid')

for u in range(user_count+1):
	hist = train_df[train_df['uid']==u]
	#hist = hist[hist['label']>3]
	u_read = hist['iid'].unique().tolist()
	if u_read==[]:
		u_read_list.append([0])
	else:
		u_read_list.append(u_read)

train_df = train_df.sort_values(axis = 0, ascending = True, by = 'iid')

for i in range(item_count+1):
	hist = train_df[train_df['iid']==i]
	#hist = hist[hist['label']>3]
	i_read = hist['uid'].unique().tolist()
	if i_read==[]:
		i_read_list.append([0])
	else:
		i_read_list.append(i_read)

for s in trust_f:
	uid = s[0]
	fid = s[1]
	if uid > user_count or fid > user_count:
		continue
	trust_list.append([uid, fid])

trust_df = pd.DataFrame(trust_list, columns = ['uid', 'fid'])
trust_df = trust_df.sort_values(axis = 0, ascending = True, by = 'uid')

for u in range(user_count+1):
	hist = trust_df[trust_df['uid']==u]
	u_friend = hist['fid'].unique().tolist()
	if u_friend==[]:
		u_friend_list.append([0])
		uf_read_list.append([[0]])
	else:
		u_friend_list.append(u_friend)
		uf_read_f = []
		for f in u_friend:
			uf_read_f.append(u_read_list[f])
		uf_read_list.append(uf_read_f)

for i in range(item_count+1):
	if len(i_read_list[i])<=30:
		i_friend_list.append([0])
		if_read_list.append([[0]])
		i_link_list.append([0])
		continue
	i_friend = []
	for j in range(item_count+1):
		if len(i_read_list[j])<=30:
			sim_ij = 0
		else:
			sim_ij = 0
			for s in i_read_list[i]:
				sim_ij += np.sum(i_read_list[j]==s)
		i_friend.append([j, sim_ij])
	i_friend_cd = sorted(i_friend, key=lambda d:d[1], reverse=True)
	i_friend_i = []
	i_link_i = []
	for k in range(20):
		if i_friend_cd[k][1]>5:
			i_friend_i.append(i_friend_cd[k][0])
			i_link_i.append(i_friend_cd[k][1])
	if i_friend_i==[]:
		i_friend_list.append([0])
		if_read_list.append([[0]])
		i_link_list.append([0])
	else:
		i_friend_list.append(i_friend_i)
		i_link_list.append(i_link_i)
		if_read_f = []
		for f in i_friend_i:
			if_read_f.append(i_read_list[f])
		if_read_list.append(if_read_f)
	
with open(workdir+'/data.pkl', 'wb') as f:
	pickle.dump(u_friend_list, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(u_read_list, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(uf_read_list, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(i_friend_list, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(i_read_list, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(if_read_list, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(i_link_list, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump((user_count, item_count), f, pickle.HIGHEST_PROTOCOL)


