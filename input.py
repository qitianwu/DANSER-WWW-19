import numpy as np
import random

class DataInput:
	def __init__(self, data, u_read_list, u_friend_list, uf_read_list, i_read_list, i_friend_list, if_read_list, \
		i_link_list, batch_size, trunc_len):
		self.batch_size = batch_size
		self.data = data
		self.u_read_list = u_read_list
		self.u_friend_list = u_friend_list
		self.uf_read_list = uf_read_list
		self.i_read_list = i_read_list
		self.i_friend_list = i_friend_list
		self.if_read_list = if_read_list
		self.i_link_list = i_link_list
		self.epoch_size = len(self.data) // self.batch_size
		self.trunc_len = trunc_len
		if self.epoch_size * self.batch_size < len(self.data):
			self.epoch_size += 1
		self.i = 0
	
	def __iter__(self):
		return self

	def __next__(self):
		if self.i == self.epoch_size:
			raise StopIteration

		ts = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size,
                                                  len(self.data))]
		self.i += 1

		iid, uid, label = [], [], []
		u_read, u_friend, uf_read = [], [], []
		u_read_l, u_friend_l, uf_read_l = [], [], []
		i_read, i_friend, if_read = [], [], []
		i_read_l, i_friend_l, if_read_l, i_link = [], [], [], []

		for t in ts:
			#if len(self.friend_list[t[1]]) <= 1:
			#	continue
			#if min([len(self.read_list[t[1]][j]) for j in range(len(self.read_list[t[1]]))]) <= 1:
			#	continue
			uid.append(t[0])
			iid.append(t[1])
			label.append(t[2])

			def sample(data, n_sample):
				loc = []
				select = []
				r = random.randint(0, len(data)-1)
				for i in range(n_sample):
					while r in loc:
						r = random.randint(0, len(data)-1)
					loc.append(r)
					select.append(data[r])
				return select
      
			u_read_u = self.u_read_list[t[0]]
			u_read.append(u_read_u)
			u_read_l.append(len(u_read_u))
			u_friend_u = self.u_friend_list[t[0]]
			if len(u_friend_u) <= self.trunc_len:
				u_friend.append(u_friend_u)
			else:
				u_friend.append(sample(u_friend_u, self.trunc_len))
			u_friend_l.append(min(len(u_friend_u), self.trunc_len))
			uf_read_u = self.uf_read_list[t[0]]
			uf_read.append(uf_read_u)
			uf_read_l_u = []
			for f in range(len(uf_read_u)):
				uf_read_l_u.append(min(len(uf_read_u[f]), self.trunc_len))
			uf_read_l.append(uf_read_l_u)
			

			i_read_i = self.i_read_list[t[1]]
			i_read.append(i_read_i)
			i_friend_i = self.i_friend_list[t[1]]
			if len(i_friend_i) <= self.trunc_len:
				i_friend.append(i_friend_i)
			else:
				i_friend.append(sample(i_friend_i, self.trunc_len))
			if_read_i = self.if_read_list[t[1]]
			if_read.append(if_read_i)
			i_link_i = self.i_link_list[t[1]]
			i_link.append(i_link_i)
			if len(i_read_i)<=1:
				i_read_l.append(0)
			else:
				i_read_l.append(len(i_read_i))
			if len(i_friend_i)<=1:
				i_friend_l.append(0)
			else:
				i_friend_l.append(min(len(i_friend_i), self.trunc_len))
			if_read_l_i = []
			for f in range(len(if_read_i)):
				if len(if_read_i[f])<=1:
					if_read_l_i.append(0)
				else:
					if_read_l_i.append(min(len(if_read_i[f]), self.trunc_len))
			if_read_l.append(if_read_l_i)

		data_len = len(iid)
    
		#padding
		u_read_maxlength = max(u_read_l)
		u_friend_maxlength = min(self.trunc_len, max(u_friend_l)) #500
		uf_read_maxlength = min(self.trunc_len, max(max(uf_read_l)))
		u_readinput = np.zeros([data_len, u_read_maxlength], dtype = np.int32)
		for i, ru in enumerate(u_read):
			u_readinput[i, :len(ru)] = ru[:len(ru)]
		u_friendinput = np.zeros([data_len, u_friend_maxlength], dtype = np.int32)
		for i, fi in enumerate(u_friend):
			u_friendinput[i, :min(len(fi), u_friend_maxlength)] = fi[:min(len(fi), u_friend_maxlength)]
		uf_readinput = np.zeros([data_len, u_friend_maxlength, u_read_maxlength], dtype = np.int32)
		for i in range(len(uf_read)):
			for j, rj in enumerate(uf_read[i][:u_friend_maxlength]): 
				uf_readinput[i, j, :min(len(rj), u_read_maxlength)] = rj[:min(len(rj), u_read_maxlength)]
		uf_read_linput = np.zeros([data_len, u_friend_maxlength], dtype = np.int32)
		for i, fr in enumerate(uf_read_l):
			uf_read_linput[i, :min(len(fr), u_friend_maxlength)] = fr[:min(len(fr), u_friend_maxlength)]

		i_read_maxlength = max(i_read_l)
		i_friend_maxlength = min(10, max(i_friend_l)) #500
		if_read_maxlength = min(self.trunc_len,max(max(if_read_l)))
		i_readinput = np.zeros([data_len, i_read_maxlength], dtype = np.int32)
		for i, ru in enumerate(i_read):
			i_readinput[i, :len(ru)] = ru[:len(ru)]
		i_friendinput = np.zeros([data_len, i_friend_maxlength], dtype = np.int32)
		for i, fi in enumerate(i_friend):
			i_friendinput[i, :min(len(fi), i_friend_maxlength)] = fi[:min(len(fi), i_friend_maxlength)]
		if_readinput = np.zeros([data_len, i_friend_maxlength, i_read_maxlength], dtype = np.int32)
		for i in range(len(if_read)):
			for j, rj in enumerate(if_read[i][:i_friend_maxlength]):
				if_readinput[i, j, :min(len(rj), if_read_maxlength)] = rj[:min(len(rj), if_read_maxlength)]
		if_read_linput = np.zeros([data_len, i_friend_maxlength], dtype = np.int32)
		for i, fr in enumerate(if_read_l):
			if_read_linput[i, :min(len(fr), i_friend_maxlength)] = fr[:min(len(fr), i_friend_maxlength)]
		i_linkinput = np.zeros([data_len, i_friend_maxlength, 1], dtype = np.int32)
		for i, li in enumerate(i_link):
			li = np.reshape(np.array(li), [-1, 1])
			i_linkinput[i, :min(len(li), i_friend_maxlength)] = li[:min(len(li), i_friend_maxlength)]
		

		return self.i, (iid, uid, label), u_readinput, u_friendinput, uf_readinput, u_read_l, u_friend_l, uf_read_linput, \
		i_readinput, i_friendinput, if_readinput, i_linkinput, i_read_l, i_friend_l, if_read_linput
