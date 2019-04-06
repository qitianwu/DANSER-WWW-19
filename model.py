import tensorflow as tf
import pickle

class Model(object):
    
	def __init__(self, user_count, item_count):

		self.user = tf.placeholder(tf.int32, [None,]) # [B]
		self.item = tf.placeholder(tf.int32, [None,]) # [B]
		self.label = tf.placeholder(tf.float32, [None,]) # [B]

		self.u_read = tf.placeholder(tf.int32, [None, None]) # [B, R]
		self.u_read_l = tf.placeholder(tf.int32, [None,]) # [B]
		self.u_friend = tf.placeholder(tf.int32, [None, None]) # [B, F]
		self.u_friend_l = tf.placeholder(tf.int32, [None,]) # [B]
		self.uf_read = tf.placeholder(tf.int32, [None, None, None]) # [B, F, R]
		self.uf_read_l = tf.placeholder(tf.int32, [None, None]) # [B, F]

		self.i_read = tf.placeholder(tf.int32, [None, None]) # [B, R]
		self.i_read_l = tf.placeholder(tf.int32, [None,]) # [B]
		self.i_friend = tf.placeholder(tf.int32, [None, None]) # [B, R]
		self.i_friend_l = tf.placeholder(tf.int32, [None,]) # [B]
		self.if_read = tf.placeholder(tf.int32, [None, None, None]) # [B, F, R]
		self.if_read_l = tf.placeholder(tf.int32, [None, None]) # [B, F]
		self.i_link = tf.placeholder(tf.float32, [None, None, 1]) # [B, F, 1]

		self.learning_rate = tf.placeholder(tf.float32)  
		self.training = tf.placeholder(tf.int32)
		self.keep_prob = tf.placeholder(tf.float32)
		self.lambda1 = tf.placeholder(tf.float32)
		self.lambda2 = tf.placeholder(tf.float32)

		#--------------embedding layer-------------------
        
		hidden_units_u = 10
		hidden_units_i = 10

		user_emb_w = tf.get_variable("norm_user_emb_w", [user_count+1, hidden_units_u], initializer = None)
		item_emb_w = tf.get_variable("norm_item_emb_w", [item_count+1, hidden_units_i], initializer = None)
		item_b = tf.get_variable("norm_item_b", [item_count+1],
                             initializer=tf.constant_initializer(0.0))

		#--------------Embedding Layer-------------------
		#self embedding
		uid_emb = tf.nn.embedding_lookup(user_emb_w, self.user)
		iid_emb = tf.nn.embedding_lookup(item_emb_w, self.item)
		i_b = tf.gather(item_b, self.item)

		ur_emb = tf.nn.embedding_lookup(item_emb_w, self.u_read) # [B, R, H]
		key_masks = tf.sequence_mask(self.u_read_l, tf.shape(ur_emb)[1])   # [B, R]
		key_masks = tf.expand_dims(key_masks, axis = 2) # [B, R, 1]
		key_masks = tf.tile(key_masks, [1, 1, tf.shape(ur_emb)[2]]) # [B, R, H]
		key_masks = tf.reshape(key_masks, [-1, tf.shape(ur_emb)[1], tf.shape(ur_emb)[2]]) # [B, R, H]
		paddings = tf.zeros_like(ur_emb) # [B, R, H]
		ur_emb = tf.where(key_masks, ur_emb, paddings)  # [B, R, H]

		ir_emb = tf.nn.embedding_lookup(user_emb_w, self.i_read) # [B, R, H]
		key_masks = tf.sequence_mask(self.i_read_l, tf.shape(ir_emb)[1])   # [B, R]
		key_masks = tf.expand_dims(key_masks, axis = 2) # [B, R, 1]
		key_masks = tf.tile(key_masks, [1, 1, tf.shape(ir_emb)[2]]) # [B, R, H]
		key_masks = tf.reshape(key_masks, [-1, tf.shape(ir_emb)[1], tf.shape(ir_emb)[2]]) # [B, R, H]
		paddings = tf.zeros_like(ir_emb) # [B, R, H]
		ir_emb = tf.where(key_masks, ir_emb, paddings)  # [B, R, H]
		
		#friend embedding
		fuid_emb = tf.nn.embedding_lookup(user_emb_w, self.u_friend)
		key_masks = tf.sequence_mask(self.u_friend_l, tf.shape(fuid_emb)[1])   # [B, F]
		key_masks = tf.expand_dims(key_masks, axis = 2) # [B, F, 1]
		key_masks = tf.tile(key_masks, [1, 1, tf.shape(fuid_emb)[2]]) # [B, F, H]
		paddings = tf.zeros_like(fuid_emb) # [B, F, H]
		fuid_emb = tf.where(key_masks, fuid_emb, paddings)  # [B, F, H]

		fiid_emb = tf.nn.embedding_lookup(item_emb_w, self.i_friend)
		key_masks = tf.sequence_mask(self.i_friend_l, tf.shape(fiid_emb)[1])   # [B, F]
		key_masks = tf.expand_dims(key_masks, axis = 2) # [B, F, 1]
		key_masks = tf.tile(key_masks, [1, 1, tf.shape(fiid_emb)[2]]) # [B, F, H]
		paddings = tf.zeros_like(fiid_emb) # [B, F, H]
		fiid_emb = tf.where(key_masks, fiid_emb, paddings)  # [B, F, H]

		ufr_emb = tf.nn.embedding_lookup(item_emb_w, self.uf_read)
		key_masks = tf.sequence_mask(self.uf_read_l, tf.shape(ufr_emb)[2])   # [B, F, R]
		key_masks = tf.expand_dims(key_masks, axis = 3) # [B, F, R, 1]
		key_masks = tf.tile(key_masks, [1, 1, 1, tf.shape(ufr_emb)[3]]) # [B, F, R, H]
		paddings = tf.zeros_like(ufr_emb) # [B, F, R, H]
		ufr_emb = tf.where(key_masks, ufr_emb, paddings)  # [B, F, R, H]

		ifr_emb = tf.nn.embedding_lookup(user_emb_w, self.if_read) # [B, F, R, H]
		key_masks = tf.sequence_mask(self.if_read_l, tf.shape(ifr_emb)[2])   # [B, F, R]
		key_masks = tf.expand_dims(key_masks, axis = 3) # [B, F, R, 1]
		key_masks = tf.tile(key_masks, [1, 1, 1, tf.shape(ifr_emb)[3]]) # [B, F, R, H]
		paddings = tf.zeros_like(ifr_emb) # [B, F, R, H]
		ifr_emb = tf.where(key_masks, ifr_emb, paddings)  # [B, F, R, H]
		
		#--------------Dual GCN/GAT Layer-------------------

		uid_emb_exp1 = tf.tile(uid_emb, [1, tf.shape(fuid_emb)[1]+1])
		uid_emb_exp1 = tf.reshape(uid_emb_exp1, [-1, tf.shape(fuid_emb)[1]+1, hidden_units_u]) # [B, F, H]
		iid_emb_exp1 = tf.tile(iid_emb, [1, tf.shape(fiid_emb)[1]+1])
		iid_emb_exp1 = tf.reshape(iid_emb_exp1, [-1, tf.shape(fiid_emb)[1]+1, hidden_units_i]) # [B, F, H]
		uid_emb_ = tf.expand_dims(uid_emb, axis = 1)
		iid_emb_ = tf.expand_dims(iid_emb, axis = 1)

		uid_in = tf.layers.dense(uid_emb_exp1, hidden_units_u, use_bias = False, name = 'trans_uid')
		fuid_in = tf.layers.dense(tf.concat([uid_emb_, fuid_emb], axis = 1), hidden_units_u, use_bias = False, reuse=True, name = 'trans_uid')
		din_gat_uid = tf.concat([uid_in, fuid_in], axis = -1)
		d1_gat_uid = tf.layers.dense(din_gat_uid, 1, activation=tf.nn.leaky_relu, name='gat_uid')
		d1_gat_uid = tf.nn.dropout(d1_gat_uid, keep_prob=self.keep_prob)
		d1_gat_uid = tf.reshape(d1_gat_uid, [-1, tf.shape(ufr_emb)[1]+1, 1]) # [B, F, 1]
		weights_uid = tf.nn.softmax(d1_gat_uid, axis=1)  # [B, F, 1]
		weights_uid = tf.tile(weights_uid, [1, 1, hidden_units_u]) # [B, F, H]
		uid_gat = tf.reduce_sum(tf.multiply(weights_uid, fuid_in), axis = 1)
		uid_gat = tf.reshape(uid_gat, [-1, hidden_units_u])

		iid_in = tf.layers.dense(iid_emb_exp1, hidden_units_i, use_bias = False, name = 'trans_iid')
		fiid_in = tf.layers.dense(tf.concat([iid_emb_, fiid_emb], axis = 1), hidden_units_i, use_bias = False, reuse=True, name = 'trans_iid')
		din_gat_iid = tf.concat([iid_in, fiid_in], axis = -1)
		d1_gat_iid = tf.layers.dense(din_gat_iid, 1, activation=tf.nn.leaky_relu, name='gat_iid')
		d1_gat_iid = tf.nn.dropout(d1_gat_iid, keep_prob=self.keep_prob)
		d1_gat_iid = tf.reshape(d1_gat_iid, [-1, tf.shape(ifr_emb)[1]+1, 1]) # [B, F, 1]
		weights_iid = tf.nn.softmax(d1_gat_iid, axis=1)  # [B, F, 1]
		weights_iid = tf.tile(weights_iid, [1, 1, hidden_units_i]) # [B, F, H]
		iid_gat = tf.reduce_sum(tf.multiply(weights_iid, fiid_in), axis = 1)
		iid_gat = tf.reshape(iid_gat, [-1, hidden_units_i])

		uid_emb_exp2 = tf.tile(uid_emb, [1, tf.shape(ir_emb)[1]])
		uid_emb_exp2 = tf.reshape(uid_emb_exp2, [-1, tf.shape(ir_emb)[1], hidden_units_u]) # [B, R, H]
		iid_emb_exp2 = tf.tile(iid_emb, [1, tf.shape(ur_emb)[1]])
		iid_emb_exp2 = tf.reshape(iid_emb_exp2, [-1, tf.shape(ur_emb)[1], hidden_units_i]) # [B, R, H]
		uid_emb_exp3 = tf.expand_dims(uid_emb, axis = 1)
		uid_emb_exp3 = tf.expand_dims(uid_emb_exp3, axis = 2) # [B, 1, 1, H]
		uid_emb_exp3 = tf.tile(uid_emb_exp3, [1, tf.shape(ifr_emb)[1], tf.shape(ifr_emb)[2], 1]) # [B, F, R, H]
		iid_emb_exp3 = tf.expand_dims(iid_emb, axis = 1)
		iid_emb_exp3 = tf.expand_dims(iid_emb_exp3, axis = 2) # [B, 1, 1, H]
		iid_emb_exp3 = tf.tile(iid_emb_exp3, [1, tf.shape(ufr_emb)[1], tf.shape(ufr_emb)[2], 1]) # [B, F, R, H]

		uint_in = tf.multiply(ur_emb, iid_emb_exp2) # [B, R, H]
		uint_in = tf.reduce_max(uint_in, axis = 1) # [B, H]
		uint_in = tf.layers.dense(uint_in, hidden_units_i, use_bias = False, name = 'trans_uint') # [B, H]
		uint_in_ = tf.expand_dims(uint_in, axis = 1) # [B, 1, H]
		uint_in = tf.tile(uint_in, [1, tf.shape(ufr_emb)[1]+1])
		uint_in = tf.reshape(uint_in, [-1, tf.shape(ufr_emb)[1]+1, hidden_units_i]) # [B, F, H]
		fint_in = tf.multiply(ufr_emb, iid_emb_exp3) # [B, F, R, H]
		fint_in = tf.reduce_max(fint_in, axis = 2) # [B, F, H]
		fint_in = tf.layers.dense(fint_in, hidden_units_i, use_bias = False, reuse = True, name = 'trans_uint')
		fint_in = tf.concat([uint_in_, fint_in], axis = 1) # [B, F, H]
		din_gat_uint = tf.concat([uint_in, fint_in], axis = -1)
		d1_gat_uint = tf.layers.dense(din_gat_uint, 1, activation=tf.nn.leaky_relu, name='gat_uint')
		d1_gat_uint = tf.nn.dropout(d1_gat_uint, keep_prob=self.keep_prob)
		d1_gat_uint = tf.reshape(d1_gat_uint, [-1, tf.shape(ufr_emb)[1]+1, 1]) # [B, F, 1]
		weights_uint = tf.nn.softmax(d1_gat_uint, axis=1)  # [B, F, 1]
		weights_uint = tf.tile(weights_uint, [1, 1, hidden_units_i]) # [B, F, H]
		uint_gat = tf.reduce_sum(tf.multiply(weights_uint, fint_in), axis = 1)
		uint_gat = tf.reshape(uint_gat, [-1, hidden_units_i])

		iinf_in = tf.multiply(ir_emb, uid_emb_exp2) # [B, R, H]
		iinf_in = tf.reduce_max(iinf_in, axis = 1) # [B, H]
		iinf_in = tf.layers.dense(iinf_in, hidden_units_u, use_bias = False, name = 'trans_iinf') # [B, H]
		iinf_in_ = tf.expand_dims(iinf_in, axis = 1) # [B, 1, H]
		iinf_in = tf.tile(iinf_in, [1, tf.shape(ifr_emb)[1]+1])
		iinf_in = tf.reshape(iinf_in, [-1, tf.shape(ifr_emb)[1]+1, hidden_units_u]) # [B, F, H]
		finf_in = tf.multiply(ifr_emb, uid_emb_exp3) # [B, F, R, H]
		finf_in = tf.reduce_max(finf_in, axis = 2) # [B, F, H]
		finf_in = tf.layers.dense(finf_in, hidden_units_u, use_bias = False, reuse = True, name = 'trans_iinf')
		finf_in = tf.concat([iinf_in_, finf_in], axis = 1) # [B, F, H]
		din_gat_iinf = tf.concat([iinf_in, finf_in], axis = -1)
		d1_gat_iinf = tf.layers.dense(din_gat_iinf, 1, activation=tf.nn.leaky_relu, name='gat_iinf')
		d1_gat_iinf = tf.nn.dropout(d1_gat_iinf, keep_prob=self.keep_prob)
		d1_gat_iinf = tf.reshape(d1_gat_iinf, [-1, tf.shape(ifr_emb)[1]+1, 1]) # [B, F, 1]
		weights_iinf = tf.nn.softmax(d1_gat_iinf, axis=1)  # [B, F, 1]
		weights_iinf = tf.tile(weights_iinf, [1, 1, hidden_units_u]) # [B, F, H]
		iinf_gat = tf.reduce_sum(tf.multiply(weights_iinf, finf_in), axis = 1)
		iinf_gat = tf.reshape(iinf_gat, [-1, hidden_units_u])

		#--------------Pairwise Neural Interaction Layer---------------
		
		din_ui = tf.multiply(uid_gat, iid_gat)
		if self.training is True: 
			din_ui = tf.layers.batch_normalization(inputs=din_ui, name='norm_ui_b1', training = True)
		else:
			din_ui = tf.layers.batch_normalization(inputs=din_ui, name='norm_ui_b1', training = False)
		d1_ui = tf.layers.dense(din_ui, 16, activation=tf.nn.tanh, use_bias = True, name='norm_ui_1')
		d2_ui = tf.nn.dropout(d1_ui, keep_prob=self.keep_prob)
		d2_ui = tf.layers.dense(d2_ui, 8, activation=tf.nn.tanh, use_bias = True, name='norm_ui_2')
		d3_ui = tf.nn.dropout(d2_ui, keep_prob=self.keep_prob)
		d3_ui = tf.layers.dense(d3_ui, 4, activation=tf.nn.tanh, use_bias = True, name='norm_ui_3')
		d4_ui = tf.layers.dense(d3_ui, 1, activation=None, use_bias = True, name='norm_merge', reuse=tf.AUTO_REUSE)
		d4_ui = tf.reshape(d4_ui, [-1, 1])
		d3_ui_ = tf.reshape(d3_ui, [-1, tf.shape(d3_ui)[-1], 1])

		din_uf = tf.multiply(uid_gat, iinf_gat)
		if self.training is True:
			din_uf = tf.layers.batch_normalization(inputs=din_uf, name='norm_uf_b1', training = True)
		else:
			din_uf = tf.layers.batch_normalization(inputs=din_uf, name='norm_uf_b1', training = False)
		d1_uf = tf.layers.dense(din_uf, 16, activation=tf.nn.tanh, use_bias = True, name='norm_uf_1')
		d2_uf = tf.nn.dropout(d1_uf, keep_prob=self.keep_prob)
		d2_uf = tf.layers.dense(d2_uf, 8, activation=tf.nn.tanh, use_bias = True, name='norm_uf_2')
		d3_uf = tf.nn.dropout(d2_uf, keep_prob=self.keep_prob)
		d3_uf = tf.layers.dense(d3_uf, 4, activation=tf.nn.tanh, use_bias = True, name='norm_uf_3')
		d4_uf = tf.layers.dense(d3_uf, 1, activation=None, use_bias = True, name='norm_merge', reuse=tf.AUTO_REUSE)
		d4_uf = tf.reshape(d4_uf, [-1, 1])
		d3_uf_ = tf.reshape(d3_uf, [-1, tf.shape(d3_uf)[-1], 1])

		din_fi = tf.multiply(uint_gat, iid_gat)
		if self.training is True:
			din_fi = tf.layers.batch_normalization(inputs=din_fi, name='norm_fi_b1', training = True)
		else:
			din_fi = tf.layers.batch_normalization(inputs=din_fi, name='norm_fi_b1', training = False)
		d1_fi = tf.layers.dense(din_fi, 16, activation=tf.nn.tanh, use_bias = True, name='norm_fi_1')
		d2_fi = tf.nn.dropout(d1_fi, keep_prob=self.keep_prob)
		d2_fi = tf.layers.dense(d2_fi, 8, activation=tf.nn.tanh, use_bias = True, name='norm_fi_2')
		d3_fi = tf.nn.dropout(d2_fi, keep_prob=self.keep_prob)
		d3_fi = tf.layers.dense(d3_fi, 4, activation=tf.nn.tanh, use_bias = True, name='norm_fi_3')
		d4_fi = tf.layers.dense(d3_fi, 1, activation=None, use_bias = True, name='norm_merge', reuse=tf.AUTO_REUSE)
		d4_fi = tf.reshape(d4_fi, [-1, 1])
		d3_fi_ = tf.reshape(d3_fi, [-1, tf.shape(d3_fi)[-1], 1])

		din_ff = tf.multiply(uint_gat, iinf_gat)
		if self.training is True:
			din_ff = tf.layers.batch_normalization(inputs=din_ff, name='norm_ff_b1', training = True)
		else:
			din_ff = tf.layers.batch_normalization(inputs=din_ff, name='norm_ff_b1', training = False)
		d1_ff = tf.layers.dense(din_ff, 16, activation=tf.nn.tanh, use_bias = True, name='norm_ff_1')
		d2_ff = tf.nn.dropout(d1_ff, keep_prob=self.keep_prob)
		d2_ff = tf.layers.dense(d2_ff, 8, activation=tf.nn.tanh, use_bias = True, name='norm_ff_2')
		d3_ff = tf.nn.dropout(d2_ff, keep_prob=self.keep_prob)
		d3_ff = tf.layers.dense(d3_ff, 4, activation=tf.nn.tanh, use_bias = True, name='norm_ff_3')
		d4_ff = tf.layers.dense(d3_ff, 1, activation=None, use_bias = True, name='norm_merge', reuse=tf.AUTO_REUSE)
		d4_ff = tf.reshape(d4_ff, [-1, 1])
		d3_ff_ = tf.reshape(d3_ff, [-1, tf.shape(d3_ff)[-1], 1])

		d3 = tf.concat([d3_ui_, d3_uf_, d3_fi_, d3_ff_], axis = 2)

		#--------------Policy-Based Fusion Layer---------------

		def policy(uid_emb, iid_emb, l_name = 'policy_1'):
			din_policy = tf.concat([uid_emb, iid_emb, tf.multiply(uid_emb, iid_emb)], axis = -1)
			policy = tf.layers.dense(din_policy, 4, activation=None, name=l_name)
			policy = tf.nn.softmax(policy)
			return policy

		policy1 = policy(uid_emb, iid_emb, 'policy_1')
		policy2 = policy(uid_emb, iid_emb, 'policy_2')
		policy3 = policy(uid_emb, iid_emb, 'policy_3')
		policy4 = policy(uid_emb, iid_emb, 'policy_4')
		policy = (policy1 + policy2 + policy3 + policy4) / 4
		policy_exp = tf.tile(policy, [1, tf.shape(d3_ui)[-1]])
		policy_exp = tf.reshape(policy_exp, [-1, tf.shape(d3_ui)[-1], 4])
		if self.training == True:
			dist = tf.distributions.Multinomial(total_count = 1., probs = policy)
			t = dist.sample(1)
			t = tf.reshape(t, [-1, 4]) #[B, 4]
			t_exp = tf.tile(t, [1, tf.shape(d3_ui)[-1]])
			t_exp = tf.reshape(t_exp, [-1, tf.shape(d3_ui)[-1], 4])
			dmerge = tf.reduce_sum(tf.multiply(t_exp, d3), axis = 2)
		else:
			dmerge = tf.reduce_sum(tf.multiply(policy_exp, d3), axis = 2)
		dmerge = tf.reshape(dmerge, [-1, 4])
		dmerge = tf.layers.dense(dmerge, 1, activation=None, use_bias = True, name='norm_merge', reuse=tf.AUTO_REUSE)
		dmerge = tf.reshape(dmerge, [-1])

		#--------------Output Layer---------------
		self.logits = i_b + dmerge
		self.score = self.logits
		i_b_exp = tf.reshape(i_b, [-1, 1])
		logits_policy = tf.concat([i_b_exp + d4_ui, i_b_exp + d4_uf, i_b_exp + d4_fi, i_b_exp + d4_ff], axis = -1)
		score_policy = logits_policy

		loss_emb_reg = tf.reduce_sum(tf.abs(i_b)) + tf.reduce_sum(tf.abs(iid_emb)) + tf.reduce_sum(tf.abs(uid_emb)) + tf.reduce_sum(tf.abs(fuid_emb)) 
		self.loss = tf.reduce_mean(tf.square(self.score-self.label)) + self.lambda1*loss_emb_reg

		labels_exp = tf.reshape(self.label, [-1, 1])
		self.loss_p1 = tf.reduce_mean(tf.reduce_sum(tf.multiply(-tf.log(policy1), -tf.square(score_policy-labels_exp)), axis = -1))
		self.loss_p2 = tf.reduce_mean(tf.reduce_sum(tf.multiply(-tf.log(policy2), -tf.square(score_policy-labels_exp)), axis = -1))
		self.loss_p3 = tf.reduce_mean(tf.reduce_sum(tf.multiply(-tf.log(policy3), -tf.square(score_policy-labels_exp)), axis = -1))
		self.loss_p4 = tf.reduce_mean(tf.reduce_sum(tf.multiply(-tf.log(policy4), -tf.square(score_policy-labels_exp)), axis = -1))

		# Step variable
		self.global_step = tf.Variable(0, trainable=False, name='global_step')
		self.global_epoch_step = \
		tf.Variable(0, trainable=False, name='global_epoch_step')
		self.global_epoch_step_op = \
		tf.assign(self.global_epoch_step, self.global_epoch_step+1)

		self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
		trainable_params = tf.trainable_variables(scope = 'norm')
		gradients = tf.gradients(self.loss, trainable_params)
		clip_gradients, _ = tf.clip_by_global_norm(gradients, 5*self.learning_rate)
		self.train_op = self.opt.apply_gradients(zip(clip_gradients, trainable_params), global_step=self.global_step)
		
		trainable_params1 = tf.trainable_variables(scope = 'policy_1')
		gradients1 = tf.gradients(self.loss_p1, trainable_params1)
		clip_gradients1, _ = tf.clip_by_global_norm(gradients1, 5*self.learning_rate)
		self.train_op1 = self.opt.apply_gradients(zip(clip_gradients1, trainable_params1))

		trainable_params2 = tf.trainable_variables(scope = 'policy_2')
		gradients2 = tf.gradients(self.loss_p2, trainable_params2)
		clip_gradients2, _ = tf.clip_by_global_norm(gradients2, 5*self.learning_rate)
		self.train_op2 = self.opt.apply_gradients(zip(clip_gradients2, trainable_params2))

		trainable_params3 = tf.trainable_variables(scope = 'policy_3')
		gradients3 = tf.gradients(self.loss_p3, trainable_params3)
		clip_gradients3, _ = tf.clip_by_global_norm(gradients3, 5*self.learning_rate)
		self.train_op3 = self.opt.apply_gradients(zip(clip_gradients3, trainable_params3))

		trainable_params4 = tf.trainable_variables(scope = 'policy_4')
		gradients4 = tf.gradients(self.loss_p4, trainable_params4)
		clip_gradients4, _ = tf.clip_by_global_norm(gradients4, 5*self.learning_rate)
		self.train_op4 = self.opt.apply_gradients(zip(clip_gradients4, trainable_params4))
		
		#--------------end model---------------

	def train(self, sess, datainput, u_readinput, u_friendinput, uf_readinput, u_read_l, u_friend_l, uf_read_l, i_readinput, i_friendinput, if_readinput, i_linkinput, i_read_l, i_friend_l, if_read_l, lr, keep_prob, lambda1, lambda2):
		loss, _ = sess.run([self.loss, self.train_op], feed_dict={
		self.item: datainput[0], self.user: datainput[1], self.label: datainput[2], \
		self.u_read: u_readinput, self.u_friend: u_friendinput, self.uf_read: uf_readinput, self.u_read_l: u_read_l, self.u_friend_l: u_friend_l, self.uf_read_l: uf_read_l,
		self.i_read: i_readinput, self.i_friend: i_friendinput, self.if_read: if_readinput, self.i_link: i_linkinput, self.i_read_l: i_read_l, self.i_friend_l: i_friend_l, self.if_read_l: if_read_l,
		self.training: 1, self.learning_rate: lr, self.keep_prob: keep_prob, self.lambda1: lambda1, self.lambda2: lambda2,
		})
		return loss
	
	def policy_update(self, sess, datainput, u_readinput, u_friendinput, uf_readinput, u_read_l, u_friend_l, uf_read_l, i_readinput, i_friendinput, if_readinput, i_linkinput, i_read_l, i_friend_l, if_read_l, lr, keep_prob, lambda1, lambda2):
		_ = sess.run([self.train_op1], feed_dict={
		self.item: datainput[0], self.user: datainput[1], self.label: datainput[2], \
		self.u_read: u_readinput, self.u_friend: u_friendinput, self.uf_read: uf_readinput, self.u_read_l: u_read_l, self.u_friend_l: u_friend_l, self.uf_read_l: uf_read_l,
		self.i_read: i_readinput, self.i_friend: i_friendinput, self.if_read: if_readinput, self.i_link: i_linkinput, self.i_read_l: i_read_l, self.i_friend_l: i_friend_l, self.if_read_l: if_read_l,
		self.training: 1, self.learning_rate: lr, self.keep_prob: keep_prob, self.lambda1: lambda1, self.lambda2: lambda2,
		})
		_ = sess.run([self.train_op2], feed_dict={
		self.item: datainput[0], self.user: datainput[1], self.label: datainput[2], \
		self.u_read: u_readinput, self.u_friend: u_friendinput, self.uf_read: uf_readinput, self.u_read_l: u_read_l, self.u_friend_l: u_friend_l, self.uf_read_l: uf_read_l,
		self.i_read: i_readinput, self.i_friend: i_friendinput, self.if_read: if_readinput, self.i_link: i_linkinput, self.i_read_l: i_read_l, self.i_friend_l: i_friend_l, self.if_read_l: if_read_l,
		self.training: 1, self.learning_rate: lr, self.keep_prob: keep_prob, self.lambda1: lambda1, self.lambda2: lambda2,
		})
		_ = sess.run([self.train_op3], feed_dict={
		self.item: datainput[0], self.user: datainput[1], self.label: datainput[2], \
		self.u_read: u_readinput, self.u_friend: u_friendinput, self.uf_read: uf_readinput, self.u_read_l: u_read_l, self.u_friend_l: u_friend_l, self.uf_read_l: uf_read_l,
		self.i_read: i_readinput, self.i_friend: i_friendinput, self.if_read: if_readinput, self.i_link: i_linkinput, self.i_read_l: i_read_l, self.i_friend_l: i_friend_l, self.if_read_l: if_read_l,
		self.training: 1, self.learning_rate: lr, self.keep_prob: keep_prob, self.lambda1: lambda1, self.lambda2: lambda2,
		})
		_ = sess.run([self.train_op4], feed_dict={
		self.item: datainput[0], self.user: datainput[1], self.label: datainput[2], \
		self.u_read: u_readinput, self.u_friend: u_friendinput, self.uf_read: uf_readinput, self.u_read_l: u_read_l, self.u_friend_l: u_friend_l, self.uf_read_l: uf_read_l,
		self.i_read: i_readinput, self.i_friend: i_friendinput, self.if_read: if_readinput, self.i_link: i_linkinput, self.i_read_l: i_read_l, self.i_friend_l: i_friend_l, self.if_read_l: if_read_l,
		self.training: 1, self.learning_rate: lr, self.keep_prob: keep_prob, self.lambda1: lambda1, self.lambda2: lambda2,
		})

	def eval(self, sess, datainput, u_readinput, u_friendinput, uf_readinput, u_read_l, u_friend_l, uf_read_l, i_readinput, i_friendinput, if_readinput, i_linkinput, i_read_l, i_friend_l, if_read_l, lambda1, lambda2):
		score, loss = sess.run([self.score, self.loss], feed_dict={
		self.item: datainput[0], self.user: datainput[1], self.label: datainput[2], \
		self.u_read: u_readinput, self.u_friend: u_friendinput, self.uf_read: uf_readinput, self.u_read_l: u_read_l, self.u_friend_l: u_friend_l, self.uf_read_l: uf_read_l,
		self.i_read: i_readinput, self.i_friend: i_friendinput, self.if_read: if_readinput, self.i_link: i_linkinput, self.i_read_l: i_read_l, self.i_friend_l: i_friend_l, self.if_read_l: if_read_l,
		self.training: 0, self.keep_prob: 1, self.lambda1: lambda1, self.lambda2: lambda2,
		})
		return score, loss

	def save(self, sess, path):
		saver = tf.train.Saver()
		saver.save(sess, save_path=path)

	def restore(self, sess, path):
		saver = tf.train.Saver()
		saver.restore(sess, save_path=path)
