import tensorflow as tf
from utils.logger import logger
import json
import os
from model.util import embedding_layer, sentence_encoding_models
from context_fusion.nn import linear


path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
params_path = os.path.join(path, 'config', 'params.json')
data_path = os.path.join(path, 'data')
PAD = '<pad>'
UNK = '<unk>'
method_name_list = [
    'cnn_kim',
    'no_ct',
    'lstm', 'gru', 'sru', 'sru_normal',  # rnn
    'multi_cnn', 'hrchy_cnn',
    'multi_head', 'multi_head_git', 'disa',
    'block'
]


class Model(object):
    def __init__(self, token_seq, labels, is_train,
                 config, token_embedding_matrix=None, **kwargs):
        self.methods = config.methods
        self.activation_function = config.activation_function
        self.wd = config.wd
        self.keep_prob = config.keep_prob
        self.vocab_size = config.vocab_size
        self.seq_max_length = config.seq_max_length
        self.cls_num = config.cls_num
        self.batch_size = config.batch_size
        self.big_cls_num = config.big_cls_num
        self.token_embedding_matrix = token_embedding_matrix
        self.token_embedding_dim = config.token_embedding_dim
        self.block_len = config.block_len
        self.hidden_size = config.hidden_size
        self.small_cls_num = config.small_cls_num
        self.var_decay = config.var_decay
        self.optimizer = config.optimizer
        self.decay = config.decay
        self.n_gpu = config.n_gpu
        self.learning_rate = config.learning_rate
        self.token_seq = token_seq
        self.labels = labels
        self.is_train = is_train
        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)
        self.ema = tf.train.ExponentialMovingAverage(self.decay)

        self.var_ema = tf.train.ExponentialMovingAverage(self.var_decay)

        if self.check_method():
            logger.info('check ok!')


        self.token_mask = tf.cast(self.token_seq, tf.bool)
        self.token_len = tf.reduce_sum(tf.cast(self.token_mask, tf.int32), -1)

        self.get_token_embedding()
        self.get_rep()
        self.get_logits()

    def get_token_embedding(self):
        with tf.variable_scope('emb', reuse=tf.AUTO_REUSE):
            self.emb = embedding_layer(token_indices=self.token_seq,
                                  token_embedding_matrix=self.token_embedding_matrix,
                                  n_tokens=self.vocab_size,
                                  token_embedding_dim=self.token_embedding_dim,
                                  name="emb_mat",
                                  trainable=True)

    def get_rep(self):
        with tf.variable_scope("sent_encoding", reuse=tf.AUTO_REUSE):
            res_rep = []
            for method in self.methods:
                rep = sentence_encoding_models(
                self.emb, self.token_mask, method, 'relu',
                'based_sent2vec', self.wd, self.keep_prob,
                block_len=self.block_len)
                res_rep.append(rep)
            self.rep = tf.concat(res_rep, -1)

    def get_logits(self):
        with tf.variable_scope('output'):
            pre_logits = tf.nn.relu(linear([self.rep], self.hidden_size, True, scope='pre_logits_linear',
                                           wd=self.wd, input_keep_prob=self.keep_prob,
                                        is_train=self.is_train))  # bs, hn
            self.logits = linear([pre_logits], self.cls_num, False, scope='get_output',
                            wd=self.wd, input_keep_prob=self.keep_prob,
                                 is_train=self.is_train)




    def check_method(self):
        for method in self.methods:
            if method not in method_name_list:
                raise Exception("each method in self.methods: {} must be in method_name_list: {}".format(self.methods, method_name_list))
        return True

    def build_loss(self):
        # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels,
        #                                                                     logits=self.logits))
        print("====================================================================")
        labels_split = tf.split(self.labels, num_or_size_splits=self.big_cls_num, axis=-1)
        logits_split = tf.split(self.logits, num_or_size_splits=self.big_cls_num, axis=-1)
        print("labels_split: ", labels_split)
        print("logits_split: ", logits_split)
        with tf.name_scope("weight_decay"):
            for var in set(tf.get_collection('reg_vars')):
                weight_decay = tf.multiply(tf.nn.l2_loss(var), self.wd,
                                           name="{}-wd".format('-'.join(str(var.op.name).split('/'))))
                tf.add_to_collection('losses', weight_decay)
        reg_vars = tf.get_collection('losses')
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        logger.info('regularization var num: %d' % len(reg_vars))
        logger.info('trainable var num: %d' % len(trainable_vars))
        for logit, label in zip(logits_split, labels_split):
            tf.add_to_collection('losses', tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=label,
                logits=logit
            )))
        self.loss = tf.add_n(tf.get_collection('losses'), name='loss')
        tf.summary.scalar(self.loss.op.name, self.loss)
        tf.add_to_collection('ema/scalar', self.loss)




    def get_metrics(self):
        self.labels_split = tf.split(self.labels, num_or_size_splits=self.big_cls_num, axis=-1)
        self.logits_split = tf.split(self.logits, num_or_size_splits=self.big_cls_num, axis=-1)
        res = []
        res_prec = []
        res_recall = []
        for logit, label in zip(self.logits_split, self.labels_split):
            pre = tf.argmax(logit, axis=-1)
            pre = tf.one_hot(pre, depth=self.small_cls_num)
            recall = tf.metrics.recall(label, pre)
            res_recall.append(recall)
            prec = tf.metrics.precision(label, pre)
            res_prec.append(prec)
            res.append(2* recall[0] * prec[0] /(recall[0] + prec[0]))
        f1_score = tf.reduce_sum(tf.stack(res))/ 20.
        return f1_score, res_recall, res_prec
        # print(self.f1_score)

    def build_var_ema(self):
        ema_op = self.var_ema.apply(tf.trainable_variables(), )
        with tf.control_dependencies([ema_op]):
            self.loss = tf.identity(self.loss)

    def build_ema(self):
        tensors = tf.get_collection("ema/scalar") + \
                  tf.get_collection("ema/vector")
        ema_op = self.ema.apply(tensors)
        for var in tf.get_collection("ema/scalar"):
            ema_var = self.ema.average(var)
            tf.summary.scalar(ema_var.op.name, ema_var)
        for var in tf.get_collection("ema/vector"):
            ema_var = self.ema.average(var)
            tf.summary.histogram(ema_var.op.name, ema_var)

        with tf.control_dependencies([ema_op]):
            self.loss = tf.identity(self.loss)

    def get_optimizer(self):
        if self.optimizer.lower() == 'adadelta':
            assert self.learning_rate > 0.1 and self.learning_rate < 1.
            self.opt = tf.train.AdadeltaOptimizer(self.learning_rate)
        elif self.optimizer.lower() == 'adam':
            assert self.learning_rate < 0.1
            self.opt = tf.train.AdamOptimizer(self.learning_rate)
        elif self.optimizer.lower() == 'rmsprop':
            assert self.learning_rate < 0.1
            self.opt = tf.train.RMSPropOptimizer(self.learning_rate)
        else:
            raise AttributeError('no optimizer named as \'%s\'' % self.optimizer)

        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        all_params_num = 0
        for elem in trainable_vars:
            var_name = elem.name.split(':')[0]
            if var_name.endswith('emb_mat'): continue
            params_num = 1
            for l in elem.get_shape().as_list(): params_num *= l
            all_params_num += params_num
        logger.info('Trainable Parameters Number: %d' % all_params_num)

        self.train_op = self.opt.minimize(self.loss, global_step=self.global_step,
                                          var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))


    def step(self, sess, batch_data, batch_labels):
        assert isinstance(sess, tf.Session)
        feed_dict = {self.token_seq:batch_data, self.labels:batch_labels}
        loss, train_op = sess.run([self.loss, self.train_op], feed_dict=feed_dict)
        return loss, train_op


