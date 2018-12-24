import tensorflow as tf
from utils.logger import logger
import json
import os
from model.util import embedding_layer, context_fusion_layers, sentence_encoding_models
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

with open(params_path) as param:
    params_dict = json.load(param)
config = tf.contrib.training.HParams(**params_dict)


class Model(object):
    def __init__(self, config, token_embedding_matrix=None, scope=None, is_train=None, **kwargs):
        self.methods = config.methods
        self.activation_function = config.activation_function
        self.scope = scope
        self.wd = config.wd
        self.keep_prob = config.keep_prob
        self.vocab_size = config.vocab_size
        self.seq_max_length = config.seq_max_length
        self.cls_num = config.cls_num
        self.big_cls_num = config.big_cls_num
        self.token_embedding_matrix = token_embedding_matrix
        self.token_embedding_dim = config.token_embedding_dim
        self.block_len = config.block_len
        self.hidden_size = config.hidden_size
        self.small_cls_num = config.small_cls_num
        self.is_train = is_train

        if self.check_method():
            logger.info('check ok!')

        self.init_placeholders()
        self.token_mask = tf.cast(self.token_seq, tf.bool)
        self.token_len = tf.reduce_sum(tf.cast(self.token_mask, tf.int32), -1)

    def get_token_embedding(self):
        with tf.variable_scope('emb', reuse=tf.AUTO_REUSE):
            self.emb = embedding_layer(token_indices=self.token_seq,
                                  token_embedding_matrix=self.token_embedding_matrix,
                                  n_tokens=self.vocab_size,
                                  token_embedding_dim=self.token_embedding_dim,
                                  name="emb",
                                  trainable=True)

    def get_rep(self):
        with tf.variable_scope("sent_encoding", reuse=tf.AUTO_REUSE):
            res_rep = []
            for method in self.methods:
                rep = sentence_encoding_models(
                self.emb, self.token_mask, method, 'relu',
                'based_sent2vec', self.wd, self.is_train, self.keep_prob,
                block_len=self.block_len)
                res_rep.append(rep)
            self.rep = tf.concat(res_rep, 3)

    def get_logits(self):
        with tf.variable_scope('output'):
            pre_logits = tf.nn.relu(linear([self.rep], self.hidden_size, True, scope='pre_logits_linear',
                                           wd=self.wd, input_keep_prob=self.keep_prob,
                                           is_train=self.is_train))  # bs, hn
            self.logits = linear([pre_logits], self.cls_num, False, scope='get_output',
                            wd=self.wd, input_keep_prob=self.keep_prob, is_train=self.is_train)




    def check_method(self):
        for method in self.methods:
            if method not in method_name_list:
                raise Exception("each method in self.methods: {} must be in method_name_list: {}".format(self.methods, method_name_list))
        return True

    def init_placeholders(self):
        self.token_seq = tf.placeholder(tf.int32, [None, self.seq_max_length], name='token_seq')

        self.labels = tf.placeholder(tf.int32, [None, self.cls_num], name='gold_label')

    def build_loss(self):
        with tf.name_scope("weight_decay"):
            for var in set(tf.get_collection('reg_vars', self.scope)):
                weight_decay = tf.multiply(tf.nn.l2_loss(var), self.wd,
                                           name="{}-wd".format('-'.join(str(var.op.name).split('/'))))
                tf.add_to_collection('losses', weight_decay)
        reg_vars = tf.get_collection('losses', self.scope)
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
        logger.info('regularization var num: %d' % len(reg_vars))
        logger.info('trainable var num: %d' % len(trainable_vars))
        losses = tf.nn.softmax_cross_entropy_with_logits(
            labels=self.labels,
            logits=self.logits
        )
        tf.add_to_collection('losses', tf.reduce_mean(losses, name='xentropy_loss_mean'))
        loss = tf.add_n(tf.get_collection('losses', self.scope), name='loss')
        tf.summary.scalar(loss.op.name, loss)
        tf.add_to_collection('ema/scalar', loss)
        return loss

    def get_metrics(self):
        self.labels_split = tf.split(self.labels, num_or_size_splits=self.big_cls_num)
        self.logits_split = tf.split(self.logits, num_or_size_splits=self.big_cls_num)

        for logit in self.logits_split:
            pre = tf.argmax(logit, axis=-1)
            pre = tf.one_hot(pre, depth=self.small_cls_num)
            




if __name__ == '__main__':
    pass




    # ===============测试check_method============================
    # model = Model(rep_tensor=None, rep_mask=None, config=config)

