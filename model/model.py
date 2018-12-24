import tensorflow as tf
from utils.logger import logger
import json
import os
from model.util import embedding_layer, context_fusion_layers, sentence_encoding_models

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
    def __init__(self, rep_tensor, rep_mask, config, scope=None, is_train=None, **kwargs):
        self.rep_tensor = rep_tensor
        self.rep_mask = rep_mask
        self.methods = config.methods
        self.activation_function = config.activation_function
        self.scope = scope
        self.wd = config.wd
        self.keep_prob = config.keep_prob
        self.vocab_size = config.vocab_size

        if self.check_method():
            logger.info('check ok!')

    def check_method(self):
        for method in self.methods:
            if method not in method_name_list:
                raise Exception("each method in self.methods: {} must be in method_name_list: {}".format(self.methods, method_name_list))
        return True


def get_model(token_indices=None, token_embedding_matrix=None, vocab_size=None, trainable=True):
    rep_tensor = embedding_layer(token_indices=token_indices, token_embedding_matrix=token_embedding_matrix, vocab_size=vocab_size, trainable=trainable)


if __name__ == '__main__':
    pass




    # ===============测试check_method============================
    # model = Model(rep_tensor=None, rep_mask=None, config=config)

