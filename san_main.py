import tensorflow as tf
from utils.logger import logger
from model.san_model import Model
import os
import json
from sklearn.externals import joblib
import random
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report

path = os.path.dirname(os.path.realpath(__file__))
# print(path)
params_path = os.path.join(path, 'config', 'params.json')
data_path = os.path.join(path, 'data')
with open(params_path) as param:
    params_dict = json.load(param)
config = tf.contrib.training.HParams(**params_dict)
print(config)


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
        res.append(2 * recall[0] * prec[0] / (recall[0] + prec[0]))
    f1_score = tf.reduce_sum(tf.stack(res)) / 20.
    return f1_score, res_recall, res_prec

def metrics(labels, logits):
    labels_split = np.hsplit(labels, 20)
    logis_split = np.hsplit(logits, 20)
    f1_score = 0
    for logit, label in zip(logis_split, labels_split):
        logit = np.argmax(logit, axis=-1)
        label = np.argmax(label, axis=-1)
        report = classification_report(logit, label)
        avg = report.split('\n\n')[-1]
        f1_score += avg.split('      ')[3]
        # print(report)
        # precision = average_precision_score(label, logit)
        # recall = average_precision_score(label, logit)
        # f1_score += 2 * precision * recall / (precision + recall)
    return f1_score / 20




def train():
    train_data = joblib.load(os.path.join(data_path, "train_data_500.m"))
    train_labels = joblib.load(os.path.join(data_path, "train_label_500.m"))
    indices = np.random.permutation(range(len(train_data)))
    train_data = train_data[indices]
    train_labels = train_labels[indices]

    test_data = joblib.load(os.path.join(data_path, "validation_data_500.m"))
    test_labels = joblib.load(os.path.join(data_path, "validation_label_500.m"))
    indices = np.random.permutation(range(len(test_data)))
    test_data = test_data[indices]
    test_labels = test_labels[indices]

    token_embedding_matrix = joblib.load(os.path.join(data_path, 'tokens_embedding.m'))

    g = tf.Graph()
    if config.gpu_mem is None:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config.gpu_mem,
                                    allow_growth=True)
        graph_config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)

    else:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config.gpu_mem)
        graph_config = tf.ConfigProto(gpu_options=gpu_options)


    # model = None
    with g.as_default():
        with tf.variable_scope("%s" % config.model_name, reuse=tf.AUTO_REUSE) as scope:
            sess = tf.Session(config=graph_config)
            model = Model(config, token_embedding_matrix=token_embedding_matrix)
            model.build_loss()
            # model.get_metrics()
            model.get_optimizer()

            if True:
                model.build_var_ema()

            if config.model_type == "train":
                model.build_ema()
            # summary = tf.summary.merge_all()
            # f1_score, res_prec, res_recall = model.get_metrics()
            sess.run(tf.global_variables_initializer())
            for _ in range(config.n_epochs):
                for batch_data, batch_label in model.get_batch_data(train_data, train_labels):
                    global_step = sess.run(model.global_step) + 1
                    print(global_step)
                    loss, train_op = model.step(sess, batch_data, batch_label)
                    # print(train_op)
                    print("loss: ", loss)

                    logits = sess.run(model.logits, feed_dict={model.token_seq:test_data, model.labels:test_labels, model.is_train: False})
                    # print("logits: ", logits)
                    print("f1_score: ", metrics(test_labels, logits))










if __name__ == '__main__':
    train()
