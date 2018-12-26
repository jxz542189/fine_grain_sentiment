import tensorflow as tf
from utils.logger import logger
from model.san_model import Model
import os
import json
from sklearn.externals import joblib
import random
import numpy as np
from sklearn.metrics import f1_score
os.environ['CUDA_VISIBLE_DEVICES'] ='1'
path = os.path.dirname(os.path.realpath(__file__))
# print(path)
params_path = os.path.join(path, 'config', 'params.json')
data_path = os.path.join(path, 'data')
with open(params_path) as param:
    params_dict = json.load(param)
config = tf.contrib.training.HParams(**params_dict)
print(config)


def metrics(labels, logits):
    labels_split = np.hsplit(labels, 20)
    logis_split = np.hsplit(logits, 20)
    f_score = 0
    for logit, label in zip(logis_split, labels_split):
        logit = np.argmax(logit, axis=-1)
        label = np.argmax(label, axis=-1)
        temp_f1_score = f1_score(logit, label, average=None)
        f_score += np.average(temp_f1_score)
    return f_score / 20


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

    with g.as_default():
        with tf.variable_scope("%s" % config.model_name, reuse=tf.AUTO_REUSE) as scope:
            sess = tf.Session(config=graph_config)
            model = Model(config, token_embedding_matrix=token_embedding_matrix)
            model.build_loss()
            model.get_optimizer()

            if True:
                model.build_var_ema()

            if config.model_type == "train":
                model.build_ema()
            writer = tf.summary.FileWriter(logdir=os.path.join(path, config.summary_dir))
            summary = tf.summary.merge_all()
            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver(max_to_keep=3)
            for _ in range(config.n_epochs):
                for batch_data, batch_label in model.get_batch_data(train_data, train_labels):
                    global_step = sess.run(model.global_step) + 1

                    loss, train_op = model.step(sess, batch_data, batch_label)
                    temp_summary = sess.run(summary,
                                            feed_dict={model.token_seq: batch_data, model.labels: batch_label,
                                                       model.is_train: False})
                    writer.add_summary(temp_summary, global_step)
                    if global_step % config.eval_period == 0:
                        pre = []
                        true_labels = []
                        for batch_test, test_label in model.get_batch_data(test_data, test_labels):
                            logits = sess.run(model.logits,
                                              feed_dict={model.token_seq: batch_test, model.labels: test_label,
                                                         model.is_train: False})
                            pre.append(logits)
                            true_labels.append(test_label)
                        test_label = np.concatenate(true_labels, axis=0)
                        logits = np.concatenate(pre, axis=0)
                        f1_score = metrics(test_label, logits)
                        logger.info("global_step: {}".format(global_step))
                        logger.info("loss: {}".format(loss))
                        logger.info("f1_score: {}".format(f1_score))
                        saver.save(sess, os.path.join(path, config.ckpt_path), global_step)
            pre = []
            true_labels = []
            total_loss = 0
            for batch_test, test_label in model.get_batch_data(test_data, test_labels):
                logits, loss = sess.run([model.logits, model.loss],
                                  feed_dict={model.token_seq: batch_test, model.labels: test_label,
                                             model.is_train: False})
                total_loss += loss
                pre.append(logits)
                true_labels.append(test_label)
            test_label = np.concatenate(true_labels, axis=0)
            logits = np.concatenate(pre, axis=0)
            f1_score = metrics(test_label, logits)
            logger.info("loss: {}".format(total_loss))
            logger.info("f1_score: {}".format(f1_score))
            saver.save(sess, os.path.join(path, config.ckpt_path), 1000000)


if __name__ == '__main__':
    train()
