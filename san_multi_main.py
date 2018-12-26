import tensorflow as tf
from utils.logger import logger
from model.san_multi_model import Model
import os
import json
from sklearn.externals import joblib
import numpy as np
from sklearn.metrics import f1_score


os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
path = os.path.dirname(os.path.realpath(__file__))
# print(path)
params_path = os.path.join(path, 'config', 'params.json')
data_path = os.path.join(path, 'data')
with open(params_path) as param:
    params_dict = json.load(param)
config = tf.contrib.training.HParams(**params_dict)
logger.info(config)

n_gpus = config.n_gpu

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


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


PS_OPS = ['Variable', 'VariableV2', 'AutoReloadVariable']

def assign_to_device(device, ps_device='cpu:0'):
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return "/" + ps_device
        else:
            return device

    return _assign


def get_optimizer(config):
    opt = None
    if config.optimizer.lower() == 'adadelta':
        assert config.learning_rate > 0.1 and config.learning_rate < 1.
        opt = tf.train.AdadeltaOptimizer(config.learning_rate)
    elif config.optimizer.lower() == 'adam':
        assert config.learning_rate < 0.1
        opt = tf.train.AdamOptimizer(config.learning_rate)
    elif config.optimizer.lower() == 'rmsprop':
        assert config.learning_rate < 0.1
        opt = tf.train.RMSPropOptimizer(config.learning_rate)
    else:
        raise AttributeError('no optimizer named as \'%s\'' % config.optimizer)
    return opt

def get_batch_data(train_data, train_labels):
    batch_num = int(len(train_data) / config.batch_size)
    for i in range(batch_num):
        yield (train_data[i * config.n_gpu * config.batch_size: (i+1)*config.n_gpu *config.batch_size],
               train_labels[i *config.n_gpu * config.batch_size: (i+1) *config.n_gpu * config.batch_size])


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

    # g = tf.Graph()
    if config.gpu_mem is None:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config.gpu_mem,
                                    allow_growth=True)
        graph_config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)

    else:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config.gpu_mem)
        graph_config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    # sess = tf.Session(config=graph_config)
    # sess.run(tf.global_variables_initializer())

    with tf.device('/cpu:0'):
        tower_grads = []
        with tf.variable_scope("%s" % config.model_name, reuse=tf.AUTO_REUSE):
            sess = tf.Session(config=graph_config)

            token_seq = tf.placeholder(tf.int32, [None, config.seq_max_length], name='token_seq')

            token_labels = tf.placeholder(tf.int32, [None, config.cls_num], name='gold_label')
            is_train = tf.constant(config.is_train, dtype=tf.bool,name='is_train')
            opt = get_optimizer(config)
            # logits = None
            for i in range(n_gpus):
                with tf.device(assign_to_device('/gpu:{}'.format(i), ps_device='/cpu:0')):
                    _x = token_seq[i * config.batch_size: (i + 1) * config.batch_size]
                    _y = token_labels[i * config.batch_size: (i + 1) * config.batch_size]
                    model = Model(token_seq=_x, labels=_y, is_train=is_train, config=config, token_embedding_matrix=token_embedding_matrix)
                    model.build_loss()
                    if True:
                        model.build_var_ema()

                    if config.model_type == "train":
                        model.build_ema()
                    grads = opt.compute_gradients(model.loss)
                    tower_grads.append(grads)

            tower_grads = average_gradients(tower_grads)
            train_op = opt.apply_gradients(tower_grads)

            writer = tf.summary.FileWriter(logdir=os.path.join(path, config.summary_dir))
            summary = tf.summary.merge_all()

            saver = tf.train.Saver(max_to_keep=3)
            test_train = tf.constant(False, tf.bool)
            model = Model(token_seq, token_labels, is_train=test_train, config=config, token_embedding_matrix=token_embedding_matrix)
            model.build_loss()
            n = 0
            sess.run(tf.global_variables_initializer())
            for k in range(config.n_epochs):
                for batch_data, batch_label in get_batch_data(train_data, train_labels):
                    n += 1
                    _ = sess.run(train_op, feed_dict={token_seq: batch_data, token_labels: batch_label})

                    loss, train_op = model.step(sess, batch_data, batch_label)
                    temp_summary = sess.run(summary,
                                            feed_dict={token_seq: batch_data, token_labels: batch_label})
                    global_step = n
                    writer.add_summary(temp_summary, global_step)
                    if global_step % config.eval_period == 0:
                        pre = []
                        true_labels = []
                        total_loss = 0
                        for batch_test, test_label in get_batch_data(test_data, test_labels):
                            logits, loss = sess.run([model.logits, model.loss],
                                                    feed_dict={model.token_seq: batch_test,
                                                               model.labels: test_label})
                            total_loss += loss
                            pre.append(logits)
                            true_labels.append(test_label)
                        test_label = np.concatenate(true_labels, axis=0)
                        logits = np.concatenate(pre, axis=0)
                        f1_score = metrics(test_label, logits)
                        logger.info("global_step: {}".format(global_step))
                        logger.info("loss: {}".format(total_loss))
                        logger.info("f1_score: {}".format(f1_score))
                        saver.save(sess, os.path.join(path, config.ckpt_path), global_step)
            pre = []
            true_labels = []
            total_loss = 0
            for batch_test, test_label in get_batch_data(test_data, test_labels):
                logits, loss = sess.run([model.logits, model.loss],
                                        feed_dict={model.token_seq: batch_test, model.labels: test_label})
                total_loss += loss
                pre.append(logits)
                true_labels.append(test_label)
            test_label = np.concatenate(true_labels, axis=0)
            logits = np.concatenate(pre, axis=0)
            f1_score = metrics(test_label, logits)
            logger.info("final global_step: {}".format(n + 1))
            logger.info("loss: {}".format(total_loss))
            logger.info("f1_score: {}".format(f1_score))
            saver.save(sess, os.path.join(path, config.ckpt_path), 1000000)


if __name__ == '__main__':
    train()
