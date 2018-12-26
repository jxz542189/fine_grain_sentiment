import tensorflow as tf
from utils.logger import logger
from model.san_multi_model import Model
import os
import json
from sklearn.externals import joblib
import numpy as np
from sklearn.metrics import f1_score

PS_OPS = ['Variable', 'VariableV2', 'AutoReloadVariable']
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
        yield (train_data[i * config.n_gpu * config.batch_size: (i+1) * config.n_gpu *config.batch_size],
               train_labels[i * config.n_gpu * config.batch_size: (i+1) * config.n_gpu *config.batch_size])

def _deduplicate_indexed_slices(values, indices):
    """Sums `values` associated with any non-unique `indices`.
    Args:
      values: A `Tensor` with rank >= 1.
      indices: A one-dimensional integer `Tensor`, indexing into the first
      dimension of `values` (as in an IndexedSlices object).
    Returns:
      A tuple of (`summed_values`, `unique_indices`) where `unique_indices` is a
      de-duplicated version of `indices` and `summed_values` contains the sum of
      `values` slices associated with each unique index.
    """
    unique_indices, new_index_positions = tf.unique(indices)
    summed_values = tf.unsorted_segment_sum(
      values, new_index_positions,
      tf.shape(unique_indices)[0])
    return (summed_values, unique_indices)


def average_gradients(tower_grads):
    # calculate average gradient for each shared variable across all GPUs
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        # We need to average the gradients across each GPU.

        g0, v0 = grad_and_vars[0]

        if g0 is None:
            # no gradient for this variable, skip it
            average_grads.append((g0, v0))
            continue

        if isinstance(g0, tf.IndexedSlices):
            # If the gradient is type IndexedSlices then this is a sparse
            #   gradient with attributes indices and values.
            # To average, need to concat them individually then create
            #   a new IndexedSlices object.
            indices = []
            values = []
            for g, v in grad_and_vars:
                indices.append(g.indices)
                values.append(g.values)
            all_indices = tf.concat(indices, 0)
            avg_values = tf.concat(values, 0) / len(grad_and_vars)
            # deduplicate across indices
            av, ai = _deduplicate_indexed_slices(avg_values, all_indices)
            grad = tf.IndexedSlices(av, ai, dense_shape=g0.dense_shape)

        else:
            # a normal tensor can just do a simple average
            grads = []
            for g, v in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)
                # Append on a 'tower' dimension which we will average over
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)

        # the Variables are redundant because they are shared
        # across towers. So.. just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)

        average_grads.append(grad_and_var)

    assert len(average_grads) == len(list(zip(*tower_grads)))

    return average_grads

def clip_grads(grads, config):
    # grads = [(grad1, var1), (grad2, var2), ...]
    def _clip_norms(grad_and_vars, val, name):
        # grad_and_vars is a list of (g, v) pairs
        grad_tensors = [g for g, v in grad_and_vars]
        vv = [v for g, v in grad_and_vars]
        scaled_val = val

        clipped_tensors, g_norm = tf.clip_by_global_norm(
            grad_tensors, scaled_val)

        ret = []
        for t, (g, v) in zip(clipped_tensors, grad_and_vars):
            ret.append((t, v))

        return ret

    all_clip_norm_val = config.all_clip_norm_val
    ret = _clip_norms(grads, all_clip_norm_val, 'norm_grad')

    assert len(ret) == len(grads)

    return ret

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
    with tf.device('/cpu:0'):
        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0))
        token_seq = tf.placeholder(tf.int32, [None, config.seq_max_length], name='token_seq')

        token_labels = tf.placeholder(tf.int32, [None, config.cls_num], name='gold_label')
        is_train = tf.constant(config.is_train, dtype=tf.bool, name='is_train')

        opt = get_optimizer(config)
        tower_grads = []
        models = []
        norm_summaries = []
        total_loss = tf.get_variable('total_loss',
                                     [],
                                     initializer=tf.constant_initializer(0))
        for k in range(n_gpus):
            with tf.device(assign_to_device('/gpu:{}'.format(k), ps_device='/cpu:0')):
                with tf.variable_scope('san_another_multi_main', reuse=k>0):
                    _x = token_seq[k * config.batch_size: (k + 1) * config.batch_size]
                    _y = token_labels[k * config.batch_size: (k + 1) * config.batch_size]
                    model = Model(token_seq=_x, labels=_y, is_train=is_train, config=config,
                                  token_embedding_matrix=token_embedding_matrix)
                    model.build_loss()
                    loss = model.loss
                    models.append(model)
                    grads = opt.compute_gradients(loss,
                                                  aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
                    tower_grads.append(grads)
                    total_loss += loss

        grads = average_gradients(tower_grads)
        grads = clip_grads(grads, config)
        train_op = opt.apply_gradients(grads, global_step=global_step)
        init = tf.global_variables_initializer()

    with tf.Session(config=graph_config) as sess:
        sess.run(init)
        saver = tf.train.Saver(max_to_keep=3)
        writer = tf.summary.FileWriter(logdir=os.path.join(path, config.summary_dir))
        summary = tf.summary.merge_all()
        test_train = tf.constant(False, tf.bool)
        model_test = Model(token_seq=token_seq, labels=token_labels, is_train=test_train, config=config,
                      token_embedding_matrix=token_embedding_matrix)
        for k in range(config.n_epochs):
            for batch_data, batch_label in get_batch_data(train_data, train_labels):
                # n += 1
                feed_dict = {}
                for i in range(n_gpus):
                    model = models[i]
                    min_batch_data = batch_data[i * config.batch_size:(i+1) * config.batch_size]
                    min_batch_label = batch_label[i * config.batch_size:(i+1)* config.batch_size]
                    feed_dict.update({model.token_seq: min_batch_data, model.labels: min_batch_label})

                global_n, _, batch_total_loss, temp_summary = sess.run([global_step, train_op, total_loss, summary], feed_dict=feed_dict)
                writer.add_summary(temp_summary, global_n)
                print("global_step: {}, batch_total_loss: {}".format(global_n, batch_total_loss))


                if global_step % config.eval_period == 0:
                    pre = []
                    true_labels = []
                    total_loss = 0
                    for batch_test, test_label in get_batch_data(test_data, test_labels):
                        logits, loss = sess.run([model_test.logits, model_test.loss],
                                                feed_dict={model_test.token_seq: batch_test,
                                                           model_test.labels: test_label})
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
            logits, loss = sess.run([model_test.logits, model_test.loss],
                                    feed_dict={model_test.token_seq: batch_test, model_test.labels: test_label})
            total_loss += loss
            pre.append(logits)
            true_labels.append(test_label)
        test_label = np.concatenate(true_labels, axis=0)
        logits = np.concatenate(pre, axis=0)
        f1_score = metrics(test_label, logits)
        logger.info("final global_step: {}".format(global_n + 1))
        logger.info("loss: {}".format(total_loss))
        logger.info("f1_score: {}".format(f1_score))
        saver.save(sess, os.path.join(path, config.ckpt_path), 1000000)



if __name__ == '__main__':
    train()
