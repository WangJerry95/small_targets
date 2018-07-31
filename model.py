""" Convolutional Neural Network.
Build and train a convolutional neural network with TensorFlow.
Author: Jerry Wang
Date:4/10/2018
"""
import tensorflow as tf
from tensorflow.contrib import slim


class STD(object):
    def __init__(self, lrs=[0.0001, 0.00003, 0.00001], boundaries=[9000, 12000],
                 keep_prob=0.5, weight_decay=0.0, skip_step=20):
        self.keep_prob = keep_prob
        self.skip_step = skip_step
        self.g_step = tf.contrib.framework.get_or_create_global_step()
        self.lr = tf.train.piecewise_constant(self.g_step, boundaries, lrs, name='learning_rate')
        self.n_classes = 2
        self.weight_decay = weight_decay

    def inference(self, inputs):
        with tf.variable_scope('std_32', values=[inputs]):
            inputs = tf.to_float(inputs)
            net = slim.repeat(inputs, 2, slim.conv2d, 32, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1', padding='VALID')

            net = slim.repeat(net, 2, slim.conv2d, 64, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2', padding='VALID')

            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3', padding='VALID')

            # feature_dim = net.shape[1] * net.shape[2] * net.shape[3]
            # net = tf.reshape(net, [-1, feature_dim])
            net = slim.flatten(net)
            net = slim.fully_connected(net, 256, scope='fc1')
            net = slim.dropout(net, self.keep_prob, scope='dropout')
            self.logits = slim.fully_connected(net, 3, scope='fc2')
            variables_to_restore = slim.get_model_variables()
            return self.logits

    def deploy(self, inputs):
        with tf.variable_scope('std_32', values=[inputs]):
            inputs = tf.to_float(inputs)
            net = slim.repeat(inputs, 2, slim.conv2d, 32, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1',padding='VALID')

            net = slim.repeat(net, 2, slim.conv2d, 64, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2',padding='VALID')

            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3',padding='VALID')

            net = slim.conv2d(net, 256, [4, 4], padding='VALID', normalizer_fn=None, scope='conv4')
            net = slim.dropout(net, self.keep_prob, scope='dropout')
            net = slim.conv2d(net, 3, [1, 1], padding='VALID', normalizer_fn=None, scope='conv5')
            self.feature_map = net
            self.probability_map = tf.nn.softmax(net, axis=-1)[:, :, :, 1:]

        def name_in_checkpoint(var):
            if "conv4" in var.op.name:
                return var.op.name.replace("conv4", "fc1")
            elif "conv5" in var.op.name:
                return var.op.name.replace("conv5", "fc2")
            else:
                return var.op.name

        variables_to_restore = slim.get_model_variables()
        variables_to_restore = {name_in_checkpoint(var): var for var in variables_to_restore}
        restorer = tf.train.Saver(variables_to_restore, reshape=True)
        return restorer

    def arg_scope(self, is_training, weight_decay=0.0001):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            weights_initializer=slim.xavier_initializer(),
                            biases_initializer=tf.truncated_normal_initializer(stddev=0.01)):
            with slim.arg_scope([slim.conv2d],
                                padding='SAME'):
                # with slim.arg_scope([slim.conv2d],
                #                     normalizer_fn=slim.batch_norm,
                #                     normalizer_params={'is_training': is_training, 'decay': 0.90}):
                with slim.arg_scope([slim.dropout], is_training=is_training) as sc:

                    return sc


    def losses(self, logits, labels):
        with tf.name_scope('loss'):
            # labels = tf.squeeze(labels)
            class_weights = tf.constant([0.6, 0.2, 0.2])
            weights = tf.gather(class_weights, labels)
            labels = tf.one_hot(labels, depth=3, on_value=1, off_value=0)
            self.loss = tf.losses.softmax_cross_entropy(labels, logits, weights=weights)
            self.reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            self.total_loss = tf.losses.get_total_loss()

    def focal_loss(self, gamma):
        with tf.name_scope('focal_loss'):
            probability = tf.nn.softmax(self.logits, axis=-1)
            pt = tf.multiply(probability, self.Y)
            pt = tf.reduce_sum(pt, axis=1)
            log_pt = tf.log(pt)
            self.reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            self.focal_loss = tf.reduce_sum(-1*(tf.pow((1-pt), gamma))*log_pt)
            self.loss = tf.add(self.focal_loss, self.reg_loss, name='loss')

    def optimize(self):
        optimizer = tf.train.RMSPropOptimizer(self.lr)
        train_op = slim.learning.create_train_op(self.total_loss, optimizer,
                                                 global_step=self.g_step)
        return train_op

    def summary(self):
        with tf.name_scope('summaries'):
            loss_summary = tf.summary.scalar('loss', self.total_loss)
            reg_loss_summary = tf.summary.scalar('reg_loss', self.reg_loss)
            batch_accuracy = tf.summary.scalar('batch_accuracy', self.accuracy)
            histogram_loss = tf.summary.histogram('histogram loss', self.total_loss)
            train_summary = tf.summary.merge([loss_summary, reg_loss_summary,
                                             batch_accuracy, histogram_loss],
                                             name='train_summary')

            return train_summary

    def eval(self, logits, labels):
        with tf.name_scope('evaluation'):
            # labels = tf.squeeze(labels)
            preds = tf.nn.softmax(logits)
            correct_preds = tf.equal(tf.argmax(preds, 1), labels)
            self.accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

            return self.accuracy
