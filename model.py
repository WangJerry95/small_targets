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
            net = slim.max_pool2d(net, [2, 2], scope='pool1')

            net = slim.repeat(net, 2, slim.conv2d, 64, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')

            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')

            feature_dim = net.shape[1] * net.shape[2] * net.shape[3]
            net = tf.reshape(net, [-1, feature_dim])

            net = slim.fully_connected(net, 256, scope='fc1')
            net = slim.dropout(net, self.keep_prob, scope='dropout')
            self.logits = slim.fully_connected(net, 2, scope='fc2')

            return self.logits

    def arg_scope(self, is_training, weight_decay=0.0001):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            weights_initializer=slim.xavier_initializer(),
                            biases_initializer=tf.zeros_initializer()):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                padding='SAME'):
                with slim.arg_scope([slim.conv2d],
                                    normalizer_fn=slim.batch_norm,
                                    normalizer_params={'is_training': is_training, 'decay': 0.90}):
                    with slim.arg_scope([slim.dropout], is_training=is_training) as sc:

                        return sc

    # def deploy(self):
    #     self.X = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='inputs')
    #     conv1 = conv_relu(self.X,
    #                       filters=32,
    #                       ksize=5,
    #                       stride=1,
    #                       padding='SAME',
    #                       regularizer=self.regularizer,
    #                       scope_name='conv1')
    #
    #     pool1 = maxpool(conv1,
    #                     ksize=2,
    #                     stride=2,
    #                     padding='VALID',
    #                     scope_name='pool1')
    #
    #     conv2 = conv_relu(pool1,
    #                       filters=64,
    #                       ksize=5,
    #                       stride=1,
    #                       padding='SAME',
    #                       regularizer=self.regularizer,
    #                       scope_name='conv2')
    #
    #     pool2 = maxpool(conv2,
    #                     ksize=2,
    #                     stride=2,
    #                     padding='VALID',
    #                     scope_name='pool2')
    #
    #     conv3 = conv_relu(pool2,
    #                       filters=128,
    #                       ksize=5,
    #                       stride=1,
    #                       padding='SAME',
    #                       regularizer=self.regularizer,
    #                       scope_name='conv3')
    #
    #     pool3 = maxpool(conv3,
    #                     ksize=2,
    #                     stride=2,
    #                     padding='VALID',
    #                     scope_name='pool3')
    #
    #     with tf.variable_scope('fc', reuse=tf.AUTO_REUSE):
    #         conv4_kernel = tf.get_variable('weights', shape=[4, 4, 128, 64],
    #                                        initializer=tf.truncated_normal_initializer(),
    #                                        regularizer=self.regularizer)
    #         conv4_biases = tf.get_variable('biases',shape=[64],
    #                                        initializer=tf.truncated_normal_initializer())
    #
    #     with tf.variable_scope('logits', reuse=tf.AUTO_REUSE):
    #         conv5_kernel = tf.get_variable('weights',
    #                                        shape=[1, 1, 64, 2],
    #                                        initializer=tf.truncated_normal_initializer(),
    #                                        regularizer=self.regularizer)
    #         conv5_biases = tf.get_variable('biases',shape=[2],
    #                                        initializer=tf.truncated_normal_initializer())
    #
    #     conv4 = tf.nn.conv2d(pool3,
    #                          filter=conv4_kernel,
    #                          strides=[1,1,1,1],
    #                          padding='VALID',
    #                          name='conv4')
    #     conv4 = tf.nn.relu(tf.add(conv4, conv4_biases))
    #
    #     conv5 = tf.nn.conv2d(conv4,
    #                          filter=conv5_kernel,
    #                          strides=[1,1,1,1],
    #                          padding='VALID',
    #                          name='conv5')
    #     conv5 = tf.add(conv5, conv5_biases)
    #
    #     self.feature_map = conv5
    #
    #     entropy = tf.nn.softmax(conv5, axis=-1)
    #     self.probability_map = entropy[:, :, :, 0]

    def losses(self, logits, labels):
        with tf.name_scope('loss'):
            # labels = tf.squeeze(labels)
            labels = tf.one_hot(labels, depth=2, on_value=1, off_value=0)
            self.loss = tf.losses.softmax_cross_entropy(labels, logits)
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
        optimizer = tf.train.AdamOptimizer(self.lr)
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
