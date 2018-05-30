import tensorflow as tf
from model import STD
import utils
import os
from datasets import dataset_factory
from tensorflow.contrib import slim

# train_dir = '/home/jerry/Projects/one-shot_detection/datasets/small_target/train/'
# test_dir = '/home/jerry/Projects/one-shot_detection/datasets/small_target/val/'
# logdir = 'graphs/'
# ckpt_dir = 'checkpoints/'
flags = tf.app.flags
tf.app.flags.DEFINE_string(
    'train_dir', './logs/',
    'The directory where training logs are stored'
)
tf.app.flags.DEFINE_string(
    'checkpoint_dir', './checkpoints/',
    'The directory where checkpoints are stored'
)
tf.app.flags.DEFINE_integer(
    'num_readers', 2,
    'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
    'batch_size', 32,
    'The number of samples in each batch.')
FLAGS = flags.FLAGS

def train(learning_rates, boundaries, weight_decay, dropout, max_epoch):
    model = STD(lrs=learning_rates, boundaries=boundaries,
                keep_prob=dropout, weight_decay=weight_decay)
    tf.logging.set_verbosity(tf.logging.DEBUG)
    # model.build_model()
    # start training
    utils.safe_mkdir('checkpoints')
    utils.safe_mkdir('logs')
    # writer = tf.summary.FileWriter(FLAGS.train_dir, tf.get_default_graph())
    # saver = tf.train.Saver(max_to_keep=5)
    # train_img_list, train_label_list = utils.get_files('datasets/small_target/train.txt', train_dir)
    # test_img_list, test_label_list = utils.get_files('datasets/small_target/val.txt', test_dir)
    with tf.name_scope('data_provider'):
        dataset = dataset_factory.get_dataset('std', './datasets/small_target/', 'train')
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers=FLAGS.num_readers,
            common_queue_capacity=20 * FLAGS.batch_size,
            common_queue_min=10 * FLAGS.batch_size,
            shuffle=True)
        [image, label] = provider.get(['image', 'label'])
        batch_image, batch_labels = tf.train.batch([image, label],
                                                   batch_size=FLAGS.batch_size,
                                                   num_threads=4,
                                                   capacity=64)
    arg_scope = model.arg_scope(is_training=True, weight_decay=model.weight_decay)
    with slim.arg_scope(arg_scope):
        batch_logits = model.inference(batch_image)
    model.losses(batch_logits, batch_labels)
    train_op = model.optimize()
    model.eval(batch_logits, batch_labels)
    summary_op = model.summary()
    # summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    slim.learning.train(train_op, FLAGS.train_dir,
                        log_every_n_steps=10,
                        global_step=model.g_step,
                        # saver=saver,
                        save_summaries_secs=5,
                        save_interval_secs=60,
                        session_config=config
                        # summary_writer=writer,
                        )


def main(_):
    train(learning_rates=[0.00001, 0.000003, 0.000003],
          boundaries=[6000, 12000],
          weight_decay=0.00003,
          dropout=0.5,
          max_epoch=100)


if __name__ == "__main__":
    tf.app.run()
