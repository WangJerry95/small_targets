import os
import sys
import random
import cv2
import numpy as np

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'dataset_dir', './datasets/small_target/',
    'Directory where the original dataset is stored.')
tf.app.flags.DEFINE_string(
    'split_name', 'train',
    'split directory for the dataset')
tf.app.flags.DEFINE_string(
    'label_file', './datasets/small_target/train.txt',
    'The txt file where label and image name are stored'
)
tf.app.flags.DEFINE_string(
    'dataset_name', 'std',
    'The name of dataset'
)

def int64_feature(value):
    """Wrapper for inserting int64 features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    """Wrapper for inserting float features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def convert_to_example(image, label):
    """
    Build an Example proto for an image example
    :param image_data:
    :param label:
    :return: example proto
    """
    # shape = list(image.shape)
    image_data = image.tobytes()
    image_format = b'raw'
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(image_data),
        'label': int64_feature(label),
        # 'image/shape': int64_feature(shape),
        'image/format': bytes_feature(image_format)

    }))

    return example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.dataset_dir + FLAGS.dataset_name
                                         + '_' + FLAGS.split_name + '.tfrecords')
    with open(FLAGS.label_file,'r') as f:
        lines = f.readlines()
        random.shuffle(lines)
        image_num = 0
        for l in lines:
            try:
                l = l.split()
            except ValueError:
                continue
            image = cv2.imread(FLAGS.dataset_dir + FLAGS.split_name+"/" + l[0], 0)
            image = cv2.resize(image, (32, 32))
            label = int(l[1])
            example = convert_to_example(image, label)
            writer.write(example.SerializeToString())
            image_num += 1
            sys.stdout.write('\r>> Converting image %d/%d' % (image_num, len(lines)))
            sys.stdout.flush()


if __name__ == "__main__":
    tf.app.run()



