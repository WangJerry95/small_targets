import os
import tensorflow as tf

slim = tf.contrib.slim

SPLITS_TO_SIZES = {
    'train': 23328,
    'val': 1722
}

ITEMS_TO_DESCRIPTIONS = {
    'image': 'A image of fix height and width.',
    'shape': 'Shape of the image',
    'label': 'A list of labels, one per each object.',
}

def get_dataset(dataset_name, dataset_dir, split_name, reader=None, split_to_sizes=SPLITS_TO_SIZES):
    """Gets a dataset tuple with instructions for reading std dataset.

    :param dataset_name:
    :param dataset_dir:
    :param split_name:
    :param split_to_size:
    :param reader:
    :return:
    """
    if split_name not in split_to_sizes:
        raise ValueError('split name %s was not recognized.' % split_name)
    data_source = os.path.join(dataset_dir, dataset_name) + '_' + split_name + '.tfrecords'

    if reader is None:
        reader = tf.TFRecordReader

    # Features in std TFRecords.
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'label': tf.FixedLenFeature((), tf.int64, default_value=0)
    }
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format',
                                              channels=1, shape=[32, 32, 1]),
        'label': slim.tfexample_decoder.Tensor('label', shape=[])
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    return slim.dataset.Dataset(
            data_sources=data_source,
            reader=reader,
            decoder=decoder,
            num_samples=split_to_sizes[split_name],
            items_to_descriptions=ITEMS_TO_DESCRIPTIONS,
            num_classes=2)
