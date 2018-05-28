import tensorflow as tf
from tensorflow.contrib import slim
from model import STD
from datasets import dataset_factory

flags = tf.app.flags
flags.DEFINE_string('dataset_dir', './datasets/small_target/',
                    'Directory with the validation data.')
flags.DEFINE_integer('eval_interval_secs', 60,
                    'Number of seconds between evaluations.')
flags.DEFINE_integer('num_evals', 50, 'Number of batches to evaluate.')
flags.DEFINE_string('log_dir', './logs/eval',
                    'Directory where to log evaluation data.')
flags.DEFINE_string('checkpoint_dir', './logs/',
                    'Directory with the model checkpoint data.')
flags.DEFINE_integer('batch_size', 32, 'The size of batch for eval')
FLAGS = flags.FLAGS


def main(_):
    # load the dataset
    dataset = dataset_factory.get_dataset('std', FLAGS.dataset_dir, 'val')
    tf.logging.set_verbosity(tf.logging.DEBUG)
    # load batch
    provider = slim.dataset_data_provider.DatasetDataProvider(dataset,
                                                              num_readers=4,
                                                              shuffle=True)
    image, label = provider.get(['image', 'label'])
    batch_image, batch_labels = tf.train.batch([image, label],
                                               batch_size=FLAGS.batch_size,
                                               num_threads=4,
                                               capacity=64)

    # get the model and prediction
    model = STD()
    arg_scope = model.arg_scope(is_training=False)
    with slim.arg_scope(arg_scope):
        logits = model.inference(batch_image)
    predictions = tf.nn.softmax(logits)

    # convert prediction values for each class into single class prediction
    predictions = tf.to_int64(tf.argmax(predictions, 1))

    # streaming metrics to evaluate
    metrics_to_values, metrics_to_updates = slim.metrics.aggregate_metric_map({
        'mse': slim.metrics.streaming_mean_squared_error(predictions, batch_labels),
        'accuracy': slim.metrics.streaming_accuracy(predictions, batch_labels),
    })

    # write the metrics as summaries
    for metric_name, metric_value in metrics_to_values.iteritems():
        tf.summary.scalar(metric_name, metric_value)

    # evaluate on the model saved at the checkpoint directory
    # evaluate every eval_interval_secs
    slim.evaluation.evaluation_loop(
        '',
        FLAGS.checkpoint_dir,
        FLAGS.log_dir,
        num_evals=FLAGS.num_evals,
        eval_op=metrics_to_updates.values(),
        eval_interval_secs=FLAGS.eval_interval_secs,
        max_number_of_evaluations=5)


if __name__ == "__main__":
    tf.app.run()