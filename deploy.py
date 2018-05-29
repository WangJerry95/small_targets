import tensorflow as tf
from tensorflow.contrib import slim
from model import STD
import utils
import numpy as np
import cv2
import time

PROBABILITY_THRESHOLD = 0.99
WINDOW_SIZE = (38, 38)
model = STD()
arg_scope = model.arg_scope(is_training=False)
img = cv2.imread('datasets/image/1.bmp', 0)
input_tensor = img[np.newaxis, :, :, np.newaxis]
inputs = tf.placeholder(tf.float32, shape=[None, None, None, 1])
with slim.arg_scope(arg_scope):
    restorer = model.deploy(inputs)

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state('./logs')
    if ckpt and ckpt.model_checkpoint_path:
        print(ckpt.model_checkpoint_path)
        restorer.restore(sess, ckpt.model_checkpoint_path)

    time_start = time.time()
    probability_map, feature_map = sess.run([model.probability_map, model.feature_map], feed_dict={inputs: input_tensor})
    probability_map = np.reshape(probability_map, [probability_map.shape[1], probability_map.shape[2]])
    feature_map = np.reshape(feature_map, [feature_map.shape[1], feature_map.shape[2], 2])
    feature_map_pos = feature_map[:, :, 1]
    feature_map_neg = feature_map[:, :, 0]

    suspect_region = np.where(probability_map > PROBABILITY_THRESHOLD)
    coordinates = np.vstack((suspect_region[1], suspect_region[0])).T  # exchange the x and y coordinates
    scores = [feature_map_pos[y, x] for x, y in coordinates]
    coordinates = 8*coordinates  # mapping to corresponding coordinate in origin image

    suppressed_coordinate = utils.non_max_suppress(coordinates, scores, WINDOW_SIZE, 0.0)
    end_time = time.time()
    print('Time cost: %.4f seconds' % (end_time-time_start))

    detect_out = img.copy()
    for coordinate in suppressed_coordinate:
        tl = tuple(coordinate)
        br = tuple(coordinate + WINDOW_SIZE)

        cv2.rectangle(img, tl, br, (255, 255, 255))
    # cv2.imshow('result', img)
    # cv2.waitKey(0)
    cv2.imwrite('result.bmp', img)

