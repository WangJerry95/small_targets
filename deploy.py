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
img = cv2.imread('datasets/image/val_6.jpg', 0)
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
    #probability_map = np.reshape(probability_map, [probability_map.shape[1], probability_map.shape[2]])
    probability_map = np.squeeze(probability_map)
    #feature_map = np.reshape(feature_map, [feature_map.shape[1], feature_map.shape[2], 2])
    feature_map = np.squeeze(feature_map)

    suspect_region = np.where(probability_map > PROBABILITY_THRESHOLD)
    # coordinate: n*3 array, each row of which stands for (x, y, classid)
    coordinates = np.vstack((suspect_region[1], suspect_region[0], suspect_region[2])).T  # exchange the x and y coordinates
    scores = [feature_map[y, x, z+1] for x, y, z in coordinates]
    positions = np.hstack((8*coordinates[:, 0:-1], coordinates[:, -1:]))  # mapping to corresponding coordinate in origin image

    suppressed_coordinate = utils.non_max_suppress(positions, scores, WINDOW_SIZE, 0.0)
    end_time = time.time()
    print('Time cost: %.4f seconds' % (end_time-time_start))

    detect_out = img.copy()
    detect_out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for coordinate in suppressed_coordinate:
        tl = tuple(coordinate[0:-1])
        br = tuple(coordinate[0:-1] + WINDOW_SIZE)
        classid = coordinate[-1]
        if classid == 0:
            cv2.rectangle(detect_out, tl, br, (0, 0, 255))
        if classid == 1:
            cv2.rectangle(detect_out, tl, br, (0, 255, 0))
    # cv2.imshow('result', img)
    # cv2.waitKey(0)
    cv2.imwrite('result.bmp', detect_out)

