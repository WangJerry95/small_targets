import tensorflow as tf
from model import STD
import utils
import numpy as np
import cv2
import time

PROBABILITY_THRESHOLD = 0.9
WINDOW_SIZE = (38, 38)
model = STD()
model.deploy()

# test_dir = '/home/jerry/Projects/one-shot_detection/data/small_target/val/'
# test_img_list, test_label_list = utils.get_files('data/small_target/val.txt', test_dir)
# test_batch, test_batch_labels = utils.get_batch(test_img_list,
#                                                 test_label_list,
#                                                 batch_num=0, batch_size=100)
img = cv2.imread('data/image/11.bmp', 0)
# mean = np.mean(img)
# std = np.std(img)
# img_normalized = (img-mean)/std

input_tensor = img[np.newaxis, :, :, np.newaxis]

with tf.Session() as sess:
    saver = tf.train.Saver(reshape=True)
    ckpt = tf.train.get_checkpoint_state('checkpoints/gamma=1e-4')
    if ckpt and ckpt.model_checkpoint_path:
        print(ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        global_step = sess.run(model.g_step)

    time_start = time.time()
    probability_map, feature_map = sess.run([model.probability_map, model.feature_map], feed_dict={model.X: input_tensor})
    probability_map = np.reshape(probability_map, [probability_map.shape[1], probability_map.shape[2]])
    feature_map = np.reshape(feature_map, [feature_map.shape[1], feature_map.shape[2], 2])
    feature_map_pos = feature_map[:, :, 0]
    feature_map_neg = feature_map[:, :, 1]

    suspect_region = np.where(probability_map > PROBABILITY_THRESHOLD)
    coordinates = np.vstack((suspect_region[1], suspect_region[0])).T  # exchange the x and y coordinates
    scores = [feature_map_pos[y, x] for x, y in coordinates]
    coordinates = 8*coordinates  # mapping to corresponding coordinate in origin image

    suppressed_coordinate = utils.non_max_suppress(coordinates, scores, WINDOW_SIZE, 0.1)
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

