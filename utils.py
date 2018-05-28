#coding=UTF-8
import numpy as np
import os
import cv2
import random
# you need to change this to your data directory
train_dir = '/home/jerry/Projects/one-shot_detection/data/small_target/train/'
test_dir = '/home/jerry/Projects/one-shot_detection/data/small_target/test/'


def get_files(file_txt, root_dir=train_dir):
    """
    Args:
        file_txt: a txt file contains samples' name and its labels
    Returns:
        list of images and labels
    """
    image_list = []
    label_list = []

    with open(file_txt, 'r') as file:
        lines = file.readlines()
        if not lines:
            raise Exception("txt file is empty")
        for line in lines:
            try:
                img_name, img_label = line.split(' ', 1)
            except ValueError:
                continue
            img_label = img_label.strip()
            image_list.append(root_dir+img_name)
            label_list.append(int(img_label))
    sample_pairs = zip(image_list, label_list)
    random.shuffle(sample_pairs)
    image_list, label_list = zip(*sample_pairs)
    return image_list, label_list


#%%

def get_batch(images, labels, batch_num, batch_size=64, capacity=2000):
    """
    Args:
        images: list type
        labels: list type
        batch_num: the index of batch
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    """
    pair = zip(images, labels)
    pair_batch = pair[batch_num*batch_size:(batch_num+1)*batch_size]
    image_batch = []
    label_batch = []
    for (image, label) in pair_batch:
        img = cv2.imread(image, 0)
        # mean = np.mean(img)
        # std = np.std(img, keepdims=True)
        # img = (img - mean)/(std+0.0001)
        image_batch.append(img)
        label_batch.append(label)
    image_batch = np.array(image_batch, dtype=np.float32)
    label_batch = np.array(label_batch, dtype=np.float32)
    label_batch_reverse = 1 - label_batch
    label_batch = np.vstack((label_batch, label_batch_reverse)).T
    return image_batch[:, :, :, np.newaxis], label_batch


def safe_mkdir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass


def non_max_suppress(coordinates, scores, window_size, iou_threshold):
    """
    :param coordinates: ndarray of shape [n,2], each row stands for a top left point of a window
    :param scores: the scores for each point
    :param window_size: tuple of (window_width, window_height)
    :param iou_threshold: iou threshold
    :return: suppressed coordinate in origin image
    """
    is_suppressed = np.zeros([len(scores)], dtype=np.bool)
    indices = np.argsort(-1*np.array(scores))  # sort by descending order
    for i in range(len(scores)):
        if is_suppressed[i]:
            continue
        for j in range(i+1, len(scores)):
            if is_suppressed[j]:
                continue
            iou = get_iou(coordinates[indices[i]], coordinates[indices[j]], window_size)
            if iou > iou_threshold:
                is_suppressed[j] = True
    suppress_indices = indices[np.where(is_suppressed)]
    suppressed_coordinates = np.delete(coordinates, suppress_indices, axis=0)

    return suppressed_coordinates


def get_iou(coordinate1, coordinate2, window_size):
    """

    :param coordinate1: ndarray of shape [2,], the top-left corner of a window
    :param coordinate2: ndarray of shape [2,], the top-left corner of a window
    :param window_size: tuple of (window_width, window_height)
    :return: iou: intersection over union
    """
    window_width, window_height = window_size
    horizontal_distance, vertical_distance = np.abs(coordinate1-coordinate2)
    if horizontal_distance > window_width or vertical_distance > window_height:
        return 0  # two window doesn't intersect
    intersection = (window_width - horizontal_distance)*(window_height - vertical_distance)
    union = 2*window_height*window_width - intersection
    iou = float(intersection)/float(union)
    return iou



