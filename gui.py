import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from detect import Ui_MainWindow
import tensorflow as tf
from tensorflow.contrib import slim
from model import STD
import utils
import numpy as np
import cv2

PROBABILITY_THRESHOLD = 0.9
WINDOW_SIZE = (32, 32)

class App(QMainWindow):

    def __init__(self):
        super(App, self).__init__()
        self.ui =Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.actionOpen.triggered.connect(self.openFileNameDialog)
        self.ui.actionExit.triggered.connect(self.close)
        self.ui.actionBuild_model.triggered.connect(self.load_model)
        self.ui.actionDetect.triggered.connect(self.onDetect)
        self.ui.loadButton.clicked.connect(self.load_model)
        self.ui.detectButton.clicked.connect(self.onDetect)

        self.session = tf.Session()
        self.model = STD()
        self.inputs = tf.placeholder(tf.float32, shape=[None, None, None, 1])
        arg_scope = self.model.arg_scope(is_training=False)
        with slim.arg_scope(arg_scope):
            self.restorer = self.model.deploy(self.inputs)
        self.image_path = None
        self.model_path = None

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.image_path, _ = QFileDialog.getOpenFileName(self, "Select image", "./datasets/image/",
                                                         "bmp Files (*.bmp);;png Files (*.png)", options=options)
        self.image = QImage(self.image_path)
        self.ui.label.setPixmap(QPixmap.fromImage(self.image))
        self.ui.label.resize(self.image.width(), self.image.height())
        self.resize(self.image.width(), self.image.height())

    def load_model(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.model_path = QFileDialog.getExistingDirectory(self, "Select model", "./logs/",
                                                           options=options)

        ckpt = tf.train.get_checkpoint_state(self.model_path)
        if ckpt and ckpt.model_checkpoint_path:
            self.restorer.restore(self.session, ckpt.model_checkpoint_path)
            # global_step = self.session.run(self.model.g_step)
        QMessageBox.information(self, "Message", "Model loaded!", QMessageBox.Yes)

    def onDetect(self):
        if not self.image_path:
            QMessageBox.information(self, "error", "Image is not selected!", QMessageBox.Yes)
            return
        if not self.model_path:
            QMessageBox.information(self, "error", "Model is not loaded!", QMessageBox.Yes)
            return
        img = cv2.imread(self.image_path, 0)
        input_tensor = img[np.newaxis, :, :, np.newaxis]

        probability_map, feature_map = self.session.run([self.model.probability_map, self.model.feature_map],
                                                        feed_dict={self.inputs: input_tensor})
        probability_map = np.reshape(probability_map, [probability_map.shape[1], probability_map.shape[2]])
        feature_map = np.reshape(feature_map, [feature_map.shape[1], feature_map.shape[2], 2])
        feature_map_pos = feature_map[:, :, 1]

        suspect_region = np.where(probability_map > PROBABILITY_THRESHOLD)
        coordinates = np.vstack((suspect_region[1], suspect_region[0])).T  # exchange the x and y coordinates
        scores = [feature_map_pos[y, x] for x, y in coordinates]
        coordinates = 8 * coordinates  # mapping to corresponding coordinate in origin image

        suppressed_coordinate = utils.non_max_suppress(coordinates, scores, WINDOW_SIZE, 0.01)

        detect_out = img.copy()
        for coordinate in suppressed_coordinate:
            tl = tuple(coordinate)
            br = tuple(coordinate + WINDOW_SIZE)
            cv2.rectangle(detect_out, tl, br, (255, 255, 255))
        cv2.imwrite('result.bmp', detect_out)
        self.ui.label.setPixmap(QPixmap.fromImage(QImage('result.bmp')))
        # self.ui.label.resize(self.image.width(), self.image.height())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())