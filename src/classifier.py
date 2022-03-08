from abc import abstractmethod

from tensorflow.python.keras.layers import Reshape, UpSampling2D, Conv2DTranspose
from tensorflow.python.keras.models import Model

from utils import *
import tensorflow as tf
import os
from keras.models import load_model
from keras.layers import Input, Conv2D, Dense, Flatten, \
    GlobalAvgPool2D, MaxPooling2D, Softmax, BatchNormalization, Dropout


class AbstractClassifier:
    """
    Abstract Classifier
    """

    def __init__(self, save_path="../models", pretrained=True):
        """

        :param save_path: folder path to save classifier
        :param pretrained: set True to use load to load pretrained models
        """
        self.save_path = save_path
        self.classifier = None
        if not pretrained:
            self.build()
            self.compile()

    @abstractmethod
    def build(self):
        """
        Build Model.
        :return: None
        """
        pass

    def compile(self):
        """
        Compile model
        :return:  None
        """
        self.classifier.compile(optimizer='adam', loss="mse", metrics=['accuracy'])

    def train(self, epochs=50, batch_size=32, save_name='classifier.h5'):
        """

        :param epochs: epochs to train classifier.
        :param batch_size:
        :param save_name:
        :return:
        """
        x_train, y_train, x_val, y_val, x_test, y_test = data_preprocessing_mnist(0.2)
        self.classifier.fit(x_train, y_train, batch_size=batch_size, validation_data=(x_val, y_val), epochs=epochs)

        if save_name is not None:
            self.classifier.save(os.path.join(self.save_path, save_name))

    def load(self, save_name="classifier.h5", save_path=None):
        """

        :param save_name:
        :param save_path:
        :return:
        """
        if save_path is None:
            save_path = self.save_path

        path = os.path.join(save_path, save_name)
        self.classifier = load_model(path)

    def evaluate(self, data, label):
        """

        :param data:
        :param label:
        :return:
        """
        return self.classifier.evaluate(data, label)


class AlexnetClassifier(AbstractClassifier):
    """

    """

    def build(self):
        input_layer = Input(shape=(28, 28, 1))

        x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
        x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Flatten()(x)
        x = Dense(units=200)(x)
        x = Dense(units=200)(x)
        x = Dense(units=10)(x)
        output_layer = Softmax(name='softmax_output')(x)

        self.classifier = Model(input_layer, output_layer, name="decoder")
