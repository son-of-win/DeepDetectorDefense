import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.datasets import mnist, cifar10


def data_preprocessing_mnist(val_ratio):
    (x_train_all, y_train_all), (x_test, y_test) = mnist.load_data()
    x_train_all = x_train_all.reshape((x_train_all.shape[0], 28, 28, 1))

    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
    x_train_all = x_train_all.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    y_train_all = tf.keras.utils.to_categorical(y_train_all, 10)
    y_test = tf.keras.utils.to_categorical(y_test)

    x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size=val_ratio)
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def data_preprocessing_cifar10(val_ratio):
    (x_train_all, y_train_all), (x_test, y_test) = cifar10.load_data()
    x_train_all = x_train_all.reshape((x_train_all.shape[0], 32, 32, 3))

    x_test = x_test.reshape((x_test.shape[0], 32, 32, 3))
    x_train_all = x_train_all.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    y_train_all = tf.keras.utils.to_categorical(y_train_all, 10)
    y_test = tf.keras.utils.to_categorical(y_test)

    x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size=val_ratio)
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def shuffle_data(origin_data, origin_label, adv_data, adv_label):
    all_data = []
    for i in range(len(origin_label)):
        temp = [origin_label[i], origin_data[i], 0]
        all_data.append(temp)
    for i in range(len(adv_label)):
        temp = [adv_label[i], adv_data[i], 1]
        all_data.append(temp)
    random.shuffle(all_data)
    data_all_shuffle = np.array(all_data, dtype=object)
    data_train = data_all_shuffle[:, 1]
    label_train = data_all_shuffle[:, 0]
    is_adv = data_all_shuffle[:, 2]
    train_data = []
    for data in data_train:
        train_data.append(np.array(data))
    return np.array(train_data), label_train, is_adv
