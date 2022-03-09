import numpy as np
from keras.models import load_model
from classifier import AlexnetClassifier
from utils import *
from detector import *
from test_accuracy import test_mnist_data, test_mnist_only_scalar

# prepare data

# origin data
(x_train, y_train), (x_val, y_val), (x_test, y_test) = data_preprocessing_mnist(0.2)

# adv data
# adv1 = np.load('../data/fgsm_mnist_alexnet_origin=9_target=7_weight=0.35.npy')
# adv2 = np.load('../data/fgsm_mnist_alexnet_origin=9_target=7_weight=0.6.npy')
# adv3 = np.load('../data/fgsm_mnist_alexnet_origin=9_target=7_weight=0.122.npy')
# adv = np.load('../data/cw_l2.npy')
# adv = np.concatenate((adv1, adv2, adv3))
adv = np.load('../data/lbfgs_origin=7_target=9_weight=0.43.npy')
adv_label = np.full(len(adv), 9)

x_test = x_test[0:len(adv)]
y_test = y_test[0:len(adv)]
train_data, train_label, is_adv = shuffle_data(x_test, y_test, adv, adv_label)

Classifier = AlexnetClassifier()
Classifier.load("lenet.h5")
detectorFilter = DefenderModel(28, 1)
print(Classifier.evaluate(x_test, y_test))

print("num origin: %d \nnum adv: %d" % (len(x_test), len(adv)))
# test_mnist_data(train_data, train_label, is_adv, Classifier, detectorFilter)
test_mnist_only_scalar(train_data, train_label, is_adv, Classifier, detectorFilter)