import numpy as np
from keras.models import load_model
from classifier import AlexnetClassifier
from utils import *
from detector import *
from test_accuracy import *
from tuning_hyperparam import *

# prepare data

# origin data
(x_train, y_train), (x_val, y_val), (x_test, y_test) = data_preprocessing_cifar10(0.2)

# adv data
# mnist
# adv = np.load('../data/fgsm_mnist_alexnet_origin=9_target=7_weight=0.35.npy')
# adv2 = np.load('../data/fgsm_mnist_alexnet_origin=9_target=7_weight=0.6.npy')
# adv3 = np.load('../data/fgsm_mnist_alexnet_origin=9_target=7_weight=0.122.npy')
# adv4 = np.load('../data/cw_l2.npy')
# adv = np.concatenate((adv1, adv2, adv3))
# cifar
adv1 = np.load('../data/cifar10/cifar_fgsm_l2_0_250.npy')
adv2 = np.load('../data/cifar10/cifar_fgsm_l2_0_500.npy')
adv = np.concatenate((adv1, adv2))

adv_label = np.full(len(adv), 9)
x_train = x_train[0:len(adv)]
y_train = y_train[0:len(adv)]
train_data, train_label, is_adv = shuffle_data(x_train, y_train, adv, adv_label)

Classifier = AlexnetClassifier()
Classifier.load("cifar10_classifier.h5")
# detectorFilter = DefenderModel(28, 1)
detectorFilter = DefenderModel(32,3)
print(Classifier.evaluate(x_test, y_test))

print("num origin: %d \nnum adv: %d" % (len(x_train), len(adv)))
test_cifar_data(train_data, train_label, is_adv, Classifier, detectorFilter)
# test_mnist_only_scalar(train_data, train_label, is_adv, Classifier, detectorFilter)
# IntervalTuningMnist(train_data, train_label, is_adv, Classifier, detectorFilter)
# boxFilterSizeTuningMnist(train_data, train_label, is_adv, Classifier, detectorFilter)
# DiamondFilterSizeTuningMnist(train_data, train_label, is_adv, Classifier, detectorFilter)
# CrossFilterSizeTuningMnist(train_data, train_label, is_adv, Classifier, detectorFilter)
