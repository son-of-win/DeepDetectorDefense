import numpy as np
from keras.models import load_model
from classifier import AlexnetClassifier
from utils import *
from detector import *


# prepare data
(x_train, y_train), (x_val, y_val), (x_test, y_test) = data_preprocessing_mnist(0.2)
adv = np.load('../data/fgsm_mnist_alexnet_origin=9_target=7_weight=0.35.npy')
adv_label = np.full(len(adv), 9)
train_data, train_label, is_adv = shuffle_data(x_test, y_test, adv, adv_label)

Classifier = AlexnetClassifier()
Classifier.load("lenet.h5")
detectorFilter = DefenderModel(28, 1)
print(Classifier.evaluate(x_test, y_test))
TN = 0
TP = 0
FN = 0
FP = 0
for i in range(len(train_data)):
    # current_label = int(np.argmax(train_label[i]))
    current_label_no_filter = int(np.argmax(Classifier.classifier.predict(train_data[i].reshape((1, 28, 28, 1)))))
    temp = np.reshape(train_data[i], (detectorFilter.image_channel, detectorFilter.image_size, detectorFilter.image_size))
    imageEntropy = detectorFilter.caculateEntropy(temp)
    if imageEntropy < 4:
        current_image_res = detectorFilter.scalarQuantization(temp, 128)
    elif imageEntropy < 5:
        current_image_res = detectorFilter.scalarQuantization(temp, 64)
    else:
        current_image_ASQ = detectorFilter.scalarQuantization(temp, 43)
        current_image_crossMean = detectorFilter.crossAndDiamondMeanFilter(current_image_ASQ, 3, 25, 5)
        current_image_res = detectorFilter.chooseCloserFilter(imageEntropy, current_image_ASQ, current_image_crossMean)
    current_x_res = np.reshape(current_image_res, (1, 28, 28, 1))
    current_x_label_predict = np.argmax(Classifier.classifier.predict(current_x_res))

    if current_label_no_filter != current_x_label_predict:
        if is_adv[i]:
            TP += 1
        else:
            FP += 1
    else:
        if is_adv[i]:
            FN += 1
        else:
            TN += 1
print("num origin: %d \nnum adv: %d" %(len(x_test), len(adv)))
print("TP: %d, FP: %d, FN: %d, TN: %d" % (TP, FP, FN, TN))
