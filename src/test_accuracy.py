from utils import *
import time
import numpy as np

def get_output_image(image, detectorFilter):
    image_reshape = np.reshape(image,
                               (detectorFilter.image_channel, detectorFilter.image_size, detectorFilter.image_size))
    imageEntropy = detectorFilter.caculateEntropy(image_reshape)
    if imageEntropy < 4:
        current_image_res = detectorFilter.scalarQuantization(image_reshape, 128)
    elif imageEntropy < 5:
        current_image_res = detectorFilter.scalarQuantization(image_reshape, 64)
    else:
        current_image_ASQ = detectorFilter.scalarQuantization(image_reshape,8)
        current_image_crossMean = detectorFilter.MeanFilter(current_image_ASQ, 19, 'cross')
        current_image_res = detectorFilter.chooseCloserFilter(imageEntropy, current_image_ASQ,
                                                              current_image_crossMean)
        # print(abs(image_reshape - current_image_ASQ))
    current_x_res = np.reshape(current_image_res,
                               (detectorFilter.image_size, detectorFilter.image_size, detectorFilter.image_channel))
    return current_x_res

def test_mnist_data(train_data, train_label, is_adv, Classifier, detectorFilter):
    TN = 0
    TP = 0
    FN = 0
    FP = 0
    startTime = time.time()
    for i in range(len(train_data)):
        # current_label = int(np.argmax(train_label[i]))
        current_label_no_filter = int(np.argmax(Classifier.classifier.predict(train_data[i].reshape((1, 28, 28, 1)))))
        temp = np.reshape(train_data[i],
                          (detectorFilter.image_channel, detectorFilter.image_size, detectorFilter.image_size))
        imageEntropy = detectorFilter.caculateEntropy(temp)
        if imageEntropy < 4:
            current_image_res = detectorFilter.scalarQuantization(temp, 128)
        elif imageEntropy < 5:
            current_image_res = detectorFilter.scalarQuantization(temp, 64)
        else:
            current_image_ASQ = detectorFilter.scalarQuantization(temp, 43)
            current_image_crossMean = detectorFilter.MeanFilter(current_image_ASQ, 7, 'cross')
            current_image_res = detectorFilter.chooseCloserFilter(imageEntropy, current_image_ASQ,
                                                                  current_image_crossMean)
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
    # print("num origin: %d \nnum adv: %d" %(len(x_test), len(adv)))
    endTime = time.time()
    # recall = TP / (TP + FN)
    # precision = TP / (TP + FP)
    print("time run: %.4f" % (endTime - startTime))
    # print("Recall: %.4f\nPrecision: %.4f"%(recall, precision))
    print("TP: %d, FP: %d, FN: %d, TN: %d \n" % (TP, FP, FN, TN))


def test_mnist_only_scalar(train_data, train_label, is_adv, Classifier, detectorFilter):
    TN = 0
    TP = 0
    FN = 0
    FP = 0
    startTime = time.time()
    for i in range(len(train_data)):
        current_label_no_filter = int(np.argmax(Classifier.classifier.predict(train_data[i].reshape((1, 28, 28, 1)))))
        temp = np.reshape(train_data[i],
                          (detectorFilter.image_channel, detectorFilter.image_size, detectorFilter.image_size))
        imageEntropy = detectorFilter.caculateEntropy(temp)
        if imageEntropy < 4:
            current_image_res = detectorFilter.scalarQuantization(temp, 128)
        elif imageEntropy < 5:
            current_image_res = detectorFilter.scalarQuantization(temp, 64)
        else:
            current_image_res = detectorFilter.scalarQuantization(temp, 43)
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
    endTime = time.time()
    print("time run: %.4f" % (endTime - startTime))
    # print("Recall: %.4f\nPrecision: %.4f"%(recall, precision))
    print("TP: %d, FP: %d, FN: %d, TN: %d \n" % (TP, FP, FN, TN))


def test_cifar_data(train_data, train_labe, is_adv, Classifier, detectorFilter):
    TN = 0
    TP = 0
    FN = 0
    FP = 0
    startTime = time.time()
    for i in range(len(train_data)):
        # current_label = int(np.argmax(train_label[i]))
        current_label_no_filter = int(np.argmax(Classifier.classifier.predict(train_data[i].reshape((1, 32, 32, 3)))))
        temp = np.reshape(train_data[i],
                          (detectorFilter.image_channel, detectorFilter.image_size, detectorFilter.image_size))
        imageEntropy = detectorFilter.caculateEntropy(temp)
        if imageEntropy < 4:
            current_image_res = detectorFilter.scalarQuantization(temp, 128)
        elif imageEntropy < 5:
            current_image_res = detectorFilter.scalarQuantization(temp, 64)
        else:
            current_image_ASQ = detectorFilter.scalarQuantization(temp, 43)
            current_image_crossMean = detectorFilter.MeanFilter(current_image_ASQ, 7, 'cross')
            current_image_res = detectorFilter.chooseCloserFilter(imageEntropy, current_image_ASQ,
                                                                  current_image_crossMean)
        current_x_res = np.reshape(current_image_res, (1, 32, 32, 3))
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
    # print("num origin: %d \nnum adv: %d" %(len(x_test), len(adv)))
    endTime = time.time()
    # recall = TP / (TP + FN)
    # precision = TP / (TP + FP)
    print("time run: %.4f" % (endTime - startTime))
    # print("Recall: %.4f\nPrecision: %.4f"%(recall, precision))
    print("TP: %d, FP: %d, FN: %d, TN: %d \n" % (TP, FP, FN, TN))
