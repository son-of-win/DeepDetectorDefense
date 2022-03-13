from utils import *
import time
import numpy as np


def IntervalTuningMnist(train_data, train_label, is_adv, Classifier, detectorFilter):
    print("Interval scalar tuning.......")
    intervals = [128, 85, 64, 51, 43, 37, 32, 28, 26]
    for interval in intervals:
        TN = 0
        TP = 0
        FN = 0
        FP = 0
        startTime = time.time()
        for i in range(len(train_data)):
            # current_label = int(np.argmax(train_label[i]))
            current_label_no_filter = int(
                np.argmax(Classifier.classifier.predict(train_data[i].reshape((1, 28, 28, 1)))))
            temp = np.reshape(train_data[i],
                              (detectorFilter.image_channel, detectorFilter.image_size, detectorFilter.image_size))
            current_image_res = detectorFilter.scalarQuantization(temp, interval)
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
        print("with intervals: %d, TP: %d, FP: %d, FN: %d, TN: %d \n" % (interval, TP, FP, FN, TN))


def boxFilterSizeTuningMnist(train_data, train_label, is_adv, Classifier, detectorFilter):
    print("Box mean filter tuning..........")
    kernelSize = [3, 5, 7, 9, 11]
    for kernel in kernelSize:
        TN = 0
        TP = 0
        FN = 0
        FP = 0
        startTime = time.time()
        for i in range(len(train_data)):
            # current_label = int(np.argmax(train_label[i]))
            current_label_no_filter = int(
                np.argmax(Classifier.classifier.predict(train_data[i].reshape((1, 28, 28, 1)))))
            temp = np.reshape(train_data[i],
                              (detectorFilter.image_channel, detectorFilter.image_size, detectorFilter.image_size))
            current_image_res = detectorFilter.MeanFilter(temp, kernel, 'box')
            current_x_res = np.reshape(current_image_res, (1, 28, 28, 1))
            current_x_label_predict = np.argmax(Classifier.classifier.predict(current_x_res))

            if current_label_no_filter != current_x_label_predict:
                if is_adv[i] == 1:
                    TP += 1
                else:
                    FP += 1
            else:
                if is_adv[i] == 1:
                    FN += 1
                else:
                    TN += 1
        endTime = time.time()
        print("time run: %.4f" % (endTime - startTime))
        print("with kernel: %d, TP: %d, FP: %d, FN: %d, TN: %d \n" % (kernel, TP, FP, FN, TN))


def DiamondFilterSizeTuningMnist(train_data, train_label, is_adv, Classifier, detectorFilter):
    print("Diamond mean filter tuning..........")
    kernelSize = [3, 5, 7, 9, 11]
    for kernel in kernelSize:
        TN = 0
        TP = 0
        FN = 0
        FP = 0
        startTime = time.time()
        for i in range(len(train_data)):
            # current_label = int(np.argmax(train_label[i]))
            current_label_no_filter = int(
                np.argmax(Classifier.classifier.predict(train_data[i].reshape((1, 28, 28, 1)))))
            temp = np.reshape(train_data[i],
                              (detectorFilter.image_channel, detectorFilter.image_size, detectorFilter.image_size))
            current_image_res = detectorFilter.MeanFilter(temp, kernel, "diamond")
            current_x_res = np.reshape(current_image_res, (1, 28, 28, 1))
            current_x_label_predict = np.argmax(Classifier.classifier.predict(current_x_res))

            if current_label_no_filter != current_x_label_predict:
                if is_adv[i] == 1:
                    TP += 1
                else:
                    FP += 1
            else:
                if is_adv[i] == 1:
                    FN += 1
                else:
                    TN += 1
        endTime = time.time()
        print("time run: %.4f" % (endTime - startTime))
        print("with kernel: %d, TP: %d, FP: %d, FN: %d, TN: %d \n" % (kernel, TP, FP, FN, TN))


def CrossFilterSizeTuningMnist(train_data, train_label, is_adv, Classifier, detectorFilter):
    print("Cross mean filter tuning..........")
    kernelSize = [3, 5, 7, 9, 11]
    for kernel in kernelSize:
        TN = 0
        TP = 0
        FN = 0
        FP = 0
        startTime = time.time()
        for i in range(len(train_data)):
            # current_label = int(np.argmax(train_label[i]))
            current_label_no_filter = int(
                np.argmax(Classifier.classifier.predict(train_data[i].reshape((1, 28, 28, 1)))))
            temp = np.reshape(train_data[i],
                              (detectorFilter.image_channel, detectorFilter.image_size, detectorFilter.image_size))
            current_image_res = detectorFilter.MeanFilter(temp, kernel, "Cross")
            current_x_res = np.reshape(current_image_res, (1, 28, 28, 1))
            current_x_label_predict = np.argmax(Classifier.classifier.predict(current_x_res))

            if current_label_no_filter != current_x_label_predict:
                if is_adv[i] == 1:
                    TP += 1
                else:
                    FP += 1
            else:
                if is_adv[i] == 1:
                    FN += 1
                else:
                    TN += 1
        endTime = time.time()
        print("time run: %.4f" % (endTime - startTime))
        print("with kernel: %d, TP: %d, FP: %d, FN: %d, TN: %d \n" % (kernel, TP, FP, FN, TN))
