import math
from filter_mask import *
import numpy as np


class DefenderModel:
    def __init__(self, image_size, image_channel):
        self.image_size = image_size
        self.image_channel = image_channel

    def normalization(self, image):
        image[image < 0] = 0
        image[image > 1.0] = 1.0

    def scalarQuantization(self, image, interval, left=True):
        """

        :param image: input image
        :param interval:  number of interval
        :param left:
        :return:
        """
        if image.shape[0] == 1:
            image_data = image * 255
        else:
            image_data = np.array(image, dtype=np.float32)
        image_data //= interval
        image_data *= interval
        if not left:
            halfInterval = interval // 2
            image_data = image_data + halfInterval
        if image.shape[0] == 1:
            image_data /= 255.0

        return image_data

    def caculateEntropy(self, image):
        if image.shape[0] == 0:
            expandImage = np.array(image * 255.0, dtype=np.int16)
        else:
            expandImage = np.array(image, dtype=np.int16)
        H = 0
        for c in range(self.image_channel):
            fi = np.zeros(256, dtype='float32')
            for i in range(self.image_size):
                for j in range(self.image_size):
                    fi[expandImage[c][i][j]] += 1
            fi = fi / np.power(self.image_size, 2, dtype='float32')
            Hi = 0
            for i in range(256):
                if fi[i] > 0:
                    Hi += fi[i] * math.log(fi[i], 2)
            H += Hi
        return -H / self.image_channel

    def chooseCloserFilter(self, origin_image, filter1_image, filter2_image):
        """

        :param origin_image: origin
        :param filter1_image: output of filter1
        :param filter2_image: output of filter2
        :return:
        """
        result = np.zeros_like(origin_image)
        for i in range(self.image_channel):
            for j in range(self.image_size):
                for k in range(self.image_size):
                    a = abs(filter1_image[i][j][k] - origin_image[i][j][k])
                    b = abs(filter2_image[i][j][k] - origin_image[i][j][k])
                    if a < b:
                        result[i][j][k] = filter1_image[i][j][k]
                    else:
                        result[i][j][k] = filter2_image[i][j][k]
        return result

    def MeanFilter(self, input_image, filter_size, type_filter):
        """
        :param input_image:
        :param filter_size:
        :param type_filter: type of filter (box, cross or diamond)
        :return:
        """
        start = (filter_size - 1) // 2
        end = self.image_size - start
        input_image = np.array(input_image, dtype=np.float32)
        if type_filter == 'diamond':
            kernel_matrix = Filter(filter_size, "diamond")
        elif type_filter == "box":
            kernel_matrix = Filter(filter_size, "box")
        else:
            kernel_matrix = Filter(filter_size, "cross")
        print(input_image[0].shape)
        for row in range(start, end):
            for col in range(start, end):
                for channel in range(self.image_channel):
                    input_image[channel][row][col] = sum(sum(input_image[channel, row - start:row + start + 1,
                                                             col - start:col + start + 1] * kernel_matrix.get_mask_matrix()))
                    input_image[channel][row][col] /= kernel_matrix.coefficient
        return input_image
