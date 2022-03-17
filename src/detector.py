import time
from filter_mask import *
import numpy as np
from skimage.measure import shannon_entropy


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
        image = image * 255
        image_data = np.array(image, dtype=np.float32)
        image_data //= interval
        image_data *= interval
        if not left:
            halfInterval = interval // 2
            image_data = image_data + halfInterval
        image_data /= 255.0

        return image_data

    def caculateEntropy(self, image):
        return shannon_entropy(image)

    def chooseCloserFilter(self, origin_image, filter1_image, filter2_image):
        """
        :param origin_image: origin
        :param filter1_image: output of filter1
        :param filter2_image: output of filter2
        :return:
        """
        distance1 = np.absolute(filter1_image - origin_image)
        distance2 = np.absolute(filter2_image - origin_image)
        result = filter1_image
        smaller_index = np.where(distance1 > distance2)
        result[smaller_index] = filter2_image[smaller_index]
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
