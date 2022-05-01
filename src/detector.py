import math
from filter_mask import *
import numpy as np
import tensorflow as tf
from skimage.measure import shannon_entropy
from scipy.signal import convolve2d

class DefenderModel:
    def __init__(self, image_size, image_channel):
        self.image_size = image_size
        self.image_channel = image_channel

    def normalization(self, image):
        image[image < 0] = 0
        image[image > 1.0] = 1.0

    def scalarQuantization(self, images, interval):
        """

        :param image: input image
        :param interval:  number of interval
        :param left:
        :return:
        """
        images = images * 255
        images_data = np.array(images, dtype=np.float32)
        images_data //= interval
        images_data *= interval
        images_data /= 255.0
        return images_data

    def entropy(self, image):
        p = np.array([(image==v).sum() for v in range(256)])
        p = p/p.sum()

        # compute entropy
        e = -(p[p>0]*np.log2(p[p>0])).sum()
        return e

    def caculateEntropy(self, image):
        return shannon_entropy(image)
    def oneDEntropy(self, inputDigit):
        expandDigit = np.array((inputDigit-0.5)*255,dtype=np.int16)
        f1 = np.zeros(256)
        f2 = np.zeros(256)
        f3 = np.zeros(256)
        for i in range(32):
            for j in range(32):
                f1[expandDigit[i][j][0]]+=1
                f2[expandDigit[i][j][1]]+=1
                f3[expandDigit[i][j][2]]+=1
        f1/=1024.0
        f2/=1024.0
        f3/=1024.0
        H1 = 0
        H2 = 0
        H3 = 0
        for i in range(256):
            if f1[i] > 0:
                H1+=f1[i]*math.log(f1[i],2)
            if f2[i] > 0:
                H2+=f2[i]*math.log(f2[i],2)
            if f3[i] > 0:
                H3+=f3[i]*math.log(f3[i],2)
        return -(H1+H2+H3)/3.0

    def caculateEntropyBatch(self, images, batch_size):
        entropys = []
        for image in images:
          entropys.append(shannon_entropy(image))
        return np.array(entropys)

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
        #smaller_index
        # smaller_index = np.where(distance1 > distance2)
        # result[smaller_index] = filter2_image[smaller_index]

        # bigger_index
        # bigger_index = np.where(distance1 < distance2)
        # result[bigger_index] = filter2_image[bigger_index]
        result = (filter1_image + filter2_image) / 2
        return result

    def MeanFilter(self, input_image, filter_size, type_filter):
        """
        :param input_image:
        :param filter_size:
        :param type_filter: type of filter (box, cross or diamond)
        :return:
        """
        image = np.array(input_image, dtype=np.float32)
        if type_filter == 'diamond':
            kernel_matrix = Filter(filter_size, "diamond")
        elif type_filter == "box":
            kernel_matrix = Filter(filter_size, "box")
        else:
            kernel_matrix = Filter(filter_size, "cross")
        for channel in range(self.image_channel):
            image[channel, :] = convolve2d(image[channel, :], kernel_matrix.get_mask_matrix(), mode='same')
            image[channel, :] = image[channel, :] / kernel_matrix.coefficient
        return image

    def MeanFilterBatch(self, images, filter_size, type_filter):
        kernel_core = Filter(filter_size, type_filter).get_mask_matrix()
        kernel = np.array([kernel_core, kernel_core, kernel_core])
        print(kernel.shape)
        return tf.nn.conv2d(input=images, filters=np.array(kernel).reshape((filter_size,filter_size,3,1)), strides=1,padding="SAME")
        # return tf.nn.conv2d(input=images, filters=kernel.reshape((filter_size,filter_size,3,1)), strides=1,padding="SAME")
    

