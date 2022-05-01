import numpy as np
class Filter:
    def __init__(self, filter_size, filter_type):
        """
        init filter mask with size, type and coefficient
        :param filter_size:
        :param filter_type:
        """
        self.filter_size = filter_size
        self.filter_type = filter_type
        if filter_type == 'cross':
            self.coefficient = filter_size * 2 - 1
        elif filter_type == 'diamond':
            halfSize = filter_size // 2
            self.coefficient = 2 * (halfSize + 2 * (np.sum([x for x in range(halfSize)]))) + filter_size
        elif filter_type == "box":
            self.coefficient = filter_size ** 2

    def get_mask_matrix(self):
        """

        :return: matrix of 0 and 1 for each filter type  (cross, box or diamond)
        """
        filter_matrix = np.zeros((self.filter_size, self.filter_size), dtype=np.int16)
        if self.filter_type == 'cross':
            for i in range(self.filter_size):
                filter_matrix[i][self.filter_size // 2] = 1
            filter_matrix[self.filter_size // 2] = [1]
        elif self.filter_type == "box":
            filter_matrix = np.ones((self.filter_size, self.filter_size), dtype=np.int16)
        else:
            half = self.filter_size // 2
            for i in range(half):
                filter_matrix[i][half] = 1
                filter_matrix[self.filter_size - 1 - i][half] = 1
                if i > 0:
                    for j in range(self.filter_size):
                        if half - i <= j <= half + i:
                            filter_matrix[i][j] = 1
                            filter_matrix[self.filter_size - 1 - i][j] = 1
            filter_matrix[half] = [1]
        return filter_matrix