import numpy as np
from itertools import product
from abc import ABC, abstractmethod


class KernelElement:
    def __init__(self, data, center=None) -> None:
        self._data = np.array(data)
        self._center = center
        self._value = np.sum(self._data)

    def x_y_points(self, x, y, width, height):
        for i, j in self.flat_points:
            if (i + x) <= width and (j + y) <= height:
                yield np.array([i+x, j+y])

    @property
    def flat_points(self):
        return np.transpose(np.nonzero(self._data))

    @property
    def data(self):
        return self._data

    @property
    def center(self):
        return self._center

    @property
    def width(self):
        return self.data.shape[1]

    @property
    def height(self):
        return self.data.shape[0]

    def is_inside(self, binary_matrix, x, y):
        m = binary_matrix
        return (x + self.width <= m.width) and (y + self.height <= m.height)

    def fits(self, binary_matrix, x, y):
        if not self.is_inside(binary_matrix, x, y):
            return False

        num_fits = (
            binary_matrix.data[y : y + self.height, x : x + self.width] * self.data
        )
        fits = np.sum(num_fits) == self._value

        return fits

    def hits(self, binary_matrix, x, y):
        if not self.is_inside(binary_matrix, x, y):
            return False

        x_p, y_p = x + self.center[0], y + self.center[1]
        return binary_matrix.data[y_p, x_p] == 1

    @abstractmethod
    def transform(self, binary_matrix):
        pass
