from itertools import product
from morph.kernel_elements.kernel_element import KernelElement


class BinaryDilation(KernelElement):
    def __init__(self, data, center) -> None:
        super().__init__(data, center)

    def dilate(self, binary_matrix, x, y):
        if self.hits(binary_matrix, x, y):
            for i, j in product(range(self.width), range(self.height)):
                yield (i + x, j + y)

    def transform(self, binary_matrix):
        copy_data = binary_matrix.copy().data
        w, h = copy_data.shape
        for i in range(w):
            for j in range(h):
                for x, y in self.dilate(binary_matrix, j, i):
                    copy_data[y, x] = 1

        return copy_data

