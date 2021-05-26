from morph.kernel_elements.kernel_element import KernelElement


class BinaryErosion(KernelElement):
    def __init__(self, data, center) -> None:
        super().__init__(data, center)

    def erode(self, binary_matrix, x, y):
        if self.fits(binary_matrix, x, y):
            return [[x + self.width, y + self.height]]

        return []

    def transform(self, binary_matrix):
        copy_data = binary_matrix.zeros().data
        w, h = copy_data.shape
        for y in range(w):
            for x in range(h):
                for i, j in self.erode(binary_matrix, x, y):
                    copy_data[j, i] = 1

        return copy_data
