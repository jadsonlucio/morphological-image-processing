from morph.kernel_elements.kernel_element import KernelElement


class GrayScaleFlatErosion(KernelElement):
    def __init__(self, data) -> None:
        super().__init__(data)

    def transform(self, grayscale_matrix):
        copy_data = grayscale_matrix.copy().data
        m = grayscale_matrix
        w, h = copy_data.shape
        for i in range(w):
            for j in range(h):
                copy_data[i, j] = max(
                    map(
                        lambda p: grayscale_matrix[p[1], p[0]],
                        self.x_y_points(i, j, m.width, m.height),
                    )
                )

        return copy_data
