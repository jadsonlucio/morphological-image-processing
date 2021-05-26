from morph.kernel_elements.kernel_element import KernelElement


class GrayScaleFlatDilation(KernelElement):
    def __init__(self, data) -> None:
        super().__init__(data)

    def transform(self, grayscale_matrix):
        data = grayscale_matrix.data
        copy_data = grayscale_matrix.copy().data
        m = grayscale_matrix
        w, h = copy_data.shape
        print(copy_data.shape)
        for i in range(w):
            for j in range(h):
                print(i,j)
                try:
                    copy_data[j, i] = min(
                        map(
                            lambda p: data[p[1], p[0]],
                            self.x_y_points(i, j, m.width, m.height),
                        )
                    )
                except:
                    pass

        return copy_data
