import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from morph.kernel_elements.kernel_element import KernelElement


matplotlib.rcParams["backend"] = "TkAgg"


class MorphMatrix:
    def __init__(self, data) -> None:
        self._data = np.array(data)

    @property
    def data(self):
        return self._data

    @property
    def width(self):
        return self._data.shape[1]

    @property
    def height(self):
        return self._data.shape[0]

    @property
    def _constructor(self):
        return MorphMatrix

    def apply(self, kernel_element: KernelElement):
        new_data = kernel_element.transform(self)
        return self._constructor(new_data)

    def erosion(self, kernel_element: KernelElement):
        pass

    def dilation(self, kernel_element: KernelElement):
        pass

    def opening(self, kernel_element: KernelElement):
        s = kernel_element
        return self.erosion(s).dilation(s)

    def closing(self, kernel_element: KernelElement):
        s = kernel_element
        return self.dilation(s).erosion(s)

    def copy(self):
        return self._constructor(self._data.copy())

    def zeros(self):
        return self._constructor(np.zeros(self._data.shape).astype("bool"))

    def ones(self):
        return self._constructor(np.ones(self._data.shape).astype("bool"))

    def show(self, use_grid=False, cmap="gray", *args, **kwargs):
        if use_grid:
            plt.pcolormesh(self._data, edgecolors="k", linewidth=2)
            ax = plt.gca()
            ax.invert_yaxis()
            ax.set_aspect("equal")
        else:
            plt.imshow(self._data, cmap=cmap, *args, **kwargs)
        plt.show()

