from typing import Union

from PIL import Image
import numpy as np
from .morph_matrix import MorphMatrix
from .kernel_elements.grayscale.grayscale_dilation import GrayScaleFlatDilation
from .kernel_elements.grayscale.grayscale_erosion import GrayScaleFlatErosion
from .kernel_elements.kernel_element import KernelElement


class GrayscaleMorphMatrix(MorphMatrix):
    def erosion(self, kernel_element: KernelElement):
        s = kernel_element
        erosion_element = GrayScaleFlatErosion(data=s.data)
        return self.apply(erosion_element)

    def dilation(self, kernel_element: KernelElement):
        s = kernel_element
        dilation_element = GrayScaleFlatDilation(data=s.data)
        return self.apply(dilation_element)

    @property
    def _constructor(self):
        return GrayscaleMorphMatrix

    @classmethod
    def from_image(self, image: Union[Image.Image, str]):
        if isinstance(image, str):
            image = Image.open(image)

        grayscale_image_matrix = image.convert("L")
        new_arr = (np.array(grayscale_image_matrix)).astype("int")
        return GrayscaleMorphMatrix(new_arr)
