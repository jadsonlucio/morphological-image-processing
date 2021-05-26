from morph.kernel_elements.binary.binary_erosion import BinaryErosion
from morph.kernel_elements.binary.binary_dilation import BinaryDilation
from morph.kernel_elements.kernel_element import KernelElement
from typing import Union

from PIL import Image
import numpy as np
from .morph_matrix import MorphMatrix


class BinaryMorphMatrix(MorphMatrix):
    def __init__(self, data) -> None:
        super().__init__(data)

    def erosion(self, kernel_element: KernelElement):
        s = kernel_element
        erosion_element = BinaryErosion(data=s.data, center=s.center)
        return self.apply(erosion_element)

    def dilation(self, kernel_element: KernelElement):
        s = kernel_element
        dilation_element = BinaryDilation(data=s.data, center=s.center)
        return self.apply(dilation_element)

    def complement(self):
        copy = ~self.data
        return BinaryMorphMatrix(copy)

    def union(self, other):
        copy = self.data | other.data
        return BinaryMorphMatrix(copy)

    def intersection(self, other):
        copy = self.data & other.data
        return BinaryMorphMatrix(copy)

    def subtraction(self, other):
        copy = (self.data | other.data) ^ other.data
        return BinaryMorphMatrix(copy)

    @classmethod
    def from_image(self, image: Union[Image.Image, str]):
        if isinstance(image, str):
            image = Image.open(image)

        binary_image_matrix = image.convert("1")
        new_arr = (np.array(binary_image_matrix)).astype("int")
        return BinaryMorphMatrix(new_arr)

    @property
    def _constructor(self):
        return BinaryMorphMatrix
