import unittest
import numpy as np
from morph.binary_morph_matrix import BinaryMorphMatrix

element1 = [[1, 1, 1], [0, 1, 0], [0, 1, 0]]


class TestBinaryMorphMatrix(unittest.TestCase):
    def setUp(self) -> None:
        self.element1 = BinaryMorphMatrix(np.array(element1))

    def test_complement(self):
        self.assertEqual(
            self.element1.complement().data.tolist(), (~np.array(element1)).tolist()
        )