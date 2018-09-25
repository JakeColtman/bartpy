import unittest

import pandas as pd
import numpy as np

from bartpy.data import Data
from bartpy.errors import NoSplittableVariableException


class TestData(unittest.TestCase):

    def setUp(self):
        self.y = np.array([1, 2, 3, 4, 5])
        self.X = pd.DataFrame({"a": [1, 2, 3, 3, 4], "b": [1, 1, 1, 1, 1]}).values
        self.data = Data(self.X, self.y, normalize=True)

    def test_n_obsv(self):
        self.assertEqual(self.data.n_obsv, 5)

    def test_normalization(self):
        self.assertEqual(-0.5, self.data.y.min())
        self.assertEqual(0.5, self.data.y.max())

    def test_splittable_variables(self):
        self.assertListEqual(list(self.data.splittable_variables()), [0])

    def test_random_splittable_value(self):
        for a in range(100):
            self.assertIn(self.data.random_splittable_value(0), [1, 2, 3])
        self.assertIsNone(self.data.random_splittable_value(1))

    def test_random_splittable_variable(self):
        self.assertEqual(self.data.random_splittable_variable(), 0)
        self.filtered_data = Data(self.data.X[:,[1]], self.data.y)
        with self.assertRaises(NoSplittableVariableException):
            self.filtered_data.random_splittable_variable()


if __name__ == '__main__':
    unittest.main()
