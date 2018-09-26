import unittest

import pandas as pd
import numpy as np

from bartpy.data import Data, is_not_constant
from bartpy.errors import NoSplittableVariableException


class TestData(unittest.TestCase):

    def setUp(self):
        self.y = np.array([1, 2, 3, 4, 5])
        self.X = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [1, 1, 1, 1, 1], "c": [1, 2, 3, 3, 4]})
        self.data = Data(self.X, self.y, normalize=True)

    def test_unnormalization(self):
        self.assertListEqual(list(self.data.unnormalized_y), list(self.y))
        self.assertListEqual(list(self.data.unnormalize_y(np.array([0, 0.25, 0.5, 0.75]))), [3, 4, 5, 6])

    def test_unique_proportion_of_value_in_variable(self):
        self.assertEqual(self.data.proportion_of_value_in_variable(0, 1), 0.2)

    def test_non_unique_proportion_of_value_in_variable(self):
        self.assertEqual(self.data.proportion_of_value_in_variable(2, 1), 0.2)
        self.assertEqual(self.data.proportion_of_value_in_variable(2, 3), 0.4)

    def test_unique_columns(self):
        self.assertEqual(self.data.unique_columns, [0])

    def test_covariates_stored_as_matrix(self):
        self.assertEqual(type(self.data.X), np.ndarray)

    def test_is_not_constant(self):
        self.assertTrue(is_not_constant(np.array([1, 1, 2, 3])))
        self.assertFalse(is_not_constant(np.array([1, 1, 1, 1])))

    def test_n_obsv(self):
        self.assertEqual(self.data.n_obsv, 5)

    def test_normalization(self):
        self.assertEqual(-0.5, self.data.y.min())
        self.assertEqual(0.5, self.data.y.max())

    def test_splittable_variables(self):
        self.assertListEqual(list(self.data.splittable_variables()), [0, 2])

    def test_random_splittable_value(self):
        for a in range(10000):
            self.assertIn(self.data.random_splittable_value(0), [1, 2, 3, 4])
        self.assertIsNone(self.data.random_splittable_value(1))

    def test_random_splittable_variable(self):
        for a in range(100):
            self.assertIn(self.data.random_splittable_variable(), [0, 2])
        self.filtered_data = Data(self.data.X[:,[1]], self.data.y)
        with self.assertRaises(NoSplittableVariableException):
            self.filtered_data.random_splittable_variable()

    def test_n_splittable_variables(self):
        self.assertEqual(self.data.n_splittable_variables, 2)

    def test_variables(self):
        self.assertEqual(self.data.variables, [0, 1, 2])

if __name__ == '__main__':
    unittest.main()
