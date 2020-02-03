import unittest

import pandas as pd
import numpy as np

from bartpy.data import CovariateMatrix, Data, Target, is_not_constant, format_covariate_matrix
from bartpy.errors import NoSplittableVariableException


class TestIsNotConstant(unittest.TestCase):

    def setUp(self):
        self.non_constant_array = np.array([1, 1, 2, 3]).view(np.ma.MaskedArray)
        self.non_constant_array.mask = np.zeros_like(self.non_constant_array)
        self.constant_array = np.array([1, 1, 1, 1]).view(np.ma.MaskedArray)
        self.constant_array.mask = np.zeros_like(self.constant_array)

    def test_unmasked(self):
        self.assertTrue(is_not_constant(self.non_constant_array))
        self.assertFalse(is_not_constant(self.constant_array))

    def test_masked(self):
        self.non_constant_array.mask = np.array([False, False, True, True])
        self.assertFalse(is_not_constant(self.non_constant_array))


class TestDataNormalization(unittest.TestCase):

    def setUp(self):
        self.y_raw = [1, 2, 3, 4, 5]
        self.y = Target(np.array(self.y_raw), mask=np.zeros(5).astype(bool), n_obsv=5, normalize=True)

    def test_unnormalization(self):
        self.assertListEqual(list(self.y.unnormalized_y), self.y_raw)
        self.assertListEqual(list(self.y.unnormalize_y(np.array([0, 0.25, 0.5, 0.75]))), [3, 4, 5, 6])

    def test_normalization(self):
        self.assertEqual(-0.5, self.y.values.min())
        self.assertEqual(0.5, self.y.values.max())


class TestTargetCaching(unittest.TestCase):

    def setUp(self):
        self.y_raw = np.array([1, 2, 3, 4, 5])
        self.y = Target(self.y_raw, np.zeros(5).astype(bool), 5, False)

    def test_summed_y(self):
        self.assertEqual(self.y.summed_y(), np.sum(self.y_raw))
        self.y.update_y(np.array(self.y_raw * 2))
        self.assertEqual(self.y.summed_y(), np.sum(self.y_raw) * 2)

    def test_y(self):
        self.assertListEqual(list(self.y.values), list(self.y_raw))
        updated_y = np.array(self.y_raw * 2)
        self.y.update_y(updated_y)
        self.assertListEqual(list(self.y.values), list(updated_y))


class TestMasking(unittest.TestCase):

    def setUp(self):
        self.y = np.array([1, 2, 3, 4, 5])
        self.X = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [1, 1, 1, 1, 1], "c": [1, 2, 3, 3, 4]})
        self.X = format_covariate_matrix(self.X)
        self.mask = np.array([True, True, False, False, False])
        self.data = Data(self.X, self.y, self.mask, normalize=False)

    def test_y_sum(self):
        self.assertEqual(self.data.y.summed_y(), 12)

    def test_updating_y_sum(self):
        self.data.update_y(self.y * 2)
        self.assertEqual(self.data.y.summed_y(), 24)

    def test_n_obsv(self):
        self.assertEqual(self.data.X.n_obsv, 3)

    def test_updating_mask(self):
        from bartpy.splitcondition import SplitCondition
        from operator import le
        s = SplitCondition(0, 4, le)
        updated_data = self.data + s

        self.assertListEqual(list(updated_data.mask), [True, True, False, False, True])
        self.assertListEqual(list(updated_data.X.mask), [True, True, False, False, True])
        self.assertListEqual(list(updated_data.y._mask), [True, True, False, False, True])
        self.assertEqual(updated_data.X.n_obsv, 2)
        self.assertEqual(updated_data.X._n_obsv, 2)
        self.assertEqual(updated_data.y.summed_y(), 7)


class TestCovariateMatrix(unittest.TestCase):

    def setUp(self):
        self.X = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [1, 1, 1, 1, 1], "c": [1, 2, 3, 3, 4]})
        self.X = format_covariate_matrix(self.X)
        self.X = CovariateMatrix(self.X, mask=np.zeros(5).astype(bool), n_obsv=5, unique_columns=None, splittable_variables=None)

    def test_unique_proportion_of_value_in_variable(self):
        self.assertEqual(self.X.proportion_of_value_in_variable(0, 1), 0.2)

    def test_non_unique_proportion_of_value_in_variable(self):
        self.assertEqual(self.X.proportion_of_value_in_variable(2, 1), 0.2)
        self.assertEqual(self.X.proportion_of_value_in_variable(2, 3), 0.4)

    def test_unique_columns(self):
        unique_columns = [i for i in self.X.variables if self.X.is_column_unique(i)]
        self.assertEqual(unique_columns, [0])

    def test_n_obsv(self):
        self.assertEqual(self.X.n_obsv, 5)

    def test_splittable_variables(self):
        self.assertListEqual(list(self.X.splittable_variables()), [0, 2])

    def test_random_splittable_value(self):
        for _ in range(10000):
            self.assertIn(self.X.random_splittable_value(0), [1, 2, 3, 4])
        with self.assertRaises(NoSplittableVariableException):
            self.assertIsNone(self.X.random_splittable_value(1))

    def test_random_splittable_variable(self):
        for _ in range(100):
            self.assertIn(self.X.random_splittable_variable(), [0, 2])

        filtered_X = CovariateMatrix(self.X.values[:, [1]], mask=np.zeros(5).astype(bool), n_obsv=5, unique_columns=None, splittable_variables=None)
        with self.assertRaises(NoSplittableVariableException):
            filtered_X.random_splittable_variable()

    def test_n_splittable_variables(self):
        self.assertEqual(self.X.n_splittable_variables, 2)

    def test_variables(self):
        self.assertEqual(self.X.variables, [0, 1, 2])


if __name__ == '__main__':
    unittest.main()
