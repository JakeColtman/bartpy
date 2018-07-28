import unittest

import pandas as pd

from bartpy.data import Data, SplitData, Split, sample_split
from bartpy.errors import NoSplittableVariableException


class TestData(unittest.TestCase):

    def setUp(self):
        self.y = pd.Series([1, 2, 3, 4, 5])
        self.X = pd.DataFrame({"a": [1, 2, 3, 3, 4], "b": [1, 1, 1, 1, 1]})
        self.data = Data(self.X, self.y, normalize=True)

    def test_n_obsv(self):
        self.assertEqual(self.data.n_obsv, 5)

    def test_normalization(self):
        self.assertEqual(-0.5, self.data.y.min())
        self.assertEqual(0.5, self.data.y.max())

    def test_splittable_variables(self):
        self.assertListEqual(list(self.data.splittable_variables()), ["a"])

    def test_unique_values(self):
        self.assertListEqual([1, 2, 3, 4], list(self.data.unique_values("a")))
        self.assertListEqual([1], list(self.data.unique_values("b")))

    def test_splitting_data(self):
        split = Split("a", 2)
        split_data = self.data.split_data(split)
        self.assertListEqual(list(split_data.left_data.X["a"]), [1, 2])
        self.assertListEqual(list(split_data.left_data.X["a"]), [1, 2])
        self.assertListEqual(list(split_data.left_data.y), [-0.5, -0.25])
        self.assertListEqual(list(split_data.right_data.X["a"]), [3, 3, 4])

    def test_random_splittable_value(self):
        for a in range(100):
            self.assertIn(self.data.random_splittable_value("a"), [1, 2, 3])
        self.assertIsNone(self.data.random_splittable_value("b"))

    def test_random_splittable_variable(self):
        self.assertEqual(self.data.random_splittable_variable(), "a")
        self.data._X = self.data.X.drop("a", axis=1)
        with self.assertRaises(NoSplittableVariableException):
            self.data.random_splittable_variable()

    def test_sample_split(self):
        split = sample_split(self.data)
        self.assertIn(split.splitting_variable, self.data.splittable_variables())
        self.assertIn(split.splitting_value, self.data.unique_values(split.splitting_variable))


if __name__ == '__main__':
    unittest.main()
