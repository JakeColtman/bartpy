import unittest

import pandas as pd

from bartpy.data import Data
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

    def test_random_splittable_value(self):
        for a in range(100):
            self.assertIn(self.data.random_splittable_value("a"), [1, 2, 3])
        self.assertIsNone(self.data.random_splittable_value("b"))

    def test_random_splittable_variable(self):
        self.assertEqual(self.data.random_splittable_variable(), "a")
        self.data._X = self.data.X.drop("a", axis=1)
        with self.assertRaises(NoSplittableVariableException):
            self.data.random_splittable_variable()


if __name__ == '__main__':
    unittest.main()
