import unittest

import pandas as pd

from bartpy.data import Data, SplitData, Split, sample_split_condition, SplitCondition, LTESplitCondition, GTSplitCondition
from bartpy.errors import NoSplittableVariableException
from bartpy.tree import SplitCondition, SplitNode, split_node, LeafNode


class TestSplit(unittest.TestCase):

    def test_null_split_returns_all_values(self):
        split = Split([])
        data = Data(pd.DataFrame({"a": [1, 2]}), pd.Series([1, 2]))
        conditioned_data = split.split_data(data)
        self.assertListEqual(list(data.X["a"]), list(conditioned_data.X["a"]))

    def test_single_condition_data(self):
        condition = LTESplitCondition("a", 1)
        split = Split([condition])
        data = Data(pd.DataFrame({"a": [1, 2]}), pd.Series([1, 2]))
        conditioned_data = split.split_data(data)
        self.assertListEqual([1], list(conditioned_data.X["a"]))

    def test_combined_condition_data(self):
        first_condition = LTESplitCondition("a", 3)
        second_condition = GTSplitCondition("a", 1)

        split = Split([first_condition, second_condition])
        data = Data(pd.DataFrame({"a": [1, 2, 3, 4]}), pd.Series([1, 2, 1, 1]))
        conditioned_data = split.split_data(data)
        self.assertListEqual([2, 3], list(conditioned_data.X["a"]))

    def test_split(self):
        data = Data(pd.DataFrame({"a": [1, 2, 3]}), pd.Series([1, 2, 3]))
        self.a = split_node(LeafNode(data), SplitCondition("a", 1))
        left_data = self.a.left_child.data
        self.assertListEqual([1], list(left_data.X["a"]))


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

    def test_sample_split_condition(self):
        left_split, right_split = sample_split_condition(self.data)
        self.assertIn(left_split.splitting_variable, self.data.splittable_variables())
        self.assertIn(left_split.splitting_value, self.data.unique_values(left_split.splitting_variable))


if __name__ == '__main__':
    unittest.main()
