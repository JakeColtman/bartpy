import unittest

from bartpy.data import Data
from bartpy.split import SplitCondition, Split, GTSplitCondition, LTESplitCondition

import pandas as pd


class TestSplit(unittest.TestCase):

    def setUp(self):
        self.data = Data(pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1]}), pd.Series([1, 2, 3]))
        self.a = SplitCondition("a", 1)
        self.b = SplitCondition("b", 2)
        self.split_one = Split(self.data, [self.a.left])
        self.split_two = Split(self.data, [self.b.left])
        self.split_three = self.split_one + self.b.right

    def test_condition_adding_preserves_data(self):
        self.assertEqual(self.data, self.split_three.data)

    def test_nested_splitting_on_stored_data(self):
        conditions = self.split_three.condition()
        self.assertListEqual([True, False, False], list(conditions))

    def test_nested_splitting_on_new_data(self):
        new_data = Data(pd.DataFrame({"a": [1, 2, 3, -1], "b": [3, 2, 1, 5]}), pd.Series([1, 2, 3, 4]))
        conditions = self.split_three.condition(new_data)
        self.assertListEqual([True, False, False, True], list(conditions))

    def test_most_recent_split(self):
        self.assertEqual(self.split_one.most_recent_split_condition(), self.a.left)
        self.assertEqual(self.split_three.most_recent_split_condition(), self.b.right)


if __name__ == '__main__':
    unittest.main()
