from operator import le, gt
import unittest

import pandas as pd
import numpy as np

from bartpy.data import Data, make_bartpy_data
from bartpy.split import SplitCondition, Split, CombinedCondition


class TestSplit(unittest.TestCase):

    def test_null_split_returns_all_values(self):
        data = make_bartpy_data(pd.DataFrame({"a": [1, 2]}).values, np.array([1, 2]))
        split = Split(data)
        conditioned_data = split.data
        self.assertListEqual(list(data.X.get_column(0)), list(conditioned_data.X.get_column(0)))

    def test_single_condition_data(self):
        data = make_bartpy_data(pd.DataFrame({"a": [1, 2]}).values, np.array([1, 2]))
        left_condition, right_condition = SplitCondition(0, 1, le), SplitCondition(0, 1, gt)
        left_split, right_split = Split(data) + left_condition, Split(data) + right_condition
        self.assertListEqual([1], list(left_split.data.X.get_column(0)))
        self.assertListEqual([2], list(right_split.data.X.get_column(0)))

    def test_combined_condition_data(self):
        data = make_bartpy_data(pd.DataFrame({"a": [1, 2, 3, 4]}).values, np.array([1, 2, 1, 1]))

        first_left_condition, first_right_condition = SplitCondition(0, 3, le), SplitCondition(0, 3, gt)
        second_left_condition, second_right_condition = SplitCondition(0, 1, le), SplitCondition(0, 1, gt)

        split = Split(data)
        updated_split = split + first_left_condition + second_right_condition
        conditioned_data = updated_split.data
        self.assertListEqual([2, 3], list(conditioned_data.X.get_column(0)))

    def test_most_recent_split(self):
        data = make_bartpy_data(pd.DataFrame({"a": [1, 2, 3, 4]}).values, np.array([1, 2, 1, 1]))

        first_left_condition, first_right_condition = SplitCondition(0, 3, le), SplitCondition(0, 3, gt)
        second_left_condition, second_right_condition = SplitCondition(0, 1, le), SplitCondition(0, 1, gt)

        split = Split(data)
        updated_split = split + first_left_condition + second_right_condition
        self.assertEqual((split + first_left_condition).most_recent_split_condition(), first_left_condition)
        self.assertEqual(updated_split.most_recent_split_condition(), second_right_condition)


class TestCombinedCondition(unittest.TestCase):

    def setUp(self):
        self.X = np.array([1, 2, 4, 6, 3, 5]).reshape(6, 1)

    def test_single_condition(self):
        condition = SplitCondition(0, 3, gt)
        combined_condition = CombinedCondition([0], [condition])
        self.assertListEqual(list(combined_condition.condition(self.X)), [False, False, True, True, False, True])

    def test_multiple_conditions(self):
        conditions = [
            SplitCondition(0, 2, gt),
            SplitCondition(0, 5, le)
        ]

        combined_condition = CombinedCondition([0], conditions)
        self.assertEqual(combined_condition.variables[0].min_value, 2)
        self.assertEqual(combined_condition.variables[0].max_value, 5)
        self.assertListEqual(list(combined_condition.condition(self.X)), [False, False, True, False, True, True])

    def test_multiple_variables(self):
        conditions = [
            SplitCondition(0, 2, gt),
            SplitCondition(1, 1, gt)
        ]

        X = self.X[:, 0].reshape(3, 2)
        combined_condition = CombinedCondition([0, 1], conditions)
        self.assertListEqual(list(combined_condition.condition(X)), [False, True, True])


if __name__ == '__main__':
    unittest.main()
