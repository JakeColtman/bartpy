import unittest

import pandas as pd

from bartpy.data import Data
from bartpy.split import SplitCondition, Split


class TestSplit(unittest.TestCase):

    def test_null_split_returns_all_values(self):
        data = Data(pd.DataFrame({"a": [1, 2]}), pd.Series([1, 2]))
        split = Split(data)
        conditioned_data = split.data
        self.assertListEqual(list(data.X["a"]), list(conditioned_data.X["a"]))

    def test_single_condition_data(self):
        data = Data(pd.DataFrame({"a": [1, 2]}), pd.Series([1, 2]))
        condition = SplitCondition("a", 1)
        left_split, right_split = Split(data) + condition
        self.assertListEqual([1], list(left_split.data.X["a"]))
        self.assertListEqual([2], list(right_split.data.X["a"]))

    def test_combined_condition_data(self):
        data = Data(pd.DataFrame({"a": [1, 2, 3, 4]}), pd.Series([1, 2, 1, 1]))

        first_condition = SplitCondition("a", 3)
        second_condition = SplitCondition("a", 1)
        split = Split(data)
        updated_split = ((split + first_condition)[0] + second_condition)[1]
        conditioned_data = updated_split.data
        self.assertListEqual([2, 3], list(conditioned_data.X["a"]))

    def test_most_recent_split(self):
        data = Data(pd.DataFrame({"a": [1, 2, 3, 4]}), pd.Series([1, 2, 1, 1]))

        first_condition = SplitCondition("a", 3)
        second_condition = SplitCondition("a", 1)
        split = Split(data)
        updated_split = ((split + first_condition)[0] + second_condition)[1]
        self.assertEqual((split + first_condition)[0].most_recent_split_condition(), first_condition)
        self.assertEqual(updated_split.most_recent_split_condition(), second_condition)


if __name__ == '__main__':
    unittest.main()
