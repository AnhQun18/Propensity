import unittest
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mbd_mini_eda.currency import currency_analysis_group, currency_flags


class CurrencyFlagTests(unittest.TestCase):
    def test_nullable_flags_preserve_missing_and_input(self):
        source = pd.Series([11, 3, None, 15], dtype='Int64', index=[8, 4, 9, 1])
        before = source.copy()
        result = currency_flags(source)
        pd.testing.assert_series_equal(source, before)
        self.assertEqual(result.index.tolist(), source.index.tolist())
        self.assertEqual(result.currency_missing.tolist(), [False, False, True, False])
        self.assertEqual(result.loc[[8, 4, 1], 'currency_is_foreign'].tolist(), [False, True, True])
        self.assertTrue(pd.isna(result.loc[9, 'currency_is_main']))
        self.assertTrue(pd.isna(result.loc[9, 'currency_is_foreign']))
        self.assertTrue((result.loc[~result.currency_missing, 'currency_is_main'] ^ result.loc[~result.currency_missing, 'currency_is_foreign']).all())

    def test_fractional_codes_are_rejected(self):
        with self.assertRaises(ValueError):
            currency_flags(pd.Series([11, 11.5]))

    def test_rare_currency_group_is_distinct_from_missing_and_foreign_flag(self):
        source = pd.Series([11, 9, 5, 16, 12, 3, None], dtype='Int64')
        group = currency_analysis_group(source)
        self.assertEqual(group.tolist(), ['11', 'other_currency', 'other_currency',
                                         'other_currency', 'other_currency', '3', 'missing_currency'])
        flags = currency_flags(source)
        self.assertTrue(flags.loc[1:5, 'currency_is_foreign'].all())
        self.assertTrue(pd.isna(flags.loc[6, 'currency_is_foreign']))


if __name__ == '__main__':
    unittest.main()
