"""Causal feature construction, exact calendar gaps and label availability."""
import unittest
import numpy as np
import pandas as pd
from propensity_baseline.ablation import PURCHASE_FEATURES, purchase_features


class PurchaseFeatureTests(unittest.TestCase):
    def history(self):
        rows = []
        for month, y in [('2022-02', 1.), ('2022-04', 0.), ('2022-05', 1.)]:
            rows.append(dict(client_id='a', month=month, y_any=y,
                             **{f'target_{i}': y if i == 1 else 0. for i in range(1,5)}))
        return pd.DataFrame(rows)

    def protocol(self):
        return {'assumed_label_available_at': {
            m: (pd.Period(m, freq='M') + 1).start_time.tz_localize('UTC').isoformat()
            for m in ['2022-02', '2022-03', '2022-04', '2022-05']}}

    def test_current_and_future_labels_do_not_change_past_predictors(self):
        history = self.history()
        rows = pd.DataFrame({'client_id': ['a'], 'month': ['2022-04']})
        before = purchase_features(rows, history, self.protocol())
        history.loc[history.month >= '2022-04', ['y_any','target_1']] = 1.
        after = purchase_features(rows, history, self.protocol())
        pd.testing.assert_frame_equal(before, after)
        self.assertTrue(pd.isna(before.loc[0, 'purchase_any_lag1']))
        self.assertEqual(before.loc[0, 'purchase_any_lag2'], 1.)
        self.assertEqual(before.loc[0, 'purchase_known_months_3'], 1.)
        self.assertEqual(before.loc[0, 'purchase_months_since_last_positive'], 2.)

    def test_delayed_labels_are_unavailable_and_unknown_is_not_zero(self):
        rows = pd.DataFrame({'client_id': ['a','missing'], 'month': ['2022-05']*2}, index=[9,4])
        protocol = self.protocol()
        protocol['assumed_label_available_at']['2022-04'] = '2022-05-02T00:00:00Z'
        result = purchase_features(rows, self.history(), protocol)
        self.assertEqual(result.index.tolist(), [9,4])
        self.assertEqual(result.columns.tolist(), list(PURCHASE_FEATURES))
        self.assertTrue(result.purchase_any_lag1.isna().all())
        self.assertEqual(result.loc[9, 'purchase_positive_months_3'], 1.)
        self.assertEqual(result.loc[4, 'purchase_known_months_history'], 0.)
        self.assertTrue(pd.isna(result.loc[4, 'purchase_positive_months_history']))
        self.assertEqual(result.loc[4, 'purchase_has_previous_label'], 0.)

    def test_observed_zero_and_partial_unknown_product_labels(self):
        history = self.history()
        history.loc[history.month.eq('2022-02'), 'target_2'] = np.nan
        rows = pd.DataFrame({'client_id': ['a','a'], 'month': ['2022-03','2022-05']})
        result = purchase_features(rows, history, self.protocol())
        self.assertEqual(result.loc[0, 'purchase_any_lag1'], 1.)
        self.assertTrue(pd.isna(result.loc[0, 'purchase_target_2_lag1']))
        self.assertEqual(result.loc[1, 'purchase_any_lag1'], 0.)
        self.assertEqual(result.loc[1, 'purchase_has_previous_label'], 1.)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            purchase_features(pd.concat([rows,rows]), history, self.protocol())
