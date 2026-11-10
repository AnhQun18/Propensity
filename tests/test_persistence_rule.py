"""Persistence must use exactly M-1 and compare equal populations/budgets."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from propensity_baseline.config import BaselineConfig
from propensity_baseline.persistence import compare_predictions, persistence_predictions


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.frame=pd.DataFrame({'client_id':['a','b','c'],'month':['2022-03']*3,'y_any':[0.,1.,1.]})
        self.manifest={'months':[{'month':m} for m in ['2022-02','2022-03','2022-05']]}
        self.protocol={'assumed_label_available_at':{'2022-02':'2022-03-01T00:00:00Z'}}
        self.schema={'predictors':['trx_recent_event_count']}
        self.prior=pd.DataFrame({'client_id':['a','b','c'],'y_any':[1.,0.,np.nan]})

    def test_exact_previous_month_and_unknown_are_preserved(self):
        with patch('propensity_baseline.runner.read_months',return_value=self.prior) as read:
            result=persistence_predictions(SimpleNamespace(),self.frame,self.schema,self.manifest,self.protocol)
        self.assertEqual(read.call_args.args[1],['2022-02'])
        self.assertEqual(result.hard_prediction.iloc[0],1.)
        self.assertEqual(result.hard_prediction.iloc[1],0.)
        self.assertTrue(pd.isna(result.hard_prediction.iloc[2]))
        self.assertEqual(result.rule_available.tolist(),[True,True,False])

    def test_missing_calendar_month_is_not_filled_from_last_available_month(self):
        for month in ['2022-02','2022-05']:
            with self.subTest(month=month),patch('propensity_baseline.runner.read_months') as read:
                frame=self.frame.assign(month=month)
                result=persistence_predictions(SimpleNamespace(),frame,self.schema,self.manifest,self.protocol)
                read.assert_not_called()
                self.assertFalse(result.rule_available.any())
                self.assertTrue(result.score.isna().all())

    def test_delayed_labels_unavailable_at_cutoff_are_not_read(self):
        protocol={'assumed_label_available_at':{'2022-02':'2022-03-02T00:00:00Z'}}
        with patch('propensity_baseline.runner.read_months') as read:
            result=persistence_predictions(SimpleNamespace(),self.frame,self.schema,self.manifest,protocol)
        read.assert_not_called()
        self.assertFalse(result.rule_available.any())

    def test_current_month_outcome_does_not_affect_rule_score(self):
        with patch('propensity_baseline.runner.read_months',return_value=self.prior):
            before=persistence_predictions(SimpleNamespace(),self.frame,self.schema,self.manifest,self.protocol)
            after=persistence_predictions(SimpleNamespace(),self.frame.assign(y_any=1-self.frame.y_any),self.schema,self.manifest,self.protocol)
        pd.testing.assert_series_equal(before.score,after.score)

    def test_same_cohort_and_natural_rule_budget(self):
        model=pd.DataFrame({'client_id':list('abcd'),'month':['2022-03']*4,'y_any':[0,1,1,1],
                            'model':['multi_history']*4,'score':[.1,.9,.8,.99],'p_any':[.1,.9,.8,.99]})
        rule=model[['client_id','month','y_any']].assign(prior_y_any=[1.,1.,0.,np.nan],rule_available=[True,True,True,False])
        records,_=compare_predictions(model,rule,seed=3)
        result=records[0]
        self.assertEqual(result['comparison_rows'],3)
        self.assertEqual(result['excluded_unavailable_prior'],1)
        self.assertEqual(result['rule_natural_decisions']['selected'],2)
        self.assertEqual(result['model_at_rule_budget']['selected'],2)
        self.assertEqual(result['rule_natural_decisions']['tp'],1)
        self.assertEqual(result['model_at_rule_budget']['tp'],2)
        self.assertNotIn('log_loss',result['rule_ranking_metrics'])

    def test_default_baseline_includes_all_modalities(self):
        config=BaselineConfig('synthetic_features','synthetic_models')
        self.assertEqual(config.variants,('multi_recent','multi_history'))


if __name__=='__main__':
    unittest.main()
