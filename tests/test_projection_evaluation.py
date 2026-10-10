"""Known-answer audit aggregations and measurement terminology."""
import unittest
import numpy as np
from src.vision.projection_evaluation import error_statistics,region_grid
from src.analytics.report import _table
from src.analytics.quality import pass_features
from src.analytics.config import AnalyticsConfig
import pandas as pd

class ProjectionEvaluationTests(unittest.TestCase):
    def test_statistics_and_coverage(self):
        s=error_statistics([3,4,np.nan]);self.assertEqual(s['samples'],2)
        self.assertAlmostEqual(s['rmse_m'],np.sqrt(12.5));self.assertAlmostEqual(s['coverage_pct'],200/3)
        self.assertEqual(s['p50_m'],3.5);self.assertEqual(s['median_m'],3.5)
    def test_empty_statistics(self):
        self.assertIsNone(error_statistics([np.nan])['mean_m'])
        self.assertIsNone(error_statistics([])['coverage_pct'])
    def test_actual_heatmap_counts(self):
        counts,means=region_grid([[1,1],[2,2],[8,8]],[2,4,10],[0,5,10],[0,5,10])
        np.testing.assert_array_equal(counts,[[2,0],[0,1]])
        self.assertEqual(means[0,0],3);self.assertTrue(np.isnan(means[0,1]))
    def test_truthful_headers(self):
        page=_table(pd.DataFrame([{'speed_peak_kmh':104,'difficulty_score_0_100':75,'pass_id':1}]),['speed_peak_kmh','difficulty_score_0_100','pass_id'])
        self.assertIn('estimated 2D',page);self.assertIn('heuristic',page);self.assertIn('transfer candidate',page)
    def test_numeric_heuristic_unchanged(self):
        values=pass_features([0,0],[20,0],[[10,1]],[[15,1]],1,AnalyticsConfig())
        self.assertEqual(values['inverse_geometric_difficulty_0_1'],values['heuristic_completion_score_0_1'])
        self.assertAlmostEqual(values['inverse_geometric_difficulty_0_1'],1-values['difficulty_score_0_100']/100)
