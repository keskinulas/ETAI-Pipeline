import unittest

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from unittest.mock import patch
from sklearn.model_selection import cross_validate

from src.evaluate import build_scorers, cross_validate_pipeline, oof_classification_report, fairness_report


class CrossValidationTests(unittest.TestCase):
    def test_fold_local_imputation_and_aligned_oof_predictions(self):
        X = pd.DataFrame({'value': [0, 1, np.nan, 3, 4, 50, 6, 7, 8, 9, 100, 11]},
                         index=np.arange(12) * 3)
        y = pd.Series([0, 1] * 6, index=X.index)
        pipe = Pipeline([('impute', SimpleImputer(strategy='median')),
                         ('model', LogisticRegression())])
        cv = StratifiedKFold(3, shuffle=True, random_state=42)
        captured = {}

        def capture(*args, **kwargs):
            captured.update(cross_validate(*args, **kwargs))
            return captured

        with patch('src.evaluate.cross_validate', side_effect=capture):
            scorers = build_scorers(['accuracy', 'roc_auc', 'fbeta_2'])
            scores, predictions = cross_validate_pipeline(pipe, X, y, cv, scorers, 'roc_auc')
        self.assertFalse(hasattr(pipe['impute'], 'statistics_'))
        self.assertEqual(len(scores), 3)
        np.testing.assert_allclose(scores['gap'], scores['train'] - scores['validation'])
        np.testing.assert_allclose(scores['validation'], scores['roc_auc'])
        self.assertTrue(set(scorers).issubset(scores.columns))
        seen = []
        for model, train, val in zip(captured['estimator'], captured['indices']['train'],
                                     captured['indices']['test']):
            self.assertEqual(model['impute'].statistics_[0], X.iloc[train]['value'].median())
            np.testing.assert_allclose(predictions[val], model.predict_proba(X.iloc[val])[:, 1])
            seen.extend(val)
        self.assertEqual(sorted(seen), list(range(len(X))))

    def test_reports_use_threshold_and_exclude_missing_compas_scores(self):
        y = pd.Series([0, 0, 1, 1])
        proba = np.array([0.2, 0.6, 0.7, 0.9])
        extras = pd.DataFrame({'race': ['A'] * 4, 'score_text': [None, 'Low', 'High', 'High']})
        self.assertIn('predicted 0', oof_classification_report(y, proba))
        report = fairness_report(y, proba, extras, threshold=0.8)
        self.assertIn('FPR = 0.00  (n=2)', report)
        self.assertIn('FPR = 0.00  (n=1)', report)


if __name__ == '__main__':
    unittest.main()
