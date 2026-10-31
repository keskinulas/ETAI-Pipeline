"""Checks for isolation of outer validation rows and Optuna's returned estimator."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.tuning import nested_cross_validate, tune_pipeline, tuning_report
from src.evaluate import build_scorers


class TuningTests(unittest.TestCase):
    def test_nested_tuning_only_sees_outer_training_rows(self):
        X = pd.DataFrame({'value': [0, 1, np.nan, 3, 4, 50, 6, 7, 8, 9, 100, 11]},
                         index=np.arange(12) * 3)
        y = pd.Series([0, 1] * 6, index=X.index)
        pipe = Pipeline([('impute', SimpleImputer(strategy='median')),
                         ('model', LogisticRegression())])
        outer = StratifiedKFold(3, shuffle=True, random_state=42)
        inner = StratifiedKFold(2, shuffle=True, random_state=42)
        space = {'model__C': {'type': 'float', 'low': 0.0001, 'high': 100, 'log': True}}
        captured = []
        scorers = build_scorers(['accuracy', 'roc_auc', 'fbeta_2'])

        def capture(pipeline, features, target, *args, **kwargs):
            winner, study = tune_pipeline(pipeline, features, target, *args, **kwargs)
            self.assertFalse(hasattr(winner['model'], 'coef_'))
            self.assertEqual(winner['model'].C, study.best_params['model__C'])
            captured.append((features.copy(), winner, study))
            return winner, study

        with patch('src.tuning.tune_pipeline', side_effect=capture):
            scores, predictions = nested_cross_validate(
                pipe, X, y, outer, inner, scorers, 'roc_auc', space, 2, 42
            )
        self.assertFalse(hasattr(pipe['model'], 'coef_'))
        self.assertEqual(len(scores), 3)
        seen = []
        for fold, ((train, val), (features, winner, study)) in enumerate(zip(outer.split(X, y), captured)):
            pd.testing.assert_frame_equal(features, X.iloc[train])
            self.assertEqual(winner['impute'].statistics_[0], X.iloc[train]['value'].median())
            np.testing.assert_allclose(predictions[val], winner.predict_proba(X.iloc[val])[:, 1])
            for name, scorer in scorers.items():
                self.assertAlmostEqual(scores.iloc[fold][name], scorer(winner, X.iloc[val], y.iloc[val]))
            self.assertEqual(len(study.trials), 2)
            self.assertTrue(all(np.isfinite(t.user_attrs['std']) for t in study.trials))
            seen.extend(val)
        self.assertEqual(sorted(seen), list(range(len(X))))
        np.testing.assert_allclose(scores['gap'], scores['train'] - scores['validation'])
        self.assertIn('Optimism check', tuning_report(captured[-1][2], scores, 'roc_auc'))


if __name__ == '__main__':
    unittest.main()
