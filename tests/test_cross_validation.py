import unittest

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from unittest.mock import patch
from sklearn.model_selection import cross_validate

from src.evaluate import cross_validate_pipeline


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
            scores, predictions = cross_validate_pipeline(pipe, X, y, cv)
        self.assertFalse(hasattr(pipe['impute'], 'statistics_'))
        self.assertEqual(len(scores), 3)
        np.testing.assert_allclose(scores['gap'], scores['train'] - scores['validation'])
        seen = []
        for model, train, val in zip(captured['estimator'], captured['indices']['train'],
                                     captured['indices']['test']):
            self.assertEqual(model['impute'].statistics_[0], X.iloc[train]['value'].median())
            np.testing.assert_array_equal(predictions[val], model.predict(X.iloc[val]))
            seen.extend(val)
        self.assertEqual(sorted(seen), list(range(len(X))))


if __name__ == '__main__':
    unittest.main()
