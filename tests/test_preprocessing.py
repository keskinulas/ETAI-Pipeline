import copy
import unittest

import numpy as np
import pandas as pd
import yaml
from sklearn.pipeline import Pipeline

from src.model import build_model
from src.preprocessing import (
    clean_dataset, drop_duplicate_rows, split_features_target,
    split_dev_test, build_preprocessor,
)
from src.evaluate import fairness_report


class PreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open('config.yaml') as f:
            cls.config = yaml.safe_load(f)

    def test_cleaning_preserves_rows_and_real_missing_categories(self):
        frame = pd.DataFrame({
            'id': [1, 1, None, None],
            'sex': [' Male ', None, 'Female', '-'],
            'age': [20, -1, 30, 40],
        })
        clean = clean_dataset(frame, self.config['diagnostics'])
        self.assertTrue(clean.index.equals(frame.index))
        self.assertEqual(clean.loc[0, 'sex'], 'Male')
        self.assertTrue(pd.isna(clean.loc[1, 'sex']))
        self.assertTrue(pd.isna(clean.loc[1, 'age']))
        self.assertEqual(drop_duplicate_rows(clean, 'id').index.tolist(), [0, 2, 3])

    def test_all_recipes_use_training_statistics_and_predict_unseen_rows(self):
        cfg = self.config
        raw = pd.read_csv(cfg['data']['path'])
        clean = drop_duplicate_rows(clean_dataset(raw, cfg['diagnostics']), 'id')
        X, y, extras = split_features_target(
            clean, cfg['data'], cfg['preprocessing']['mnar_indicator_sources']
        )
        self.assertEqual(len(X), len(clean))
        self.assertNotIn('race', X)
        self.assertNotIn('score_text', X)
        np.testing.assert_array_equal(
            X['priors_count_was_missing'], clean['priors_count'].isna().astype(int)
        )
        train, held, yt, yh, et, eh = split_dev_test(X, y, extras, .2, 42)
        self.assertTrue(train.index.equals(yt.index))
        self.assertTrue(held.index.equals(eh.index))
        self.assertFalse(set(train.index) & set(held.index))
        # Repeated label-free rows must each receive a prediction.
        incoming = raw.iloc[[0, 0, 1]].drop(columns=['two_year_recid', 'race', 'score_text'])
        incoming['sex'] = 'unseen-category'
        incoming['priors_count'] = np.nan
        incoming = clean_dataset(incoming, cfg['diagnostics'])
        new, labels, audit = split_features_target(
            incoming, cfg['data'], cfg['preprocessing']['mnar_indicator_sources']
        )
        self.assertIsNone(labels)
        self.assertIsNone(audit)
        for encoder in ['target', 'onehot', 'ordinal', 'count']:
            for scaler in ['robust', 'standard', 'minmax', 'none']:
                with self.subTest(encoder=encoder, scaler=scaler):
                    prep = copy.deepcopy(cfg['preprocessing'])
                    prep.update(encoder=encoder, scaler=scaler)
                    pipe = Pipeline([
                        ('prep', build_preprocessor(prep)),
                        ('model', build_model({'type': 'dummy', 'params': {'strategy': 'most_frequent'}})),
                    ]).fit(train, yt)
                    imputer = pipe['prep'].named_transformers_['numeric']['impute']
                    np.testing.assert_allclose(
                        imputer.statistics_, train[prep['numeric_features']].median().to_numpy()
                    )
                    before = imputer.statistics_.copy()
                    self.assertEqual(len(pipe.predict(new)), len(incoming))
                    np.testing.assert_array_equal(before, imputer.statistics_)
                    self.assertTrue(np.isfinite(pipe['prep'].transform(new)).all())

    def test_missing_compas_score_is_not_high_risk(self):
        extras = pd.DataFrame({'race': ['Other', 'Other'], 'score_text': [np.nan, 'Low']})
        report = fairness_report(pd.Series([0, 0]), np.array([0, 0]), extras)
        self.assertIn('FPR = 0.00  (n=1)', report)

    def test_new_models(self):
        for name in ['dummy', 'random_forest']:
            model = build_model({'type': name, 'params': None})
            model.fit([[0], [1], [2], [3]], [0, 0, 1, 1])
            self.assertEqual(len(model.predict([[1]])), 1)


if __name__ == '__main__':
    unittest.main()
