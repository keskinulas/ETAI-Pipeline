import unittest

import numpy as np
from scipy import stats

from src.evaluate import corrected_paired_test


class ModelComparisonTests(unittest.TestCase):
    def test_corrected_variance_and_pair_direction(self):
        a = np.array([0.72, 0.75, 0.73, 0.71, 0.74])
        b = np.array([0.70, 0.74, 0.72, 0.72, 0.71])
        diff = a - b
        expected_t = diff.mean() / np.sqrt((1 / 5 + 1 / 4) * diff.var(ddof=1))
        result = corrected_paired_test(a, b, 1 / 4)
        self.assertAlmostEqual(result['t'], expected_t)
        self.assertAlmostEqual(result['p_value'], 2 * stats.t.sf(abs(expected_t), 4))
        reverse = corrected_paired_test(b, a, 1 / 4)
        self.assertAlmostEqual(reverse['mean_diff'], -result['mean_diff'])
        self.assertAlmostEqual(reverse['p_value'], result['p_value'])

    def test_identical_scores_and_invalid_pairs(self):
        self.assertEqual(corrected_paired_test([0.7] * 5, [0.7] * 5, .25)['p_value'], 1)
        for a, b in [([.7], [.6]), ([.7, .8], [.6]), ([.7, np.nan], [.6, .8])]:
            with self.assertRaises(ValueError):
                corrected_paired_test(a, b, .25)
