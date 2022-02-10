import unittest

import numpy as np
from numpy.testing import assert_almost_equal
from scipy.stats import norm

from smolppl.smolppl import (
    Normal,
    LatentVariable,
    ObservedVariable,
    evaluate_log_density,
    prior_sample,
    posterior_sample,
)


class TestSmolppl(unittest.TestCase):
    def test_log_density_correct_1(self):
        # z <- x
        z = LatentVariable("z", Normal, [0.0, 5.0])
        x = ObservedVariable("x", Normal, [z, 1.0], observed=5.0)

        assert_almost_equal(
            evaluate_log_density(x, {"z": 1.5}),
            norm.logpdf(1.5, 0, 5) + norm.logpdf(5, 1.5, 1.0),
        )

    def test_log_density_correct_2(self):
        # z <-
        #    +- x
        # w <-
        z = LatentVariable("z", Normal, [0.0, 5.0])
        w = LatentVariable("w", Normal, [0.0, 4.0])
        x = ObservedVariable("x", Normal, [z, w], observed=5.0)

        assert_almost_equal(
            evaluate_log_density(x, {"z": 1.5, "w": 0.5}),
            norm.logpdf(1.5, 0, 5)
            + norm.logpdf(0.5, 0.0, 4.0)
            + norm.logpdf(5, 1.5, 0.5),
        )

    def test_log_density_correct_3(self):
        # z <- w <- x
        # ^---------+
        z = LatentVariable("z", Normal, [0.0, 5.0])
        w = LatentVariable("w", Normal, [z, 5.0])
        x = ObservedVariable("x", Normal, [z, w], observed=5.0)

        assert_almost_equal(
            evaluate_log_density(x, {"z": 1.5, "w": 0.5}),
            norm.logpdf(1.5, 0, 5)
            + norm.logpdf(0.5, 1.5, 5)
            + norm.logpdf(5, 1.5, 0.5),
        )

    def test_log_density_correct_4(self):
        mu = LatentVariable("mu", Normal, [0.0, 5.0])
        y_bar = ObservedVariable("y_bar", Normal, [mu, 1.0], observed=5.0)

        assert_almost_equal(
            evaluate_log_density(y_bar, {"mu": 4.0}),
            norm.logpdf(4.0, 0.0, 5.0) + norm.logpdf(5.0, 4.0, 1.0),
        )

    def test_prior_sampling_correct_1(self):
        x = LatentVariable("x", Normal, [0.0, 3.0])
        y = ObservedVariable("y", Normal, [x, 4.0], observed=1.5)

        prior_samples = [prior_sample(y) for _ in range(10_000)]

        assert np.isclose(np.mean(prior_samples), 0.0, atol=1e-1)
        assert np.isclose(np.std(prior_samples), 5.0, atol=1e-1)

    def test_prior_sampling_correct_2(self):
        z = LatentVariable("z", Normal, [100.0, 0.1])
        w = LatentVariable("w", Normal, [z, 0.1])
        x = ObservedVariable("x", Normal, [z, w], observed=5.0)
        # x ~ N(z, w)
        # w ~ N(z, 0.1)
        # z ~ N(100, 0.1)
        #
        # z = N(100, 0.1)
        # w = N(100, 0.14)
        # x = N(N(100, 0.1), N(100, 0.14)). Should be ok
        # x = N(100, 0.1) + N(100, 0.14) * N(0, 1) ~= N(100, 100)
        # x ~= [99.8,100.2] + [99.7,100.3] * [-2, 2]
        # x ~= [-100, +300] => N(100, 100)

        n_samples = 1_000
        samples = [prior_sample(x) for _ in range(n_samples)]
        sample_mean_mean = np.mean(samples)
        sample_mean_std = np.std(samples) / np.sqrt(n_samples)

        # in CLT we trust
        assert (
            (sample_mean_mean - 2 * sample_mean_std)
            <= 100
            <= (sample_mean_mean + 2 * sample_mean_std)
        )

    def test_posterior_sampling_correct(self):
        x = LatentVariable("x", Normal, [0.0, 3.0])
        y = ObservedVariable("y", Normal, [x, 4.0], observed=1.5)

        posterior_samples = [posterior_sample(y, {"x": -2.0}) for _ in range(10_000)]

        assert np.isclose(np.mean(posterior_samples), -2.0, atol=1e-1)
        assert np.isclose(np.std(posterior_samples), 4.0, atol=1e-1)
