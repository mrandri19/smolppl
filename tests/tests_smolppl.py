import unittest
from scipy.stats import norm
from smolppl.smolppl import (
    Normal,
    LatentVariable,
    ObservedVariable,
    evaluate_log_density,
)
from numpy.testing import assert_almost_equal


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
