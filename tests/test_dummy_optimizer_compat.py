import unittest


try:
    from skopt.space import Integer, Real
    from optimize_model import _RandomSearchOptimizer
except ImportError:
    Integer = Real = _RandomSearchOptimizer = None


@unittest.skipIf(_RandomSearchOptimizer is None, "scikit-optimize is unavailable")
class DummyOptimizerCompatibilityTests(unittest.TestCase):
    def test_seeded_random_search_is_deterministic(self):
        dimensions = [Integer(10, 20), Real(1e-4, 1e-2, prior="log-uniform")]
        first = _RandomSearchOptimizer(dimensions, random_state=142)
        second = _RandomSearchOptimizer(dimensions, random_state=142)

        first_points = [first.ask() for _ in range(5)]
        second_points = [second.ask() for _ in range(5)]

        self.assertEqual(first_points, second_points)
        for point in first_points:
            self.assertGreaterEqual(point[0], 10)
            self.assertLessEqual(point[0], 20)
            self.assertGreaterEqual(point[1], 1e-4)
            self.assertLessEqual(point[1], 1e-2)

    def test_tell_accepts_finite_score(self):
        optimizer = _RandomSearchOptimizer([Integer(1, 3)], random_state=142)
        point = optimizer.ask()
        self.assertIsNone(optimizer.tell(point, 0.25))


if __name__ == "__main__":
    unittest.main()
