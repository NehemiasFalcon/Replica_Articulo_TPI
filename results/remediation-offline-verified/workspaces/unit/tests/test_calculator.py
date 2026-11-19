import unittest
from src.calculator import add

class CalculatorTests(unittest.TestCase):
    def test_positive(self):
        self.assertEqual(add(2, 3), 5)

    def test_zero(self):
        self.assertEqual(add(0, 4), 4)

    def test_negative(self):
        self.assertEqual(add(-2, 3), 1)
