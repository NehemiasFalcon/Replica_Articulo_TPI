import unittest
from src.report import title

class ReportTests(unittest.TestCase):
    def test_title(self):
        self.assertEqual(title("  hello world "), "Hello World")
