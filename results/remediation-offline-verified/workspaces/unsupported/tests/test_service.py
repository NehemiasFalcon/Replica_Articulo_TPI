import unittest
from src.service import status

class ServiceTests(unittest.TestCase):
    def test_status(self):
        self.assertEqual(status(), "online")
