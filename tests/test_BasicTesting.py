import unittest
import autoCorrection
import inspect


class TestTesting(unittest.TestCase):

    def testing_is_working(self):
        self.assertEqual(True, True)

    def load_autoCorrection(self):
        corrector = autoCorrection.correctors.AECorrector()
        isinstance(corrector, autoCorrection.correctors.AECorrector)


if __name__ == '__main__':
    unittest.main()

