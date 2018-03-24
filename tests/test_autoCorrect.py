import autoCorrection
import numpy
import unittest


class TestEndToEnd(unittest.TestCase):

    def test_end_to_end(self):
        counts = numpy.random.negative_binomial(n=20, p=0.2, size=(10, 8))
        sf = numpy.ones((10, 8))
        corrector = autoCorrection.correctors.AECorrector()
        correction = corrector.correct(counts=counts, size_factors=sf)
        self.assertEqual(counts.shape, correction.shape)


class TestSavingAndLoading(unittest.TestCase):


    def test_loading(self):
        self.test_saving()
        counts = numpy.random.negative_binomial(n=20, p=0.2, size=(10, 8))
        sf = numpy.ones((10, 8))
        corrector = autoCorrection.correctors.AECorrector(model_name='test1', model_directory=".")
        correction = corrector.correct(counts, sf, only_predict=True)
        self.assertEqual(counts.shape, correction.shape)

    def test_saving(self):
        counts = numpy.random.negative_binomial(n=20, p=0.2, size=(10, 8))
        sf = numpy.ones((10, 8))
        corrector = autoCorrection.correctors.AECorrector(model_name='test1', model_directory=".", save_model=True)
        correction = corrector.correct(counts, sf)
        self.assertEqual(counts.shape, correction.shape)

if __name__ == '__main__':
    unittest.main()

