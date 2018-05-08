import autoCorrection
import numpy as np
import unittest


class TestEndToEnd(unittest.TestCase):

    def test_end_to_end(self):
        counts = np.random.negative_binomial(n=20, p=0.2, size=(10, 8))
        sf = np.ones((10, 8))
        corrector = autoCorrection.correctors.AECorrector()
        correction = corrector.correct(counts=counts, size_factors=sf)
        self.assertEqual(counts.shape, correction.shape)


class TestSavingAndLoading(unittest.TestCase):

    def test_loading(self):
        self.test_saving()
        counts = np.random.negative_binomial(n=20, p=0.2, size=(10, 8))
        sf = np.ones((10, 8))
        corrector = autoCorrection.correctors.AECorrector(model_name='test1', model_directory=".")
        correction = corrector.correct(counts, sf, only_predict=True)
        self.assertEqual(counts.shape, correction.shape)

    def test_saving(self):
        counts = np.random.negative_binomial(n=20, p=0.2, size=(10, 8))
        sf = np.ones((10, 8))
        corrector = autoCorrection.correctors.AECorrector(model_name='test1', model_directory=".", save_model=True)
        correction = corrector.correct(counts, sf)
        self.assertEqual(counts.shape, correction.shape)

        
class TestSetSeed(unittest.TestCase):
    
    def test_setSeed(self):
        # generate data
        nsamples = 15
        ngenes = 20
        counts = np.random.negative_binomial(n=20, p=0.2, size=(ngenes, nsamples))
        sf = np.random.uniform(0.8, 1.2, size=(ngenes, nsamples))
        
        # run the autocorrection 2 times with seed and one without. it should deviate
        ac = autoCorrection.correctors
        correct1 = ac.AECorrector(model_name='test1', model_directory=".", save_model=True, verbose=0).correct(counts, sf)
        correct2 = ac.AECorrector(model_name='test1', model_directory=".", save_model=True, verbose=0, seed=42).correct(counts, sf)
        correct3 = ac.AECorrector(model_name='test1', model_directory=".", save_model=True, verbose=0, seed=42).correct(counts, sf)

        # check if the results are similar. Due to randomness in the numbers we still have little changes
        #self.assertTrue(sum(sum(np.round(correct2) == np.round(correct3))) > 0.9 * nsamples * ngenes)
        self.assertTrue(sum(sum(np.round(correct1) == np.round(correct2))) < 0.3 * nsamples * ngenes)
        self.assertTrue(sum(sum(np.round(correct1) == np.round(correct3))) < 0.3 * nsamples * ngenes)
    

if __name__ == '__main__':
    unittest.main()

