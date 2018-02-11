import unittest
import numpy as np

from neuron_p import Neurnet


class TestNeuralNetworkP(unittest.TestCase):
    x=np.array([
        [0,0,1],
        [0,1,1],
        [1,0,1],
        [1,1,1]
    ])

    y=np.array([
        [0],
        [1],
        [1],
        [0],
    ])

    def setUpClass():
        print("Start ===> Runs setUpClass() before test suit")

    def setUp(self):
        print("           Runs setUp() before every test cases")

    def test_weidhts(self):

        myNN=Neurnet(type(self).x, type(self).y, ((4,)))

        self.assertEqual(myNN.weights[0].shape[0], 3, "1. weight-matrix has wrong 'input' dimension.")
        self.assertEqual(myNN.weights[0].shape[1], 4, "1. weight-matrix has wrong 'output' dimension.")

        self.assertEqual(myNN.weights[1].shape[0], 4, "2. weight-matrix has wrong 'input' dimension.")
        self.assertEqual(myNN.weights[1].shape[1], 1, "2. weight-matrix has wrong 'output' dimension.")

        np.testing.assert_array_almost_equal(myNN.weights[0], np.array(
            [[-0.16595599,  0.44064899, -0.99977125, -0.39533485],
             [-0.70648822, -0.81532281, -0.62747958, -0.30887855],
             [-0.20646505,  0.07763347, -0.16161097,  0.370439  ]] ), decimal=8, err_msg="1. weight-matrix has wrong values")
        np.testing.assert_array_almost_equal(myNN.weights[1], np.array(
            [[-0.5910955 ],
             [ 0.75623487],
             [-0.94522481],
             [ 0.34093502]] ), decimal=8, err_msg="2. weight-matrix has wrong values")


    def test_layers(self):

        myNN=Neurnet(type(self).x, type(self).y, ((4,)))
        myNN.feedForward()

        self.assertEqual(myNN.layers[0].shape[0], 4, "1. layer-matrix has wrong 'training' dimension.")
        self.assertEqual(myNN.layers[0].shape[1], 3, "1. layer-matrix has wrong 'neurons' dimension.")

        self.assertEqual(myNN.layers[1].shape[0], 4, "2. layer-matrix has wrong 'training' dimension.")
        self.assertEqual(myNN.layers[1].shape[1], 4, "2. layer-matrix has wrong 'neurons' dimension.")

        self.assertEqual(myNN.layers[2].shape[0], 4, "3. layer-matrix has wrong 'training' dimension.")
        self.assertEqual(myNN.layers[2].shape[1], 1, "3. layer-matrix has wrong 'neurons' dimension.")

        np.testing.assert_array_equal(myNN.layers[0], type(self).x, "1. layer has wrong values")
        np.testing.assert_array_almost_equal(myNN.layers[1], np.array(
            [[0.44856632, 0.51939863, 0.45968497, 0.59156505],
             [0.28639589, 0.32350963, 0.31236398, 0.51538526],
             [0.40795614, 0.62674606, 0.23841622, 0.49377636],
             [0.25371248, 0.42628115, 0.14321233, 0.41732254]]), decimal=8, err_msg="2. layer has wrong values")
        np.testing.assert_array_almost_equal(myNN.layers[2], np.array(
            [[0.47372957],
             [0.48895696],
             [0.54384086],
             [0.54470837]] ), decimal=8, err_msg="3. layer has wrong values")

    def test_cycle(self):
        pass

    def tearDown(self):
        print("          Runs tearDown() anyway after every cases")

    def tearDownClass():
        print("Stop ===> Runs tearDownClass() before test suit")

if __name__ == '__main__':
    unittest.main()