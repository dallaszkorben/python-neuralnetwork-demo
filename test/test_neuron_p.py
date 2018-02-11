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

    def test_a(self):
        pass
        myNN=Neurnet(type(self).x, type(self).y, ((3,)))

#if __name__ == '__main__':
#    unittest.main()