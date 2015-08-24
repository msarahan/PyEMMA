__author__ = 'noe'

import unittest
import numpy as np

class TestBAR(unittest.TestCase):

    def test_metropolis(self):
        from pyemma.thermo.util import metropolis
        P = metropolis([2000, 1.0, 1.0, 0.0], [100.0, 1.0, 2.0, 10.0])
        ref = np.array([1.00000000e+00, 1.00000000e+00, 3.67879441e-01, 4.53999298e-05])
        assert np.allclose(P, ref)

    def test_bar(self):
        from pyemma.thermo.util import bar
        # basic test 1: the free energy difference should be zero when both samples are from the same state
        U1 = np.array([[0.0, 1.0],
                       [1.0, 2.0],
                       [0.5, 1.5]])
        U2 = U1[:, np.array([1,0])]
        assert bar(U1, U2) == 0.0
        # basic test 2: this free energy difference should be -1.0
        U2 = U1.copy()
        assert bar(U1, U2) == -1.0
        # basic test 3: and this 1.0
        U1 = U1[:, np.array([1,0])]
        U2 = U2[:, np.array([1,0])]
        assert bar(U1, U2) == 1.0


if __name__ == "__main__":
    unittest.main()
