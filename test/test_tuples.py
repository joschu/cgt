import cgt, numpy as np
import unittest

class TupleTestCase(unittest.TestCase):
    def runTest(self):
        x = cgt.vector()
        xval = np.random.randn(1)
        ytrue = ((xval,(xval,)),xval,)
        f = cgt.make_function([x], ((x,(x,)),x,))
        y = f(xval)
        assert y==ytrue
if __name__ == "__main__":
    TupleTestCase().runTest()