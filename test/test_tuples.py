import cgt, numpy as np
import unittest

class TupleTestCase(unittest.TestCase):
    def runTest(self):
        f1 = cgt.make_function([], ())
        assert f1() == ()

        x = cgt.vector()
        xval = np.random.randn(1)
        f2 = cgt.make_function([x], [(x,x),(x,),()])
        ytrue = [(xval,xval),(xval,),()]
        y = f2(xval)
        assert y==ytrue
if __name__ == "__main__":
    TupleTestCase().runTest()

