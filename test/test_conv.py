import numpy as np
import cgt
from cgt import nn
import unittest

class ConvTestCase(unittest.TestCase):
    def runTest(self):
        np.random.seed(0)
        x = np.random.randn(2,2,5,17)
        f = np.random.randn(3,2,4,7)

        filtrows = f.shape[2]
        filtcols = f.shape[3]

        batchsize = x.shape[0]
        outchans = f.shape[0]

        try: 
            import scipy.signal
        except ImportError:
            print "skipping because we don't have ndimage"
            return

        out = np.zeros((batchsize,outchans,x.shape[2]+filtrows-1,x.shape[3]+filtcols-1))
        for b in xrange(x.shape[0]):
            for inchan in xrange(x.shape[1]):
                for outchan in xrange(outchans):
                    out[b,outchan] += scipy.signal.convolve2d(x[b,inchan],f[outchan,inchan],mode='full')

        cgt.set_precision('double')
        out1 = cgt.numeric_eval1(nn.conv2d(cgt.constant(x), cgt.constant(f)), {})
        np.testing.assert_allclose(out, out1)

if __name__ == "__main__":
    ConvTestCase().runTest()
