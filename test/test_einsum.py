import cgt, numpy as np,numpy.random as nr
import unittest


class EinsumTestCase(unittest.TestCase):    
    def runTest(self):
        cgt.set_precision("double")
        x = cgt.tensor3()
        y = cgt.tensor3()

        sizes = {'i':2,'j':3,'k':5,'l':7}
        xaxes = 'ijk'
        yaxes = 'ikl'
        zaxes = 'ijl'
        for i in xrange(10):
            xperm = xaxes
            (yperm,zperm) = permaxes = [[chars[i] for i in np.random.permutation(3)] for chars in [yaxes,zaxes]]
            desc = "%s,%s->%s"%tuple("".join(chars) for chars in [xperm] + permaxes)
            z = cgt.einsum(desc, x, y)
            xval = nr.randn(*(sizes[c] for c in xperm))
            yval = nr.randn(*(sizes[c] for c in yperm))
            np.testing.assert_allclose(
                cgt.numeric_eval(z, {x : xval, y : yval}),
                np.einsum(desc, xval, yval))


if __name__ == "__main__":
    EinsumTestCase().runTest()
