import numpy as np
import cgt
import unittest


class LinearRegressionTestCase(unittest.TestCase):
    def test_linreg(self):
        cgt.set_precision('double')
        N = 10
        K = 3

        Xval = np.random.randn(N,K)
        wval = np.random.randn(K)
        bval = np.random.randn()
        yval = np.random.randn(N)

        X_nk = cgt.matrix("X")
        y_n = cgt.vector("y")
        w_k = cgt.vector("w")
        b = cgt.scalar(name="b")

        ypred = cgt.dot(X_nk, w_k) + b

        err = cgt.sum(cgt.square(ypred - y_n))
        g = cgt.grad(err, [w_k, b])

        g_simple,an = cgt.simplify_and_analyze(g)


        print "Loss function:"
        cgt.print_tree([err])
        print "Gradient:"
        cgt.print_tree(g)

        print "Gradient simplified"
        cgt.print_tree(g_simple, nodefn=lambda node,o: o.write(" " + an["node2hash"][node][:5]))

        print "-------"

        d = {X_nk : Xval, w_k : wval, b : bval, y_n : yval}

        np.testing.assert_allclose(cgt.numeric_eval(err,d), np.linalg.norm(Xval.dot(wval) + bval - yval)**2)
        np.testing.assert_allclose(cgt.numeric_eval(g[0],d), 2 * Xval.T.dot(Xval.dot(wval) + bval - yval))
        np.testing.assert_allclose(cgt.numeric_eval(g[1],d), 2 *  np.sum(Xval.dot(wval) + bval - yval, 0))
        # add_log_entry("linreg", collect(values(d)), collect(keys(d)), [err], [g])


if __name__ == "__main__":
    unittest.main()
