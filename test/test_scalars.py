import cgt, numpy as np, numpy.random as nr, itertools as it
from numeric_diff import numeric_grad
import unittest


class ScalarTestCase(unittest.TestCase):
    def test_scalars(self):
        np.random.seed(0)
        cgt.set_precision('double')

        x = cgt.scalar('x')
        y = cgt.scalar('y')
        z = cgt.scalar('z')
        vars = [x,y,z] #pylint: disable=W0622
        vals = nr.rand(len(vars))+1

        PROB2RESULT = {}

        for ((key,_), cls) in it.chain(
                it.izip(cgt.UNARY_INFO.items(),it.repeat(cgt.ElwiseUnary)),
                it.izip(cgt.BINARY_INFO.items(),it.repeat(cgt.ElwiseBinary))
                ):
            cgt.utils.colorprint(cgt.utils.Color.YELLOW, "Testing %s\n"%key)
            if cls == cgt.ElwiseUnary:
                n_in = 1
                op = cls(key)
            else:
                n_in = 2
                op = cls(key, (True,True))
            inputvars = vars[0:n_in]
            inputvals = vals[0:n_in]
            out = cgt.Result(op, inputvars)
            f = cgt.make_function(inputvars, out)
            try:
                grads = cgt.grad(out, inputvars)
            except cgt.NonDifferentiable:
                print "nondiff"
                continue
            print "Function:"
            cgt.print_tree(out)
            print "Gradient original:"
            cgt.print_tree(grads)
            print "Gradient simplified:"
            grads_simple = cgt.simplify(grads)
            cgt.print_tree(grads_simple)
            gradf = cgt.make_function(inputvars, grads)
            nugrad = numeric_grad(lambda li: f(*li), inputvals) #pylint: disable=W0640
            cgtgrad = gradf(*inputvals)
            np.testing.assert_almost_equal(nugrad,cgtgrad,decimal=6)

            grad_count = cgt.count_nodes(grads_simple)
            PROB2RESULT[key] = {}
            PROB2RESULT[key]["grad"] = grad_count

        from tabulate import tabulate
        print tabulate([[key,val["grad"]] for (key,val) in PROB2RESULT.iteritems()],headers=["funcname","gradcount"])    


if __name__ == "__main__":
    unittest.main()