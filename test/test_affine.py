import cgt
import numpy as np
import unittest
import numpy.random as nr

PROB2RESULT = {}

def gradients_affine(f, xs):
    grads = [np.zeros_like(x) for x in xs]
    xs = map(np.copy, xs)
    yorig = f(*xs)
    h=0.1
    for (x,g) in zip(xs, grads):
        for i in xrange(x.size):
            xiorig = x.flat[i]
            x.flat[i] = xiorig + h
            y = f(*xs)
            g.flat[i] = (y - yorig)/h
            x.flat[i] = xiorig
    return grads

def tensor_like(x):
    return cgt.tensor(x.dtype, x.ndim)

def broadcast(opname,x,y,bcpat):
    return cgt.broadcast(opname, x, y, bcpat) if isinstance(x, cgt.Node) else eval("x %s y"%opname)

def check_affine(f, *nu_inputs):
    types = ",".join(["{%s,%s}"%(x.dtype, x.ndim) for x in nu_inputs])
    cgt.colorprint(cgt.Color.YELLOW,"Testing %s(%s)\n"%(f.__name__, types))
    sy_inputs = map(tensor_like, nu_inputs)
    for (i,sy) in enumerate(sy_inputs):
        sy.name = "x%i"%i

    sy_result = f(*sy_inputs)

    print("Function:")
    cgt.print_tree([sy_result])

    f_cgt = cgt.make_function(sy_inputs, sy_result)
    sy_grads = cgt.grad(sy_result, sy_inputs)
    gradf_cgt = cgt.make_function(sy_inputs, sy_grads)

    sy_result_simple = cgt.simplify([sy_result])
    sy_grads_simple = cgt.simplify(sy_grads)

    print "Gradient:"
    cgt.print_tree(sy_grads)

    print "Gradient after simplification:"
    cgt.print_tree(sy_grads_simple)

    out_true = f(*nu_inputs)
    out_cgt = f_cgt(*nu_inputs)

    grads_true = gradients_affine(f_cgt, nu_inputs)
    grads_cgt = gradf_cgt(*nu_inputs)

    np.testing.assert_allclose(out_cgt, out_true)

    for (g_cgt, g_true) in zip(grads_cgt, grads_true):
        np.testing.assert_allclose(g_cgt, g_true)

    result_count = cgt.count_nodes(sy_result_simple)
    grad_count = cgt.count_nodes(sy_grads_simple)
    print "Result before: %i. after: %i"%(cgt.count_nodes([sy_result]), result_count)
    print "Grad before: %i. after: %i"%(cgt.count_nodes(sy_grads), grad_count)

    PROB2RESULT[f.__name__] = {}    
    PROB2RESULT[f.__name__]["fn"] = result_count
    PROB2RESULT[f.__name__]["grad"] = grad_count

    # add_log_entry(f,nu_inputs, sy_inputs, Var[sy_result],sy_grads)

def type_dispatch(name):
    def newfn(*args):
        if isinstance(args[0], cgt.Node):
            return cgt.__dict__[name](*args)
        else:
            return np.__dict__[name](*args)
    return newfn


sum = type_dispatch("sum")
dot = type_dispatch("dot")
max = type_dispatch("max")
repeat = type_dispatch("repeat")

def xplusx(x):
    return sum(x+x)

def _2x_plus_3x(x):
    v = 2*x + 3*x
    return sum(2*x + 3*x)

def pyramid(x,y,z):
    return sum( (x+y) + (y+z) )

def elem_mult1(x,y,z):
    return sum( (x * y ) )

def elem_mult2(x,y,z):
    return sum( (x * y ) * z)

def matmat00(X,Y):
    return sum(X.dot(Y))

def matmat01(X,Y):
    return sum(X.dot(Y.T))

def matmat10(X,Y):
    return sum(X.T.dot(Y))

def matmat11(X,Y):
    return sum(X.T.dot(Y.T))



def matvec(X,y):
    return sum(X.dot(y))

def vecvec(x,y):
    return dot(x,y)

def matmatplusvec(X,Y,z1):
    return sum(broadcast("+",X.dot(Y),z1,"xx,1x"))

def bcadd(X, y):
    return sum(broadcast("+",X,y,"xx,1x"))

def slisum1(X):
    return sum(X[:,0:1])

def slisum2(X):
    return sum(X[0:1,:])

def slisum3(X):
    return sum(X[0,:])

def slisum4(X):
    return sum(X[:,0])

def max0(X):
    return max(X*1e4)

def max1(X):
    return sum(max(X*1e4,0))

def max2(X):
    return sum(max(X*1e4,1))

def fancysli0(X):
    return sum(X[np.array([1,0]),np.array([0,1])])
    
def xm1(x):
    return sum(x-1)

def onemx(x):
    return sum(1-x)

def sum01(x):
    return sum(sum(x,0),0)

def sum10(x):
    return sum(sum(x,1),0)

def transpose(x,y):
    return sum(x.T*y)

def repeat0(x,y):
    return (repeat(x,7,0)*y).sum()

def repeat1(x,y):
    return (repeat(x,7,1)*y).sum()

################################################################
### Tests 
################################################################

class AffineTestCase(unittest.TestCase):
    def setUp(self):
        cgt.set_precision('double')
        nr.seed(303)
    def test_affine(self):

        sA = np.array(nr.rand())
        sB = np.array(nr.rand())
        sC = np.array(nr.rand())
        mA = nr.randn(2,3)
        mB = nr.randn(2,3)
        mC = nr.randn(2,3)

        for fn in [xplusx, _2x_plus_3x, xm1, onemx]:
            for arg in [sA, mA]:
                check_affine(fn, arg)

        check_affine(elem_mult2, mA, mB, mC)
        check_affine(elem_mult2, sA, sB, sC)
        check_affine(pyramid, sA, sB, sC)
        check_affine(pyramid, mA, mB, mC)
        check_affine(slisum1, mA)
        check_affine(slisum2, mA)
        check_affine(slisum3, mA)
        check_affine(slisum4, mA)
        check_affine(max0, mA)
        check_affine(max1, mA)
        check_affine(max2, mA)
        check_affine(fancysli0, mA)
        check_affine(sum10, mA)
        check_affine(sum01, mA)
        check_affine(repeat0, mA[0:1, :], nr.randn(7,3))
        check_affine(repeat1, mA[:, 0:1], nr.randn(2,7))

        M23 = mA
        M35 = nr.randn(3,5)
        v3 = nr.randn(3)
        v13 = v3.reshape(1,3) #XXX
        v5 = nr.randn(5)
        v15 = v5.reshape(1,5) #XXX
        v3b = nr.randn(3)

        check_affine(matmat00, M23, M35)
        check_affine(matmat01, M23, M35.T)
        check_affine(matmat10, M23.T, M35)
        check_affine(matmat11, M23.T, M35.T)
        check_affine(matvec, M23, v3)
        check_affine(vecvec, v3, v3b)
        check_affine(bcadd, M23, v13)
        check_affine(matmatplusvec, M23, M35, v15) # Wrong because we have no broadcasting
        check_affine(transpose, M23, nr.randn(3,2))

        # TODO: examples with constants
        # TODO: examples that mix scalar and matrix types

        from tabulate import tabulate
        print tabulate([[key,val["fn"],val["grad"]] for (key,val) in sorted(PROB2RESULT.items())],headers=["funcname","fncount","gradcount"])


if __name__ == "__main__":
    unittest.main()
