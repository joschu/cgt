import cgt
from cgt import core
import numpy as np
import unittest
import numpy.random as nr


DISPLAY = False


PROB2RESULT = {}

def gradients_affine(f, xs, h=1e-1):
    grads = [np.zeros_like(x) for x in xs]
    xs = map(np.copy, xs)
    yorig = f(*xs)
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
    return cgt.broadcast(opname, x, y, bcpat) if isinstance(x, core.Node) else eval("x %s y"%opname)

def check_affine(f, *nu_inputs):
    types = ",".join(["{%s,%s}"%(x.dtype, x.ndim) for x in nu_inputs])
    cgt.utils.colorprint(cgt.utils.Color.YELLOW,"Testing %s(%s)\n"%(f.__name__, types))
    sy_inputs = map(tensor_like, nu_inputs)
    for (i,sy) in enumerate(sy_inputs):
        sy.name = "x%i"%i

    sy_result = f(*sy_inputs)

    def maybeprint(msg):
        if DISPLAY: print msg

    maybeprint("Function:")
    if DISPLAY: cgt.print_tree([sy_result])

    f_cgt = cgt.function(sy_inputs, sy_result)
    sy_grads = cgt.grad(sy_result, sy_inputs)
    gradf_cgt = cgt.function(sy_inputs, sy_grads)

    sy_result_simple = core.simplify([sy_result])
    sy_grads_simple = core.simplify(sy_grads)

    maybeprint("Gradient:")
    if DISPLAY: cgt.print_tree(sy_grads)

    maybeprint("Gradient after simplification:")
    if DISPLAY: cgt.print_tree(sy_grads_simple)

    out_true = f(*nu_inputs)
    out_cgt = f_cgt(*nu_inputs)

    grads_true = gradients_affine(f_cgt, nu_inputs, h=1e-4 if "max" in f.__name__ else 1e-1)
    grads_cgt = gradf_cgt(*nu_inputs)

    np.testing.assert_allclose(out_cgt, out_true, rtol=1e-5)

    for (g_cgt, g_true) in zip(grads_cgt, grads_true):
        np.testing.assert_allclose(g_cgt, g_true,rtol=1e-5)

    result_count = core.count_nodes(sy_result_simple)
    grad_count = core.count_nodes(sy_grads_simple)
    maybeprint("Result before: %i. after: %i"%(core.count_nodes([sy_result]), result_count))
    maybeprint("Grad before: %i. after: %i"%(core.count_nodes(sy_grads), grad_count))

    PROB2RESULT[f.__name__] = {}    
    PROB2RESULT[f.__name__]["fn"] = result_count
    PROB2RESULT[f.__name__]["grad"] = grad_count


def type_dispatch(name, argind=0):
    def newfn(*args):
        if isinstance(args[argind], core.Node):
            return cgt.__dict__[name](*args)
        else:
            return np.__dict__[name](*args)
    return newfn


sum = type_dispatch("sum")
dot = type_dispatch("dot")
max = type_dispatch("max")
einsum = type_dispatch("einsum",1)
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


def transpose021(x):
    return sum(x.transpose([0,2,1]))

def transpose012(x):
    return sum(x.transpose([0,1,2]))

def transpose102(x):
    return sum(x.transpose([1,0,2]))

def batchedmatmul(x,y):
    return sum(einsum("nij,njk->nik", x, y))

def rfft(x):
    if isinstance(x, np.ndarray):
        return np.real(np.fft.rfft2(x,(10,10),[0,1])).sum()
    else:
        return cgt.real(cgt.rfft(x, (10,10), [0,1])).sum()

################################################################
### Tests 
################################################################

class AffineTestCase(unittest.TestCase):
    def setUp(self):
        cgt.set_precision('double')
        nr.seed(303)
    def runTest(self):

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
        check_affine(matmatplusvec, M23, M35, v15)
        check_affine(transpose, M23, nr.randn(3,2))


        T235 = nr.randn(2,3,5)
        T257 = nr.randn(2,5,7)
        check_affine(transpose012, T235)
        check_affine(transpose021, T235)
        check_affine(transpose102, T235)
        check_affine(batchedmatmul, T235, T257)

        # check_affine(rfft, M35)


        # TODO: examples with constants
        # TODO: examples that mix scalar and matrix types

        if DISPLAY:
            from thirdparty.tabulate import tabulate
            print tabulate([[key,val["fn"],val["grad"]] for (key,val) in sorted(PROB2RESULT.items())],headers=["funcname","fncount","gradcount"])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbose",action="store_true")
    args = parser.parse_args()
    if args.verbose: DISPLAY = True
    case = AffineTestCase()
    case.setUp()
    case.runTest()
