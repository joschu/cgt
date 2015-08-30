import cgt, numpy as np, numpy.random as nr, itertools as it
from cgt import core, utils
from cgt.numeric_diff import numeric_grad
from cgt.tests import across_configs

DISPLAY=False

@across_configs
def test_scalars():
    np.random.seed(0)
    x = cgt.scalar('x')
    y = cgt.scalar('y')
    z = cgt.scalar('z')
    vars = [x,y,z] #pylint: disable=W0622
    vals = nr.rand(len(vars))+1

    PROB2RESULT = {}

    for ((key,_), cls) in it.chain(
            it.izip(core.UNARY_INFO.items(),it.repeat(core.ElwiseUnary)),
            it.izip(core.BINARY_INFO.items(),it.repeat(core.ElwiseBinary))
            ):
        if key == "conj":
            print "skipping conj"
            continue
        utils.colorprint(utils.Color.YELLOW, "Testing %s\n"%key)
        if cls == core.ElwiseUnary:
            n_in = 1
            op = cls(key)
        else:
            n_in = 2
            op = cls(key, (True,True))
        inputvars = vars[0:n_in]
        inputvals = vals[0:n_in]
        out = core.Result(op, inputvars)
        f = cgt.function(inputvars, out)
        try:
            grads = cgt.grad(out, inputvars)
        except core.NonDifferentiable:
            print "nondiff"
            continue
        if DISPLAY:
            print "Function:"
            cgt.print_tree(out)
            print "Gradient original:"
            cgt.print_tree(grads)
            print "Gradient simplified:"
        grads_simple = core.simplify(grads)
        if DISPLAY: cgt.print_tree(grads_simple)
        gradf = cgt.function(inputvars, grads)
        eps = {"single":1e-4,"double":1e-9}[cgt.get_precision()]
        nugrad = numeric_grad(lambda li: f(*li), inputvals,eps=eps) #pylint: disable=W0640
        cgtgrad = gradf(*inputvals)
        np.testing.assert_almost_equal(nugrad,cgtgrad,decimal={"single":3,"double":6}[cgt.get_precision()])

        grad_count = core.count_nodes(grads_simple)
        PROB2RESULT[key] = {}
        PROB2RESULT[key]["grad"] = grad_count

    if DISPLAY:
        from thirdparty.tabulate import tabulate
        print tabulate([[key,val["grad"]] for (key,val) in PROB2RESULT.iteritems()],headers=["funcname","gradcount"])    

if __name__ == "__main__":
    import nose
    nose.runmodule()
