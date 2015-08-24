import cgt, numpy as np

def test_array_wrapper():
    for backend in ("python","native"):
        for precision in ("single","double"):
            yield runtest, backend, precision

def runtest(backend, precision):
    with cgt.scoped_update_config(backend='native',precision=precision):
        xval = np.zeros(10)
        x = cgt.shared(xval)
        f = cgt.function([],[],updates=[(x,x+1)])
        f()
        g = cgt.function([],x.sum())
        assert np.allclose(x.op.get_value(), xval+1)
        xval2 = np.arange(10)
        x.op.set_value(xval2)
        print x.op.get_value()
        assert np.allclose(x.op.get_value(), xval2)
        assert g() == xval2.sum()
        f()
        assert np.allclose(x.op.get_value(), xval2+1)
        assert g() == (xval2+1).sum()

if __name__ == "__main__":
    test_array_wrapper()