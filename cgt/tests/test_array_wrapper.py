import cgt, numpy as np
from cgt.tests import across_configs


@across_configs
def test_array_wrapper():
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
    import nose
    nose.runmodule()