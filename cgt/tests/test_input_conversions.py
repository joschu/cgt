from cgt.tests import across_configs
import cgt, numpy as np

@across_configs
def test_noncontiguous_matrix():

    x = np.arange(1,7).reshape(2,3).astype(cgt.floatX)
    result = np.log(x.sum(axis=0)).sum()


    xvar = cgt.matrix()
    f = cgt.function([xvar],cgt.log(xvar.sum(axis=0)).sum())


    assert np.allclose( f(np.asarray(x, order='C')), result)
    assert np.allclose( f(np.asarray(x, order='C', dtype='int64')), result)
    assert np.allclose( f(np.asarray(x, order='F')), result)

    X = np.zeros((4,6))
    X[::2,::2] = x
    assert np.allclose( f(X[::2,::2]), result)

@across_configs
def test_scalar_input():
    x = cgt.scalar()
    f = cgt.function([x], x**2)
    xval = 2
    yval = 4
    assert np.allclose(f(2), 4)
    assert np.allclose(f(2.0), 4)    
    assert np.allclose(f(np.array(2)), 4)      
    assert np.allclose(f(np.array(2.0)), 4)    
    assert np.allclose(f(np.array([2])[0]), 4)        
    assert np.allclose(f(np.array([2.0])[0]), 4)        
