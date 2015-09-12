import cgt
import numpy as np
from cgt.tests import across_configs

@across_configs(backends=("python","native"), precisions=("single","double"))
def test_incsubtensor0():
    # First let's test fancy slice along zeroth dimension

    W = cgt.shared(np.zeros((5,3)), name="W")
    inc = cgt.matrix() # we'll increment W by this matrix
    incval = np.arange(9).reshape(3,3)
    

    inds = cgt.vector(dtype='i8')
    updates = {W : cgt.inc_subtensor(W, inds, inc)}
    f = cgt.function([inds,inc],[],updates=updates)
    f([1,2,4],incval)

    assert np.allclose(W.op.get_value(), 
        np.array(
        [[ 0.,  0.,  0.],
         [ 0.,  1.,  2.],
         [ 3.,  4.,  5.],
         [ 0.,  0.,  0.],
         [ 6.,  7.,  8.]]))


    # Now let's test non-fancy slice along zeroth dimension

@across_configs(backends=("python","native"), precisions=("single","double"))
def test_incsubtensor1():
    W = cgt.shared(np.zeros((5,3)), name="W")
    inc = cgt.matrix() # we'll increment W by this matrix
    incval = np.arange(9).reshape(3,3)

    start = cgt.scalar(dtype='i8')
    stop = cgt.scalar(dtype='i8')
    updates = {W : cgt.inc_subtensor(W, slice(start, stop), inc)}
    f = cgt.function([start,stop,inc],[],updates=updates)
    f(0,3,incval)
    assert np.allclose(W.op.get_value(), 
        np.array(
        [
         [ 0.,  1.,  2.],
         [ 3.,  4.,  5.],
         [ 6.,  7.,  8.],
         [ 0.,  0.,  0.],
         [ 0.,  0.,  0.],
         ]))

    # Now let's test the last kind of slice, where we have int arrays on each dimension

@across_configs(backends=("python","native"), precisions=("single","double"))
def test_incsubtensor2():
    W = cgt.shared(np.zeros((5,3)), name="W")
    i0 = cgt.vector(dtype='i8')
    i1 = cgt.vector(dtype='i8')
    inc = cgt.vector()

    updates2 = {W : cgt.inc_subtensor(W, (i0,i1), inc)}
    f2 = cgt.function([i0,i1,inc],[],updates=updates2)
    f2([0,1,2,2],[0,1,2,2],[1,2,3,4])
    assert np.allclose(W.op.get_value(), 
        np.array(
        [
         [ 1.,  0.,  0.],
         [ 0.,  2.,  0.],
         [ 0.,  0.,  7.],
         [ 0.,  0.,  0.],
         [ 0.,  0.,  0.],
         ]))



