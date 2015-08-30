import cgt
from cgt.core import Op
from cgt.tests import across_configs
import time
import numpy as np
from numpy.random import randn, seed

# NOTE observe differences clearly if add a time.sleep to Mul21


class SleepFor(Op):
    return_type="byval"
    available_impls=("native_cpu",)
    def get_native_compile_info(self, _, __):
        code=r"""
            CGT_EXPORT_C cgtArray* $function(void* cldata, cgtArray** reads) {
                float t = reads[1]->at<float>(0);
                usleep(t * 1000000);
                return reads[0];
            }"""
        return cgt.core.NativeCompileInfo(code,includes=["unistd.h"])
    def typ_apply(self, input_types):
        assert input_types[1].dtype == cgt.floatX
        return input_types[0]
    def shp_apply(self, inputs):
        return cgt.shape(inputs[0])


def sleepfor(x, t):
    return cgt.core.Result(SleepFor(), [x, t])

@across_configs(backends=("native",))
def test_sleeps():
    with cgt.scoped_update_config(parallel=True):
        x = cgt.scalar('x')
        y1 = sleepfor(x, .1)
        y2 = sleepfor(x, .1)

        z=y1+y2
        fpar = cgt.function([x],z)
        
        tstart = time.time()
        fpar(0)
        elapsed = time.time() - tstart
        assert elapsed < .11


@across_configs(backends=("native",))
def test_matmuls():
    with cgt.scoped_update_config(parallel=True):

        m = 8
        d = 1000

        # build graph

        X = cgt.matrix("X")
        Y = cgt.matrix("Y")
        loss=0
        for k in xrange(m):
            # loss = loss+cgt.sin(X*Y+k).sum()
            loss = loss+(X.dot(Y+k)).sum()

            f = cgt.function([X,Y], loss)

        # test things out!

        seed(0)

        X_val = randn(d, d)
        Y_val = randn(d, d)
        vals = [X_val, Y_val]

        tic = time.time()
        out = f(*vals)
        toc = time.time()

        print toc-tic
    

@across_configs(backends=("native",))
def test_update():
    with cgt.scoped_update_config(parallel=True):
        xval = np.array(1.5)
        x = cgt.shared(xval)
        f = cgt.function([], x.sum(), updates=[(x,x+1)])
        before = x.op.get_value().copy()
        f()
        after = x.op.get_value()
        assert np.allclose(after , before+1)

