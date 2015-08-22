import cgt
from cgt.core import get_config,Op
from time import time
import numpy as np
from numpy.random import randn, seed

# NOTE observe differences clearly if add a time.sleep to Mul21

class SleepByRef(Op):
    call_type="byref"
    available_impls=("native_cpu",)
    def get_native_compile_info(self, _, __):
        code=r"""
            CGT_EXPORT_C cgtArray* $function(void* cldata, cgtArray** reads) {
                printf("going to sleep...\n");
                usleep(1000000);
                printf("top of the morning to ya!\n");
                return reads[0];
            }"""
        return cgt.core.NativeCompileInfo(code,includes=["unistd.h"])
    def typ_apply(self, input_types):
        return input_types[0]
    def shp_apply(self, inputs):
        return cgt.shape(inputs[0])
    # params

def noop_byref(x):
    return cgt.core.Result(SleepByRef(), [x])

def sleeps():
    with cgt.scoped_update_config(enable_simplification=False,enable_inplace_opt=False):
        x = cgt.scalar('x')
        y1 = noop_byref(x)
        y2 = noop_byref(x)
        z=y1+y2
        with cgt.scoped_update_config(parallel_interp = True):
            f = cgt.function([x],z)

        f(0)



def matmuls(seq):
    m = 8
    d = 1000

    # build graph

    X = cgt.matrix("X")
    Y = cgt.matrix("Y")
    loss=0
    for k in xrange(m):
        # loss = loss+cgt.sin(X*Y+k).sum()
        loss = loss+(X.dot(Y+k)).sum()

    with cgt.scoped_update_config(parallel_interp = not seq):
        f = cgt.function([X,Y], loss)

    # test things out!

    seed(0)

    X_val = randn(d, d)
    Y_val = randn(d, d)
    vals = [X_val, Y_val]

    times = list()
    for k in xrange(1):
        tic = time()
        out = f(*vals)
        toc = time()
        times.append(toc - tic)
        print(out)

    print("median time: %f" % np.median(times))
    

def update():
    xval = randn(1,1)
    x = cgt.shared(xval)
    f = cgt.function([], x.sum(), updates=[(x,x+1)])
    before = x.get_value()
    print before
    f()
    print "called"
    after = x.get_value()
    print after
    print "done"

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", action="store_true",
            help="run sequential instead of parallel for profiling")
    args = parser.parse_args()
    # sleeps()
    # matmuls(args.seq)
    update()



if __name__ == "__main__":
    main()
