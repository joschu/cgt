import cgt
from cgt.core import get_config
from time import time
import numpy as np
from numpy.random import randn, seed

# NOTE observe differences clearly if add a time.sleep to Mul21

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", action="store_true",
            help="run sequential instead of parallel for profiling")
    args = parser.parse_args()

    # params

    m = 8
    d = 1000

    # build graph

    X = cgt.matrix("X")
    Y = cgt.matrix("Y")
    Ws = list()
    for k in xrange(m):
        Ws.append(cgt.matrix("W_%d" % k))
    b = cgt.scalar("b")
    Ypred = b
    for k in xrange(m):
        Ypred = Ypred + X.dot(Ws[k])
    loss = cgt.sum(cgt.square(Ypred - Y))

    inputs = [X, Y, b]
    for k in xrange(m):
        inputs.append(Ws[k])
    outputs = [loss]

    # construct parallel/sequential interpreter

    get_config()["parallel_interp"] = not args.seq
    f = cgt.execution.function(inputs, outputs)

    # test things out!

    seed(0)

    X_val = randn(d, d)
    Y_val = randn(d, d)
    b_val = 5.0
    vals = [X_val, Y_val, b_val]
    for k in xrange(m):
        vals.append(randn(d, d))

    times = list()
    for k in xrange(10):
        tic = time()
        out = f(*vals)
        toc = time()
        times.append(toc - tic)
        print(out)

    print("median time: %f" % np.median(times))


if __name__ == "__main__":
    main()
