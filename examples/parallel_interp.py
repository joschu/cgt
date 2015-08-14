import cgt
from cgt.core import load_config
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

    m = 10
    d = 10000

    # build graph

    X = cgt.matrix("X")
    y = cgt.vector("y")
    ws = list()
    for k in xrange(m):
        ws.append(cgt.vector("w_%d" % k))
    b = cgt.scalar("b")
    ypred = b
    for k in xrange(m):
        ypred = ypred + X.dot(ws[k])
    loss = cgt.sum(cgt.square(ypred - y))

    inputs = [X, y, b]
    for k in xrange(m):
        inputs.append(ws[k])
    outputs = [loss]

    # construct parallel/sequential interpreter

    load_config()["parallel_interp"] = not args.seq
    f = cgt.execution.function(inputs, outputs)

    # test things out!

    seed(0)

    X_val = randn(d, d)
    y_val = randn(d)
    b_val = 5.0
    vals = [X_val, y_val, b_val]
    for k in xrange(m):
        vals.append(randn(d))

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
