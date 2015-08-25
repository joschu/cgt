import cgt
from gru import GRUCell
import time
from cgt.utils import Message
import numpy as np

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon",type=int)
    args = parser.parse_args()
    horizon = args.horizon
    assert horizon is not None    
    size=128
    batchsize=64
    cell = GRUCell([size],size)
    X = cgt.tensor3()
    init = cgt.matrix()

    prev_h = init
    for i in xrange(horizon):
        prev_h = cell(X[i], prev_h)
    loss = prev_h.sum()

    with Message("compiling"):
        f = cgt.function([X, init],cgt.grad(loss, cell.params()))
    with Message("running"):
        xval = np.zeros((horizon,batchsize,size),cgt.floatX)
        initval = np.zeros((batchsize, size), cgt.floatX)
        for i in xrange(100): 
            f(xval, initval)


# # No speedup -- why?
# with Message("split loss. compiling"):
#     from cgt import nn
#     m = cgt.nn.Module([X, init], [loss])
#     split_loss = 0
#     X1 = cgt.tensor3()
#     init1 = cgt.matrix()
#     for start in xrange(0, batchsize, batchsize//4):
#         sli = slice(start, start+batchsize//4)
#         split_loss += m([X1[:, sli], init1[sli]])[0]
#     f = cgt.function([X1, init1],cgt.grad(split_loss, cell.params()))
# with Message("running"):
#     for i in xrange(100): 
#         f(xval,initval)
