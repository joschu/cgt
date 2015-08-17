# Based on tutorial by Alec Radford
# https://github.com/Newmu/Theano-Tutorials/blob/master/4_modern_net.py

import cgt
from cgt import nn
from cgt.distributions import categorical
import numpy as np
from example_utils import fmt_row, fetch_dataset
import time, sys

def init_weights(*shape):
    return cgt.shared(np.random.randn(*shape) * 0.01, fixed_shape_mask='all')

def rmsprop_updates(cost, params, stepsize=0.001, rho=0.9, epsilon=1e-6):
    grads = cgt.grad(cost, params)
    updates = []
    for p, g in zip(params, grads):
        acc = cgt.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * cgt.square(g)
        gradient_scaling = cgt.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - stepsize * g))
    return updates

def dense_model(X, w_h, w_h2, w_o, p_drop_input, p_drop_hidden):
    X = nn.dropout(X, p_drop_input)
    h = nn.rectify(cgt.dot(X, w_h))

    h = nn.dropout(h, p_drop_hidden)
    h2 = nn.rectify(cgt.dot(h, w_h2))

    h2 = nn.dropout(h2, p_drop_hidden)
    py_x = nn.softmax(cgt.dot(h2, w_o))
    return py_x

def convnet_model(X, w, w2, w3, w4, w_o, p_drop_conv, p_drop_hidden):
    l1a = nn.rectify(nn.conv2d(X, w, kernelshape=(3,3), pad=(1,1)))
    l1 = nn.max_pool_2d(l1a, kernelshape=(2, 2), stride=(2,2))
    l1 = nn.dropout(l1, p_drop_conv)

    l2a = nn.rectify(nn.conv2d(l1, w2, kernelshape=(3,3), pad=(1,1)))
    l2 = nn.max_pool_2d(l2a, kernelshape=(2, 2), stride=(2,2))
    l2 = nn.dropout(l2, p_drop_conv)

    l3a = nn.rectify(nn.conv2d(l2, w3, kernelshape=(3,3), pad=(1,1)))
    l3b = nn.max_pool_2d(l3a, kernelshape=(2, 2), stride=(2,2))
    batchsize,channels,rows,cols = l3b.shape
    l3 = cgt.reshape(l3b, [batchsize, channels*rows*cols])
    l3 = nn.dropout(l3, p_drop_conv)

    l4 = nn.rectify(cgt.dot(l3, w4))
    l4 = nn.dropout(l4, p_drop_hidden)
    
    pyx = nn.softmax(cgt.dot(l4, w_o))
    return pyx

def tinyconv_model(X, w, w2, p_drop):
    l1 = nn.conv2d(X, w, kernelshape=(3,3), pad=(1,1),stride=(3,3))
    l1a = nn.dropout(l1, p_drop)
    batchsize,channels,rows,cols = l1.shape
    l1flat = cgt.reshape(l1, [batchsize,channels*rows*cols])
    pyx = nn.softmax(l1flat.dot(w2))
    return l1, pyx



def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--epochs",type=int,default=10)
    parser.add_argument("--profile",action="store_true")
    parser.add_argument("--dropout",action="store_true")
    parser.add_argument("--stepsize",type=float, default=.001)
    parser.add_argument("--model",choices=["dense","conv"],default="dense")
    parser.add_argument("--test",action="store_true")
    parser.add_argument("--grad_check",action="store_true")
    args = parser.parse_args()

    if args.grad_check: cgt.set_precision("quad")

    # from mldata.org http://mldata.org/repository/data/viewslug/mnist-original/
    # converted to npz
    mnist = fetch_dataset("http://rll.berkeley.edu/cgt-data/mnist.npz")

    Xdata = (mnist["X"]/255.).astype(cgt.floatX)
    ydata = mnist["y"]

    np.random.seed(0)


    if args.model=="conv":
        Xdata = Xdata.reshape(-1, 1, 28, 28)

    Xtrain = Xdata[0:60000]
    ytrain = ydata[0:60000]

    Xtest = Xdata[60000:70000]
    ytest = ydata[60000:70000]

    sortinds = np.random.permutation(60000)
    Xtrain = Xtrain[sortinds]
    ytrain = ytrain[sortinds]

    X = cgt.tensor4("X",fixed_shape=(None,1,28,28)) if args.model=="conv" else cgt.matrix("X", fixed_shape=(None,28*28))
    y = cgt.vector("y",dtype='i8')

    if args.model == "dense":
        p_drop_input,p_drop_hidden = (0.2, 0.5) if args.dropout else (0,0)    
        w_h = init_weights(784, 256)
        w_h2 = init_weights(256, 256)
        w_o = init_weights(256, 10)
        pofy_drop = dense_model(X, w_h, w_h2, w_o, p_drop_input, p_drop_hidden)
        pofy_nodrop = dense_model(X, w_h, w_h2, w_o, 0., 0.)
        params = [w_h, w_h2, w_o]        
    elif args.model == "conv":
        p_drop_conv,p_drop_hidden = (0.2, 0.5) if args.dropout else (0,0)            
        w = init_weights(32, 1, 3, 3)
        w2 = init_weights(64, 32, 3, 3)
        w3 = init_weights(128, 64, 3, 3)
        w4 = init_weights(128 * 2 * 2, 625)
        w_o = init_weights(625, 10)
        pofy_drop = convnet_model(X, w, w2, w3, w4, w_o, p_drop_conv, p_drop_hidden)
        pofy_nodrop = convnet_model(X, w, w2, w3, w4, w_o, 0., 0.)
        params = [w, w2, w3, w4, w_o]
    else:
        raise RuntimeError("Unreachable")

    cost_drop = -cgt.mean(categorical.loglik(y, pofy_drop))
    updates = rmsprop_updates(cost_drop, params, stepsize=args.stepsize)

    y_nodrop = cgt.argmax(pofy_nodrop, axis=1)
    cost_nodrop = -cgt.mean(categorical.loglik(y, pofy_nodrop))
    err_nodrop = cgt.cast(cgt.not_equal(y_nodrop, y), cgt.floatX).mean()

    train = cgt.function(inputs=[X, y], outputs=[], updates=updates)
    computeloss = cgt.function(inputs=[X, y], outputs=[err_nodrop,cost_nodrop])

    batch_size=128


    from cgt.tests import gradcheck_model
    if args.grad_check:
        cost_nodrop = cgt.core.clone(cost_nodrop, {X:Xtrain[:1],y:ytrain[:1]})
        print "doing gradient check..."
        print "------------------------------------"
        gradcheck_model(cost_nodrop, params[0:1])
        print "success!"
        return

    if args.profile: cgt.execution.profiler.start()

    print fmt_row(10, ["Epoch","Train NLL","Train Err","Test NLL","Test Err","Epoch Time"])
    for i_epoch in xrange(args.epochs):
        tstart = time.time()
        for start in xrange(0, Xtrain.shape[0], batch_size):
            end = start+batch_size
            train(Xtrain[start:end], ytrain[start:end])
            if args.test: return
        elapsed = time.time() - tstart
        trainerr, trainloss = computeloss(Xtrain[:len(Xtest)], ytrain[:len(Xtest)])
        testerr, testloss = computeloss(Xtest, ytest)
        print fmt_row(10, [i_epoch, trainloss, trainerr, testloss, testerr, elapsed])
    if args.profile: cgt.execution.profiler.print_stats()

if __name__ == "__main__":
    main()