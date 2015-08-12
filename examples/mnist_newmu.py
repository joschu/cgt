# Based on tutorial by Alec Radford
# https://github.com/Newmu/Theano-Tutorials/blob/master/4_modern_net.py

import cgt
from cgt import nn
from cgt.distributions import categorical
import numpy as np
from example_utils import fmt_row, fetch_dataset, train_val_test_slices
import time

def init_weights(*shape):
    return cgt.shared(np.random.randn(*shape) * 0.01)

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

def model(X, w_h, w_h2, w_o, p_drop_input, p_drop_hidden):
    X = nn.dropout(X, p_drop_input)
    h = cgt.tanh(cgt.dot(X, w_h))

    h = nn.dropout(h, p_drop_hidden)
    h2 = cgt.tanh(cgt.dot(h, w_h2))

    h2 = nn.dropout(h2, p_drop_hidden)
    py_x = nn.softmax(cgt.dot(h2, w_o))
    return h, h2, py_x

def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--epochs",type=int,default=10)
    parser.add_argument("--profile",action="store_true")
    parser.add_argument("--dropout",action="store_true")
    parser.add_argument("--stepsize",type=float, default=.001)
    args = parser.parse_args()

    # from mldata.org http://mldata.org/repository/data/viewslug/mnist-original/
    # converted to npz
    mnist = fetch_dataset("http://rll.berkeley.edu/cgt-data/mnist.npz")

    Xdata = mnist["X"]/255.
    ydata = mnist["y"]



    Xtrain = Xdata[0:60000]
    ytrain = ydata[0:60000]

    Xtest = Xdata[60000:70000]
    ytest = ydata[60000:70000]

    sortinds = np.random.permutation(60000)
    Xtrain = Xtrain[sortinds]
    ytrain = ytrain[sortinds]

    np.random.seed(0)

    w_h = init_weights(784, 625)
    w_h2 = init_weights(625, 625)
    w_o = init_weights(625, 10)

    X = cgt.matrix("X")
    y = cgt.vector("y",dtype='i8')

    p_drop_input,p_drop_hidden = (0.2, 0.5) if args.dropout else (0,0)

    _h_drop, _h2_drop, pofy_drop = model(X, w_h, w_h2, w_o, p_drop_input, p_drop_hidden)
    _h_nodrop, _h2_nodrop, pofy_nodrop = model(X, w_h, w_h2, w_o, 0., 0.)

    params = [w_h, w_h2, w_o]

    cost_drop = -cgt.mean(categorical.loglik(y, pofy_drop))
    updates = rmsprop_updates(cost_drop, params, stepsize=args.stepsize)

    y_nodrop = cgt.argmax(pofy_drop, axis=1)
    cost_nodrop = -cgt.mean(categorical.loglik(y, pofy_nodrop))
    err_nodrop = cgt.cast(cgt.not_equal(y_nodrop, y), cgt.floatX).mean()

    train = cgt.function(inputs=[X, y], outputs=[], updates=updates)
    computeloss = cgt.function(inputs=[X, y], outputs=[err_nodrop,cost_nodrop])

    batch_size=128

    if args.profile: cgt.execution.profiler.start()

    print fmt_row(10, ["Epoch","Train NLL","Train Err","Test NLL","Test Err","Epoch Time"])
    for i_epoch in xrange(args.epochs):
        tstart = time.time()
        for start in xrange(0, Xtrain.shape[0], batch_size):
            end = start+batch_size
            train(Xtrain[start:end], ytrain[start:end])
        elapsed = time.time() - tstart
        trainerr, trainloss = computeloss(Xtrain, ytrain)
        testerr, testloss = computeloss(Xtest, ytest)
        print fmt_row(10, [i_epoch, trainloss, trainerr, testloss, testerr, elapsed])
    if args.profile: cgt.execution.profiler.print_stats()

if __name__ == "__main__":
    main()