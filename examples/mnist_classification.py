# Initially copied from https://github.com/Newmu/Theano-Tutorials/blob/master/4_modern_net.py
import cgt
from cgt import nn
from cgt.distributions import Categorical
import numpy as np
from sklearn.datasets import fetch_mldata
import time

def init_weights(*shape):
    return cgt.shared(np.random.randn(*shape) * 0.01)

def rmsprop_updates(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = cgt.grad(cost, params)
    updates = []
    for p, g in zip(params, grads):
        acc = cgt.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * cgt.square(g)
        gradient_scaling = cgt.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

def model(X, w_h, w_h2, w_o, p_drop_input, p_drop_hidden):
    X = nn.dropout(X, p_drop_input)
    h = nn.rectify(cgt.dot(X, w_h))

    h = nn.dropout(h, p_drop_hidden)
    h2 = nn.rectify(cgt.dot(h, w_h2))

    h2 = nn.dropout(h2, p_drop_hidden)
    py_x = nn.softmax(cgt.dot(h2, w_o))
    return h, h2, py_x

def train_test_val_slices(n, trainfrac, testfrac, valfrac):
    assert trainfrac+testfrac+valfrac==1.0
    ntrain = int(np.round(n*trainfrac))
    ntest = int(np.round(n*testfrac))
    nval = n - ntrain - ntest
    return slice(0,ntrain), slice(ntrain,ntrain+ntest), slice(ntrain+ntest,ntrain+ntest+nval)

# helper methods to print nice table
def fmt_item(x, l):
    if isinstance(x, np.ndarray):
        assert x.ndim==0
        x = x.item()
    if isinstance(x, (str,int)): rep = str(x)
    elif isinstance(x, float): rep = "%g"%x
    return " "*(l - len(rep)) + rep
def fmt_row(*row):
    return " | ".join(fmt_item(x, 10) for x in row)

def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--epochs",type=int,default=10)
    parser.add_argument("--profile",action="store_true")
    args = parser.parse_args()

    print "fetching..."
    mnist = fetch_mldata('MNIST original')
    print "done"

    Xdata = mnist["data"]/255.
    ydata = mnist["target"].astype('i4')

    # Xdata = Xdata[:100]
    # ydata = ydata[:100]

    randperm = np.random.permutation(Xdata.shape[0])
    trainsli,testsli,valsli = train_test_val_slices(Xdata.shape[0],.8,.1,.1)
    traininds,testinds,valinds = randperm[trainsli],randperm[testsli],randperm[valsli]
    Xtrain = Xdata[traininds]
    ytrain = ydata[traininds]
    Xtest = Xdata[testinds]
    ytest = ydata[testinds]
    Xval = Xdata[valinds]
    yval = ydata[valinds]

    w_h = init_weights(784, 625)
    w_h2 = init_weights(625, 625)
    w_o = init_weights(625, 10)

    X = cgt.matrix("X")
    y = cgt.vector("y",dtype='i4')

    _noise_h, _noise_h2, noise_py_x = model(X, w_h, w_h2, w_o, 0, 0.5)
    _h, _h2, py_x = model(X, w_h, w_h2, w_o, 0., 0.)
    y_x = cgt.argmax(py_x, axis=1)

    cost = -cgt.mean(Categorical.loglik(y, noise_py_x))
    params = [w_h, w_h2, w_o]
    updates = rmsprop_updates(cost, params, lr=0.001)

    err = cgt.cast(cgt.not_equal(y_x, y), cgt.floatX).mean()

    train = cgt.function(inputs=[X, y], outputs=[], updates=updates)
    computeloss = cgt.function(inputs=[X, y], outputs=[err,cost])

    batch_size=64

    if args.profile: cgt.execution.profiler.start()



    print fmt_row("Epoch","Train NLL","Train Err","Test NLL","Test Err","Epoch Time")
    for i_epoch in xrange(args.epochs):
        tstart = time.time()
        for start in xrange(0, Xtrain.shape[0], batch_size):
            end = start+batch_size
            train(Xtrain[start:end], ytrain[start:end])
        trainerr, trainloss = computeloss(Xtrain, ytrain)
        testerr, testloss = computeloss(Xtest, ytest)
        print fmt_row(i_epoch, trainloss, trainerr, testloss, testerr, time.time()-tstart)
    if args.profile: cgt.execution.profiler.print_stats()

if __name__ == "__main__":
    main()