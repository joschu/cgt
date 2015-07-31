import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata

mnist = fetch_mldata('MNIST original', data_home='~/cgt/data')  # XXX

print(mnist.data.shape)
print(mnist.target.shape)

np.unique(mnist.target)

#plt.imshow(mnist.data[1, :].reshape(28, 28))
#plt.show()

# do some preprocessing

X = mnist.data
y = mnist.target
X = X.astype('float64')
X = X / 255

# train-test split (as [Joachims, 2006])
# TODO can define own validation split...

n_train = 60000
X_train = X[:n_train, :]
X_test = X[n_train:, :]
y_train = y[:n_train]
y_test = y[n_train:]

# construct the network

import nn
import cgt
from opt import sgd_update

N_LAYERS = 2
hid_size = X.shape[1]  # 28 * 28
out_size = 10

inps = [cgt.matrix(dtype=cgt.floatX)]

param_list = []
for k in xrange(N_LAYERS):
    tmp = nn.Affine(hid_size, hid_size)#(inps[k])
    param_list.extend([tmp.weight, tmp.bias])
    inps.append(cgt.tanh(tmp(inps[k])))

tmp = nn.Affine(hid_size, out_size)
param_list.extend([tmp.weight, tmp.bias])
logprobs = nn.logsoftmax(tmp(inps[-1]))

#dnn = nn.Module(inps[0:1], [logprobs])
#params = dnn.get_parameters()
# XXX think should just make this part of get_parameters
theta = nn.setup_contiguous_storage(param_list)
# XXX initialize
theta[:] = np.random.uniform(-0.08, 0.08, theta.shape)

# XXX taken from other demo, move
def ind2onehot(inds, n_cls):
    out = np.zeros(list(inds.shape)+[n_cls,], cgt.floatX)
    for k in xrange(inds.shape[0]):
        out[k, inds[k].astype('int32')] = 1
    #out.flat[np.arange(inds.size)*n_cls + inds.ravel()] = 1
    return out

b_size = 25

def make_loss_and_grad(net):
    X_b = inps[0] #cgt.matrix(dtype=cgt.floatX)
    y_onehot = cgt.matrix(dtype='i4')
    outputs = [logprobs]

    loss = nn.crossent(outputs[0], y_onehot) / b_size
    #gradloss = cgt.grad(loss, params)
    gradloss = cgt.grad(loss, param_list)

    # XXX use flatcat function
    grad = cgt.concatenate([x.flatten() for x in gradloss])
    #grad = gradloss
    return cgt.make_function([X_b, y_onehot], [loss, grad, logprobs])

f_loss_and_grad = make_loss_and_grad(None)

# train loop

# shuffle data

perm = np.random.permutation(np.arange(X_train.shape[0]))
X_train = X_train[perm, :]
y_train = y_train[perm]

class Table(object):
    pass
state = Table()
state.theta = theta
state.step_size = 0.1
exploss = None
for k in xrange(X_train.shape[0] / b_size):
    X_batch, y_batch = X_train[k*b_size:(k+1)*b_size, :], y_train[k*b_size:(k+1)*b_size]
    loss, grad, logprobs = f_loss_and_grad(X_batch, ind2onehot(y_batch, 10))
    exploss = loss if k == 0 else 0.99*exploss + 0.01*loss
    print('iter %d, loss %f, exploss %f' % (k + 1, loss, exploss))
    sgd_update(state, grad)


# test code

correct = 0
total = 0
print(X_test.shape)
print(y_test.shape)
for k in xrange(X_test.shape[0] / b_size):
    X_batch, y_batch = X_test[k*b_size:(k+1)*b_size, :], y_test[k*b_size:(k+1)*b_size]
    loss, grad, logprobs = f_loss_and_grad(X_batch, ind2onehot(y_batch, 10))
    preds = logprobs.argmax(axis=1).flatten()
    correct = correct + (preds == y_batch).sum()
    total = total + b_size

print('%d/%d correct', correct, total)
