
from example_utils import download, fmt_row
import cPickle, numpy as np
import cgt
from cgt import nn
import argparse, time

def rmsprop_updates(cost, params, stepsize=0.001, rho=0.9, epsilon=1e-6):
    grads = cgt.grad(cost, params)
    updates = []
    for p, g in zip(params, grads):
        acc = cgt.shared(p.op.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * cgt.square(g)
        gradient_scaling = cgt.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - stepsize * g))
    return updates

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile",action="store_true")
    parser.add_argument("--unittest",action="store_true")
    parser.add_argument("--epochs",type=int,default=10)
    args = parser.parse_args()

    batchsize = 64
    Xshape = (batchsize, 3, 32, 32)
    X = cgt.tensor4("X", fixed_shape = Xshape)
    y = cgt.vector("y", fixed_shape = (batchsize,), dtype='i4')

    conv1 = nn.SpatialConvolution(3, 32, kernelshape=(5,5), pad=(2,2), 
        weight_init=nn.IIDGaussian(std=1e-4))(X)
    relu1 = nn.rectify(conv1)
    pool1 = nn.max_pool_2d(relu1, kernelshape=(3,3), stride=(2,2))
    conv2 = nn.SpatialConvolution(32, 32, kernelshape=(5,5), pad=(2,2), 
        weight_init=nn.IIDGaussian(std=0.01))(relu1)
    relu2 = nn.rectify(conv2)
    pool2 = nn.max_pool_2d(relu2, kernelshape=(3,3), stride=(2,2))
    conv3 = nn.SpatialConvolution(32, 64, kernelshape=(5,5), pad=(2,2), 
        weight_init=nn.IIDGaussian(std=0.01))(pool2)
    pool3 = nn.max_pool_2d(conv3, kernelshape=(3,3), stride=(2,2))
    relu3 = nn.rectify(pool3)
    d0,d1,d2,d3 = relu3.shape
    flatlayer = relu3.reshape([d0,d1*d2*d3])
    nfeats = cgt.infer_shape(flatlayer)[1]
    ip1 = nn.Affine(nfeats, 10)(flatlayer)
    logprobs = nn.logsoftmax(ip1)
    loss = -logprobs[cgt.arange(batchsize), y].mean()

    params = nn.get_parameters(loss)
    updates = rmsprop_updates(loss, params, stepsize=1e-3)
    
    train = cgt.function(inputs=[X, y], outputs=[loss], updates=updates)

    if args.profile: cgt.profiler.start()

    data = np.load("/Users/joschu/Data/cifar-10-batches-py/cifar10.npz")
    Xtrain = data["X_train"]
    ytrain = data["y_train"]

    print fmt_row(10, ["Epoch","Train NLL","Train Err","Test NLL","Test Err","Epoch Time"])
    for i_epoch in xrange(args.epochs):
        for start in xrange(0, Xtrain.shape[0], batchsize):
            tstart = time.time()
            end = start+batchsize
            print train(Xtrain[start:end], ytrain[start:end]), time.time()-tstart
            if start > batchsize*5: break
        # elapsed = time.time() - tstart
        # trainerr, trainloss = computeloss(Xtrain[:len(Xtest)], ytrain[:len(Xtest)])
        # testerr, testloss = computeloss(Xtest, ytest)
        # print fmt_row(10, [i_epoch, trainloss, trainerr, testloss, testerr, elapsed])
        if args.profile: 
            cgt.profiler.print_stats()
            return
        if args.unittest:
            break



if __name__ == "__main__":
    main()