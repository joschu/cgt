import cgt, numpy as np
from cgt import nn
from example_utils import fetch_dataset
import time
try:
    import theano.tensor as TT, theano, theano.tensor.signal.downsample #pylint: disable=F0401
    have_theano=True
except ImportError:
    have_theano=False

# ================================================================
# Replicate nn API in theano for the sake of comparison 
# ================================================================

class AffineTheano(object):
    """
    Theano equivalent of nn.affine
    """
    def __init__(self, input_size, output_size, name=None, weight_init=nn.Constant(0), bias_init=nn.Constant(0)):
        input_size = int(input_size)
        output_size = int(output_size)
        name = "unnamed" if name is None else name

        self.weight = theano.shared(nn.init_array(weight_init, (input_size, output_size)),
            name=name+".W")
        self.bias = theano.shared(nn.init_array(bias_init, (1, output_size)), 
            name=name+".b")
        self.bias.type.broadcastable = (True,False)

    def __call__(self, x):
        return x.dot(self.weight) +  self.bias

        
class SpatialConvolutionTheano(object):
    def __init__(self, input_channels, output_channels, kernelshape, pad, stride=(1,1), name=None, weight_init=nn.Constant(0), bias_init=nn.Constant(0)):
        # type conversion
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.kernelshape = tuple(map(int, kernelshape))
        self.pad = tuple(map(int,pad))
        self.stride = tuple(map(int,stride))
        name = "unnamed" if name is None else name

        self.weight = theano.shared(nn.init_array(weight_init, (self.output_channels, self.input_channels) + self.kernelshape),
            name=name+".W")
        self.bias = theano.shared(nn.init_array(bias_init, (1, self.output_channels, 1, 1)), 
            name=name+".b")
        self.bias.type.broadcastable = (True,False,True,True)

    def __call__(self, x):
        tmp = TT.nnet.conv2d(x, self.weight, 
            image_shape = (None, self.input_channels, None,None),
            filter_shape = (self.output_channels, self.input_channels) + self.kernelshape,
            border_mode='valid')
        return tmp + self.bias
def softmax_theano(x,axis=1):
    out = TT.exp(x)
    out = out / out.sum(axis=axis,keepdims=True)
    return out
def logsoftmax_theano(x, axis=1):
    return TT.log(softmax_theano(x, axis=axis))


# ================================================================
# Main script
# ================================================================


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--unittest", action="store_true")
    args = parser.parse_args()

    # Load data
    # -----------------------
    mnist = fetch_dataset("http://rll.berkeley.edu/cgt-data/mnist.npz")
    Xdata = (mnist["X"]/255.).astype(cgt.floatX)
    ydata = mnist["y"]

    Ntrain = 1000 if args.unittest else 10000
    Xtrain = Xdata[0:Ntrain]
    ytrain = ydata[0:Ntrain]
    sortinds = np.random.permutation(Ntrain)
    Xtrain = Xtrain[sortinds]
    ytrain = ytrain[sortinds]
    batch_size = 128
    cgt.update_config(backend="native")

    # Make symbolic variables
    # -----------------------

    def build_fc_return_loss(X, y):
        """
        Build fully connected network and return loss
        """
        np.random.seed(0)        
        h1 = nn.rectify(nn.Affine(28*28, 256, weight_init=nn.IIDGaussian(std=.1))(X))
        h2 = nn.rectify(nn.Affine(256, 256, weight_init=nn.IIDGaussian(std=.1))(h1))
        logprobs = nn.logsoftmax(nn.Affine(256, 10, weight_init=nn.IIDGaussian(std=.1))(h2))
        neglogliks = -logprobs[cgt.arange(X.shape[0]), y]
        loss = neglogliks.mean()
        return loss


    def make_updater_fc():
        X = cgt.matrix("X", fixed_shape=(None,28*28))
        y = cgt.vector("y",dtype='i8')
        stepsize = cgt.scalar("stepsize")
        loss = build_fc_return_loss(X, y)
        params = nn.get_parameters(loss)
        gparams = cgt.grad(loss, params)
        updates = [(p, p-stepsize*gp) for (p, gp) in zip(params, gparams)]
        return cgt.function([X,y, stepsize], loss, updates=updates)

    updater_fc = make_updater_fc()

    def run_sgd_epochs(Xs, ys, updater, stepsizeval = .001, n_epochs=1):
        for epoch in range(n_epochs):
            t_start = time.time()
            epoch_losses = []
            for start in range(0, Xs.shape[0]-(Xs.shape[0]%batch_size), batch_size):
                lossval = updater(Xs[start:start+batch_size], ys[start:start+batch_size], stepsizeval)
                epoch_losses.append(lossval)
            print("epoch %i took %5.2fs. mean train loss %8.3g"%(epoch, time.time() - t_start, np.mean(epoch_losses)))
    
    print("CGT Fully-Connected Network")
    run_sgd_epochs(Xtrain, ytrain, updater_fc)

    def make_updater_fc_theano():
        X = TT.matrix("X")
        y = TT.ivector("y")
        np.random.seed(0)        
        stepsize = TT.scalar("stepsize")
        layer1 = AffineTheano(28*28, 256, weight_init=nn.IIDGaussian(std=.1))
        h1 = nn.rectify(layer1(X))
        layer2 = AffineTheano(256, 256, weight_init=nn.IIDGaussian(std=.1))
        h2 = nn.rectify(layer2(h1))
        logprobs = logsoftmax_theano(AffineTheano(256, 10, weight_init=nn.IIDGaussian(std=.1))(h2))
        neglogliks = -logprobs[TT.arange(X.shape[0]), y]
        loss = neglogliks.mean()

        params = [layer1.weight, layer1.bias, layer2.weight, layer2.bias]
        gparams = TT.grad(loss, params)
        updates = [(p, p-stepsize*gp) for (p, gp) in zip(params, gparams)]
        return theano.function([X,y, stepsize], loss, updates=updates, allow_input_downcast=True)


    if have_theano and not args.unittest:
        updater_fc_theano = make_updater_fc_theano()
        print("Theano Fully-Connected Network")
        run_sgd_epochs(Xtrain, ytrain, updater_fc_theano)


    # Data parallelism
    # -----------------------

    def make_updater_fc_parallel():
        X = cgt.matrix("X", fixed_shape=(None,28*28))
        y = cgt.vector("y",dtype='i8')
        stepsize = cgt.scalar("stepsize")

        loss = build_fc_return_loss(X,y)
        params = nn.get_parameters(loss)        
        m = nn.Module([X,y], [loss])
        split_loss = 0
        for start in range(0, batch_size, batch_size//4):
            sli = slice(start, start+batch_size//4)
            split_loss += m([X[sli], y[sli]])[0]
        split_loss /= 4
        gparams = cgt.grad(split_loss, params)
        updates2 = [(p, p-stepsize*gp) for (p, gp) in zip(params, gparams)]
        return cgt.function([X,y, stepsize], split_loss, updates=updates2)
    
    with cgt.scoped_update_config(parallel=True, num_threads=4):
        updater_fc_par = make_updater_fc_parallel()


    print("Fully-connected Network with Split Input for Data Parallelism")
    run_sgd_epochs(Xtrain, ytrain, updater_fc_par)


    # Convnet on CPU
    # -----------------------

    Xtrainimg = Xtrain.reshape(-1,1,28,28)

    def build_convnet_return_loss(X, y):
        np.random.seed(0)        
        conv1 = nn.rectify(
            nn.SpatialConvolution(1, 32, kernelshape=(3,3), pad=(0,0), 
            weight_init=nn.IIDGaussian(std=.1))(X))
        pool1 = nn.max_pool_2d(conv1, kernelshape=(3,3), stride=(2,2))
        conv2 = nn.rectify(
            nn.SpatialConvolution(32, 32, kernelshape=(3,3), pad=(0,0), 
            weight_init=nn.IIDGaussian(std=.1))(pool1))
        pool2 = nn.max_pool_2d(conv2, kernelshape=(3,3), stride=(2,2))
        d0,d1,d2,d3 = pool2.shape
        flatlayer = pool2.reshape([d0,d1*d2*d3])
        nfeats = cgt.infer_shape(flatlayer)[1]
        logprobs = nn.logsoftmax(nn.Affine(nfeats, 10)(flatlayer))
        loss = -logprobs[cgt.arange(X.shape[0]), y].mean()
        return loss

    def make_updater_convnet():
        X = cgt.tensor4("X", fixed_shape=(None,1,28,28)) # so shapes can be inferred
        y = cgt.vector("y",dtype='i8')
        stepsize = cgt.scalar("stepsize")
        loss = build_convnet_return_loss(X, y)

        params = nn.get_parameters(loss)
        gparams = cgt.grad(loss, params)
        updates = [(p, p-stepsize*gp) for (p, gp) in zip(params, gparams)]
        return cgt.function([X,y, stepsize], loss, updates=updates)

    updater_convnet = make_updater_convnet()

    print("CGT Convnet")
    Xtrainimgs = Xtrain.reshape(-1,1,28,28)
    run_sgd_epochs(Xtrainimgs, ytrain, updater_convnet)

    def make_updater_convnet_theano():
        X = TT.tensor4("X") # so shapes can be inferred
        y = TT.ivector("y")
        np.random.seed(0)        
        stepsize = TT.scalar("stepsize")
        layer1 = SpatialConvolutionTheano(1, 32, kernelshape=(3,3), pad=(0,0), 
            weight_init=nn.IIDGaussian(std=.1))
        conv1 = nn.rectify(layer1(X))
        pool1 = theano.tensor.signal.downsample.max_pool_2d(conv1, ds=(3,3), st=(2,2))
        layer2 = SpatialConvolutionTheano(32, 32, kernelshape=(3,3), pad=(0,0), 
            weight_init=nn.IIDGaussian(std=.1))
        conv2 = nn.rectify(layer2(pool1))
        pool2 = theano.tensor.signal.downsample.max_pool_2d(conv2, ds=(3,3), st=(2,2))
        d0,d1,d2,d3 = pool2.shape
        flatlayer = pool2.reshape([d0,d1*d2*d3])
        nfeats = 800 # theano doens't know how to calculate shapes before compiling 
        # the function, so this needs to be computed by hand
        layer3 = AffineTheano(nfeats, 10)
        ip1 = layer3(flatlayer)
        logprobs = logsoftmax_theano(ip1)
        loss = -logprobs[TT.arange(X.shape[0]), y].mean()

        params = [layer1.weight, layer1.bias, layer2.weight, layer2.bias, layer3.weight, layer3.bias]
        gparams = TT.grad(loss, params)
        updates = [(p, p-stepsize*gp) for (p, gp) in zip(params, gparams)]
        return theano.function([X,y, stepsize], loss, updates=updates, allow_input_downcast=True)

    if False:#have_theano and not args.unittest:

        updater_convnet_theano = make_updater_convnet_theano()
        print("Theano Convnet")
        Xtrainimgs = Xtrain.reshape(-1,1,28,28)
        run_sgd_epochs(Xtrainimgs, ytrain, updater_convnet_theano)

    # Data parallelism again
    # -----------------------

    def make_updater_convnet_parallel():
        X = cgt.tensor4("X", fixed_shape=(None,1,28,28)) # so shapes can be inferred
        y = cgt.vector("y",dtype='i8')
        stepsize = cgt.scalar("stepsize")
        loss = build_convnet_return_loss(X, y)

        m = nn.Module([X,y], [loss])
        split_loss = 0
        for start in range(0, batch_size, batch_size//4):
            sli = slice(start, start+batch_size//4)
            split_loss += m([X[sli], y[sli]])[0]
        split_loss /= 4
        params = nn.get_parameters(loss)
        gparams = cgt.grad(split_loss, params)
        updates2 = [(p, p-stepsize*gp) for (p, gp) in zip(params, gparams)]
        return cgt.function([X,y, stepsize], split_loss, updates=updates2)
    
    with cgt.scoped_update_config(parallel=True, num_threads=4):
        updater_convnet_parallel = make_updater_convnet_parallel()

    # Run SGD
    # -----------------------
    print("CGT Convnet with Data Parallelism")    
    run_sgd_epochs(Xtrainimgs, ytrain, updater_convnet_parallel)

