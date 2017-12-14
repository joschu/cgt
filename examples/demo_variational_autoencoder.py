import cgt
from cgt import core
from cgt import nn
import numpy as np
import pickle as pickle
from scipy.stats import norm
import matplotlib.pyplot as plt
from example_utils import fetch_dataset

'''
MNIST manifold demo (with 2-dimensional latent z) using variational autoencoder
'''

rng = np.random.RandomState(1234)

def kld_unit_mvn(mu, var):
    # KL divergence from N(0, I)
    return (mu.shape[1] + cgt.sum(cgt.log(var), axis=1) - cgt.sum(cgt.square(mu), axis=1) - cgt.sum(var, axis=1)) / 2.0

def log_diag_mvn(mu, var):
    # log probability of x under N(mu, diag(var))
    def f(x):
        # expects batches
        k = mu.shape[1]
        logp = (-k / 2.0) * np.log(2 * np.pi) - 0.5 * cgt.sum(cgt.log(var), axis=1) - cgt.sum(0.5 * (1.0 / var) * (x - mu) * (x - mu), axis=1)
        return logp
    return f

class HiddenLayer(object):

    # adapted from http://deeplearning.net/tutorial/mlp.html

    def __init__(self, input, n_in, n_out, W=None, b=None,
                 activation=cgt.tanh, prefix=""):
        self.n_in = n_in
        self.n_out = n_out

        if W is None:
            # XXX replace with nn init
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=cgt.floatX
            )
            if activation == cgt.sigmoid:
                W_values *= 4

            W = cgt.shared(W_values, name=prefix+"_W")

        if b is None:
            b_values = np.zeros((n_out,), dtype=cgt.floatX)
            b = cgt.shared(b_values, name=prefix+"_b")

        self.W = W
        self.b = b

        # XXX broadcast api may change
        lin_output = cgt.broadcast("+", cgt.dot(input, self.W),
                cgt.dimshuffle(self.b, ["x", 0]), "xx,1x")
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]

class _MLP(object):

    # building block for MLP instantiations defined below

    def __init__(self, x, n_in, n_hid, nlayers=1, prefix=""):
        self.nlayers = nlayers
        self.hidden_layers = list()
        inp = x
        for k in range(self.nlayers):
            hlayer = HiddenLayer(
                input=inp,
                n_in=n_in,
                n_out=n_hid,
                activation=cgt.tanh,
                prefix=prefix + ("_%d" % (k + 1))
            )
            n_in = n_hid
            inp = hlayer.output
            self.hidden_layers.append(hlayer)

        self.params = [param for l in self.hidden_layers for param in l.params]
        self.input = input
        # NOTE output layer computed by instantations

class GaussianMLP(_MLP):

    def __init__(self, x, n_in, n_hid, n_out, nlayers=1, y=None, eps=None):
        super(GaussianMLP, self).__init__(x, n_in, n_hid, nlayers=nlayers, prefix="GaussianMLP_hidden")
        self.mu_layer = HiddenLayer(
            input=self.hidden_layers[-1].output,
            n_in=self.hidden_layers[-1].n_out,
            n_out=n_out,
            activation=None,
            prefix="GaussianMLP_mu"
        )
        # log(sigma^2)
        self.logvar_layer = HiddenLayer(
            input=self.hidden_layers[-1].output,
            n_in=self.hidden_layers[-1].n_out,
            n_out=n_out,
            activation=None,
            prefix="GaussianMLP_logvar"
        )
        self.mu = self.mu_layer.output
        self.var = cgt.exp(self.logvar_layer.output)
        self.sigma = cgt.sqrt(self.var)
        self.params = self.params + self.mu_layer.params +\
            self.logvar_layer.params
        # for use as encoder
        if eps is not None:
            assert(y is None)
            self.out = self.mu + self.sigma * eps
        # for use as decoder
        if y:
            assert(eps is None)
            self.out = cgt.sigmoid(self.mu)
            self.cost = -cgt.sum(log_diag_mvn(self.out, self.var)(y))

class BernoulliMLP(_MLP):

    def __init__(self, x, n_in, n_hid, n_out, nlayers=1, y=None):
        super(BernoulliMLP, self).__init__(x, n_in, n_hid, nlayers=nlayers, prefix="BernoulliMLP_hidden")
        self.out_layer = HiddenLayer(
            input=self.hidden_layers[-1].output,
            n_in=self.hidden_layers[-1].n_out,
            n_out=n_out,
            activation=cgt.sigmoid,
            prefix="BernoulliMLP_y_hat"
        )
        self.params = self.params + self.out_layer.params
        if y is not None:
            self.out = self.out_layer.output
            self.cost = cgt.sum(nn.binary_crossentropy(self.out, y))

class VAE(object):

    def __init__(self, xdim, args, dec="bernoulli"):
        self.xdim = xdim
        self.hdim = args.hdim
        self.zdim = args.zdim
        self.lmbda = args.lmbda  # weight decay coefficient * 2
        self.x = cgt.matrix("x", dtype=cgt.floatX)
        self.eps = cgt.matrix("eps", dtype=cgt.floatX)

        self.enc_mlp = GaussianMLP(self.x, self.xdim, self.hdim, self.zdim, nlayers=args.nlayers, eps=self.eps)
        if dec == "bernoulli":
            # log p(x | z) defined as -CE(x, y) = dec_mlp.cost(y)
            self.dec_mlp = BernoulliMLP(self.enc_mlp.out, self.zdim, self.hdim, self.xdim, nlayers=args.nlayers, y=self.x)
        elif dec == "gaussian":
            self.dec_mlp = GaussianMLP(self.enc_mlp.out, self.zdim, self.hdim, self.xdim, nlayers=args.nlayers, y=self.x)
        else:
            raise RuntimeError("unrecognized decoder %" % dec)

        self.cost = (-cgt.sum(kld_unit_mvn(self.enc_mlp.mu, self.enc_mlp.var)) + self.dec_mlp.cost) / args.batch_size
        self.params = self.enc_mlp.params + self.dec_mlp.params
        # L2 regularization
        self.gparams = [cgt.grad(self.cost, [p])[0] + self.lmbda * p for p in self.params]
        self.gaccums = [cgt.shared(np.zeros(p.op.get_value().shape, dtype=cgt.floatX)) for p in self.params]

        # XXX replace w/ adagrad update from nn
        ADAGRAD_EPS = 1e-10  # for stability
        self.updates = [
            (param, param - args.lr * gparam / cgt.sqrt(gaccum + cgt.square(gparam) + ADAGRAD_EPS))
            for param, gparam, gaccum in zip(self.params, self.gparams, self.gaccums)
        ]
        self.updates += [
            (gaccum, gaccum + cgt.square(gparam))
            for gaccum, gparam in zip(self.gaccums, self.gparams)
        ]

        self.train = cgt.function(
            [self.x, self.eps],
            self.cost,
            updates=self.updates
        )
        self.test = cgt.function(
            [self.x, self.eps],
            self.cost,
            updates=None
        )
        # can be used for semi-supervised learning for example
        self.encode = cgt.function(
            [self.x, self.eps],
            self.enc_mlp.out
        )

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=100)
    parser.add_argument("--nlayers", default=1, type=int, help="number of hidden layers in MLP before output layers")
    parser.add_argument("--hdim", default=500, type=int, help="dimension of hidden layer")
    parser.add_argument("--zdim", default=2, type=int, help="dimension of continuous latent variable")
    parser.add_argument("--lmbda", default=0.001, type=float, help="weight decay coefficient")
    parser.add_argument("--lr", default=0.01, type=float, help="learning rate")
    parser.add_argument("--epochs", default=1000, type=int, help="number of passes over dataset")
    parser.add_argument("--print_every", default=100, type=int, help="how often to print cost")
    parser.add_argument("--outfile", default="vae_model.pk", help="output file to save model to")
    args = parser.parse_args()
    print(args)

    if args.epochs > 100:
        print("NOTE: training might take a while. You may want to first sanity check by setting --epochs to something like 20 (manifold will be fuzzy).")

    # set up dataset

    mnist = fetch_dataset("http://rll.berkeley.edu/cgt-data/mnist.npz")
    X = (mnist["X"]/255.).astype(cgt.floatX)
    y = mnist["y"]
    np.random.seed(0)
    sortinds = np.random.permutation(70000)
    X = X[sortinds]
    y = y[sortinds]
    train_x = X[0:50000]
    train_y = y[0:50000]
    valid_x = X[50000:60000]
    valid_y = y[50000:60000]

    # run SGVB algorithm

    model = VAE(train_x.shape[1], args, dec="bernoulli")

    expcost = None
    num_train_batches = train_x.shape[0] / args.batch_size
    num_valid_batches = valid_x.shape[0] / args.batch_size
    valid_freq = num_train_batches

    for b in range(args.epochs * num_train_batches):
        k = b % num_train_batches
        x = train_x[k * args.batch_size:(k + 1) * args.batch_size, :]
        eps = np.random.randn(x.shape[0], args.zdim).astype(cgt.floatX)
        cost = model.train(x, eps)
        if not expcost:
            expcost = cost
        else:
            expcost = 0.01 * cost + 0.99 * expcost
        if (b + 1) % args.print_every == 0:
            print(("iter %d, cost %f, expcost %f" % (b + 1, cost, expcost)))
        if (b + 1) % valid_freq == 0:
            valid_cost = 0
            for l in range(num_valid_batches):
                x_val = valid_x[l * args.batch_size:(l + 1) * args.batch_size, :]
                eps_val = np.zeros((x_val.shape[0], args.zdim), dtype=cgt.floatX)
                valid_cost = valid_cost + model.test(x_val, eps_val)
            valid_cost = valid_cost / num_valid_batches
            print(("valid cost: %f" % valid_cost))

    # XXX fix pickling of cgt models
    #print("saving final model")
    #with open(args.outfile, "wb") as f:
        #pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    # XXX use this to sample, should later be able to compile f(z) = y directly (See Issue #18)
    newz = cgt.matrix("newz", dtype=cgt.floatX)
    newy = cgt.core.clone(model.dec_mlp.out, {model.enc_mlp.out:newz})
    decode = cgt.function(
        [newz],
        newy
    )

    S = (28, 28)
    M = 20
    manifold = np.zeros((S[0]*M, S[1]*M), dtype=cgt.floatX)

    for z1 in range(M):
        for z2 in range(M):
            print(z1, z2)
            z = np.zeros((1, 2))
            # pass unit square through inverse Gaussian CDF
            z[0, 0] = norm.ppf(z1 * 1.0/M + 1.0/(M * 2))
            z[0, 1] = norm.ppf(z2 * 1.0/M + 1.0/(M * 2))
            z = np.array(z, dtype=cgt.floatX)
            x_hat = decode(z)
            x_hat = x_hat.reshape(S)
            manifold[z1 * S[0]:(z1 + 1) * S[0],
                     z2 * S[1]:(z2 + 1) * S[1]] = x_hat

    plt.imshow(manifold, cmap="Greys_r")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()
