__doc__ = """
Neural network library, drawing inspiration from Torch's nn and nngraph
"""

import cgt
from cgt import core, size
import numpy as np
from .nn_ops.im2col import im2col
from .nn_ops.max_pool_2d import max_pool_2d #pylint: disable=W0611
from .nn_ops.cross_channel_lrn import cross_channel_lrn #pylint: disable=W0611
from .nn_ops import cudnn_ops #pylint: disable=W0611
from collections import namedtuple

class Module(object):
    def __init__(self, inputs, outputs):
        self.c = core.Composition(inputs, outputs)
    def __call__(self, inputs):
        return self.c.expand(inputs)        
        # assert all(isinstance(x,core.Node) for x in inputs)
        # tup_out = core.Result(self.c, inputs)
        # return [core.Result(core.TupleIndex(i),[tup_out]) for i in xrange(self.c.n_out)]
    def get_parameters(self):
        return list(node for node in self.c.get_nodes() if node.is_data())
    def expand(self, inputs):
        return self.c.expand(inputs)

def is_parameter(node):
    return node.is_data() and node.props["is_parameter"]

def get_parameters(loss):
    return list(node for node in cgt.core.topsorted([loss]) if is_parameter(node))

def parameter(val, name=None, device=None):
    fixed_shape_mask = "all"
    out = cgt.shared(val, name=name, device=device, fixed_shape_mask=fixed_shape_mask)
    out.props["is_parameter"] = True
    return out


# ================================================================
# Math functions
# ================================================================

def rectify(x):
    return x * (x >= 0)

def _nu_softplus(x,out):
    absx = np.abs(x)
    out[:] = (absx+x)/2 + np.log(1 + np.exp(-absx))

def softplus(x):
    op = core.ElwiseUnary("softplus",core.UnaryInfo("SoftPlus", _nu_softplus, True, 'f', lambda x, g, gy: gy/(cgt.exp(-x)+1.0), "(x > 0) ? (x + log(exp(-x) + 1)) : log(1+exp(x))"))
    return core.Result(op, [x])

def softmax(x,axis=1):
    # x = cgt.broadcast("-", x, x.max(axis=1,keepdims=True),"xx,x1")
    out = cgt.exp(x)
    out = cgt.broadcast("/", out, out.sum(axis=axis,keepdims=True), "xx,x1")
    return out

def logsoftmax(x, axis=1):
    return cgt.log(softmax(x, axis=axis))

def zero_one_loss(x, y):
    assert x.ndim == 2 and y.ndim in (1,2) and core.dtype_kind(y.dtype)=='i'
    return cgt.equal(x.argmax(axis=1,keepdims=False),y.flatten())

def dropout(x, p=0):
    if p==0: 
        return x
    else:
        mask = cgt.greater(cgt.rand(*cgt.shape(x)), p)
        x = x * mask
        x = x /(1.0-p)
        return x

def conv2d_fft(x_BKRC, f_LKrc, subsample, pad):
    # TODO add shape assertion
    f_LKrc = cgt.flip(f_LKrc, [2,3])
    padnrows = size(x_BKRC, 2) + size(f_LKrc, 2) - 1
    padncols = size(x_BKRC, 3) + size(f_LKrc, 3) - 1
    tx = cgt.rfft(x_BKRC, (padnrows,padncols), (2,3))
    tf = cgt.rfft(f_LKrc, (padnrows,padncols), (2,3))
    out = cgt.irfft( cgt.einsum("BKrc,LKrc->BLrc",tx, tf), (2,3))
    out = out[:,:,pad[0]:(padnrows-pad[0]):subsample[0],pad[1]:(padncols-pad[1]):subsample[1]] #pylint: disable=E1127
    return out

def conv2d(x_BKRC, f_LKrc, kernelshape, pad=(0,0), stride=(1,1)):
    col_BmnZ = im2col(x_BKRC, kernelshape, pad, stride)
    L,K,r,c = f_LKrc.shape
    f_LZ = f_LKrc.reshape([L, K*r*c])
    B,m,n,Z = col_BmnZ.shape
    col_Bmn_Z = col_BmnZ.reshape([B*m*n, Z])
    col_Bmn_L = core.Result(core.Mul22(False,True), [col_Bmn_Z, f_LZ])
    return col_Bmn_L.reshape([B,m,n,L]).transpose([0,3,1,2])


# ================================================================
# Initializations
# ================================================================


IIDGaussian = namedtuple("IIDGaussian", ["mean","std"])
IIDGaussian.__new__.__defaults__ = (0, 1)
IIDUniform = namedtuple("IIDUniform", ["low","high"])
Zeros = namedtuple("Zeros",[])

def init_array(init, shape):
    if isinstance(init, IIDGaussian):
        return (np.random.randn(*shape)*init.std + init.mean).astype(cgt.floatX)
    elif isinstance(init, IIDUniform):
        return (np.random.rand(*shape)*(init.high-init.low) + init.low).astype(cgt.floatX)
    elif isinstance(init, Zeros):
        return np.zeros(shape, cgt.floatX)
    else:
        raise ValueError("Invalid initializer %s"%init)



# ================================================================
# Layer constructors
# ================================================================

class Affine(object):
    """
    Like torch's nn.Linear
    """
    def __init__(self, input_size, output_size, name=None, weight_init=Zeros(), bias_init=Zeros()):
        input_size = int(input_size)
        output_size = int(output_size)
        name = "unnamed" if name is None else name

        self.weight = parameter(init_array(weight_init, (input_size, output_size)),
            name=name+".W")
        self.bias = parameter(init_array(bias_init, (1, output_size)), 
            name=name+".b")

    def __call__(self, x):
        return cgt.broadcast("+", x.dot(self.weight), self.bias, "xx,1x")


class SpatialConvolution(object):
    def __init__(self, input_channels, output_channels, kernelshape, pad, stride=(1,1), name=None, weight_init=Zeros(), bias_init=Zeros()):
        # type conversion
        input_channels = int(input_channels)
        output_channels = int(output_channels)
        self.kernelshape = tuple(map(int, kernelshape))
        self.pad = tuple(map(int,pad))
        self.stride = tuple(map(int,stride))
        name = "unnamed" if name is None else name

        self.weight = parameter(init_array(weight_init, (output_channels, input_channels) + self.kernelshape),
            name=name+".W")
        self.bias = parameter(init_array(bias_init, (1, output_channels, 1, 1)), 
            name=name+".b")

    def __call__(self, x):
        tmp = conv2d(x, self.weight, self.kernelshape, self.pad, self.stride)
        return cgt.broadcast("+", tmp, self.bias, "xxxx,1x11")



# ================================================================
# Optimization
# ================================================================

def sgd(cost, params, learning_rate):
    updates = []
    grads = cgt.grad(cost, params)
    for param, grad in zip(params, grads):
        updates.append((param, param - learning_rate * grad))

    return updates


def momentum(cost, params, learning_rate, momentum=0.9):
    updates = []
    grads = cgt.grad(cost, params)
    for param, grad in zip(params, grads):
        value = param.op.get_value()
        velocity = cgt.shared(np.zeros(value.shape, dtype=value.dtype))
        x = momentum * velocity + param - learning_rate * grad
        updates.append((velocity, x-param))
        updates.append((param, x))

    return updates


def nesterov_momentum(cost, params, learning_rate, momentum=0.9):
    updates = []
    grads = cgt.grad(cost, params)

    for param, grad in zip(params, grads):
        value = param.op.get_value()
        velocity = cgt.shared(np.zeros(value.shape, dtype=value.dtype))
        x = momentum * velocity + param - learning_rate * grad - param
        updates.append((velocity, x))
        updates.append((param, momentum*x + param - learning_rate * grad))

    return updates


def adagrad(cost, params, learning_rate=1.0, epsilon=1e-6):

    updates = []
    grads = cgt.grad(cost, params)

    for param, grad in zip(params, grads):
        value = param.op.get_value()
        accu = cgt.shared(np.zeros(value.shape, dtype=value.dtype))
        accu_new = accu + grad ** 2
        updates.append((accu, accu_new))
        updates.append((param, param - (learning_rate * grad) / cgt.sqrt(accu_new + epsilon)))

    return updates


def rmsprop(cost, params, learning_rate=1.0, rho=0.9, epsilon=1e-6):

    updates = []
    grads = cgt.grad(cost, params)

    for param, grad in zip(params, grads):
        value = param.op.get_value()
        accu = cgt.shared(np.zeros(value.shape, dtype=value.dtype))
        accu_new = rho * accu + (1 - rho) * grad ** 2
        updates.append((accu, accu_new))
        updates.append((param, param - (learning_rate * grad / cgt.sqrt(accu_new + epsilon))))

    return updates


def adadelta(cost, params, learning_rate=1.0, rho=0.95, epsilon=1e-6):

    updates = []
    grads = cgt.grad(cost, params)

    for param, grad in zip(params, grads):
        value = param.op.get_value()
        accu = cgt.shared(np.zeros(value.shape, dtype=value.dtype))
        delta_accu = cgt.shared(np.zeros(value.shape, dtype=value.dtype))

        accu_new = rho * accu + (1 - rho) * grad ** 2
        updates.append((accu, accu_new))

        update = (grad * cgt.sqrt(delta_accu + epsilon) / cgt.sqrt(accu_new + epsilon))
        updates.append((param, param - learning_rate * update))

        delta_accu_new = rho * delta_accu + (1 - rho) * update ** 2
        updates.append((delta_accu, delta_accu_new))

    return updates