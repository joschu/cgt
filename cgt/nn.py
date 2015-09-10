__doc__ = """
Neural network library, drawing inspiration from Torch's nn and nngraph
"""

import cgt
from cgt import core, size
import numpy as np
from .nn_ops.im2col import im2col
from .nn_ops.cross_channel_lrn import cross_channel_lrn #pylint: disable=W0611
from .nn_ops import cudnn_ops #pylint: disable=W0611
from .nn_ops.max_pool_2d import MaxPool
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

def binary_crossentropy(x, y):
    return -(y * cgt.log(x) + (1 - y) * cgt.log(1 - x))

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


# ================================================================
# Image processing functions
# ================================================================

PoolInfo = namedtuple("PoolInfo", ["kernel_h", "kernel_w", "pad_h", "pad_w", "stride_h", "stride_w"])

def conv2d(x_BKRC, f_LKrc, kernelshape, pad=(0,0), stride=(1,1)):
    devtype = cgt.get_config()["default_device"].devtype
    if devtype == "gpu":        
        return cudnn_ops.CudnnConvForward(pad[0],pad[1],stride[0],stride[1])
    else:
        assert devtype == "cpu"
        col_BmnZ = im2col(x_BKRC, kernelshape, pad, stride)
        L,K,r,c = f_LKrc.shape
        f_LZ = f_LKrc.reshape([L, K*r*c])
        B,m,n,Z = col_BmnZ.shape
        col_Bmn_Z = col_BmnZ.reshape([B*m*n, Z])
        col_Bmn_L = core.Result(core.Mul22(False,True), [col_Bmn_Z, f_LZ])
        return col_Bmn_L.reshape([B,m,n,L]).transpose([0,3,1,2])

def max_pool_2d(x, kernelshape, pad = (0,0), stride=(1,1)):
    devtype = cgt.get_config()["default_device"].devtype
    kernel_h, kernel_w = kernelshape
    pad_h, pad_w = pad
    stride_h, stride_w = stride
    info = PoolInfo(kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w)
    if devtype == "gpu":        
        return core.Result(cudnn_ops.CudnnPoolForward(info), [x])
    else:
        return core.Result(MaxPool(info), [x])[0]

# ================================================================
# Initializations
# ================================================================


IIDGaussian = namedtuple("IIDGaussian", ["mean", "std"])
IIDGaussian.__new__.__defaults__ = (0, 1)
IIDUniform = namedtuple("IIDUniform", ["low", "high"])
Constant = namedtuple("Constant", ["constant"])
XavierNormal = namedtuple("XavierNormal", ["scale"])
XavierUniform = namedtuple("XavierUniform", ["scale"])
XavierNormal.__new__.__defaults__ = (1,)
XavierUniform.__new__.__defaults__ = (1,)
HeNormal = namedtuple("HeNormal", ["scale"])
HeUniform = namedtuple("HeUniform", ['scale'])


def init_array(init, shape):
    if isinstance(init, IIDGaussian):
        return (np.random.randn(*shape)*init.std + init.mean).astype(cgt.floatX)
    elif isinstance(init, IIDUniform):
        return (np.random.rand(*shape)*(init.high-init.low) + init.low).astype(cgt.floatX)
    elif isinstance(init, Constant):
        return init.constant*np.ones(shape, cgt.floatX)
    elif isinstance(init, XavierNormal):
        std = get_xavier_weight(init, shape)
        return (np.random.randn(*shape)*std).astype(cgt.floatX)
    elif isinstance(init, XavierUniform):
        std = get_xavier_weight(init, shape)
        high = -np.sqrt(3) * std
        low = np.sqrt(3) * std
        return (np.random.rand(*shape)*(high-low) + low).astype(cgt.floatX)
    elif isinstance(init, HeNormal):
        std = get_he_weight(init, shape)
        return (np.random.randn(*shape)*std).astype(cgt.floatX)
    elif isinstance(init, HeUniform):
        std = get_he_weight(init, shape)
        low = -np.sqrt(3) * std
        high = np.sqrt(3) * std
        return (np.random.rand(*shape)*(high-low) + low).astype(cgt.floatX)
    else:
        raise ValueError("Invalid initializer %s"%init)


def get_xavier_weight(init, shape):
    """For relu activation scale (init.scale) should be sqrt(2). For sigmoid and tanh 1.0 should be used.
           Math depends on chosen underlying distribution (Normal, Uniform, etc) and activation function.
           For uniform with RELU you obtain
           a = sqrt{frac{6}{fan_{in}+fan_{out}}
           W &\sim U[-a, a]. See reference for full details.
           Reference: Xavier Glorot and Yoshua Bengio (2010):
           Understanding the difficulty of training deep feedforward neural
           networks. International conference on artificial intelligence and
           statistics."""
    if len(shape) < 2:
        raise RuntimeError("Shape length must at least 2")
    n1, n2 = shape[:2]
    field_size = np.prod(shape[2:])
    scale = init.scale
    std = scale * np.sqrt(2.0 / ((n1 + n2) * field_size))
    return std


def get_he_weight(init, shape):
    """Use sqrt(2) for RELU and 1 for sigmoid/linear/tanh for init.scale
           Weights are initialized with a standard deviation of
           sigma = scale*sqrt{1/fan_{in}}
           Reference: Kaiming He et al. (2015):
           Delving deep into rectifiers: Surpassing human-level performance on
           imagenet classification. arXiv preprint arXiv:1502.01852."""
    if len(shape) == 2:
        fan_in = shape[0]
    elif len(shape) > 2:
        fan_in = np.prod(shape[1:])
    else:
        raise RuntimeError("This initializer does not work with shapes of length less than two")

    std = init.scale * np.sqrt(1.0 / fan_in)
    return std


# ================================================================
# Layer constructors
# ================================================================

class Affine(object):
    """
    Like torch's nn.Linear
    """
    def __init__(self, input_size, output_size, name=None, weight_init=Constant(0), bias_init=Constant(0)):
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
    def __init__(self, input_channels, output_channels, kernelshape, pad, stride=(1,1), name=None, weight_init=Constant(0), bias_init=Constant(0)):
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
    """Stochastic Gradient Descent (SGD) updates
    Math:
    * ``param := param - learning_rate * gradient``
    Parameters
    ----------
    cost : a scalar loss.
    params : a list of cgt shared variables. We generate update
            expressions w.r.t. these variables.
    learning_rate : float
        Tunes the size of the update step.
    Returns
    -------
    list of tuples of the form (param, updates)
    """
    updates = []
    grads = cgt.grad(cost, params)
    for param, grad in zip(params, grads):
        updates.append((param, param - learning_rate * grad))

    return updates


def momentum(cost, params, learning_rate, momentum=0.9):
    """Stochastic Gradient Descent (SGD) updates with momentum
    Math:
    * ``velocity := momentum * velocity - learning_rate * grad``
    * ``param := param + velocity``
    Parameters
    ----------
    cost : a scalar loss.
    params : a list of cgt shared variables. We generate update
            expressions w.r.t. these variables.
    learning_rate : float
        Tunes the size of the update step.
    momentum: float
        Tunes the weight given to the velocity term.
    Returns
    -------
    list of tuples of the form [(param, updates) (velocity, velocity_update)]
    """
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
    """Stochastic Gradient Descent (SGD) updates with Nesterov momentum

    Math:
    * ``velocity := momentum * velocity - learning_rate * grad``
    * ``param := momentum*velocity + param - learning_rate * grad``

    Parameters
    ----------
    cost : a scalar loss.
    params : a list of cgt shared variables. We generate update
            expressions w.r.t. these variables.
    learning_rate : float
        Tunes the size of the update step.
    momentum: float
        Tunes the weight given to the velocity term.

    Returns
    -------
    list of tuples of the form [(param, updates) (velocity, velocity_update)]
    """
    updates = []
    grads = cgt.grad(cost, params)

    for param, grad in zip(params, grads):
        value = param.op.get_value()
        velocity = cgt.shared(np.zeros(value.shape, dtype=value.dtype))
        x = momentum * velocity - learning_rate * grad
        updates.append((velocity, x))
        updates.append((param, momentum*x + param - learning_rate * grad))

    return updates


def adagrad(cost, params, learning_rate=1.0, epsilon=1e-6):
    """Adagrad updates
    The learning rate will be scaled by dividing it by the sqaure root of the sum of accumulated squared gradients.

    Math:
    * ``accu_new = accu + grad ** 2``
    * ``param = param - (learning_rate * grad) / cgt.sqrt(accu_new + epsilon)``

    Parameters
    ----------
    cost : a scalar loss.
    params : a list of cgt shared variables. We generate update
            expressions w.r.t. these variables.
    learning_rate : float
        Tunes the size of the update step.
    epsilon: avoids division close to zero. Small float.

    Returns
    -------
    list of tuples of the form [(param, updates), (accumulated_grads, accumulated_grads_new)]

    References
    ----------
    .. [1] Duchi, J., Hazan, E., & Singer, Y. (2011):
           Adaptive subgradient methods for online learning and stochastic
           optimization. JMLR, 12:2121-2159.
    """

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
    """RMSProp updates
    Divide learning rate by moving average of RMS gradients. See [1]

    Math:
    * ``accu_new = rho * accu + (1 - rho) * grad ** 2``
    * ``param = param - (learning_rate * grad / cgt.sqrt(accu_new + epsilon))``

    Parameters
    ----------
    cost : a scalar loss.
    params : a list of cgt shared variables. We generate update
            expressions w.r.t. these variables.
    learning_rate : float
        Tunes the size of the update step.
    rho : float
        Controls decay of gradient moving average.
    epsilon : float
        Avoid division by 0 while scaling. Small constant.

    Returns
    -------
    list of tuples of the form [(param, updates), (accumulated_RMS_grads, accumulated_RMS_grads_new)]

    References
    ----------
    .. [1] Yann N. Dauphin, Harm de Vries, Junyoung Chung, Yoshua Bengio (2015):
           RMSProp and equilibrated adaptive learning rates for non-convex optimization
           arXiv:1502.04390 http://arxiv.org/abs/1502.04390
    """

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
    """ Adadelta updates
    The learning rate is scaled by the ratio of accumulated gradients to the ratio of accumulated step sizes.

    Math:
    * ``accu_new = rho * accu + (1 - rho) * grad ** 2``
    * ``update = (grad * cgt.sqrt(delta_accu + epsilon) / cgt.sqrt(accu_new + epsilon))``
    * ``param = param - learning_rate * update``
    * ``delta_accu_new = rho * delta_accu + (1 - rho) * update ** 2``

    Parameters
    ----------
    cost : a scalar loss.
    params : a list of cgt shared variables. We generate update
            expressions w.r.t. these variables.
    learning_rate : float
        Tunes the size of the update step.
    rho : float
        Controls decay of gradient moving average.
    epsilon : float
        Avoid division by 0 while scaling. Small constant.

    Returns
    -------
    list of tuples of the form
    [(param, updates), (accumulated_grads, accumulated_grads_new), (step_accum, step_accum_new)]

    References
    ----------
    .. [1] Zeiler, M. D. (2012):
           ADADELTA: An Adaptive Learning Rate Method.
           arXiv Preprint arXiv:1212.5701.
    """
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
