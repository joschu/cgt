
import operator
import numpy as np
import sys
if sys.argv[0] != "gen_py.py":
    from api_autogen import *
import cgt
from . import core, utils
# Every non-underscored function in this file should have a docstring, and it should enforce that the input data is valid

# ================================================================
# Variable Constructors
# ================================================================

_tensor_doc_template = """
Creates a symbolic variable representing a %s argument (i.e., a tensor of rank %s).

Inputs
------
name: (optional) string name of this variable, which will be displayed by printing functions.
dtype: string (e.g., 'float','int', 'float32') or numpy dtype object. Note that float precision
    will be ignored and cgt.floatX will be used.
fixed_shape: a tuple of either int or None, e.g., (None, 3, 10), representing the known
    shape components of this argument. This argument allows CGT to infer the shape of
    variables depending on this one, and also apply optimization that depend on the shape
    being known.
"""

def scalar(name=None, dtype=None, fixed_shape=None):    
    return core.Argument(core.TensorType(cgt.floatX if dtype is None else dtype, 0), name, fixed_shape=fixed_shape)
scalar.__doc__ = _tensor_doc_template%("scalar",0)
def vector(name=None, dtype=None, fixed_shape=None):
    return core.Argument(core.TensorType(cgt.floatX if dtype is None else dtype, 1), name, fixed_shape=fixed_shape)
vector.__doc__ = _tensor_doc_template%("vector",1)
def matrix(name=None, dtype=None, fixed_shape=None):
    return core.Argument(core.TensorType(cgt.floatX if dtype is None else dtype, 2), name, fixed_shape=fixed_shape)
matrix.__doc__ = _tensor_doc_template%("matrix",2)
def tensor3(name=None, dtype=None, fixed_shape=None):
    return core.Argument(core.TensorType(cgt.floatX if dtype is None else dtype, 3), name, fixed_shape=fixed_shape)
tensor3.__doc__ = _tensor_doc_template%("3-tensor",3)
def tensor4(name=None, dtype=None, fixed_shape=None):
    return core.Argument(core.TensorType(cgt.floatX if dtype is None else dtype, 4), name, fixed_shape=fixed_shape)
tensor4.__doc__ = _tensor_doc_template%("4-tensor",4)

def tensor(dtype, ndim, name=None, fixed_shape=None):
    return core.Argument(core.TensorType(cgt.floatX if dtype is None else dtype, ndim), name, fixed_shape=fixed_shape)
tensor.__doc__ = _tensor_doc_template%("k-tensor","k")

# ================================================================
# Symbolic functions
# ================================================================

def add_multi(xs):
    """
    xs -> xs[0] + xs[1] + ... + xs[len(xs)-1]
    """
    return reduce(operator.add, xs)

def arange(start, stop=None, step=1, dtype=None):
    """
    Like numpy.arange, but arguments can be symbolic
    """
    if (stop is None):
        (start, stop) = (0, start)
    if (dtype is None):
        dtype = 'i8'
    return core.Result(core.Arange(dtype), [start, stop, step])

def argmax(x, axis=None, keepdims=False):
    """
    Like numpy.argmax, but arguments can be symbolic
    """
    if (axis is None):
        out = flatten(x).argmax(axis=0)
    else:
        assert isinstance(axis, int)
        out = core.Result(core.Argmax(axis), [x])
        if (not keepdims):
            out = _dropdims(out, [axis])
    return out

def batched_matmul(x, y):
    r"""
    Given two 3-tensors x_nij, and y_njk, loop over 'n' and contract along 'j'
        x_nij, y_njk -> z_nik := \sum_j x_nij y_njk
    A variety of useful tensor contraction operations can be written in this form
    after permuting axes and reshaping.
    """
    return core.Result(core.BatchedMul22(False,False), [x,y])


def broadcast(opname, a, b, bcpat):
    """
    Perform elementwise binary operation such as addition or multiplication, and expand
    singleton dimensions when appropriate.

    opname: string name of operation: *,+,-,/,<,>,<=,>=,**,==,!=
    a, b: variables
    bcpat: a string of x,1 specifying which dimensions are singletons in both a and b. Here are some examples:
        "x1,1x":        a.shape[1] == 1 and b.shape[0] == 1
        "xx1,xxx":      a.shape[2] == 1, but we should have a.shape[0]==b.shape[0] and a.shape[1]==b.shape[1]

    E.g., here's an example of using this function
    a = np.zeros((2,3))
    b = np.zeros((2,1))
    z = cgt.broadcast("+", a, b, "xx,x1")


    """
    x,y = a,b # switched x,y -> a,b so 'x' in bcpat would be less confusing
    (xpat, ypat) = bcpat.split(',')
    (xbcaxes, ybcaxes) = [[i for (i, letter) in enumerate(pat) if (letter == '1')] for pat in (xpat, ypat)]
    assert (x.ndim == y.ndim)
    if xbcaxes:
        # for i in xbcaxes: core.assertequal1(size(x,i), 1, "you mislabeled axis %i as singleton"%i) # @SHAPE_CHECK
        x = core.Result(core.Repeat(xbcaxes), [x] + [size(y, ax) for ax in xbcaxes])
    if ybcaxes:
        # for i in ybcaxes: core.assertequal1(size(y,i), 1, "you mislabeled axis %i as singleton"%i) # @SHAPE_CHECK
        y = core.Result(core.Repeat(ybcaxes), [y] + [size(x, ax) for ax in ybcaxes])
    return core.elwise_binary(opname, x, y)

def _get_nu_cast(dtype):
    castfunc = np.cast[dtype]
    def _nu_cast(x, out=None):
        if out is None:
            return castfunc(x)
        else:
            out[...] = castfunc(x)
    return _nu_cast

def cast(x, dtype):
    """
    Convert variable x to the desired datatype
    """
    x = core.as_node(x)
    if (x.dtype == dtype):
        return x
    else:
        diff = (core.dtype_kind(dtype) in 'cf')
        opname = 'cast_to_%s' % dtype
        ui = core.UnaryInfo(opname, _get_nu_cast(dtype), diff, dtype,  
            lambda x,y,gy : cast(gy, x.dtype) if diff else None, 'x')
        return core.Result(core.ElwiseUnary(opname, ui), [x])

def ceil_divide(x, y):
    return iceil(x / y)

def concatenate(xs, axis=0):
    """
    Like np.concatenate
    xs: a list of variables with the same shape along all axes other than `axis`
    """
    return core.Result(core.Concatenate(axis), xs)

def constant(val):
    """
    creates a symbolic expression with constant value
    val: numpy array or python scalar
    """
    if isinstance(val, tuple):
        val = core.as_valid_tuple(val)
        op = core.ConstantTuple(val)
    else:
        val = core.as_valid_array(val)
        op = core.ConstantTensor(val)
    return core.Result(op, [])

def dot(x, y):
    """
    Like numpy.dot
    x,y: variables
    """
    x = core.as_node(x)
    y = core.as_node(y)
    xdim = x.ndim
    ydim = y.ndim
    if (xdim == 1):
        if (ydim == 1):
            return core.Result(core.Dot(), [x, y])
        elif (ydim == 2):
            return core.Result(core.Mul21(True), [y, x])
        else:
            raise NotImplementedError
    elif (xdim == 2):
        if (ydim == 1):
            return core.Result(core.Mul21(False), [x, y])
        elif (ydim == 2):
            return core.Result(core.Mul22(False, False), [x, y])
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

def diag(v, k=0):
    """
    see numpy.diag
    """
    assert isinstance(k, int)
    assert v.ndim == 1
    n = size(v,0)+abs(k)
    out = cgt.zeros((n,n), v.dtype)
    out = inc_subtensor(out, (cgt.arange(n), cgt.arange(n)+k), v)
    return out


def einsum(desc, x, y):
    """
    Like numpy.einsum except x and y are symbolic variables
    desc: string like "nij,njk->nik"
    x,y: symbolic variables
    """
    import re
    pat = '(\\w+),(\\w+)->(\\w+)'
    match = re.match(pat, desc)
    if (match is None):
        raise ValueError('einsum error: desc should match regexp %s' % pat)
    (xdesc, ydesc, zdesc) = match.groups()
    if ((not _is_unique(xdesc)) and _is_unique(ydesc) and _is_unique(zdesc)):
        raise ValueError('Invalid tensor description %s passed into einsum. Tensor indices should be unique' % desc)
    if (not _is_subset_of(xdesc + ydesc, zdesc)):
        raise ValueError('Invalid tensor description %s passed into einsum. Unrecognized index in output.' % desc)
    loop = []
    justx = []
    contr = []
    justy = []
    for c in xdesc:
        if (c in ydesc):
            if (c in zdesc):
                loop.append(c)
            else:
                contr.append(c)
        else:
            justx.append(c)
    for c in ydesc:
        if (not (c in xdesc)):
            justy.append(c)
    (ixloop, ijustx, ixcontr) = [[xdesc.index(c) for c in chars] for chars in [loop, justx, contr]]
    (iyloop, ijusty, iycontr) = [[ydesc.index(c) for c in chars] for chars in [loop, justy, contr]]
    xshp = shape(x)
    yshp = shape(y)
    xt = transpose(x, ixloop + ijustx + ixcontr).reshape([mul_multi([xshp[i] for i in icol]) for icol in [ixloop, ijustx, ixcontr]])
    yt = transpose(y, iyloop + iycontr + ijusty).reshape([mul_multi([yshp[i] for i in icol]) for icol in [iyloop, iycontr, ijusty]])
    zt = batched_matmul(xt, yt)
    return transpose(zt.reshape([size(x, xdesc.index(c)) for c in loop] + [size(x, xdesc.index(c)) for c in justx] + [size(y, ydesc.index(c)) for c in justy]), utils.invert_perm([zdesc.index(c) for c in loop + justx + justy]))

def fill(val, shape):
    """
    Create an array of shape `shape` filled with scalar `val`
    """
    assert isinstance(shape, list)
    val = core.as_node(val)
    # if val is a constant, use a Fill Op, which includes the value as a attribute
    if isinstance(val.op, core.Constant):
        return core.Result(core.Fill(val.op.value), shape)
    # if val is a non-constant variable, we can use a Repeat Op
    else:
        singleton = reshape(val, [1]*len(shape))
        return core.Result(core.Repeat(range(len(shape))), [singleton] + shape)

def flatten(x):
    """
    Like numpy.flatten
    """
    return reshape(x, [mul_multi(shape(x))])

def flip(x, axes):
    """
    Reverse array along specified axes
    e.g. 
    flip(x,0) == x[::-1], 
    flip(x, 1) == x[:,::-1]
    """
    x = core.as_node(x)
    assert isinstance(axes, list)
    return core.Result(core.Flip(axes), [x])

def floor_divide(x, y):
    """
    returns floor(x/y), with integer dtype
    """
    return ifloor(x / y)


def _iscolon(sli):
    return isinstance(sli, slice) and sli.start is None and sli.stop is None and sli.step is None


def subtensor(x, slis):
    """
    Index or slice array `x` with the sequence of slices `slis`
    Corresponds to the numpy code x[slis]

    x : cgt.Node or ndarray
    slis : list of slices, which is one of the following format:

    1. sequence of ":" or None
    2. sequence of ":", slice, or int
    3. sequence of intarray of length x.ndim
    4. sequence of intarray or ":", with a single intarray

    where intarray denotes either an
    - cgt node with ndim=1 and dtype=int
    - numpy ndarray with ndim=1 and dtype=int

    """
    return _subtensor(x, slis)

def inc_subtensor(x, slis, y):
    """
    Returns the array that is obtained by incrementing x[slis] by y
    This function corresponds to the following numpy code:
        out = x.copy()
        out[slis] += y
        return out
    Note that due to an in-place optimization, the copy operation is
    usually not performed.

    See subtensor docstring for a list of appropriate formats for `slis`
    Only formats 2-4 are allowed for inc_subtensor
    """
    return _subtensor(x, slis, y)

def _subtensor(x, slis, y=None):
    """
    To reduce code duplication, we use the same underlying function for
    subtensor and inc_subtensor
    if y is None, we're just slicing
    otherwise, we're incrementing the slice
    """

    x = core.as_node(x)
    if y is not None: y = core.as_node(y)
    assert x.is_tensor()

    # Make sure slis is an int
    if not isinstance(slis, (list, tuple)):
        slis = [slis]
    else:
        slis = list(slis)


    # Case 1. (see subtensor docstring)
    if all(_iscolon(sli) or sli is None  for sli in slis):
        assert y is None
        return _subtensor1(x, slis)
    # Case 2.
    elif all(_iscolon(sli) or isinstance(sli, slice) or _is_int_scalar(sli) for sli in slis):
        if y is not None:
            # check that slice is only along one axis:
            assert np.sum([not _iscolon(sli) for sli in slis])==1, "currently can only increment subtensor involving a slice along one axis"
        return _subtensor2(x, slis, y)
    # Case 3.
    elif len(slis) == x.ndim and all(_is1dintarray(sli) for sli in slis):
        return _subtensor3(x, slis, y)
    # Case 4.
    elif all(_is1dintarray(sli) or _iscolon(sli) for sli in slis) and np.sum([_is1dintarray(sli) for sli in slis])==1:        
        return _subtensor4(x, slis, y)
    else:
        raise ValueError('Tried to index with slices %s. See cgt.subtensor docstring for description of valid indexing expressions'%slis)

def _is_int_scalar(x):
    return isinstance(x,int) \
            or np.isscalar(x) and x.dtype.kind=='i' \
            or (isinstance(x, core.Node) and x.ndim == 0 and core.dtype_kind(x.dtype)=='i')
     
def _subtensor1(x, slis):

    axidx = 0 # number of axes from x matched to ":"s so far
    newshape = []

    for sli in slis:
        if _iscolon(sli):
            newshape.append(size(x, axidx))
            axidx += 1
        else:
            assert sli is None
            newshape.append(1)

    return x.reshape(newshape)

def _subtensor2(x, slis, y):

    dims2drop = []
    for (ax,sli) in enumerate(slis):
        if _is_int_scalar(sli):
            sli = slice(sli, sli + 1, 1)
            dims2drop.append(ax)

        assert isinstance(sli.step, int) or sli.step is None
        step = 1 if sli.step is None else sli.step

        if step < 0:
            start = size(x, ax)-1 if sli.start is None else sli.start
            stop = -1 if sli.stop is None else sli.stop
        else:
            start = 0 if sli.start is None else sli.start
            stop = size(x, ax) if sli.stop is None else sli.stop

        assert isinstance(step, (int, core.Node)), "step argument of a slice should be an integer or a symbolic variable"

        if y is None:
            x = core.Result(core.GetSli(ax), [x, start, stop, step])
        else:
            x = core.Result(core.IncSli(ax), [x, start, stop, step, y])

    x = _dropdims(x, dims2drop)

    return x

def _subtensor3(x, slis, y):
    """
    Index x with x.ndim arrays of int.
    """
    assert all(((indarr.ndim == 1) for indarr in slis))
    slis = map(core.as_node, slis)
    flatinds = sub2ind(slis, shape(x))
    if y is None:
        return core.Result(core.GetFlatIndices(), [x, flatinds])
    else:
        return core.Result(core.IncFlatIndices(), [x, flatinds, y])

def _subtensor4(x, slis, y):
    for (ax,sli) in enumerate(slis):
        if _is1dintarray(sli):
            if y is None:
                return core.Result(core.GetFancySli(ax), [x, sli])
            else:
                return core.Result(core.IncFancySli(ax), [x, sli, y])
    assert 0, "should be unreachable"




def _is1dintarray(x):
    if isinstance(x, core.Node) and x.ndim == 1 and core.dtype_kind(x.dtype)=='i':
        return True
    elif isinstance(x, np.ndarray) and x.ndim == 1 and x.dtype.kind == 'i':
        return True
    else:
        return False


def irfft(x, axes):
    """
    like np.fft.irfft
    """
    return core.Result(core.IRFFT(axes),[x])

def make_tuple(*xs):
    """
    Create a symbolic tuple variable out of a collection of symbolic variables
    """
    return core.Result(core.MakeTuple(), list(xs))

def max(x, axis=None, keepdims=False): #pylint: disable=W0622
    """
    Like numpy.max
    """
    axes = _red_axes(axis, x.ndim)
    out = core.Result(core.Max(axes), [x])
    if (not keepdims):
        out = _dropdims(out, axes)
    return out

def mean(x, axis=None, keepdims=False):
    """
    Like numpy.mean
    """    
    if x.dtype == 'i1': x = cgt.cast(x, cgt.floatX)
    axes = _red_axes(axis, x.ndim)
    return sum(x, axis=axes, keepdims=keepdims) / mul_multi([size(x, ax) for ax in axes])

def mul_multi(xs):
    """
    returns xs[0] * xs[1] * ... * xs[len(xs)-1]
    """
    return reduce(operator.mul, xs) if (len(xs) > 0) else constant(np.array(1, dtype='i8'))

def norm(x, axis=None, p=2, keepdims=False):
    """
    Computes p-norm of vectors formed by varying `axis`
    """
    if p==2:
        return sqrt(square(x).sum(axis=axis,keepdims=keepdims))
    else:
        return pow(pow(x, p).sum(axis=axis,keepdims=keepdims), 1.0 / p)

def ones(shape, dtype=None): #pylint: disable=W0621
    """
    Like numpy.ones
    """
    if (dtype is None):
        dtype = cgt.floatX
    return core.Result(core.Fill(np.array(1, dtype)), shape)

def ones_like(x):
    """
    Like numpy.ones_like
    """
    return ones(shape(x), x.dtype)

def outer(x, y):
    """
    Like numpy.outer
    """
    assert (x.ndim == y.ndim == 1)
    return core.Result(core.Outer(), [x, y])

def _validate_shape(shp,funcname):
    if len(shp)==1 and isinstance(shp[0],tuple):
        raise ValueError("you called %s(x) where x is a tuple. You should call %s(a,b,c...) instead."%(funcname,funcname))

def rand(*shp):
    """
    Like numpy.random.rand
    """
    _validate_shape(shp,"rand")
    shp = map(core.as_node, shp)
    return core.Result(core.ScalarRng('uniform'), shp)

def randn(*shp):
    """
    Like numpy.random.randn
    """
    _validate_shape(shp,"randn")
    shp = map(core.as_node, shp)
    return core.Result(core.ScalarRng('gaussian'), shp)

def real(x):
    """
    Like numpy.real
    """
    assert (core.dtype_kind(x.dtype) == 'c')
    return cast(x, cgt.floatX)

def repeat(x, repeats, axis):
    """
    Like numpy.repeat
    """
    return core.Result(core.Repeat([axis]), [x, core.as_node(repeats)])

def reshape(x, shp):
    """
    Like numpy.reshape
    """
    shp = map(core.as_node, shp)
    assert all(s.ndim==0 and core.dtype_kind(s.dtype)=='i' for s in shp)
    return core.Result(core.Reshape(), [x] + list(shp))

def rfft(x, periods, axes):
    """
    Like numpy.rfft
    """
    return core.Result(core.RFFT(axes),[x]+list(periods))

def set_precision(prec):
    """
    prec in {"single", "double"}
    globally set floating point precision for float and complex types
    """    
    assert prec in ("half","single", "double","quad")
    if prec == "half":
        cgt.floatX = 'f2'
        cgt.complexX = None
        utils.warn("half precision not yet supported")
    elif prec == "single":
        cgt.floatX = 'f4'
        cgt.complexX = 'c8'
    elif prec == "double":
        cgt.floatX = 'f8'
        cgt.complexX = 'c16'
    elif prec == "quad":
        cgt.floatX = 'f16'
        cgt.complexX = 'c32'

def get_precision():
    """
    Returns the current global precision, "half","single","double","quad"
    """
    return {"f2":"half","f4":"single","f8":"double","f16":"quad"}[cgt.floatX]

def shape(x):
    """
    Like numpy.shape
    """
    x = core.as_node(x)
    if isinstance(x.typ, core.TensorType):
        return [size(x, i) for i in xrange(x.ndim)]
    else:
        return tuple(map(shape, x.parents))

def shared(value, name=None, device=None, fixed_shape_mask=None):
    """
    Creates a variable that has an underlying data value, which can be changed externally
    """
    op = core.InMemoryData(value, device=device,fixed_shape_mask=fixed_shape_mask)
    return core.Result(op, [], name=name)

def size(x, axis):
    """
    size(x, axis) == x.shape[axis]
    """
    return core.Result(core.Size(axis), [x])

def stack(tensors, axis=0):
    """
    Take a sequence of tensors and stack them on given axis to make a single
    tensor. The size in dimension `axis` of the result will be equal to the number
    of tensors passed.

    tensors : a list of tensor or scalar variables with the same shape
    along all axes
    axis : an int specifying the index of the new axis, default value is 0

    If a, b, c have shape (2, 2, 2) then stack([a,b,c], axis=0) builds
    a tensor with shape (3, 2, 2, 2) and stack([a,b,c], axis=1) builds
    a tensor with shape (2, 3, 2, 2)
    """
    assert (len(tensors) > 0)
    assert 0 <= axis and axis <= tensors[0].ndim
    shp = shape(tensors[0])
    newshp = shp[:axis] + [1] + shp[axis:]
    return concatenate([reshape(t, newshp) for t in tensors], axis=axis)

def sub2ind(subs, shp):
    """
    Like matlab sub2ind
    """
    ndim = len(shp)
    assert ndim >= 1
    strides = [None]*(ndim-1) + [1]
    for i in xrange(ndim-2, -1, -1):
        strides[i] = shp[i+1] * strides[i+1]
    return add_multi([stride*sub for (stride,sub) in utils.safezip(strides, subs)])

def prod(x, axis=None, keepdims=False):
    """
    Like numpy.prod
    """
    return cgt.exp(cgt.sum(cgt.log(x), axis=axis, keepdims=keepdims))

def sum(x, axis=None, keepdims=False): #pylint: disable=W0622
    """
    Like numpy.sum
    """
    if x.dtype == 'i1':
        utils.warn("Called sum() on a one-byte integer, so you risk overflow. Might want to cast to float.")
    axes = _red_axes(axis, x.ndim)
    if (len(axes) == 0):
        return x
    out = core.Result(core.Sum(axes), [x])
    if (not keepdims):
        out = _dropdims(out, axes)
    return out

# def transport(x):
#     return core.Result(core.Transport(), [x])

def transpose(arr, axes=None):
    """
    Like numpy.transpose
    """    
    if axes is None: 
        assert arr.ndim == 2
        axes = [1,0]
    else:
        assert _is_list_or_tuple(axes) and len(axes) == arr.ndim
        axes = list(axes)
    if axes == range(arr.ndim):
        return arr
    else:
        return core.Result(core.Transpose(axes), [arr])

def tuple_index(x, i):
    """
    If x is a symbolic variable with isinstance(x.typ, TupleType), return x[i]
    """
    return core.Result(core.TupleIndex(i), [x])

def zeros(shape, dtype=None): #pylint: disable=W0621
    """
    Like numpy.zeros
    """
    if (dtype is None):
        dtype = cgt.floatX
    return core.Result(core.Fill(np.array(0, dtype)), shape)

def zeros_like(x):
    """
    Like numpy.zeros_like
    """
    return zeros(shape(x), x.dtype)

def _dropdims(x, axes):
    return reshape(x, [size(x, i) for i in xrange(x.ndim) if (i not in axes)])

def _is_list_or_tuple(xs):
    return isinstance(xs, (list, tuple))

def _is_subset_of(maybesub, bigset):
    return (len(set(bigset).difference(set(maybesub))) == 0)

def _is_unique(col):
    return (len(set(col)) == len(col))

def _red_axes(axis, ndim):
    if (axis is None):
        return range(ndim)
    elif isinstance(axis, int):
        return [axis]
    elif isinstance(axis, (list, tuple)):
        return list(axis)
    else:
        raise ValueError("invalid argument 'axis'=%s" % axis)

def dimshuffle(arr, pattern):
    """
    Based on theano's dimshuffle function.
    Permutes the axes of an array and inserts new singleton axes.
    e.g. if x has shape (2,3), then dimshuffle(x, [1,0,'x']) has shape (3,2,1)
    note that dimshuffle can not drop dimensions.
    
    inputs
    ------
    pattern: list of ints and 'x'
    arr: symbolic array to dimshuffle

    returns
    --------
    variable with permuted axes
    """
    assert isinstance(pattern, list)
    if pattern == range(arr.ndim):
        return arr
    else:
        shapes = []
        array_shape = arr.shape
        x_idxs = []
        clean_pattern = []
        for idx in range(0, len(pattern)):
            if pattern[idx] == 'x':
                shapes.append(1)
                x_idxs.append(idx)
            else:
                shapes.append(array_shape[pattern[idx]])
                clean_pattern.append(pattern[idx])
        assert len(clean_pattern) == arr.ndim
        result = transpose(arr, clean_pattern)
        if x_idxs:
            result = reshape(result, tuple(shapes))
        return result

