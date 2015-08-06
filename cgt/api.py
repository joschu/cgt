
import operator
import numpy as np
import sys

# Every non-underscored function in this file should have a docstring, and it should enforce that the input data is valid

# ================================================================
# Variable Constructors
# ================================================================

def scalar(name=None, dtype=None, fixed_shape=None):
    return core.Argument(core.TensorType(cgt.floatX if dtype is None else dtype, 0), name, fixed_shape=fixed_shape)
def vector(name=None, dtype=None, fixed_shape=None):
    return core.Argument(core.TensorType(cgt.floatX if dtype is None else dtype, 1), name, fixed_shape=fixed_shape)
def matrix(name=None, dtype=None, fixed_shape=None):
    return core.Argument(core.TensorType(cgt.floatX if dtype is None else dtype, 2), name, fixed_shape=fixed_shape)
def tensor3(name=None, dtype=None, fixed_shape=None):
    return core.Argument(core.TensorType(cgt.floatX if dtype is None else dtype, 3), name, fixed_shape=fixed_shape)
def tensor4(name=None, dtype=None, fixed_shape=None):
    return core.Argument(core.TensorType(cgt.floatX if dtype is None else dtype, 4), name, fixed_shape=fixed_shape)

def tensor(dtype, ndim, name=None, fixed_shape=None):
    return core.Argument(core.TensorType(cgt.floatX if dtype is None else dtype, ndim), name, fixed_shape=fixed_shape)

# ================================================================
# Symbolic functions
# ================================================================

def add_multi(xs):
    return reduce(operator.add, xs)

def arange(start, stop=None, step=1, dtype=None):
    if (stop is None):
        (start, stop) = (0, start)
    if (dtype is None):
        dtype = 'i8'
    return core.Result(core.Arange(dtype), [start, stop, step])

def argmax(x, axis=None, keepdims=False):
    if (axis is None):
        out = flatten(x).argmax(axis=0)
    else:
        assert isinstance(axis, int)
        out = core.Result(core.Argmax(axis), [x])
        if (not keepdims):
            out = _dropdims(out, [axis])
    return out

def batched_matmul(x, y):
    return core.Result(core.BatchedMul22(False,False), [x,y])


def broadcast(opname, x, y, bcpat):
    (xpat, ypat) = bcpat.split(',')
    (xbcaxes, ybcaxes) = [[i for (i, letter) in enumerate(pat) if (letter == '1')] for pat in (xpat, ypat)]
    assert (x.get_ndim() == y.get_ndim())
    if xbcaxes:
        for i in xbcaxes: core.assertequal1(size(x,i), 1, "you mislabeled axis %i as singleton"%i)
        x = core.Result(core.Repeat(xbcaxes), [x] + [size(y, ax) for ax in xbcaxes])
    if ybcaxes:
        for i in ybcaxes: core.assertequal1(size(y,i), 1, "you mislabeled axis %i as singleton"%i)
        y = core.Result(core.Repeat(ybcaxes), [y] + [size(x, ax) for ax in ybcaxes])
    return elwise_binary(opname, x, y)

def _get_nu_cast(dtype):
    castfunc = np.cast[dtype]
    def _nu_cast(x, out=None):
        if out is None:
            return castfunc(x)
        else:
            out[...] = castfunc(x)
    return _nu_cast

def cast(x, dtype):
    x = core.as_node(x)
    if (x.dtype == dtype):
        return x
    else:
        diff = (core.dtype_kind(dtype) in 'cf')
        opname = 'cast_to_%s' % dtype
        ui = core.UnaryInfo(opname, _get_nu_cast(dtype), diff, dtype,  
            lambda x,y,gy : (cast(gy, x.dtype) if diff else core._nondiff()), 'x')
        return core.Result(core.ElwiseUnary(opname, ui), [x])

def ceil_divide(x, y):
    return iceil(x / y)

def concatenate(xs, axis=0):
    return core.Result(core.Concatenate(axis), xs)

def constant(val):
    return core.Result(core.Constant(val), [])

def dot(x, y):
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

def einsum(desc, x, y):
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

def elwise_binary(opname, x, y):
    (x, y) = map(core.as_node, (x, y))
    scalar_mask = ((x.ndim == 0), (y.ndim == 0))
    op = core.ElwiseBinary(opname, scalar_mask)
    if (scalar_mask == (False, False)):
        assert (x.ndim == y.ndim)
    return core.Result(op, [x, y])

def fill(val, shape):
    assert isinstance(shape, list)
    val = core.as_node(val)
    # if val is a constant, use a Fill Op, which includes the value as a attribute
    if isinstance(val.op, core.Constant):
        return core.Result(core.Fill(val.op.value), shape)
    # if val is a non-constant variable, we can use a Repeat Op
    else:
        singleton = reshape(val, [1]*len(shape))
        return core.Result(core.Repeat(range(len(shape))), [singleton] + shape)

def flatcat(xs):
    return concatenate([flatten(x) for x in xs])

def flatten(x):
    return reshape(x, [mul_multi(shape(x))])

def floor_divide(x, y):
    return ifloor(x / y)

def getitem(arr, slis):
    arr = core.as_node(arr)
    if isinstance(arr.get_type(), core.TupleType):
        assert isinstance(slis, int)
        return tuple_index(arr, slis)
    if (not _is_list_or_tuple(slis)):
        slis = [slis]
    if all(isinstance(sli, (int, slice, type(None))) for sli in slis):
        return getitem_nonfancy(arr, slis)
    elif all((isinstance(sli, (np.ndarray, core.Node)) for sli in slis)):
        return getitem_fancy(arr, slis)
    else:
        raise ValueError('Tried to index with slices %s. Either all should be in {slice,int,colon} or all must be a ndarray of ints' % str(slis))

def getitem_fancy(arr, indarrs):
    assert all(((indarr.ndim == 1) for indarr in indarrs))
    indarrs = map(core.as_node, indarrs)
    flatinds = sub2ind(indarrs, shape(arr))
    return core.Result(core.GetFlatIndices(), [arr, flatinds])

def getitem_nonfancy(arr, slis):
    out = arr
    ax = 0
    shapedesc = []
    if (not _is_list_or_tuple(slis)):
        slis = [slis]
    for sli in slis:
        if isinstance(sli, slice) and all(x is None for x in (sli.start, sli.stop, sli.step)):
            shapedesc.append(ax)
        elif (sli is None):
            shapedesc.append('+')
            ax -= 1
        elif isinstance(sli, bool):
            raise ValueError('tried to index with a bool')
        else:
            if isinstance(sli, slice):
                shapedesc.append('k')
            elif isinstance(sli, int):
                sli = slice(sli, sli + 1, 1)
                shapedesc.append('-')
            else:
                raise NotImplementedError
            start = (0 if sli.start is None else sli.start)
            stop = size(arr, ax) if (sli.stop is None) else sli.stop
            step = (1 if sli.step is None else sli.step)
            if (isinstance(stop, int) and (stop < 0)):
                stop = size(arr, ax) - stop
            out = core.Result(core.GetSli(ax), [out, start, stop, step])
        ax += 1
    if all(((x == 'k') for x in shapedesc)):
        return out
    else:
        axidx = 0
        newshape = []
        for d in shapedesc:
            if (d == '+'):
                newshape.append(1)
            elif (d == '-'):
                axidx += 1
            else:
                newshape.append(size(out, axidx))
                axidx += 1
        for axidx in xrange(axidx, out.ndim):
            newshape.append(size(out, axidx))
        out = reshape(out, newshape)
    return out

def irfft(x, axes):
    return core.Result(core.IRFFT(axes),[x])

def make_tuple(*xs):
    return core.Result(core.MakeTuple(), list(xs))

def max(x, axis=None, keepdims=False): #pylint: disable=W0622
    axes = _red_axes(axis, x.ndim)
    out = core.Result(core.Max(axes), [x])
    if (not keepdims):
        out = _dropdims(out, axes)
    return out

def mean(x, axis=None, keepdims=False):
    axes = _red_axes(axis, x.ndim)
    return sum(x, axis=axes, keepdims=keepdims) / mul_multi([size(x, ax) for ax in axes])

def mul_multi(xs):
    return reduce(operator.mul, xs) if (len(xs) > 0) else constant(np.array(1, dtype='i8'))

def norm(x, axis=None, p=2, keepdims=False):
    if p==2:
        return sqrt(square(x).sum(axis=axis,keepdims=keepdims))
    else:
        return pow(pow(x, p).sum(axis=axis,keepdims=keepdims), 1.0 / p)

def outer(x, y):
    assert (x.ndim == y.ndim == 1)
    return core.Result(core.Outer(), [x, y])

def _validate_shape(shp,funcname):
    if len(shp)==1 and isinstance(shp[0],tuple):
        raise ValueError("you called %s(x) where x is a tuple. You should call %s(a,b,c...) instead."%(funcname,funcname))

def rand(*shp):
    _validate_shape(shp,"rand")
    shp = map(core.as_node, shp)
    return core.Result(core.ScalarRng('uniform'), shp)

def randn(*shp):
    _validate_shape(shp,"randn")
    shp = map(core.as_node, shp)
    return core.Result(core.ScalarRng('gaussian'), shp)

def real(x):
    assert (core.dtype_kind(x.dtype) == 'c')
    return cast(x, cgt.floatX)

def repeat(x, repeats, axis):
    return core.Result(core.Repeat([axis]), [x, constant(repeats)])

def reshape(x, shp):
    shp = map(core.as_node, shp)
    assert all(s.ndim==0 and core.dtype_kind(s.dtype)=='i' for s in shp)
    return core.Result(core.Reshape(), [x] + list(shp))

def rfft(x, periods, axes):
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

def shape(x):
    x = core.as_node(x)
    typ = x.get_type()
    if isinstance(typ, core.TensorType):
        return [size(x, i) for i in xrange(x.ndim)]
    else:
        return tuple(map(shape, x.parents))

def shared(val, name='', device=None, fixed_shape_mask=None):
    return core.Data(val, name=name, device=device, fixed_shape_mask=fixed_shape_mask)

def size(x, axis):
    return core.Result(core.Size(axis), [x])

def stack(scalars):
    assert (len(scalars) > 0)
    return core.Result(core.Stack(), scalars)

def sub2ind(subs, shp):
    ndim = len(shp)
    assert ndim >= 1
    strides = [None]*(ndim-1) + [1]
    for i in xrange(ndim-2, -1, -1):
        strides[i] = shp[i+1] * strides[i+1]
    return add_multi([stride*sub for (stride,sub) in utils.safezip(strides, subs)])

def prod(x, axis=None, keepdims=False):
    return cgt.exp(cgt.sum(cgt.log(x), axis=axis, keepdims=keepdims))

def sum(x, axis=None, keepdims=False): #pylint: disable=W0622
    if x.dtype == 'i1':
        utils.warn("Called sum() on a one-byte integer, so you risk overflow. Might want to cast to float.")
    axes = _red_axes(axis, x.ndim)
    if (len(axes) == 0):
        return x
    out = core.Result(core.Sum(axes), [x])
    if (not keepdims):
        out = _dropdims(out, axes)
    return out

def transport(x, src, targ):
    return core.Result(core.Transport(src, targ), [x])

def transpose(arr, axes=None):
    if axes is None: 
        assert arr.ndim == 2
        axes = [1,0]
    else:
        assert _is_list_or_tuple(axes) and len(axes) == arr.get_ndim()
        axes = list(axes)
    if axes == range(arr.get_ndim()):
        return arr
    else:
        return core.Result(core.Transpose(axes), [arr])

def tuple_index(x, i):
    return core.Result(core.TupleIndex(i), [x])

def zeros(shape, dtype=None): #pylint: disable=W0621
    if (dtype is None):
        dtype = cgt.floatX
    return core.Result(core.Fill(np.array(0, dtype)), shape)

def zeros_like(x):
    return zeros(shape(x), x.get_dtype())

def _dropdims(x, axes):
    return reshape(x, [size(x, i) for i in xrange(x.get_ndim()) if (i not in axes)])

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


if sys.argv[0] != "gen_py.py":
    from api_autogen import *
import cgt
from . import core, utils