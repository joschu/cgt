import sys, numpy as np, hashlib, copy, cPickle, ctypes, warnings, os
from collections import defaultdict,namedtuple
import __builtin__
import traceback
import cgt
from . import utils

# ================================================================
# Datatypes
# ================================================================

class Dtype: #pylint: disable=W0232
    @staticmethod
    def canon(dt):
        """
        Return canonical string representation of dtype,
        using the floating point type that CGT is currently configured for

        The following string representations are used: i1,i2,i4,i8,  f4,f8,  c8,c16
        Either all floats are single precision or all are double.
        So either we're using single (f4, c8) or double (f8, c16).
        """
        dt = np.dtype(dt)
        if dt.char in 'fdg':
            return cgt.floatX
        elif dt.char in 'iulBb?':
            return 'i'+str(dt.itemsize)
        elif dt.char in 'FDG':
            return 'c'+str(dt.itemsize)
        else:
            raise ValueError("Invalid dtype %s"%dt)

def as_valid_array(x, dtype=None):
    x = np.asarray(x)
    x = x.astype(Dtype.canon(x.dtype) if dtype is None else dtype)
    return x

def as_valid_tuple(x):
    return tuple(as_valid_array(a) for a in x)

class Type(object):
    """
    Represents a datatype for Nodes
    """
    pass

class TensorType(Type):
    """
    Type used to represent computation results (Nodes in the graph) 
    that are n-dimensional arrays. 
    Scalars are represented as zero-dimensional arrays
    [though we may create a scalar type later for efficiency]
    """
    def __init__(self, dtype, ndim):
        self.dtype = Dtype.canon(dtype)
        self.ndim = ndim
    def __str__(self):
        return "Tensor(%s,%s)"%(self.dtype, self.ndim)
    def __eq__(self, other):
        return self.dtype == other.dtype and self.ndim == other.ndim

class TupleType(Type):
    """
    A compount type
    For now, we will only allow tuples of tensors, though we may allow tuple elements in the future
    """
    def __init__(self, *eltypes):
        assert all(isinstance(eltype, TensorType) for eltype in eltypes)
        self.eltypes = eltypes
        self.dtype = 'O'
    def __len__(self):
        return len(self.eltypes)
    def __getitem__(self, i):
        return self.eltypes[i]
    def __iter__(self):
        return iter(self.eltypes)
    def __str__(self):
        return "Tup(" + ",".join(map(str,self.eltypes))+")"
    def __eq__(self, other):
        return len(self.eltypes) == len(other.eltypes)\
            and all(typ0 == typ1 for (typ0, typ1) in zip(self.eltypes, other.eltypes))

class Device(object):
    """
    Represents a location where a computation is performed
    machine: which computer on a network
    devtype: cpu vs gpu
    idx: which gpu, or possibly which process
    """
    def __init__(self, machine="default", devtype="cpu", idx=0):
        assert isinstance(machine,str) and isinstance(devtype,str) and isinstance(idx,int)
        self.machine = machine
        self.devtype = devtype
        self.idx = idx
    def __eq__(self, other):
        return self.machine == other.machine and self.devtype == other.devtype and self.idx == other.idx
    def __str__(self):
        return "%s/%s/%s"%(self.machine,self.devtype,self.idx)

def _promote(typ1, typ2):
    """
    Output type of a floating point operation involving these input types
    """
    d1 = typ1[0]
    s1 = typ1[1:]
    d2 = typ2[0]
    s2 = typ2[1:]
    if d1 == 'c' or d2 == 'c':
        return cgt.complexX
    elif d1 == 'f' or d2 == 'f': 
        return cgt.floatX
    elif d1 == 'i' and d2 == 'i':
        assert d1 == d2
        return d1 + __builtin__.max(s1,s2)
    else:
        raise ValueError("Don't know what to do with dtypes %s,%s"%(typ1, typ2))

def _promote_multi(xtypes):
    return reduce(_promote, xtypes)

def dtype_kind(dtype):
    """
    one of f,c,i
    """
    return dtype[0]

def _dtype_itemsize(dtype):
    """
    size in bytes
    """
    return int(dtype[1:])

def _type_to_int(typ1):
    """
    integer type of result of operation such as floor that converts to integer
    """
    d1 = dtype_kind(typ1)
    if d1 == 'f' or d1 == 'c':
        return 'i8'
    else:
        return typ1

# ================================================================
# Computation Graph Nodes
# ================================================================

class Node(object):
    """
    Node in the computation graph    
    """
    def __init__(self, typ, op, parents):
        self.typ = typ
        self.op = op
        self.parents = parents
    def is_input(self):
        return False
    def is_argument(self):
        return False
    def is_data(self):
        return False
    def get_type(self):
        return self.typ
    def get_diff(self):
        return [] if self.op is None else self.op.get_diff(len(self.parents))
    # Unary ops
    def __neg__(self):
        return Result(ElwiseUnary("neg"), [self])
    # Binary ops
    def __add__(self, other):
        return elwise_binary("+", self, other)
    def __sub__(self, other):
        return elwise_binary("-", self, other)
    def __mul__(self, other):
        return elwise_binary("*", self, other)
    def __div__(self, other):
        return elwise_binary("/", self, other)
    def __truediv__(self, other):
        return self.__div__(other)
    def __pow__(self, other):
        return elwise_binary("**", self, other)
    def __floordiv__(self, other):
        return cgt.floor_divide(self, other)
    def __gt__(self, other):
        return cgt.greater(self, other)
    def __ge__(self, other):
        return cgt.greater_equal(self, other)
    def __lt__(self, other):
        return cgt.less(self, other)
    def __le__(self, other):
        return cgt.less_equal(self, other)
    # def __eq__(self, other):
    #     return equal(self, other)
    # def __ne__(self, other):
    #     return not_equal(self, other)

    # TODO make these all match
    def __radd__(self, other):
        return self.__add__(other)
    def __rsub__(self, other):
        return cgt.constant(other).__sub__(self)
    def __rmul__(self, other):
        return self.__mul__(other)
    def __rdiv__(self, other):
        return cgt.constant(other).__div__(self)
    def __rtruediv__(self, other):
        return cgt.constant(other).__rtruediv__(self)
    def __rfloordiv__(self, other):
        return cgt.constant(other).__floordiv__(self)

    @property
    def shape(self):
        return cgt.shape(self)
    @property
    def ndim(self):
        return self.typ.ndim if isinstance(self.typ, TensorType) else 0
    @property
    def dtype(self):
        return self.typ.dtype
    @property
    def T(self):
        return cgt.transpose(self)
    
    __array_priority__ = 1000 # precedence over numpy operators

    def __getitem__(self, slis):
        return cgt.getitem(self, slis)
    def __iter__(self):
        raise TypeError("Node is not iterable")
    def __len__(self):
        if isinstance(self.typ, TupleType):
            return len(self.typ)
        else:
            raise ValueError("Node of type Tensor has no __len__")

    def reshape(self, shp):
        assert isinstance(shp, (list,tuple))
        return cgt.reshape(self, shp)
    def dot(self, other):
        return cgt.dot(self, other)
    def sum(self, axis=None, keepdims=False):
        return cgt.sum(self, axis=axis, keepdims=keepdims)
    def prod(self, axis=None, keepdims=False):
        return cgt.prod(self, axis=axis, keepdims=keepdims)
    def max(self, axis=None, keepdims=False):
        return cgt.max(self, axis=axis, keepdims=keepdims)
    def argmax(self, axis=None, keepdims=False):
        return cgt.argmax(self, axis=axis, keepdims=keepdims)
    def mean(self, axis=None, keepdims=False):
        return cgt.mean(self, axis=axis, keepdims=keepdims)
    def transpose(self, axes=None):
        return cgt.transpose(self, axes=axes)
    def flatten(self):
        return cgt.flatten(self)

def _ndarray_type(value):
    assert isinstance(value, np.ndarray)
    return TensorType(value.dtype, value.ndim)

def _get_value_type(value):
    if isinstance(value, np.ndarray):
        return TensorType(value.dtype, value.ndim)
    elif isinstance(value, tuple):
        return TupleType(*map(_get_value_type, value))

def _get_value_device(value):
    warnings.warn("todo: properly implement")
    return Device()

def _put_on_device(value, device):
    warnings.warn("not putting value on device")
    return value

def num_components(node):
    return len(node.get_type()) if isinstance(node.get_type(), TupleType) else 1


class OpImpl(object):
    def is_py(self):
        raise NotImplementedError
    def is_c(self):
        raise NotImplementedError
    def is_cuda(self):
        raise NotImplementedError

class PyImpl(OpImpl):
    def __init__(self, inplace_func=None, valret_func=None):
        assert (inplace_func is None) != (valret_func is None)
        self.inplace_func = inplace_func
        self.valret_func = valret_func
    def is_py(self):
        return True
    def is_c(self):
        return False
    def is_cuda(self):
        return False

class CImpl(OpImpl):
    def __init__(self, code, includes=None, link_flags=""):
        self.code = code
        self.includes = [] if includes is None else includes
        self.link_flags = link_flags
    def is_py(self):
        return False
    def is_c(self):
        return True
    def is_cuda(self):
        return False

class CUDAImpl(CImpl):
    def is_py(self):
        return False
    def is_c(self):
        return False
    def is_cuda(self):
        return True

class Op(object):
    """
    Describes an operation that will be performed on some data.
    """

    # attributes that can be overwritten in subclasses
    call_type = 'inplace' # or valret
    inplace_alias_ok = True
    writes_to_input = -1

    # pylint: disable=W0613

    def shp_apply(self, parents):
        """
        Return output shapes as a function of input nodes
        """
        raise NotImplementedError
    def typ_apply(self, parent_typs):
        """
        Return output types as a function of input types
        """
        raise NotImplementedError        
    def get_diff(self, num_inputs):
        """
        Return a list of length len(inputs), specifying which inputs the Op is differentiable with respect to.
        """
        assert isinstance(num_inputs, int)
        return [True]*num_inputs
    def get_expr(self, parent_exprs):
        """
        Return string expression for this operation, built from the parent expressions
        """
        return "%s(%s)"%(str(self), ",".join(parent_exprs))
    def get_hash(self):
        """
        Return a string that uniquely identifies the value of this Op.
        Should ideally be fixed across program runs
        """
        return cPickle.dumps(self.__dict__, -1)+self.__class__.__name__
    def get_name(self):
        """
        Get a human-readable description of the Op, including its attributes
        """
        return type(self).__name__.lower()
    def get_py_impl(self):
        raise MethodNotDefined
    def get_c_impl(self, inputs):
        raise MethodNotDefined
    def get_cuda_impl(self, inputs):
        raise MethodNotDefined
    def get_closure(self, _inputs):
        """        
        Return a list (specifying the closure data) or None (indicating that no closure data is needed.)

        For an Op with several attributes, some of them might be incorporated directly into the code of the Op,
        whereas others can be provided at runtime. This function provides the data used to create a closure
        containing the information that should be passed at runtime.
        CGT will generate code for a struct with the desired field.

        Closure data is specified in a list of triples of the form [(fieldname, fieldtype, fieldval)]
        Fieldtype is a string
        Fieldtypes is a ctypes type like ctypes.c_int or ctypes.c_void_p
        Fieldval is the python value.

        For example, let's say we want to store an int and a bool in the closure. Then we can return the triples
        [("myint",ctypes.c_int, 3), ("mybool",ctypes.c_bool, True)]
        """
        return None
    def get_replacement(self, _newparents, _analysis):
        """
        Return the name of this node
        """
        return None
    def pullback(self, inputs, output, goutput): #pylint: disable=W0613
        """
        Compute symbolic expressions for derivatives obtained by backpropagation on this Op
        Given a function y = f(x_1, x_2, ..., x_k), let J_k denote the Jacobian dy/dx_k
        pullback(...) computes gradx_k = J_k^T grady
        """
        raise MethodNotDefined
    def pushforward(self, inputs, output, goutput):
        r"""
        Compute symbolic expressions for derivatives obtained by "tangent propagation" on this Op
        Given a function y = f(x_1, x_2, ..., x_k), let J_k denote the Jacobian dy/dx_k
        pullback([x_1, ..., x_k], y, grady) := \sum_k J_k gradx_k
        """
        raise MethodNotDefined
    def spliting(self, inputs):
        """
        Return a list [tensor_type_sig, split_specs]
        where tensor_type_sig is a string labeling the input and output axes
        and split_specs is a list of tuples (axis, split_type, split_args...) 

        tensor_type_sig is easiest to illustrate with a few examples:
        Mul22: i.j , j.k-> i.k
        Sum{1} i.j -> i.1
        GetSli{0} ij.1.1

        """
        raise MethodNotDefined

    def __str__(self):
        """
        Get a human-readable description of the Op, including its attributes
        """
        return type(self).__name__.lower()


def as_node(val_or_node):
    """    
    If numeric data received, convert to a constant node
    """
    if isinstance(val_or_node, Node):
        return val_or_node
    elif isinstance(val_or_node, np.ndarray) or np.isscalar(val_or_node):
        return cgt.constant(val_or_node)
    elif isinstance(val_or_node, tuple):
        return cgt.make_tuple(*val_or_node)
    else:
        raise ValueError("expected numeric data or Node, got object of type %s"%type(val_or_node))

class Result(Node):
    """
    Node representing an intermediate computational result, which depends on its parents in the graph
    (TODO: be more precise about semantics. Does it depend ONLY on the parents, or can it depend on 
    some exogenous input too? What about random variables?)
    """
    def __init__(self, op, parents, typ = None):
        parents = map(as_node, parents)
        typ = op.typ_apply(parents) if typ is None else typ
        assert op is not None
        # self.stackinfo = traceback.extract_stack()
        Node.__init__(self, typ, op, parents)
    def get_hash(self, node2hash):
        hashobj = hashlib.md5(self.op.get_hash())
        for p in self.parents: hashobj.update(node2hash[p])
        return hashobj.hexdigest()
    def __repr__(self):
        return "Result{%s}"%(str(self.op))

class Input(Node):
    """
    Abstract class representing an input to the graph -- a node with no parents, which does not
    correspond to a computation.
    """
    def __init__(self, typ, name=None):
        self.name = "" if name is None else name
        assert isinstance(self.name, (str,unicode))
        Node.__init__(self, typ, None, [])
    def is_input(self):
        return True
    def get_hash(self, _node2hash):
        hashobj = hashlib.md5(str(id(self)))
        # XXX
        return hashobj.hexdigest()
    def __repr__(self):
        raise NotImplementedError
    def get_fixed_shape(self):
        raise NotImplementedError

class Argument(Input):
    """
    Input to the graph that is an argument to a function call
    """
    def __init__(self, typ, name=None, fixed_shape=None):    
        assert isinstance(typ, Type)
        Input.__init__(self, typ, name)
        if fixed_shape is not None:
            assert isinstance(fixed_shape, tuple) and len(fixed_shape) == self.ndim
            self.fixed_shape = fixed_shape
        else:
            self.fixed_shape = (None,)*self.ndim
    def is_argument(self):
        return True
    def __repr__(self):
        return "Argument{%s,name='%s'}"%(self.typ,self.name)
    def get_fixed_shape(self):
        return self.fixed_shape


# Just here as a temporary  hack so node.op does the right thing in cython
class GetData(Op):
    call_type="valret"
    def __init__(self, datanode):
        self.datanode = datanode
    def get_py_impl(self):
        def f(reads): return self.datanode.get_value()
        return PyImpl(valret_func=f)
    def get_c_impl(self, *args):
        return CImpl(code=r"""
extern "C" cgtArray* $function($closure* cldata, cgtArray** reads) {
    return (cgtArray*)cldata->ptr;
}""")
    def get_closure(self, _):
        ptr = self.datanode._value.get_pointer()
        return [("ptr", ctypes.c_void_p, ptr)]

class Data(Input):
    """
    An input to the graph, which is associated with a value and implicitly provided
    during function calls.
    Data is similar to global variables in standard programming languages.
    """
    def __init__(self, value,name=None,device=None, fixed_shape_mask=None):
        value = as_valid_array(value)
        if device is None:
            self.device = Device()
        # TODO only if backend=cython
        import cycgt2

        self.use_numpy = cgt.load_config()["backend"] == "python"

        if self.use_numpy:
            self._value = value
        else:
            self._value = cycgt2.CppArrayWrapper.from_numpy(value, self.device.devtype)

        self.name = "unnamed" if name is None else name
        assert self._value.dtype != object
        if fixed_shape_mask is None:  self.fixed_shape_mask = (False,)*self._value.ndim
        elif fixed_shape_mask == "all": self.fixed_shape_mask = (True,)*self._value.ndim
        else: self.fixed_shape_mask = fixed_shape_mask
        Input.__init__(self, _ndarray_type(value), name)
        self.op = GetData(self)
    def is_data(self):
        return True
    def __repr__(self):
        return "Data{%s,name='%s'}"%(self.typ,self.name)
    def get_device(self):
        return self.device
    def get_fixed_shape(self):
        shp = self._value.shape
        return [s if bfixed else None for (bfixed,s) in utils.safezip(self.fixed_shape_mask, shp)]
    def get_value(self):
        return self._value if self.use_numpy else self._value.to_numpy()

    # TODO: remove external accesses to .value

def _singleton_ones(dtype, ndim):
    return cgt.constant(np.ones((1,)*ndim, dtype))

def make_argument(typ):
    if isinstance(typ, TupleType):
        return Argument(TupleType(typ))
    elif isinstance(typ, TensorType):
        return Argument(TensorType(typ.dtype, typ.ndim))
    else:
        raise ValueError("expected Tuple or Tensor. Got %s"%typ)

# ================================================================
# Differentiation
# ================================================================

def differentiably_influences(outputs, nodelist=None):
    """
    Return the set of nodes that differentiably influence `outputs`
    in reverse topological sorted order
    """
    if nodelist is None: nodelist = list(topsorted(outputs))
    # find which inputs we're differentiable wrt
    diset = set(outputs)
    for node in reversed(nodelist):
        # print node, dio.get(node, False),id(node),"updating",[map(id, node.parents)],[map(str, node.parents)]
        if node in diset and not node.is_input():
            for (p,d) in utils.safezip(node.parents, node.get_diff()):
                if d: diset.add(p)
    return diset

def differentiably_influenced_by(wrt, outputs=None, nodelist=None):
    assert (outputs is None) != (nodelist is None) # one of these are provided
    if nodelist is None: nodelist = list(topsorted(outputs))
    dibset = set(wrt)
    for node in nodelist:
        if any(p in dibset and d for (p,d) in utils.safezip(node.parents, node.get_diff())):
            dibset.add(node)
    return dibset


def influences(outputs):
    return list(topsorted(outputs))


def pullback(outputs, goutputs, wrt):
    """    
    This function propagates derivative information backwards from the outputs of a computation
    to the inputs. 
    All of these operations are performed symbolically, and we construct expressions for derivatives
    of inputs in terms of derivatives of outputs.
    This function is called 'pullback' as a reference to the similar concept in differential geometry.
    
    More precisely, suppose f is a function with (y_1, y_2, ..., y_k) = f(x_1, x_2, ..., x_n)
    Then pullback([x_1,...,x_n], [y_1,...,y_k], [gy_1, ..., gy_k]) := [gx_1, ..., gx_n]
    """
    nodelist = list(topsorted(outputs))

    dio = differentiably_influences(outputs,nodelist=nodelist)
    dibw = differentiably_influenced_by(wrt, nodelist=nodelist)

    # Some checks
    badwrtset = set(wrt).difference(dio)
    if badwrtset:
        raise NonDifferentiable("Outputs not differentiable wrt %s"%badwrtset)

    badoutset = set(outputs).difference(dibw)
    if badoutset:
        raise NonDifferentiable("Outputs %s not differentiable wrt any of %s"%(badoutset, badwrtset))

    var2gs = defaultdict(list)
    for (node, gnode) in utils.safezip(outputs, goutputs):
        var2gs[node] = [gnode]

    active = dio.intersection(dibw)

    for node in reversed(nodelist):
        if node not in active: continue
        # once we reach a node, we have already backpropagated from all parents
        # so now we can sum up the gradients
        if len(var2gs[node]) > 1:
            var2gs[node] = [cgt.add_multi(var2gs[node])]
        # only one gradient at this point
        gnode = var2gs[node][0]
        if isinstance(node, Result):
            if isinstance(node.op, TupleIndex):
                par = node.parents[0]
                if par not in var2gs: var2gs[par] = [tuple([] for _ in len(par.get_type())) ]
                var2gs[par][0][node.op.idx].append(gnode)
                # XXX wil fail with unused outputs
            else:
                gpars = node.op.pullback(node.parents, node, gnode)
                diffs = node.get_diff()
                for (par,gpar,d) in utils.safezip3(node.parents, gpars,diffs):
                    assert (gpar is not None) == d # grad is None iff not diff wrt input
                    if d: var2gs[par].append(gpar)

    # only we already summed up the gradients for the input nodes, so just take
    # 0th element
    return [var2gs[node][0] for node in wrt]

# XXX fails when non-constant
def infer_shape(arr):
    return tuple(x.op.value for x in  CACHER.simplify(cgt.shape(arr)))

def grad(cost, wrt):    
    """
    Compute the gradient of scalar-valued `cost` with respect to a list of variables `wrt`
    """
    assert cost.ndim == 0
    assert all(x.is_input() for x in wrt), "Can only differentiate wrt Input nodes. (Or do you have a good reason to differentiate wrt a non-input? Let us know, we can probably implement it."
    gout = _singleton_ones(cost.dtype, 0)
    return pullback([cost], [gout], wrt)


# ================================================================
# Ops 
# ================================================================


# Constants
# ----------------------------------------------------------------

class Constant(Op): #pylint: disable=W0223
    def __init__(self, value):
        self.value = value
    def get_value(self):
        return self.value

class ConstantTensor(Constant):
    call_type = "inplace"
    # XXX for some reason valret version gives rare segfaults
    def __init__(self, value):
        Constant.__init__(self, as_valid_array(value))
    def get_expr(self, parent_exprs):
        return self._value_str()
    def __str__(self):
        return "const{%s}"%self._value_str()
    def _value_str(self):
        ndim = self.value.ndim
        return "%g"%self.value if ndim==0 else "%s%g...%s"%("["*ndim, self.value.flat[0], "]"*ndim)        
    def get_py_impl(self):
        # def valret_func(reads):
        #     return self.value
        def inplace_func(reads, write):
            if isinstance(write, tuple):
                for (arrfrom,arrto) in utils.safezip(self.value,write):
                    np.copyto(arrto, arrfrom)
            else:
                np.copyto(write,self.value)
        return PyImpl(inplace_func=inplace_func)#PyImpl(valret_func=valret_func)
    def pullback(self, _inps, _out, _gout):
        return []
    def shp_apply(self, _inputs):
        return [cgt.constant(x) for x in self.value.shape] 
    def typ_apply(self, inputs):
        assert len(inputs)==0
        return _ndarray_type(self.value)
    def get_hash(self):
        return str(id(self))
    def get_closure(self, _):
        assert isinstance(self.value, np.ndarray)
        shapeptr = ctypes.cast(self.value.ctypes.shape, ctypes.c_void_p).value
        shit.append(self.value)
        return [
        ("ndim", ctypes.c_int,self.value.ndim),
        ("shape",ctypes.c_void_p,shapeptr),
        ("dtype",ctypes.c_byte,self.value.dtype.num),
        ("data",ctypes.c_void_p,self.value.ctypes.data)]
    def get_c_impl(self, *args):
        code = None
        if self.call_type == "valret": code = self._c_code_valret(*args)
        elif self.call_type == "inplace": code = self._c_code_inplace(*args)
        else: raise ValueError
        return CImpl(code=code)
    def _c_code_inplace(self, inputs):
        if isinstance(self.value, tuple):
            raise MethodNotDefined
        return r"""
extern "C" void $function($closure* cldata, cgtArray** reads, cgtArray* write) {
    cgt_memcpy(cgtCPU, cgtCPU, write->data(), cldata->data, write->nbytes());
}
"""
    def _c_code_valret(self, inputs):
        return r"""
extern "C" cgtArray* $function($closure* cldata, cgtArray** reads) {
        auto out = new cgtArray(cldata->ndim, (size_t*)cldata->shape, 
            (cgtDtype)cldata->dtype, cgtCPU, (void*)cldata->data, false);
        return out;
}"""


class ConstantTuple(Constant):
    call_type = "valret"
    def __init__(self, value):
        Constant.__init__(value)
    def get_expr(self, parent_exprs):
        return str(self.value)
    def __str__(self):
        return "const{%s}"%str(self.value)
    def py_apply_valret(self, reads):
        return self.value
    def shp_apply(self, _inputs):
        return tuple(map(cgt.constant, x.shape) for x in self.value)
        return cgt.shape(as_node(self.value))
    def typ_apply(self, inputs):
        assert len(inputs)==0
        return _get_value_type(self.value)
    def get_hash(self):
        return str(id(self))



shit = [] # XXX just so this stuff doesn't get dereferenced.
# this is a memory leak, will fix later


class Fill(Op):
    """
    (value, shape...) -> array filled with `value`, with shape `shape`
    """
    def __init__(self, value):
        self.value = as_valid_array(value)
        assert self.value.ndim ==0
        assert self.value.dtype != "O"
        self.dtype = self.value.dtype
        assert self.value.ndim==0
    def get_diff(self, num_inputs):
        return [False]*num_inputs
    def __str__(self):
        return "fill{%g}"%self.value
    def get_py_impl(self):
        def f(reads, write):
            write[...] = self.value
        return PyImpl(inplace_func=f)
    def pullback(self, inputs, output, goutput):
        raise NonDifferentiable
    def shp_apply(self, inputs):
        return inputs
    def typ_apply(self, inputs):
        assert all(x.dtype == 'i8' for x in inputs)
        return TensorType(self.dtype, len(inputs))
    def get_closure(self, inputs):
        typ = ctypes.c_long if self.value.dtype.kind=='i' else ctypes.c_double
        return [("value", typ, self.value.item())]
    def get_c_impl(self, inputs):
        outdtype = Dtype.canon(self.value.dtype)
        return CImpl(code=r"""
extern "C" void $function($closure* cldata, cgtArray** reads, cgtArray* write) {
    size_t s = write->size();
    %(cdtype)s value = cldata->value;
    for (int i=0; i < s; ++i) write->at<%(cdtype)s>(i) = value;
}"""%dict(cdtype = np2c[outdtype]))
def _is_int(node):
    return dtype_kind(node.dtype)=='i'

def _list_is_valid_sli(inputs):
    return len(inputs)==3 and all(x.ndim==0 and _is_int(x) for x in inputs)

class Arange(Op):
    """
    (start,stop,step) -> 1D array, just like numpy
    """
    call_type='valret'
    def __init__(self, dtype='i8'):
        self.dtype = dtype
    def get_diff(self, num_inputs):
        return [False]*num_inputs
    def get_py_impl(self):
        def f((start, stop, step)):
            return np.arange(start, stop, step, self.dtype)
        return PyImpl(valret_func=f)
    def pullback(self, inputs, output, goutput):
        raise NonDifferentiable
    def shp_apply(self, inputs):
        start,stop,step = inputs
        return [(stop - start)//step]
    def typ_apply(self, inputs):
        assert _list_is_valid_sli(inputs)
        return TensorType(self.dtype, 1)

class ScalarRng(Op):
    """
    (shape...) -> array filled with iid random numbers, from either uniform or normal distribution
    """
    def __init__(self, kind):
        assert kind in ("uniform","gaussian")
        self.kind = kind
    def get_diff(self, num_inputs):
        return [False]*num_inputs
    def __str__(self):
        return "rng{%s}"%self.kind
    def get_py_impl(self):
        def f(reads, write):
            if self.kind == "uniform": write[...] = np.random.rand(*reads)
            elif self.kind == "gaussian": write[...] = np.random.randn(*reads)
            else: raise RuntimeError
        return PyImpl(inplace_func=f)
    def pullback(self, inputs, output, goutput):
        raise NonDifferentiable
    def shp_apply(self, inputs):
        return inputs
    def typ_apply(self, inputs):
        return TensorType(cgt.floatX, len(inputs))

# Elementwise
# ----------------------------------------------------------------

def _no_grad():
    raise NonDifferentiable()

def _nu_sigmoid(x, out=None):
    return np.reciprocal(1+np.exp(-x), out=out)

def _nu_iceil(x,out=None):
    return np.ceil(x, out=out)

def _nu_ifloor(x,out=None):
    return np.floor(x,out=out)

def _nu_divide(x, y, out=None):
    if x.dtype.kind != 'f': x = x.astype(cgt.floatX)
    return np.divide(x, y, out=out)

UnaryInfo = namedtuple("UnaryInfo", ("short","pyfunc","diff","typeinfo", "gradexpr", "cexpr"))

UNARY_INFO = {
    "abs" : UnaryInfo(   "abs", np.abs,  True,   's', lambda x, y, gy: gy*cgt.sign(x), "fabs(x)"),
    "ceil" : UnaryInfo(  "ceil", np.ceil, False,  'i',  lambda x, y, gy: _no_grad(), "ceil(x)"),
    "cos" : UnaryInfo(   "cos", np.cos,  True,   'f',   lambda x, y, gy: -gy*cgt.sin(x), "cos(x)"),
    "exp" : UnaryInfo(   "exp", np.exp,  True,   'f',   lambda x, y, gy: gy*cgt.exp(x), "exp(x)"),
    "iceil" : UnaryInfo( "iceil", _nu_iceil, False, 'i',   lambda x, y, gy: _no_grad(), "(int)ceil(x)"),
    "ifloor" : UnaryInfo( "ifloor", _nu_ifloor, False, 'i',   lambda x, y, gy: _no_grad(), "(int)floor(x)"),
    "log" : UnaryInfo(   "log", np.log,  True,   'f', lambda x, y, gy: gy/x, "log(x)"),
    "neg" : UnaryInfo(   "negative", np.negative, True, 's', lambda x, y, gy: -gy, "(-x)"),
    "sign" : UnaryInfo(   "sign", np.sign, False,   's',  lambda x, y, gy: _no_grad(), "2*(x>0)-1"),
    "sin" : UnaryInfo(    "sin", np.sin,    True, 'f',  lambda x, y, gy: gy*cgt.cos(x), "sin(x)"),
    "square" : UnaryInfo( "square", np.square, True, 's',  lambda x, y, gy: 2.0*gy*x, "x*x"),
    "sqrt" : UnaryInfo( "sqrt", np.sqrt, True, 'f', lambda x, y, gy: gy/(2.0*y), "sqrt(x)"),
    "tanh" : UnaryInfo(   "tanh", np.tanh, True,   'f', lambda x, y, gy: gy*(1-cgt.square(y)), "tanh(x)"),
    "sigmoid" : UnaryInfo( "sigmoid", _nu_sigmoid, True, 'f', lambda x, y, gy: gy*y*(1-y), "1.0/(1.0+exp(-x))"),
    "conj" : UnaryInfo( "conj", np.conj, True, 'c', lambda x, y, gy: cgt.conj(gy), "conj(x)")
}

BinaryInfo = namedtuple("BinaryInfo", ("short","pyfunc","commutes","diff","typeinfo","gradexpr", "cexpr"))


BINARY_INFO = {
    #infix             short      pyfunc    commutes     diff        typeinfo
    "*"   : BinaryInfo("multiply",  np.multiply, True,    (True,True),    'p',        lambda x, y, z, gz: [y*gz,x*gz], "x*y"),
    "+"   : BinaryInfo("add",  np.add,   True,    (True,True),    'p',        lambda x, y, z, gz: [gz,gz], "x+y"),
    "-"   : BinaryInfo("subtract",  np.subtract, False,    (True,True),   'p',       lambda x, y, z, gz: [gz,-gz], "x-y"),
    "/"   : BinaryInfo("divide",  _nu_divide,  False,    (True,True),    'f',       lambda x, y, z, gz: [gz/y,-gz*z/y], "(x+0.0)/y"),
    "<"   : BinaryInfo("less",   np.less,    False,    (False,False),  'i1',     lambda x, y, z, gz: _no_grad(), "x<y"),
    ">"   : BinaryInfo("greater",   np.greater,    False,    (False,False),  'i1',     lambda x, y, z, gz: _no_grad(), "x>y"),
    "<="   : BinaryInfo("less_equal",   np.less_equal,    False,    (False,False),  'i1',     lambda x, y, z, gz: _no_grad(), "x<=y"),
    ">="   : BinaryInfo("greater_equal",   np.greater_equal,    False,    (False,False),  'i1',     lambda x, y, z, gz: _no_grad(), "x>=y"),
    "**"   : BinaryInfo("power",  np.power,      False,    (True,True), 'p',      lambda x, y, z, gz: [gz*y*z/x,z*cgt.log(x)],"pow(x,y)"),
    "=="  : BinaryInfo("equal", lambda x,y,out : np.equal(x,y,out=out),      True,      (False, False), 'i1',  lambda x, y, z, gz: _no_grad(), "x==y"),
    "!="  : BinaryInfo("not_equal", lambda x,y,out : np.not_equal(x,y,out=out),      True,      (False, False), 'i1',  lambda x, y, z, gz: _no_grad(), "x!=y"),
}


np2c = {"i1":"int8_t","i2":"int16_t","i4":"int32_t","i8":"int64_t",
        "f4":"float","f8":"double","f16":"long double",
        "c4" : "float complex", "c8" : "double complex", "c16" : "long double complex"}


class ElwiseUnary(Op):
    """
    Elementwise unary operation
    """

    def __init__(self, opname, info=None):
        self.opname = opname
        self.info = UNARY_INFO[opname] if info is None else info
    def get_diff(self, _):
        return [self.info.diff]
    def __str__(self):
        return self.opname
    def get_hash(self):
        return utils.hash_seq1(self.opname)
    def get_py_impl(self):
        def f(reads, write):
            self.info.pyfunc(reads[0], out=write)
        return PyImpl(inplace_func=f)
    def get_replacement(self, _newparents, _analysis):
        return None
    def pullback(self, (x,), y, gy): #pylint: disable=W0613
        return [self.info.gradexpr(x, y, gy)]
    def shp_apply(self, inputs):
        return cgt.shape(inputs[0])
    def typ_apply(self, inputs):
        typeinfo = self.info.typeinfo
        intype = inputs[0].dtype
        if typeinfo == 's':
            out_type = intype
        elif typeinfo == 'i':
            out_type = _type_to_int(intype)
        elif typeinfo == 'f':
            out_type = cgt.floatX
        elif typeinfo == 'c':
            out_type = cgt.complexX
        else:
            assert typeinfo in (cgt.floatX, cgt.complexX, 'i1','i2','i4','i8')
            out_type = typeinfo
        return TensorType(out_type, inputs[0].ndim)
    def get_c_impl(self, inputs):
        info = self.info
        out_dtype = self.typ_apply(inputs).dtype
        code = r"""
static inline %(cdtype1)s scalar_CGT_FUNCNAME(%(cdtype0)s x) {return %(cexpr)s;}
extern "C" void $function(void* cldata, cgtArray** reads, cgtArray* write) {
    cgtArray* read = reads[0];
    int s = read->size();
    %(cdtype0)s* readdata = (%(cdtype0)s*)read->data();
    %(cdtype1)s* writedata = (%(cdtype1)s*)write->data();
    for (int i=0; i < s; ++i) {
        writedata[i] = scalar_CGT_FUNCNAME(readdata[i]);
    }
}
"""%dict(cdtype0=np2c[inputs[0].dtype], cdtype1=np2c[out_dtype], cexpr=info.cexpr)
        return CImpl(code, includes=["math.h"], link_flags="-lm")
    def cuda_code(self, inputs):
        raise RuntimeError # move this to get_cuda_impl
        info = self.info
        npdtype = inputs[0].dtype
        return """
__forceinline__ __device__ %(cdtype)s $function(%(cdtype)s x) {return %(cexpr)s;}        
__global__ void CGT_FUNCNAME_kernel(const size_t n, const %(cdtype)s* in, %(cdtype)s* out) {
  CUDA_KERNEL_LOOP(i, n) {
    out[i] = $function(in[i]);
  }
}
extern "C" void $function(void* cldata, cgtArray** reads, cgtArray* write) {
    cgtArray* read = reads[0];
    size_t n = read->size();
    int num_blocks, num_threads;
    cgt_get_bt(n, &num_blocks, &num_threads);
    CGT_FUNCNAME_kernel<<<num_blocks, num_threads>>>(n, (%(cdtype)s*)read->data(), (%(cdtype)s*)write->data());
}
"""%dict(cdtype=np2c[npdtype],cexpr=info.cexpr)
    def cuda_includes(self):
        raise RuntimeError # move this to get_cuda_impl
        return ["cgt_cuda.h","cgt_common.h"]

class ElwiseBinary(Op):
    # +, -, *, /, <, ^, //
    def __init__(self, opname, scalar_mask, info=None):
        assert opname in BINARY_INFO        
        self.opname = opname
        self.info = BINARY_INFO[opname] if info is None else info
        self.scalar_mask = scalar_mask
    def get_diff(self, _):
        return BINARY_INFO[self.opname].diff
    def get_hash(self):
        return utils.hash_seq1(self.opname)        
    def get_expr(self, parent_exprs):
        return "(%s %s %s)"%(parent_exprs[0], self.opname, parent_exprs[1])
    def __str__(self):
        return BINARY_INFO[self.opname].short
    def get_replacement(self, parents, analysis):
        l,r = parents
        node2sv = analysis["node2sv"]
        out = None
        
        # The following replacements are allowed to return a scalar constant value
        # Before returning, we'll broadcast it back to the right shape

        if isinstance(l.op,Fill) and not self.scalar_mask[1]:
            out=Result(ElwiseBinary(self.opname, (True,False), self.info),
                [cgt.constant(l.op.value), r])
        elif isinstance(r.op,Fill) and not self.scalar_mask[0]:
            out=Result(ElwiseBinary(self.opname, (False,True), self.info),
                [l, cgt.constant(r.op.value)])
        # if both have single value, apply this operation numerically and fill the result with it
        elif l in node2sv and r in node2sv:
            out =self.info.pyfunc(node2sv[l], node2sv[r])
        # if l has has a single value, apply the operation to l and return a Constant
        elif l in node2sv and isinstance(r.op, Constant):
            out = py_numeric_apply(self, [node2sv[l], r.op.val])
        # same as previous but swapped
        elif r in node2sv and isinstance(l.op, Constant):
            out = py_numeric_apply(self, [l.op.val, node2sv[r]])
        elif self.opname == "*":
            if l in node2sv and node2sv[l] == 1: out = r
            if l in node2sv and node2sv[l] == -1: out = -r
            if r in node2sv and node2sv[r] == 1: out = l
            if r in node2sv and node2sv[r] == -1: out = -l
        elif self.opname == "+":
            if l in node2sv and node2sv[l] == 0: out = r
            if r in node2sv and node2sv[r] == 0: out = l

        if out is not None:
            outtyp = self.typ_apply(parents)
            out = cgt.cast(out, outtyp.dtype)
            if out.ndim==0 and outtyp.ndim>0:
                ind4shape = 1 if self.scalar_mask[0] else 0
                outshape = analysis["node2shape"][parents[ind4shape]]
                out = cgt.fill(out, outshape)

        return out

    def pullback(self, (x, y), z, gz): #pylint: disable=W0613
        gin = BINARY_INFO[self.opname].gradexpr(x, y, z, gz)
        return [cgt.sum(gv) if (v.ndim==0 and gv.ndim > 0) else gv for (v,gv) in utils.safezip([x,y],gin)]
    def shp_apply(self, inputs):
        ind4shape = 1 if self.scalar_mask[0] else 0
        return cgt.shape(inputs[ind4shape])
    def typ_apply(self, inputs):
        assert ((inputs[0].ndim==0) == self.scalar_mask[0]) and ((inputs[1].ndim==0) == self.scalar_mask[1])
        if self.scalar_mask==(False,False):
            assert inputs[0].ndim == inputs[1].ndim
            assertequaln(cgt.shape(inputs[0]),cgt.shape(inputs[1]),"shape mismatch at elementwise binary operation")
        typeinfo = BINARY_INFO[self.opname].typeinfo
        if typeinfo == 'p':
            out_dtype = _promote(inputs[0].dtype, inputs[1].dtype)
        elif typeinfo == 'f':
            out_dtype = cgt.floatX
        else:
            out_dtype = typeinfo
        ind4shape = 1 if self.scalar_mask[0] else 0
        return TensorType(out_dtype, inputs[ind4shape].ndim)
    def get_py_impl(self):
        def f(reads, write):
            x,y = reads
            if self.scalar_mask==(False,False):
                if x.shape != y.shape:
                    raise RuntimeError("mismatched shapes %s %s. Note that implicit broadcasting isn't allowed. Use the broadcast(...) function"%(x.shape, y.shape))
            self.info.pyfunc(x,y, out=write)
        return PyImpl(inplace_func=f)
    def get_c_impl(self, inputs):
        typ2 = self.typ_apply(inputs)
        npdtype0 = inputs[0].dtype
        npdtype1 = inputs[1].dtype
        npdtype2 = typ2.dtype
        ind4shape = 1 if self.scalar_mask[0] else 0
        index0 = "0" if self.scalar_mask[0] else "i"
        index1 = "0" if self.scalar_mask[1] else "i"
        code = r"""
static inline %(cdtype2)s scalar_CGT_FUNCNAME(%(cdtype0)s x, %(cdtype1)s y) {return %(cexpr)s;}
extern "C" void $function(void* cldata, cgtArray** reads, cgtArray* write) {
    int s = reads[%(ind4shape)s]->size();
    %(cdtype0)s* in0 = (%(cdtype0)s*)reads[0]->data();
    %(cdtype1)s* in1 = (%(cdtype1)s*)reads[1]->data();
    %(cdtype2)s* out = (%(cdtype2)s*)write->data();
    cgt_check(write->size() == s, "Shape error in elementwise binary operation. You might be missing a call to cgt.broadcast(...)");
    for (int i=0; i < s; ++i) {
        out[i] = scalar_CGT_FUNCNAME(in0[%(index0)s], in1[%(index1)s]);
    }
}
"""%dict(cdtype0=np2c[npdtype0],cdtype1=np2c[npdtype1],cdtype2=np2c[npdtype2],
    cexpr=self.info.cexpr, index0=index0, index1=index1, ind4shape=ind4shape) 
        return CImpl(code=code, includes=["cgt_common.h", "math.h"])

    def cuda_code(self, inputs):
        raise RuntimeError # move to get_*_impl
        typ2 = self.typ_apply(inputs)
        npdtype0 = inputs[0].dtype
        npdtype1 = inputs[1].dtype
        npdtype2 = typ2.dtype
        return """
__forceinline__ __device__ %(cdtype2)s $function(%(cdtype0)s x, %(cdtype1)s) {return %(cexpr)s;}
__global__ void CGT_FUNCNAME_kernel(const size_t n, const %(cdtype0)s* x, %(cdtype1)s* y, %(cdtype2)s z) { \
  CUDA_KERNEL_LOOP(i, n) {
    z[i] = $function(x[%(index0)s], y[%(index1)s]);
  }
}
extern "C" void $function(void* cldata, cgtArray** reads, cgtArray* write) {
    size_t n = reads[1]->size();
    int num_blocks,num_threads;
    cgt_get_bt(n, &num_blocks, &num_threads);
    TODO
    CGT_FUNCNAME_kernel<<<num_blocks, num_threads>>>(n, (%(cdtype0)s*)reads[0]->data(), (%(cdtype1)s*)reads[1]->data(), (%(cdtype2)s*)write->data());
}
"""%dict(cdtype0=np2c[npdtype0],cdtype1=np2c[npdtype1],cdtype2=np2c[npdtype2])  
    def cuda_includes(self):
        raise RuntimeError # move to get_*_impl
        return ["cgt_cuda.h","cgt_common.h"]

def elwise_binary(opname, x, y):
    (x, y) = map(as_node, (x, y))
    scalar_mask = ((x.ndim == 0), (y.ndim == 0))
    op = ElwiseBinary(opname, scalar_mask)
    if (scalar_mask == (False, False)):
        assert (x.ndim == y.ndim)
    return Result(op, [x, y])

# Shape manip
# ----------------------------------------------------------------

class Size(Op):
    """
    Return an element of the shape of a tensor
    """
    call_type = "valret"
    def __init__(self, axis):
        self.axis = axis
    def get_diff(self, _):
        return [False]
    def __str__(self):
        return "size{%i}"%self.axis
    def get_py_impl(self):
        def f(reads):
            return np.array(reads[0].shape[self.axis])
        return PyImpl(valret_func=f)
    def pullback(self, inputs, output, goutput):
        raise NonDifferentiable
    def shp_apply(self, _inputs):
        return []
    def typ_apply(self, _inputs):
        return TensorType('i8',0)
    def get_replacement(self, inputs, _analysis):
        x = inputs[0]
        if x.is_input():
            fixed_shape = x.get_fixed_shape()
            if fixed_shape[self.axis] is not None:
                return cgt.constant(fixed_shape[self.axis])
    def get_closure(self, inputs):
        return [("ax",ctypes.c_int,self.axis)]
    def get_c_impl(self, _):
        code = r"""
extern "C" cgtArray* $function(void* cl0, cgtArray** reads) {
$closure* cl = ($closure*)cl0;
    cgtArray* in = reads[0];
    cgtArray* out = new cgtArray(0, NULL, cgt_i8, cgtCPU);
    ((long*)out->data())[0] = in->shape()[cl->ax];
    return out;
}"""
        return CImpl(code)

class Reshape(Op):
    gpu_uses_c = True
    call_type = "valret"
    def get_diff(self, num_inputs):
        return [True] + [False]*(num_inputs-1)
    def get_py_impl(self):
        def f(reads):
            return reads[0].reshape(reads[1:])
        return PyImpl(valret_func=f)
    def pullback(self, inputs, _out, gout):
        return [cgt.reshape(gout, cgt.shape(inputs[0]))] + [None]*(len(inputs)-1)
    def shp_apply(self, inputs):
        return inputs[1:]
    def typ_apply(self, inputs):
        return TensorType(inputs[0].dtype, len(inputs)-1)
    def get_closure(self, parents):
        return [("ndim", ctypes.c_int,len(parents)-1)]
    def get_c_impl(self, _):
        code = r"""
extern "C" cgtArray* $function($closure* cldata, cgtArray** reads) {
    cgtArray* in = reads[0];
    size_t* newshape = new size_t[cldata->ndim];
    for (int i=0; i < cldata->ndim; ++i) newshape[i] = static_cast<size_t*>(reads[i+1]->data())[0];
    cgtArray* out = new cgtArray(cldata->ndim, newshape, in->dtype(), in->devtype(), in->data(), false);
    return out;
}
"""
        return CImpl(code)

class Concatenate(Op):
    call_type = "valret"
    def __init__(self, axis):
        self.axis = axis
    def get_diff(self, num_inputs):
        return [True]*num_inputs
    def get_py_impl(self):
        def f(reads): return np.concatenate(reads,axis=self.axis)
        return PyImpl(valret_func=f)
    def pullback(self, inputs, _output, gout):
        start = 0
        out = []
        for x in inputs:
            end = start + cgt.size(x, self.axis)
            out.append(Result(GetSli(self.axis), [gout, start,end, 1]))
            start = end
        return out
    def shp_apply(self, inputs):
        out = cgt.shape(inputs[0])
        out[self.axis] = cgt.add_multi([cgt.size(x,self.axis) for x in inputs])
        return out
    def typ_apply(self, inputs):
        return TensorType(_promote_multi([x.dtype for x in inputs]), inputs[0].ndim)

class Stack(Op):
    def get_diff(self, num_inputs):
        return [True for _ in xrange(num_inputs)]
    def get_numeric_py(self):
        raise RuntimeError # move to get_*_impl
        def fn(*vals):
            return np.array(vals)
        return fn
    def pullback(self, inputs, output, goutput):
        return [goutput[i] for i in xrange(len(inputs))]
    def shp_apply(self, inputs):
        return [cgt.constant(len(inputs))] + cgt.shape(inputs[0])
    def typ_apply(self, inputs):
        assert utils.allsame([x.get_type() for x in inputs])
        return TensorType(inputs[0].dtype, inputs[0].ndim+1)

class Repeat(Op):
    def __init__(self, axes):
        self.axes = axes
    def get_diff(self, num_inputs):
        return [True] + [False for _ in xrange(num_inputs-1)]
    def get_py_impl(self):
        def f(reads, write):
            arr = reads[0]
            numreps = reads[1:]
            shp = arr.shape
            assert all(shp[i] == 1 for i in self.axes)
            for (ax,numrep) in utils.safezip(self.axes, numreps):
                arr = np.repeat(arr, numrep, ax)
            np.copyto(write, arr)
        return PyImpl(inplace_func=f)
    def get_c_impl(self, inputs):
        x = inputs[0]
        openloops = " ".join(["for (int i%(ax)s=0; i%(ax)s < write->shape()[%(ax)s]; ++i%(ax)s) {"%dict(ax=ax) for ax in xrange(x.ndim)])
        closeloops = "}"*x.ndim
        outidxexpr = ",".join(["i%(ax)s"%dict(ax=ax) for ax in xrange(x.ndim)])
        inidxexpr = ",".join(["0" if ax in self.axes else "i%(ax)s"%dict(ax=ax) for ax in xrange(x.ndim)])
        code = r"""
extern "C" void $function(void* cldata, cgtArray** reads, cgtArray* write) {
    cgtArray *read=reads[0];
    %(openloops)s
        write->at<%(cdtype)s>(%(outidxexpr)s) = read->at<%(cdtype)s>(%(inidxexpr)s);
    %(closeloops)s
}
"""%dict(openloops=openloops, outidxexpr=outidxexpr, inidxexpr=inidxexpr, closeloops=closeloops,
    cdtype=np2c[inputs[0].dtype])
        return CImpl(code)
    def get_replacement(self, parents, analysis):
        if parents[0] in analysis["node2sv"]:
            value = analysis["node2sv"][parents[0]]
            shp = self.shp_apply(parents)
            return Result(Fill(value), shp)
    def pullback(self, inputs, output, goutput):
        return [cgt.sum(goutput, self.axes, keepdims=True)] + [None]*(len(inputs)-1)
    def shp_apply(self, inputs):
        out = cgt.shape(inputs[0])
        for (ax,rep) in utils.safezip(self.axes, inputs[1:]):
            out[ax] = rep
        return out
    def typ_apply(self, inputs):
        assert all(x.dtype == "i8" for x in inputs[1:])
        return inputs[0].get_type()

class Transpose(Op):
    def __init__(self, axes):
        self.axes = axes
    def get_diff(self, _):
        return [True]
    def get_py_impl(self):
        def f(reads, write):
            np.copyto(write, reads[0].transpose(self.axes))
        return PyImpl(inplace_func=f)
    def pullback(self, inputs, output, goutput):
        return [cgt.transpose(goutput, utils.invert_perm(self.axes))]
    def shp_apply(self, inputs):
        inshape = cgt.shape(inputs[0])
        return [inshape[ax] for ax in self.axes]
    def typ_apply(self, inputs):
        return inputs[0].get_type()
    def get_c_impl(self, inputs):
        x = inputs[0]
        d = {}
        d["openloops"] = " ".join(["for (int i%(ax)s=0; i%(ax)s < write->shape()[%(ax)s]; ++i%(ax)s) {"%dict(ax=ax) for ax in xrange(x.ndim)])
        d["closeloops"] = "}"*x.ndim
        d["outidxexpr"] = " + ".join(["i%i * write->stride(%i)"%(i,i) for i in xrange(x.ndim)])
        d["inidxexpr"] = " + ".join(["i%i * read->stride(%i)"%(i,ax) for (i,ax) in enumerate(self.axes)])
        d["cdtype"] = np2c[x.dtype]
        code = r"""
extern "C" void $function(void* cldata, cgtArray** reads, cgtArray* write) {
    cgtArray *read = reads[0];
    %(cdtype)s* indata = (%(cdtype)s*)read->data(), *outdata = (%(cdtype)s*)write->data();
    %(openloops)s
        outdata[%(outidxexpr)s] = indata[%(inidxexpr)s];
    %(closeloops)s
}"""%d
        return CImpl(code)

class Transport(Op):
    gpu_uses_c = True
    def __init__(self, src, targ):
        assert isinstance(src, Device)
        self.src = src
        self.targ = targ
    def typ_apply(self, inputs):
        return inputs[0].get_type()
    def get_numeric_py(self):
        raise RuntimeError # move to get_*_impl
        def fn(x):
            return x
        return fn
    def shp_apply(self, inputs):
        return cgt.shape(inputs[0])
    def get_c_impl(self, _inputs):
        code = """
void $function(void* cldata, cgtArray** reads, cgtArray* write) {
    cgt_memcpy(write->devtype, reads[0]->devtype, write->data(), reads[0]->data(), reads[0]->nbytes());
}
"""
        return CImpl(code)

# TODO save computation by removing negative freq components
class RFFT(Op):
    def __init__(self, axes):
        self.axes = axes
    def get_diff(self, num_inputs):
        return [True] + [False]*(num_inputs-1)
    def get_py_impl(self):
        def f(reads, write):
            x = reads[0]
            shp = map(int,reads[1:])
            np.copyto(write, np.fft.fftn(x,shp,self.axes))
        return PyImpl(inplace_func=f)
    def pullback(self, inputs, _outputs, goutput):
        return cgt.real(Result(RFFT(self.axes),[goutput]+inputs[1:]))
    def shp_apply(self, inputs):
        out = cgt.shape(inputs[0])
        for (ax,sz) in utils.safezip(self.axes, inputs[1:]):
            out[ax]=sz
        return out
    def typ_apply(self, inputs):
        assert inputs[0].dtype==cgt.floatX
        return TensorType(cgt.complexX,inputs[0].ndim)

class IRFFT(Op):
    def __init__(self, axes):
        self.axes = axes
    def get_diff(self, _):
        return [True]
    def get_py_impl(self):
        def f(reads, write):
            x = reads[0]
            shp = map(int,reads[1:])
            slis = [slice(0,None) for _ in xrange(x.ndim)]
            for (ax,s) in zip(self.axes,shp): slis[ax] = slice(0, s)
            np.copyto(write, np.real(np.fft.ifftn(x,axes=self.axes)[slis]))
        return PyImpl(inplace_func=f)
    def pullback(self, inputs, _outputs, goutput):
        return Result(IRFFT(self.axes),[goutput]) # XXX is this right?
    def shp_apply(self, inputs):
        return cgt.shape(inputs[0])
    def typ_apply(self, inputs):
        return TensorType(cgt.floatX,inputs[0].ndim)

# Reductions
# ----------------------------------------------------------------

class Sum(Op):
    def __init__(self, axes):
        self.axes = tuple(axes)
    def get_diff(self, _):
        return [True]
    def __str__(self):
        return "sum{%s}"%(",".join(map(str,self.axes)))
    def get_py_impl(self):
        def f(reads, write):
            reads[0].sum(axis = self.axes or None, out=write, keepdims=True)
        return PyImpl(inplace_func=f)
    def pullback(self, inputs, output, goutput):
        return [Result(Repeat(self.axes), [goutput] + [cgt.size(inputs[0],ax) for ax in self.axes])]
    def shp_apply(self, inputs):
        x = inputs[0]
        s = cgt.shape(x)
        return [(cgt.constant(1) if i in self.axes else s[i]) for i in xrange(x.ndim)]
    def typ_apply(self, inputs):
        return inputs[0].get_type()
    def get_c_impl(self, inputs):
        x = inputs[0]
        openloops = " ".join(["for (int i%(ax)s=0; i%(ax)s < read->shape()[%(ax)s]; ++i%(ax)s) {"%dict(ax=ax) for ax in xrange(x.ndim)])
        closeloops = "}"*x.ndim
        outdims = [i for i in xrange(x.ndim) if i not in self.axes]
        inidxexpr = " + ".join(["i%(ax)s * read->stride(%(ax)s)"%dict(ax=ax) for ax in xrange(x.ndim)])
        outidxexpr = " + ".join(["i%(ax)s * write->stride(%(ax)s)"%dict(ax=ax) for ax in outdims])\
            if len(outdims)>0 else "0"
        code = r"""
extern "C" void $function(void* cldata, cgtArray** reads, cgtArray* write) {
    cgtArray *read=reads[0];
    memset(write->data(), 0, write->nbytes());
    %(openloops)s
        ((%(cdtype)s*)write->data())[%(outidxexpr)s] += ((%(cdtype)s*)read->data())[%(inidxexpr)s];
    %(closeloops)s
}
"""%dict(openloops=openloops, outidxexpr=outidxexpr, inidxexpr=inidxexpr, closeloops=closeloops,
    cdtype=np2c[inputs[0].dtype])
        return CImpl(code, includes=["string.h"])

class Max(Op):
    def __init__(self, axes):
        self.axes = tuple(axes)
    def get_diff(self, _):
        return [True]
    def __str__(self):
        return "max{%s}"%(",".join(map(str,self.axes)))
    def get_py_impl(self):
        def f(reads, write):
            reads[0].max(axis=self.axes or None,keepdims=True, out=write)
        return PyImpl(inplace_func=f)
    def pullback(self, inputs, output, goutput):
        x = inputs[0]
        inputpat = "x"*x.ndim
        singpat = "".join(["1" if i in self.axes else "x" for i in xrange(x.ndim)])
        bcpat = singpat+","+inputpat
        return [cgt.broadcast("*", goutput, cgt.broadcast("==", output, x, bcpat), bcpat)]
        # XXX doesn't deal well with corner case
    def shp_apply(self, inputs):
        x = inputs[0]
        s = cgt.shape(x)
        return [(cgt.constant(1) if i in self.axes else s[i]) for i in xrange(x.ndim)]
    def typ_apply(self, inputs):
        return inputs[0].get_type()

class Argmax(Op):
    def __init__(self, axis):
        self.axis = axis
    def get_diff(self, _):
        return [False]
    def __str__(self):
        return "argmax{%s}"%self.axis
    def get_py_impl(self):
        def f(reads, write):
            write.flat[:] = reads[0].argmax(axis=self.axis)
        return PyImpl(inplace_func=f)
    def shp_apply(self, inputs):
        x = inputs[0]
        s = cgt.shape(x)
        return [(cgt.constant(1) if i == self.axis else s[i]) for i in xrange(x.ndim)]
    def typ_apply(self, inputs):
        return TensorType('i8', inputs[0].ndim)

# TODO consider reducing code duplication for e.g. shp_apply

# Casting / data movement
# ----------------------------------------------------------------

# class Copy(Op):
#     pass # TODO

# Slicing
# ----------------------------------------------------------------

class GetSli(Op):
    call_type = "valret"
    def __init__(self, axis):
        self.axis = axis
    def get_diff(self, _):
        return [True,False,False,False]
    def get_py_impl(self):
        def f(reads):
            x,start,stop,step=reads
            slices = [slice(None,None,None) for _ in xrange(x.ndim)]
            slices[self.axis] = slice(start,stop,step)
            if step < 0:
                raise NotImplementedError
            return x[slices]
        return PyImpl(valret_func=f)
    def pullback(self, inputs, output, goutput):
        ginput = cgt.zeros_like(inputs[0])
        return [Result(IncSli(self.axis), [ginput] + inputs[1:] + [goutput])] + [None]*3
    def shp_apply(self, inputs):
        arr, start, stop, step = inputs
        s = cgt.shape(arr) #pylint: disable=W0621
        newshape = copy.copy(s)
        newshape[self.axis] = cgt.ceil_divide(stop - start, step)
        return newshape
    def typ_apply(self, inputs):
        assert inputs[1].dtype == inputs[2].dtype == inputs[3].dtype == 'i8'
        return inputs[0].get_type()
    def get_c_impl(self, inputs):
        raise MethodNotDefined # XXXXXXXXX
        x = inputs[0]
        openloops = " ".join(["for (int i%(ax)s=0; i%(ax)s < write->shape()[%(ax)s]; i%(ax)s += %(step)s) {"%dict(ax=ax, step="step" if ax==self.axis else "1") for ax in xrange(x.ndim)])
        closeloops = "}"*x.ndim
        inidxexpr =  " + ".join([("(start+i%i*step)"%ax if ax==self.axis else "i%i"%ax) + "*in->stride(%i)"%ax for ax in xrange(x.ndim)])
        outidxexpr = " + ".join(["i%i"%ax + "*write->stride(%i)"%ax for ax in xrange(x.ndim)])
        code = r"""
extern "C" void $function(void* cldata, cgtArray** reads, cgtArray* write) {
    cgtArray *in=reads[0];
    long start = ((long*)reads[1]->data())[0];
    //long stop = ((long*)reads[2]->data())[0];
    long step = ((long*)reads[3]->data())[0];
    %(openloops)s
        ((%(cdtype)s*)write->data())[%(outidxexpr)s] = ((%(cdtype)s*)in->data())[%(inidxexpr)s];
    %(closeloops)s
}
"""%dict(openloops=openloops, outidxexpr=outidxexpr, inidxexpr=inidxexpr, closeloops=closeloops,
    cdtype=np2c[inputs[0].dtype])
        return CImpl(code)

class IncSli(Op):
    writes_to_input = 0
    def __init__(self, axis):
        self.axis = axis
    def get_diff(self, _):
        return [True,False,True,True]
    def get_py_impl(self):
        def f(reads, write):
            x, start, stop, step, y=reads
            slices = [slice(None,None,None) for _ in xrange(x.ndim)]
            slices[self.axis] = slice(start,stop,step)
            write[slices] += y # XXX check that it's incremental
        return PyImpl(inplace_func=f)
    def pullback(self, inputs, output, goutput):
        _x, start, stop, step, _y = inputs
        gx = goutput
        gy = Result(GetSli(self.axis), [goutput, start, stop, step])        
        return [gx, None, None, None, gy]
    def shp_apply(self, inputs):
        return cgt.shape(inputs[0])
    def typ_apply(self, inputs):
        return inputs[0].get_type()
    def get_c_impl(self, inputs):
        x = inputs[0]
        openloops = " ".join(
            ["for (int i%(ax)s=0; i%(ax)s < inc->shape()[%(ax)s]; ++i%(ax)s) {"%dict(ax=ax) for ax in xrange(x.ndim)])
        closeloops = "}"*x.ndim
        incidxexpr =  " + ".join(["i%i"%ax + "*inc->stride(%i)"%ax for ax in xrange(x.ndim)])
        outidxexpr = " + ".join([("i%i*step+start"%ax if ax == self.axis else "i%i"%ax) + "*out->stride(%i)"%ax for ax in xrange(x.ndim)])
        code = r"""
extern "C" void $function(void* cldata, cgtArray** reads, cgtArray* write) {
    cgtArray *in=reads[0], *inc = reads[4], *out=write;
    long start = ((long*)reads[1]->data())[0];
    //long stop = ((long*)reads[2]->data())[0];
    long step = ((long*)reads[3]->data())[0];
    cgt_assert(in->size() == out->size());
    if (out->data() != in->data()) cgt_memcpy(cgtCPU, cgtCPU, out, in, out->nbytes());
    %(openloops)s
        ((%(cdtype)s*)out->data())[%(outidxexpr)s] += ((%(cdtype)s*)inc->data())[%(incidxexpr)s];
    %(closeloops)s
}
"""%dict(openloops=openloops, outidxexpr=outidxexpr, closeloops=closeloops,
    cdtype=np2c[inputs[0].dtype], incidxexpr=incidxexpr)
        return CImpl(code)

class GetFlatIndices(Op):
    def get_diff(self, _):
        return [True,False]
    def get_py_impl(self):
        def f(reads, write):
            np.copyto(write, reads[0].flat[reads[1]])
        return PyImpl(inplace_func=f)
    def pullback(self, inputs, output, goutput):
        x,inds = inputs
        ginput = cgt.zeros_like(x)
        return [Result(IncFlatIndices(), [ginput, inds, goutput]), None]
    def shp_apply(self, inputs):
        return cgt.shape(inputs[1])
    def typ_apply(self, inputs):
        assert inputs[1].ndim == 1 and dtype_kind(inputs[1].dtype) == 'i'
        return TensorType(inputs[0].dtype,1)


class IncFlatIndices(Op):
    writes_to_input = 0
    def get_diff(self, _):
        return [True,False,True]
    def get_py_impl(self):
        def f(reads, write):
            x,inds,y = reads
            write.flat[inds] += y # XXX
        return PyImpl(inplace_func=f)
    def shp_apply(self, inputs):
        return cgt.shape(inputs[0])
    def typ_apply(self, inputs):
        return inputs[0].get_type()


class Flip(Op):
    def __init__(self, axes):
        self.axes = axes
    def get_diff(self, _):
        return [True]
    def get_py_impl(self):
        def f(reads, write):
            x = reads[0]
            slices = [slice(0,None,None) for _ in xrange(x.ndim)]
            for ax in self.axes: slices[ax] = slice(None,None,-1)
            np.copyto(write, x[slices])
        return PyImpl(inplace_func=f)
    def pullback(self, inputs, output, goutput):
        return [cgt.flip(goutput, self.axes)]
    def shp_apply(self, inputs):
        return cgt.shape(inputs[0])
    def typ_apply(self, inputs):
        return inputs[0].typ
    def get_c_impl(self, inputs):
        x = inputs[0]
        openloops = " ".join(["for (int i%(ax)s=0; i%(ax)s < shape[%(ax)s]; ++i%(ax)s) {"%dict(ax=ax) for ax in xrange(x.ndim)])
        closeloops = "}"*x.ndim
        inidxexpr =  ",".join(["i%i"%ax for ax in xrange(x.ndim)])
        outidxexpr =  ",".join([("shape[%(ax)s] - 1 - i%(ax)s" if ax in self.axes else "i%(ax)s")%dict(ax=ax) for ax in xrange(x.ndim)])
        code = r"""
extern "C" void $function(void* cldata, cgtArray** reads, cgtArray* write) {
    cgtArray *in=reads[0], *out=write;
    cgt_assert(in->size() == out->size());
    const size_t* shape = in->shape();
    %(openloops)s
        out->at<%(cdtype)s>(%(outidxexpr)s) = in->at<%(cdtype)s>(%(inidxexpr)s);
    %(closeloops)s
}
"""%dict(openloops=openloops, outidxexpr=outidxexpr, closeloops=closeloops, 
    inidxexpr=inidxexpr, cdtype=np2c[inputs[0].dtype])
        return CImpl(code)



# Linalg
# ----------------------------------------------------------------


class Mul21(Op):
    c_extra_includes = ["cblas.h"]
    def __init__(self, tA):
        self.tA = tA
    def get_py_impl(self):
        def f(reads, write):
            x,y = reads
            if self.tA: x = x.T
            x.dot(y, out=write)
        return PyImpl(inplace_func=f)
    def get_replacement(self, inputs, analysis):
        if inputs[1] in analysis["node2sv"]:
            return cgt.sum(inputs[0],0 if self.tA else 1) * analysis["node2sv"][inputs[1]]
    def pullback(self, inputs, _output, goutput):
        return [cgt.outer(goutput,inputs[1]), Result(Mul21(not self.tA), [inputs[0],goutput])]
    def shp_apply(self, inputs):
        assertequal1(cgt.size(inputs[0],0 if self.tA else 1),cgt.size(inputs[1],0),
            "shape mismatch at matrix-vector multiplication")         
        return [cgt.size(inputs[0], 1 if self.tA else 0)]
    def typ_apply(self, inputs):
        return TensorType(inputs[0].dtype, 1)
    def get_expr(self, (xexpr,yexpr)):        
        return u"%s%s \u00D7 %s"%(xexpr, u"\u1d57" if self.tA else "", yexpr)

class Mul22(Op):
    def __init__(self, tA, tB):
        self.tA = tA
        self.tB = tB
    def get_py_impl(self):
        def f(reads, write):
            x,y = reads
            if self.tA: x = x.T
            if self.tB: y = y.T
            x.dot(y, out=write)
        return PyImpl(inplace_func=f)
    def pullback(self, inputs, output, goutput):
        return [Result(Mul22(False, not self.tB), [goutput, inputs[1]]),
                Result(Mul22(not self.tA, False), [inputs[0], goutput])]
    def shp_apply(self, inputs):
        return [cgt.size(inputs[0], 1 if self.tA else 0),cgt.size(inputs[1],0 if self.tB else 1)]
    def typ_apply(self, inputs):
        assertequal1(cgt.size(inputs[0],0 if self.tA else 1),cgt.size(inputs[1],1 if self.tB else 0), 
            "shape mismatch at matrix-matrix multiplication")         
        assert inputs[0].dtype==cgt.floatX and inputs[1].dtype==cgt.floatX
        return inputs[0].get_type()
    def get_closure(self, inputs):
        return [("tA",ctypes.c_bool, self.tA), ("tB",ctypes.c_bool, self.tB)]
    def c_name(self, npdtype):
        raise RuntimeError # move to get_c_impl
        return "mul22_%s"%(npdtype)
    # best gemm docs: https://software.intel.com/en-us/node/520775
    def get_c_impl(self, inputs):
        npdtype = inputs[0].dtype
        try:
            letter = {"f4":"s","f8":"d","c8":"c","c16":"z"}[npdtype]
        except KeyError:
            raise MethodNotDefined("Dtype %s not supported by this BLAS. Falling back to numpy"%npdtype)
        code = """
extern "C" void $function($closure* cl, cgtArray** AB, cgtArray* C) {
    cgtArray *A=AB[0], *B=AB[1];
    int lda = A->shape()[1], ldb = B->shape()[1], ldc = C->shape()[1];
    int M = C->shape()[0];
    int N = C->shape()[1];
    int K = A->shape()[cl->tA ? 0 : 1];
    const %(cdtype)s alpha=1, beta=0;
  cblas_%(letter)sgemm(CblasRowMajor, (CBLAS_TRANSPOSE)(cl->tA + 111), (CBLAS_TRANSPOSE)(cl->tB + 111), M, N, K, alpha, (%(cdtype)s*)A->data(), lda, (%(cdtype)s*)B->data(),
      ldb, beta, (%(cdtype)s*)C->data(), ldc);
}
"""%dict(letter=letter, cdtype = np2c[npdtype])
        return CImpl(code, includes=["cblas.h"], link_flags="-lblas")

class BatchedMul22(Op):
    def __init__(self, tA, tB):
        self.tA = tA
        self.tB = tB
    def get_py_impl(self):
        def f((x,y), z):
            for (xmat, ymat, zmat) in zip(x,y, z):
                if self.tA: xmat = xmat.T
                if self.tB: ymat = ymat.T            
                xmat.dot(ymat, out=zmat)
        return PyImpl(inplace_func=f)
    def pullback(self, inputs, output, goutput):
        return [Result(BatchedMul22(False, not self.tB), [goutput, inputs[1]]),
                Result(BatchedMul22(not self.tA, False), [inputs[0], goutput])]
    def shp_apply(self, inputs):
        return [cgt.size(inputs[0],0), cgt.size(inputs[0], 2 if self.tA else 1),cgt.size(inputs[1],1 if self.tB else 2)]
    def typ_apply(self, inputs):
        # assert inputs[0].dtype==cgt.floatX and inputs[1].dtype==cgt.floatX
        return inputs[0].get_type()

class Outer(Op):
    def get_py_impl(self):
        def f(reads, write):
            np.outer(reads[0], reads[1], out=write)
        return PyImpl(inplace_func=f)
    def pullback(self, inputs, _output, goutput):
        return [goutput.dot(inputs[0]), inputs[1].dot(goutput)]
    def shp_apply(self, inputs):
        return [cgt.size(inputs[0],0), cgt.size(inputs[1],0)]
    def typ_apply(self, _inputs):
        return TensorType(cgt.floatX, 2)



# BLAS 1
# ----------------------------------------------------------------

class Dot(Op):
    call_type = "valret"
    def get_py_impl(self):
        def f(reads):
            return np.dot(reads[0], reads[1])
        return PyImpl(valret_func=f)
    def pullback(self, inputs, _output, goutput):
        x, y = inputs
        return [y*goutput, x*goutput]
    def shp_apply(self, _inputs):
        return []
    def typ_apply(self, _inputs):
        return TensorType(cgt.floatX, 0)

# Composition
# ----------------------------------------------------------------

class Composition(Op):
    call_type = "valret"
    def __init__(self, inputs, outputs):
        self._inputs = inputs
        self._outputs = outputs
        analysis = analyze(outputs)
        node2shape = analysis["node2shape"]
        self._shp = tuple(node2shape[x] for x in outputs)
        assert [x.is_input() for x in inputs]
        self._nodes = list(topsorted(outputs))

        dio = set(differentiably_influences(outputs))
        wrt = [x for x in inputs if x in dio]

        self._goutput = [Argument(x.get_type()) for x in outputs]
        gwrt = pullback(self._outputs, self._goutput, wrt)
        
        wrtidx = 0
        self._gin = []
        for x in inputs:
            if x in dio:
                self._gin.append(gwrt[wrtidx])
                wrtidx += 1
            self._gin.append(None)

        self._diff = [node in dio for node in self._inputs]
        self._out_typs = [x.get_type() for x in outputs]

    def get_diff(self, _):
        return self._diff
    def get_py_impl(self):
        from . import execution # XXX
        f = execution.function(self._inputs, self._outputs)
        def py_impl(num_inputs):
            return tuple(f(num_inputs))
        return PyImpl(valret_func=py_impl)
    def pullback(self, inputs, output, goutput):
        # repl = {}
        # repl.update(utils.safezip(self._inputs, inputs))
        # repl.update(utils.safezip(self._outputs, output))
        # repl.update(utils.safezip(self._goutput, goutput))
        # return clone(self._gin, replace=repl)
        gwrt = pullback([output], [goutput], inputs)
    def shp_apply(self, inputs):
        out = clone(self._shp, replace=dict(utils.safezip(self._inputs, inputs)))
        return out
    def typ_apply(self, inputs):
        assert [x.get_type() for x in inputs] == [x.get_type() for x in self._inputs]
        return TupleType(*self._out_typs)
    @property
    def n_out(self):
        return len(self._outputs)
    def shapes(self):
        return self._shp
    def expand(self, inputs):
        return clone(self._outputs, replace=dict(utils.safezip(self._inputs, inputs)))
    def get_nodes(self):
        return self._nodes

class TupleIndex(Op):
    call_type="valret"
    def __init__(self, idx):
        self.idx = idx
    def get_py_impl(self):
        def f(reads):
            return reads[0][self.idx]
        return PyImpl(valret_func=f)
    def shp_apply(self, inputs):
        return cgt.shape(inputs[0])[self.idx]
    def typ_apply(self, inputs):
        intype = inputs[0].get_type()
        assert isinstance(intype, TupleType)
        return inputs[0].get_type()[self.idx]

class MakeTuple(Op):
    call_type="valret"
    def get_py_impl(self):
        def f(inputs):
            return tuple(inputs)
        return PyImpl(valret_func=f)
    def shp_apply(self, inputs):
        return tuple(cgt.shape(x) for x in inputs)
    def typ_apply(self, inputs):
        return TupleType(*(x.get_type() for x in inputs))
    
def unpack(tup):
    return [Result(TupleIndex(i),[tup]) for i in xrange(len(tup.get_type()))]

# Assertion and debug operations
# ----------------------------------------------------------------
class Assertion(Op):
    """
    Assertion gets evaluated when the graph is executed, and it prints out a stack trace on failure
    """
    def __init__(self, msg):
        self.stack = traceback.extract_stack()[:-2]
        self.msg = msg
    def typ_apply(self, inputs):
        x, = inputs
        assert x.ndim==0 and x.dtype=='i1'
        return TensorType('i8',0)
    def shp_apply(self, _):
        return []
    def get_py_impl(self):
        def f(reads, write):
            x = reads[0]
            if not x.item():
                self.display_error()
        return PyImpl(inplace_func=f)
    def display_error(self):
        print "Stack trace at failed assertion:"
        print "**************************"        
        traceback.print_list(self.stack)
        print "**************************"        
        raise AssertionError("Assertion failed. Message: %s. Above, you can find the stack trace of the failed node"%self.msg)

class DebugFunc(Op):
    """
    Call a function when the graph is executed
    """
    def __init__(self, yourfunc):
        self.yourfunc = yourfunc
    def typ_apply(self, _):
        return TensorType('i8',0)
    def shp_apply(self, _):
        return []
    def get_py_impl(self):
        def f(reads, write):
            def fn(*reads):
                self.yourfunc(*reads)
        return PyImpl(inplace_func=f)

def assert_(x,msg=None):
    dbgnode = Result(Assertion(msg or "(empty)"), [x])
    print "assertion", CACHER.simplify1(dbgnode)
    # add_debug_node(dbgnode)

def dbg_call(yourfunc, *args):
    add_debug_node(Result(DebugFunc(yourfunc), list(args)))

def add_debug_node(x):
    if debug_context.global_context is not None:
        debug_context.global_context.nodes.append(x)

class debug_context(object):
    global_context = None # TODO: what is this?
    def __init__(self):
        self.nodes = []
    def __enter__(self):
        assert debug_context.global_context is None, "can only be in one debug context at a time"
        debug_context.global_context = self
        return self
    def __exit__(self, *_args):
        debug_context.global_context = None


# ================================================================
# Funcs wrapping ops (numpy-like)
# ================================================================

# ================================================================
# Graph Optimization
# ================================================================

def analyze(outputs):
    with disable_cacher():
        analysis = init_analysis()
        for node in topsorted(outputs):
            do_analysis(node, analysis)
        return analysis


def simplify_and_analyze(outputs):
    assert isinstance(outputs, list)
    analysis = init_analysis()
    repl = {}
    for output in outputs: update_simplify_map(output, analysis, repl)
    return [repl[node] for node in outputs], analysis

def process_top_stack_item_and_maybe_get_replacement(stack, analysis, repl): #pylint: disable=W0621
    """
    Helper function for update_simplify_map, which performs an update to the 
    stack, which stores the state of the simplification computation.
    
    Suppose the top element of the stack is `(orig, node)`, where `orig` is
    the original node and `node` is simpler than `orig` but not fully simplified.
    We can only guarantee that `node` is fully simplified after all of its parents are in the
    map `repl`.

    This function iterates over the parents of `node` and looks for one that is not in `repl`
    If we find one, called `par`, put `(orig, node)` back on the stack and `(par, par)` on top of it, and return.

    If all of the parents are already in `repl`, then we can try to compute a newly simplified version of `orig`.

    """
    (orig,node) = stack.pop()
    if node.is_input():
        return (orig,node)
    else:
        for par in node.parents: 
            if par not in repl:
                stack.append((orig,node))
                stack.append((par,par))
                return
        newparents = [repl[p] for p in node.parents]
        newnode = Result(node.op, newparents, typ=node.get_type())
        newnewnode = maybe_replace(newnode, analysis, repl)
        if newnewnode is None:
            return (orig,newnode)
        else:
            assert newnewnode.get_type() == orig.get_type()
            if newnewnode in repl:
                return (orig, newnewnode)
            else:
                stack.append((orig, newnewnode))

def update_simplify_map(node, analysis, repl):
    """
    Non-recursive version of simplification algorithm.
    Compute a fully simplified version of `node` and its ancestors
    When this function finishes, `repl[node]` is the simplified version of `node`,
    and repl[anc] is the simplified version of each node `anc` which is an ancestor of `node`.
    MOreover, analysis contains 

    This algorithm is most simply described recursively, and the implementation below is
    a conversion of the recursive algorithm into a stack-based algorithm (to avoid
    stack overflows). 
    (TODO: bring back recursive version for reference)

    The stack contains pairs `(orig, replacement_candidate)`, where `orig` is a node in the original
    graph (i.e., an ancestor of `node`) and `replacement_candidate` is a simplified version of it, but
    not necessarily fully simplified. We do a depth-first search on the graph, computing for each node
    the simplified version of all its parents, then we try to simplify that node.
    One tricky aspect is that once we've simplified the parents, we might apply some identity at that node.
    If that happens, we obtain a new node with non-simplified parents, so we put that on the stack.

    """
    stack = [(node,node)] #pylint: disable=W0621
    while stack:
        # Given (orig, node) on top of the stack, we visit one un-simplified parent of node,
        # putting it on the stack if necessary. If all parents are already simplified, then we can
        # check if any replacements can be applied. If we can, we return this pair and add it to our
        # dict `repl` which stores the current replacements.
        maybe_pair = process_top_stack_item_and_maybe_get_replacement(stack, analysis, repl)
        if maybe_pair:
            (orig,node) = maybe_pair                                    #pylint: disable=W0633
            # if not node.is_input():
            #     for shpcmp in node.op.shp_apply(node.parents): 
            #         update_simplify_map(shpcmp, analysis, repl, True)
            do_analysis(node, analysis)
            repl[orig] = node
            repl[node] = node
            assert orig.ndim==node.ndim

def do_analysis(node, analysis):

    node2hash = analysis["node2hash"]
    node2shape = analysis["node2shape"]    
    node2sv = analysis["node2sv"]

    # -- HASH --
    h = node.get_hash(node2hash)
    node2hash[node] = h
    analysis["hash2node"][h] = node
    # -- SHAPE --
    if node.is_input():
        node2shape[node] = cgt.shape(node)
    elif isinstance(node.op, TupleIndex):
        node2shape[node] = node2shape[node.parents[0]][node.op.idx]
    else:
        newparents = node.parents
        node2shape[node] = node.op.shp_apply(newparents)
        # assert all([s.dtype == "i8" for s in node2shape[node]])
    assert len(node2shape[node]) == node.ndim or isinstance(node.get_type(),TupleType)
    # -- SCALAR VALUE --
    if isinstance(node, Result):
        op = node.op
        if isinstance(op, Fill):
            node2sv[node] = op.value
        elif isinstance(op, ConstantTensor) and utils.is_singleton(op.value):
            node2sv[node] = op.value.flat[0]
        elif isinstance(op, Repeat) and newparents[0] in node2sv:
            node2sv[node] = node2sv[newparents[0]]
        elif isinstance(op, (ElwiseUnary, ElwiseBinary)) and all(p in node2sv for p in newparents):
            node2sv[node] = node.op.info.pyfunc(*(node2sv[p] for p in newparents))

VERBOSE_OPTIMIZATION = False

def maybe_replace(node, analysis, repl):
    if node.is_input(): return
    if isinstance(node.op, Constant): return
    # -- CSE --
    node2hash = analysis["node2hash"]
    h = node.get_hash(node2hash)
    if h in analysis["hash2node"]:
        if VERBOSE_OPTIMIZATION: print "Did CSE"
        newnode = analysis["hash2node"][h]
        assert newnode in repl and newnode.op.__class__ == node.op.__class__
        return newnode
    parents = node.parents
    # -- CONSTANT PROP --
    # ASSUMPTION: the only type of nullary ops that we can propagate this way
    # are subclasses of Constant
    if len(parents) > 0 and all(isinstance(par.op, Constant) for par in parents):
        try:
            out = cgt.constant(py_numeric_apply(node, [p.op.value for p in parents]))
            if VERBOSE_OPTIMIZATION: print "Did constant prop on %s"%node.op
            return out
        except MethodNotDefined:
            utils.warn("Couldn't get a python impl of %s"%node.op)
    # -- SIZE --
    if isinstance(node.op, Size):
        s = analysis["node2shape"][parents[0]][node.op.axis]
        if not (isinstance(s.op, Size) and s.parents[0] == node.parents[0]): 
            if VERBOSE_OPTIMIZATION: print "Did size prop"
            return s
    # -- OP IDENTITY --
    maybe_repl = node.op.get_replacement(parents, analysis)
    if maybe_repl is not None: 
        if VERBOSE_OPTIMIZATION: print "Applied op-specific identity for %s"%node.op
        return maybe_repl

    return None

def simplify(xs):
    assert isinstance(xs, list)
    return simplify_and_analyze(xs)[0]

def simplify1(x):
    return simplify([x])[0]

def init_analysis():
    return {"node2hash":{},"hash2node":{},"node2shape":{},"node2sv":{},"repl":{}}

class AnalysisCacher(object):
    def __init__(self):
        self.analysis = init_analysis()
        self.repl = {}
    def simplify(self, xs):
        with disable_cacher(): # not actually necessary but seems reasonable
            for x in xs: self.simplify1(x)
        return [self.repl[x] for x in xs]
    def simplify1(self, x):
        assert isinstance(x, Node)
        with disable_cacher():
            update_simplify_map(x, self.analysis, self.repl)
        return self.repl[x]

CACHER = AnalysisCacher()
CACHER_ENABLED = False

class disable_cacher(object):
    def __enter__(self):
        global CACHER_ENABLED
        self.prevstate = CACHER_ENABLED
        CACHER_ENABLED = False
    def __exit__(self, *args):
        global CACHER_ENABLED
        CACHER_ENABLED = self.prevstate

def assert1(x, msg=""):
    if not CACHER_ENABLED: return
    b = CACHER.simplify1(x)
    if isinstance(b.op, Constant):
        if not b.op.value:
            raise AssertionError(msg)

def assertn(xs,msg=""):
    if not CACHER_ENABLED: return
    bs = CACHER.simplify(xs)
    if isinstance(bs.op, Constant):
        if not np.all(bs.op.val):
            raise AssertionError(msg)

def _noderepr(x):
    if isinstance(x.op, ConstantTensor):
        return x.op.value.item()
    elif isinstance(x.op, ConstantTuple):
        return x.op.value
    else:
        return "?"

def assertequal1(x,y,msg):
    if not CACHER_ENABLED: return
    x = as_node(x)
    y = as_node(y)
    simpx = CACHER.simplify1(x)
    simpy = CACHER.simplify1(y)
    if isinstance(simpx.op,Constant) and isinstance(simpy.op,Constant) and simpx.op.value != simpy.op.value:
        raise AssertionError(msg + "\nlhs: %s. rhs: %s"%(_noderepr(simpx), _noderepr(simpy)))

def assertequaln(xs,ys,msg):
    if not CACHER_ENABLED: return
    xs = map(as_node,xs)
    ys = map(as_node,ys)
    simpxs = CACHER.simplify(xs)
    simpys = CACHER.simplify(ys)
    for (x,y) in utils.safezip(simpxs,simpys):
        if isinstance(x.op,Constant) and isinstance(y.op,Constant) and x.op.value != y.op.value:
            raise AssertionError(msg + "\nlhs: %s. rhs: %s"%(tuple(map(_noderepr,simpxs)), tuple(map(_noderepr,simpys))))



# ================================================================
# Graph Traversal
# ================================================================

def topsorted(outputs):
    assert isinstance(outputs, (list,tuple))
    marks = {}
    out = []
    stack = [] #pylint: disable=W0621
    # i: node
    # jidx = number of children visited so far from that node
    # marks: state of each node, which is one of
    #   0: haven't visited
    #   1: have visited, but not done visiting children
    #   2: done visiting children
    for x in outputs:
        stack.append((x,0))
        while stack:
            (i,jidx) = stack.pop()
            if jidx == 0:
                m = marks.get(i,0)
                if m == 0:
                    marks[i] = 1
                elif m == 1:
                    raise ValueError("not a dag")
                else:
                    continue
            ps = i.parents
            if jidx == len(ps):
                marks[i] = 2
                out.append(i)
            else:
                stack.append((i,jidx+1))
                j = ps[jidx]
                stack.append((j,0))
    return out

def count_nodes(outputs):
    if isinstance(outputs, Node): outputs = [outputs]
    return len(list(topsorted(outputs)))

def clone(nodes, replace=None):
    assert isinstance(nodes, list)
    if isinstance(nodes, tuple):
        return tuple(clone(x,replace) for x in nodes)
    if replace is None: replace = {}
    else:
        assert isinstance(replace, dict)
        replace = replace.copy()
    for node in topsorted(nodes):
        if node.is_input():
            if node not in replace:
                replace[node] = node
        else:
            replace[node] = Result(node.op, [replace[p] for p in node.parents], typ=node.get_type())
    return [replace[node] for node in nodes]

def alloc_from_shp(shp, typ):
    if isinstance(shp, tuple):
        return tuple([alloc_from_shp(shpel,typel) for (shpel,typel) in utils.safezip(shp,typ)])
    else:
        return np.empty(shp,typ.dtype)

def alloc_output(node, vals):
    typ = node.get_type()
    shp = get_numeric_shape_fun(node)(vals)
    return alloc_from_shp(shp,typ)

def _flatten_lists(lis):
    out = []
    sizes = []
    for li in lis:
        out.extend(li)
        sizes.append(len(li))
    return out,sizes

def _unflatten_list(li,sizes):
    start = 0
    out = []
    for sz in sizes:
        out.append(li[start:start+sz])
        start += sz
    return out


def get_numeric_shape_fun(node):
    args = [make_argument(p.get_type()) for p in node.parents]
    # outputs = simplify(node.op.shp_apply(args))
    syshape = node.op.shp_apply(args)

    if isinstance(syshape, list):
        istuple = False
    elif isinstance(syshape, tuple):
        assert all(isinstance(elem,list) for elem in syshape)
        istuple = True
        syshape,sizes = _flatten_lists(syshape)
    else:
        raise ValueError("shape should be a list or tuple of lists. got %s"%syshape)

    singletuple = not isinstance(syshape, list)
    if singletuple: # XXX
        syshape = [cgt.make_tuple(*syshape)]
    nodes = topsorted(syshape)
    def fn(vals):
        node2val = {node:val for (node,val) in utils.safezip(args, vals)}
        for node in nodes:
            if not node.is_argument():
                node2val[node] = py_numeric_apply(node, [node2val[p] for p in node.parents])
        nushape = [node2val[node] for node in syshape]
        if istuple:
            return tuple(_unflatten_list(nushape, sizes))
        else:
            return nushape 
    return fn

def py_numeric_apply(node, vals):
    try:
        py_impl = node.op.get_py_impl()
    except MethodNotDefined:
        print 'Op %s has no Python implementation' % repr(node.op)
        raise
    if node.op.call_type == "valret":
        out = py_impl.valret_func(vals)
    else:
        out = alloc_output(node,vals)
        py_impl.inplace_func(vals, out)
    return out

class NonDifferentiable(Exception):
    pass

class Disconnected(Exception):
    pass

class Todo(Exception):
    pass

class ShapeError(Exception):
    pass

class AllocationError(Exception):
    pass

class MethodNotDefined(Exception):
    pass


def get_cgt_src_root():
    osp = os.path
    return osp.dirname(osp.dirname(osp.realpath(__file__)))

_CONFIG = None
def load_config():
    osp = os.path
    global _CONFIG
    if _CONFIG is None:
        from thirdparty.configobj import ConfigObj
        from thirdparty.validate import Validator
        rcfileloc = osp.join(osp.expanduser("~/.cgtrc"))
        specfilename = osp.join(get_cgt_src_root(), "cgtrc_spec.ini")
        _CONFIG = ConfigObj(rcfileloc, configspec=specfilename)
        val = Validator()
        test = _CONFIG.validate(val,preserve_errors=True)
        if test is not True:
            for (k,v) in test.items():
                if v is not True:
                    utils.error("%s: %s in %s"%(k,v.message,rcfileloc))
            raise ValueError
        envflags = os.getenv("CGT_FLAGS")
        if envflags:
            pairs = envflags.split(",")
            for pair in pairs:
                lhs,rhs = pair.split("=")
                assert lhs in _CONFIG
                _CONFIG[lhs] = rhs
    return _CONFIG

def modify_config(**kws):
    load_config()
    for (name,val) in kws.iteritems():
        _CONFIG[name] = val
