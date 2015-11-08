import sys, numpy as np, hashlib, copy, cPickle, ctypes, os, os.path as osp
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

        The following string representations are used: i1,i2,i4,i8,  f4,f8,f16  c8,c16,c32
        So either we're using single (f4, c8) or double (f8, c16) or quad (f16, c32)
        Note that quad precision is very useful for gradient checking
        """
        dt = np.dtype(dt)
        k = dt.kind
        if k=='f':
            return cgt.floatX
        elif k in 'biu':
            return 'i'+str(dt.itemsize)
        elif k=='c':
            return cgt.complexX
        else:
            raise ValueError("Invalid dtype %s"%dt)

def as_valid_array(x, dtype=None):
    """
    Converts to numpy array and dtype with valid precision
    """
    # surprising how convoluted this function needs to be
    if not isinstance(x, np.ndarray): x = np.asarray(x,dtype=dtype)
    if dtype is None: dtype = Dtype.canon(x.dtype)
    x = x.astype(dtype)
    if not x.flags.c_contiguous: x = x.copy()
    return x

def as_valid_tuple(x):
    return tuple(as_valid_array(a) for a in x)
    # @TUPLES_OF_TENSORS

def as_valid_arg(x):
    if isinstance(x, tuple):
        return as_valid_tuple(x)
    else:
        return as_valid_array(x)

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
    def __repr__(self):
        return "Tensor(%s,%s)"%(self.dtype, self.ndim)
    def __eq__(self, other):
        return self.dtype == other.dtype and self.ndim == other.ndim
    def __hash__(self):
        return hash((self.dtype, self.ndim))

class TupleType(Type):
    """
    A compound type consisting of a tuple of other types
    Only tuples of tensors are currently supported
    """
    def __init__(self, *eltypes):
        assert all(isinstance(eltype, TensorType) for eltype in eltypes) # @TUPLES_OF_TENSORS
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
    def __hash__(self):
        return hash((self.eltypes, self.dtype))

class Device(object):
    """
    Represents a location where a computation is performed
    devtype: cpu vs gpu
    idx: index of which device
    """
    def __init__(self, devtype="cpu", idx=0):
        assert isinstance(devtype,str) and isinstance(idx,int)
        self.devtype = devtype
        self.idx = idx
    def __eq__(self, other):
        return self.devtype == other.devtype and self.idx == other.idx
    def __hash__(self):
        return hash((self.devtype, self.idx))
    def __repr__(self):
        return "%s/%s"%(self.devtype,self.idx)

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
    """
    _promote with multiple operands
    """
    return reduce(_promote, xtypes)

def dtype_kind(dtype):
    """
    one of f,c,i
    """
    assert isinstance(dtype, str)
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

    counter = 0 # allows unique identification of argument nodes

    # Constants
    # ----------------------------------------

    def __init__(self, typ, op, parents, props=None, fixed_shape=None, name=None):
        self.typ = typ
        self.op = op
        self.parents = parents        
        self.props = props or {}
        self._fixed_shape = fixed_shape
        self.name = name
        self.counter = Node.counter
        Node.counter += 1

    def __repr__(self):
        if self.op is None:
            return "Argument{%s,name='%s'}"%(self.typ,self.name)            
        else:
            return "Result{%s}"%(str(self.op))

    # CGT-specific
    # ----------------------------------------


    def is_argument(self):
        """
        Returns whether Node is an argument
        """
        return self.op is None
    def is_data(self):
        """
        Returns whether Node's Op is data
        """
        return self.op is not None and self.op.is_data_op
    def is_input(self):
        """
        Returns whether this node is either an argument or is data
        """
        return self.is_argument() or self.is_data()
    def get_diff(self):
        """
        Returns a sequence of bool indicating whether output is differentiable wrt each input
        """
        return [] if self.op is None else self.op.get_diff(len(self.parents))
    def is_tensor(self):
        """
        Returns whether this node's type (self.typ) is TensorType
        """
        return isinstance(self.typ, TensorType)
    def is_tuple(self):
        """
        Returns whether this node's type (self.typ) is TupleType
        """
        return isinstance(self.typ, TupleType)
    def is_scalar(self):
        return self.is_tensor() and self.ndim==0
    def get_hash(self, node2hash):
        """
        Return UNIQUE string identifying this Node
        """
        if self.is_input() or self.op.is_random_op:
            return str(self.counter)
        else:
            hashobj = hashlib.md5(self.op.get_hash())
            for p in self.parents: 
                hashobj.update(node2hash[p])
            return hashobj.hexdigest()
    def clone(self, newparents):
        """
        Create a new Node that applies self.op to `newparents`
        Preserve annotations on this node (.props)
        """
        if self.is_input(): return self
        else: return Node(self.typ, self.op, newparents, props = self.props)
    def get_fixed_shape(self):
        """
        Returns a tuple of int or None. You'll get ints if this is an argument or data node
        with fixed shape provided
        """
        if self.is_data():
            return self.op.get_fixed_shape()
        return (None,)*self.ndim if self._fixed_shape is None else self._fixed_shape

    # Math Overloads
    # ----------------------------------------

    __array_priority__ = 1000 # precedence over numpy operators

    def __neg__(self):
        return Result(ElwiseUnary("neg"), [self])
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
    # GOT RID OF __eq__ and __ne__ because they might lead to funny problems when
    # people want equality check. No strong opinion on whether they should be included
    # def __eq__(self, other):
    #     return equal(self, other)
    # def __ne__(self, other):
    #     return not_equal(self, other)
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
    def __getitem__(self, slis):
        if self.is_tuple():
            assert isinstance(slis, int), "TupleType can be only be indexed by an int"
            return cgt.tuple_index(self, slis)
        else:            
            return cgt.subtensor(self, slis)
    def __iter__(self):
        if self.is_tensor():
            raise TypeError("Array variable is not iterable")            
        if self.is_tuple():
            return iter(unpack(self))
        else:            
            raise NotImplementedError
    def __len__(self):
        if isinstance(self.typ, TupleType):
            return len(self.typ)
        else:
            raise ValueError("Node of type Tensor has no __len__")
    def __nonzero__(self):
        return True


    # Properties like numpy ndarray
    # ----------------------------------------

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

    # More math overloads
    # ----------------------------------------

    def reshape(self, shp):
        "see cgt.reshape"
        assert isinstance(shp, (list,tuple))
        return cgt.reshape(self, shp)
    def dot(self, other):
        "see cgt.dot"
        return cgt.dot(self, other)
    def sum(self, axis=None, keepdims=False):
        "see cgt.sum"
        return cgt.sum(self, axis=axis, keepdims=keepdims)
    def prod(self, axis=None, keepdims=False):
        "see cgt.prod"
        return cgt.prod(self, axis=axis, keepdims=keepdims)
    def max(self, axis=None, keepdims=False):
        "see cgt.max"
        return cgt.max(self, axis=axis, keepdims=keepdims)
    def argmax(self, axis=None, keepdims=False):
        "see cgt.argmax"
        return cgt.argmax(self, axis=axis, keepdims=keepdims)
    def mean(self, axis=None, keepdims=False):
        "see cgt.mean"
        return cgt.mean(self, axis=axis, keepdims=keepdims)
    def transpose(self, axes=None):
        "see cgt.transpose"
        return cgt.transpose(self, axes=axes)
    def flatten(self):
        "see cgt.flatten"
        return cgt.flatten(self)
    def dimshuffle(self, pattern):
        "see cgt.dimshuffle"
        return cgt.dimshuffle(self, pattern)


def _ndarray_type(value):
    assert isinstance(value, np.ndarray)
    return TensorType(value.dtype, value.ndim)

def _get_value_type(value):
    if isinstance(value, np.ndarray):
        return TensorType(value.dtype, value.ndim)
    elif isinstance(value, tuple):
        return TupleType(*map(_get_value_type, value))

def num_components(node):
    return len(node.typ) if isinstance(node.typ, TupleType) else 1


class Op(object):
    """
    Describes an operation that will be performed on some data.
    """

    # attributes that can be overwritten in subclasses
    return_type = "byref" # or "byval"
    writes_to_input = -1 # whether output is allowed to have same underlying data as input
    available_impls = () # python, native_cpu, native_gpu
    is_data_op = False
    is_random_op = False

    # pylint: disable=W0613

    def shp_apply(self, parents):
        """
        Return output shapes as a function of input nodes
        """
        raise NotImplementedError
    def typ_apply(self, parent_types):
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
        return type(self).__name__

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
    def get_native_compile_info(self, inputs, devtype):
        """
        returns NativeCompileInfo 
        """
        raise MethodNotDefined
    def get_py_func(self, input_types):
        """
        Returns python function that implements this operation
        """
        raise MethodNotDefined
    def get_py_callable(self, input_types):
        func = self.get_py_func(input_types)
        return PyCallable(self, len(input_types), func)
    def __repr__(self):
        """
        Get a human-readable description of the Op, including its attributes
        """
        return type(self).__name__


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

def default_props():
    props = {}
    props["default_device"] = _CONFIG["default_device"]
    if _CONFIG["debug"] and "stack" not in props: props["stack"] = traceback.extract_stack()[:-3]
    return props

def Result(op, parents, typ=None, props=None, name=None):
    """
    Just here as as "damage control" after some refactoring/renaming
    """
    parents = map(as_node, parents)
    typ = op.typ_apply([parent.typ for parent in parents]) if typ is None else typ
    return Node(typ, op, parents, props=props or default_props(), name=name)

def Argument(typ, name=None, fixed_shape=None, props=None):    
    """
    Just here as as "damage control" after some refactoring/renaming
    """
    return Node(typ, None, [], props=props or default_props(), fixed_shape=fixed_shape, name=name)

class GetData(Op):
    is_data_op=True
    return_type="byval"
    available_impls=("python","native_cpu","native_gpu")
    def __init__(self, typ):
        self.typ = typ
    def typ_apply(self, _):
        return self.typ

class InMemoryData(GetData):
    def __init__(self, value, device=None, fixed_shape_mask=None):
        value = as_valid_array(value)
        GetData.__init__(self, _ndarray_type(value))
        self.device = device or get_config()["default_device"]
        self.use_numpy = cgt.get_config()["backend"] == "python" 
        # use_numpy: whether to store the data as a numpy array or a CppArrayWrapper object
        if self.use_numpy:
            assert self.device.devtype=="cpu","can only use numpy for cpu. maybe you need to set backend=native?"
        else:
            self.dataptr = ctypes.c_long(0)
        self.set_value(value)
        assert self._value.dtype != object


        if fixed_shape_mask is None:  fixed_shape_mask = (False,)*self._value.ndim
        elif fixed_shape_mask == "all": fixed_shape_mask = (True,)*self._value.ndim
        self.fixed_shape = tuple(s if bfixed else None for (s, bfixed) in zip(value.shape, fixed_shape_mask))

    def get_py_func(self, _):
        def f(_): 
            return self.get_value()
        return f
    def get_native_compile_info(self, _input_types, _devtype):
        code=r"""
            CGT_EXPORT_C cgtArray* $function($closure* cldata, cgtArray** reads) {
                return *(cgtArray**)cldata->pptr;
            }"""
        pptr = self.get_pptr()
        return NativeCompileInfo(code, closure_triples=[("pptr", ctypes.c_void_p, pptr)], 
            store_objects=self._value)
    def __repr__(self):
        return "Data{%s}"%(self.typ)
    def get_device(self):
        return self.device
    def get_value(self):
        return self._value if self.use_numpy else self._value.to_numpy()        
        # XXX use more explicit names
    def get_shape(self):
        return self._value.shape
    def get_size(self):
        return self._value.size
    def set_value(self, value):
        value = np.asarray(value, dtype=self.typ.dtype)
        if self.use_numpy:
            self._value = value.copy()
        else:
            self._value = cgt.cycgt.CppArrayWrapper.from_numpy(value, self.device.devtype, False) #pylint: disable=E1101
            self.dataptr.value = self._value.ptr
    def get_pptr(self):
        return ctypes.addressof(self.dataptr)
    def get_fixed_shape(self):
        return self.fixed_shape

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
    i.e., the Jacobian doutputs/dnode != 0
    in reverse topological sorted order

    optionally pass in nodelist=topsorted(outputs)
    (save on recomputation of topsort)
    """
    if nodelist is None: nodelist = list(topsorted(outputs))
    diset = set(outputs)
    for node in reversed(nodelist):
        if node in diset and not node.is_input():
            for (p,d) in utils.safezip(node.parents, node.get_diff()):
                if d: diset.add(p)
    return diset

def differentiably_influenced_by(wrt, outputs=None, nodelist=None):
    """
    Return the set of nodes that are differentiably influenced by outputs,
    i.e., the set of x for which Jacobian dx/dwrt is nonzero
    """
    assert (outputs is None) != (nodelist is None) # one of these are provided
    if nodelist is None: nodelist = list(topsorted(outputs))
    dibset = set(wrt)
    for node in nodelist:
        if any(p in dibset and d for (p,d) in utils.safezip(node.parents, node.get_diff())):
            dibset.add(node)
    return dibset

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

    # Check that each output is differentiably influenced by some input
    badwrtset = set(wrt).difference(dio)
    if badwrtset:
        raise NonDifferentiable("Outputs not differentiable wrt %s"%badwrtset)

    # Check that each input differentiably influences some output
    badoutset = set(outputs).difference(dibw)
    if badoutset:
        raise NonDifferentiable("Outputs %s not differentiable wrt any of %s"%(badoutset, badwrtset))

    # Map node to a list of gradient terms
    # These gradient terms will be summed up when we visit the node, when iterating through the nodes
    # in reverse toplogical order
    var2gs = defaultdict(list)
    for (node, gnode) in utils.safezip(outputs, goutputs):
        var2gs[node] = [gnode]

    # "active" nodes are the ones that are differentially influenced by the inputs
    # and also differentiably influence the outputs. These are the nodes where we need to call the
    # "pullback" function to backpropagate derivatives
    active = dio.intersection(dibw)

    # Iterate through nodes in reverse topological order
    for node in reversed(nodelist):
        if node not in active: continue
        # Once we reach a node, we have already backpropagated from all parents
        # So now we can sum up the gradients
        if len(var2gs[node]) > 1:
            if node.is_tensor():
                var2gs[node] = [cgt.add_multi(var2gs[node])]
        # There's only one gradient in the list at this point
        gnode = var2gs[node][0]
        if not node.is_input():
            if isinstance(node.op, TupleIndex):
                # A little complication that arises when we have a node of Tuple type
                # Instead of having a list of gradient terms, we're going to store a list with one element
                # and inside that list, we have a list of gradient terms for each tuple element
                # Let's say we have a tuple node (y,z) with predecessor x                
                # x       ->      (y, z)      ->      y
                # input        Result{foo_op}      Result{TupleIndex{0}}
                # At this point in the code, we just got gy.
                # we first set the gradient at (y,z) to [[None,None]]
                # then we set the first element to gy to get
                # [[gy, None]]                
                par = node.parents[0]
                if par not in var2gs: var2gs[par] = [[None for _ in par.typ]]
                var2gs[par][0][node.op.idx] = gnode
            else:
                gpars = node.op.pullback(node.parents, node, gnode)
                diffs = node.get_diff()
                for (par,gpar,d) in utils.safezip3(node.parents, gpars,diffs):
                    assert (gpar is not None) == d # grad is None iff not diff wrt input
                    if d: var2gs[par].append(gpar)

    # only we already summed up the gradients for the input nodes, so just take
    # 0th element
    return [var2gs[node][0] for node in wrt]

def infer_shape(arr):
    """
    Infer the shape of `arr` and return a tuple of int and None
    """
    return tuple(x.op.value if isinstance(x.op, Constant) else None for x in  CACHER.simplify(cgt.shape(arr)))

def grad(cost, wrt):    
    """
    Compute the gradient of scalar-valued `cost` with respect to a list of variables `wrt`
    """
    assert cost.ndim == 0
    single_wrt = not (isinstance(wrt, list) or isinstance(wrt, tuple))
    if single_wrt:
        wrtl = [wrt]
    else:
        wrtl = wrt
    assert all(x.is_input() for x in wrtl), "Can only differentiate wrt Input nodes."
    gout = _singleton_ones(cost.dtype, 0)
    retval = pullback([cost], [gout], wrtl)
    if single_wrt:
        retval = retval[0]
    return retval

# ================================================================
# Compilation 
# ================================================================

class NativeCompileInfo(object):
    """
    Stores the information necessary to create a NativeCallable object
    """
    def __init__(self, func_code, closure_triples = None, includes=(), link_flags="", 
            setup=False, teardown=False, gpu_deref_mask=None, store_objects = (), extra_srcs=()):
        """
        func_code : code implementing function
        lang : c++ or cuda
        closure_tuples: a list of triples (fieldname, ctypes class, value) that will be provided at each call at runtime
        includes: list of strings specifying files to includes
        link flags: string specifying link flags
        setup: bool specifying if there's a setup method to call once when building a Callable, which should be called $setup in the code string
        teardown: bool specifying if there's a teardown method, called $teardown
        gpu_deref_mask : None or tuple of bools specifying which arguments to Op will have data dereferenced on the GPU (i.e., they must be moved to GPU)
        store_objects : list of python objects which should be stored somewhere as long as the Callable created from this object exists, e.g. because they own some data it uses
        """
        # To be filled in by caller of constructor
        self.op_str = None
        self.return_type = None
        self.n_in = None
        #####
        self.func_code = func_code
        self.closure_triples = closure_triples
        self.includes = list(includes)
        self.link_flags = link_flags
        self.setup = setup
        self.teardown = teardown
        self.gpu_deref_mask = gpu_deref_mask
        self.store_objects = store_objects
        self.extra_srcs = extra_srcs

    def involves_gpu(self):
        return self.gpu_deref_mask is not None

SrcFile = namedtuple("SrcFile", ["lang","code"])

class Callable(object):
    """
    Callable object built out of an Op
    """
    def call(self, *args):
        raise NotImplementedError
    @property
    def return_type(self):
        raise NotImplementedError
    @property
    def op_str(self):
        raise NotImplementedError
    @property
    def n_in(self):
        raise NotImplementedError
    
class PyCallable(Callable):
    """
    Callable object with an underlying python function acting on python objects
    """
    def __init__(self, op,  n_in, func):
        self._op_str = str(op)
        self._return_type = op.return_type
        self._n_in = n_in
        self._func = func
        self._kind = "py"
    def call(self, *args):
        return self._func(*args)    
    @property
    def op_str(self):
        return self._op_str
    @property
    def return_type(self):
        return self._return_type
    @property
    def kind(self):
        return self._kind
    @property
    def func(self):
        return self._func
    @property
    def n_in(self):
        return self._n_in
    
    
class NativeCallable(object):
    """
    Callable object with an underlying function pointer that acts on cgtObject
    """    
    def __init__(self, n_in, return_type, op_str, fptr, cldata=None, 
            store_objects=None, setup_fptr=None, teardown_fptr=None):
        self._n_in = n_in
        self._return_type = return_type
        self._op_str = op_str
        self.fptr = fptr
        self.cldata = cldata
        self.store_objects = store_objects
        self.teardown_fptr = teardown_fptr
        if setup_fptr is not None:
            setup_fptr()
        self._kind = "native"
    def __del__(self):
        if self.teardown_fptr is not None:
            self.teardown_fptr()
    @property
    def return_type(self):
        return self._return_type
    @property
    def op_str(self):
        return self._op_str
    @property
    def kind(self):
        return self._kind
    @property
    def n_in(self):
        return self._n_in
    def _call_byval(self, inputs):        
        raise Todo
        # cgt.cycgt.apply_byval(self.fptr, self.cldata, inputs) #pylint: disable=E1101
    def _call_byref(self, inputs, output):
        cgt.cycgt.apply_byref(self.fptr, self.cldata, inputs, output) #pylint: disable=E1101
    def call(self, *args):
        if self._return_type == "byval": self._call_byval(*args)
        elif self.return_type == "byref": self._call_byref(*args)
        else: raise Unreachable

    
    
# ================================================================
# Ops 
# ================================================================


# Constants
# ----------------------------------------------------------------

class Constant(Op): #pylint: disable=W0223
    available_impls = ("python","native_cpu")    
    def __init__(self, value):
        self.value = value
    def get_value(self):
        return self.value

class ConstantTensor(Constant):
    return_type = "byref"
    # XXX for some reason valret version gives rare segfaults
    def __init__(self, value):
        Constant.__init__(self, as_valid_array(value))
        self._hash = None
    def get_expr(self, parent_exprs):
        return self._value_str()
    def __str__(self):
        return "Const{%s}"%self._value_str()
    def _value_str(self):
        ndim = self.value.ndim
        return "%g"%self.value if ndim==0 else "%s%g...%s"%("["*ndim, self.value.flat[0], "]"*ndim)        
    def get_py_func(self, input_types):
        def f(_, write):
            np.copyto(write, self.value)
        return f
    # def get_py_func(self, input_types):
    #     def f(reads):
    #         return self.value
    #     return f
    # def valret_func(reads):
    #     return self.value
    # def inplace_func(reads, write):
    #     if isinstance(write, tuple):
    #         for (arrfrom,arrto) in utils.safezip(self.value,write):
    #             np.copyto(arrto, arrfrom)
    #     else:
    #         np.copyto(write,self.value)
    # return PyImpl(inplace_func=inplace_func)
    def pullback(self, _inps, _out, _gout):
        return []
    def shp_apply(self, _inputs):
        return [cgt.constant(x) for x in self.value.shape] 
    def typ_apply(self, input_types):
        assert len(input_types)==0
        return _ndarray_type(self.value)
    def get_hash(self):
        if self._hash is None: self._hash = cPickle.dumps(self.value, -1)
        return self._hash
    def get_closure(self):
        assert isinstance(self.value, np.ndarray)
        shapeptr = ctypes.cast(self.value.ctypes.shape, ctypes.c_void_p).value
        return [
        ("ndim", ctypes.c_int,self.value.ndim),
        ("shape",ctypes.c_void_p,shapeptr),
        ("dtype",ctypes.c_byte,self.value.dtype.num),
        ("data",ctypes.c_void_p,self.value.ctypes.data)]
    def get_native_compile_info(self, input_types, devtype):
        code = None
        if self.return_type == "byval": code = self._c_code_valret()
        elif self.return_type == "byref": code = self._c_code_inplace()
        else: raise ValueError
        return NativeCompileInfo(func_code=code, closure_triples=self.get_closure(),store_objects=(self.value,))
    def _c_code_inplace(self):
        if isinstance(self.value, tuple):
            raise MethodNotDefined
        return r"""
            CGT_EXPORT_C void $function($closure* cldata, cgtArray** reads, cgtArray* write) {
                cgt_memcpy(cgtCPU, cgtCPU, write->data(), cldata->data, write->nbytes());
            }
            """
    def _c_code_valret(self):
        return r"""
            CGT_EXPORT_C cgtArray* $function($closure* cldata, cgtArray** reads) {
                    auto out = new cgtArray(cldata->ndim, (long*)cldata->shape, 
                        (cgtDtype)cldata->dtype, cgtCPU, (void*)cldata->data, false);
                    return out;
            }"""

class ConstantTuple(Constant):
    return_type = "byval"
    def __init__(self, value):
        Constant.__init__(value)
    def get_expr(self, parent_exprs):
        return str(self.value)
    def __str__(self):
        return "Const{%s}"%str(self.value)
    def get_py_func(self, input_types):
        def f(_):
            return self.value
        return f
    def shp_apply(self, _inputs):
        return tuple(map(cgt.constant, x.shape) for x in self.value)
    def typ_apply(self, input_types):
        assert len(input_types)==0
        return _get_value_type(self.value)
    def get_hash(self):
        if self._hash is None: self._hash = cPickle.dumps(self.value, -1)
        return self._hash


class Fill(Op):
    """
    (value, shape...) -> array filled with `value`, with shape `shape`
    """
    available_impls = ("python","native_cpu")        
    def __init__(self, value):
        self.value = as_valid_array(value)
        assert self.value.ndim ==0
        assert self.value.dtype != "O"
        self.dtype = self.value.dtype
        assert self.value.ndim==0
        self.tag = -1 # @TAG_HACK
    def get_hash(self):
        return cPickle.dumps((self.value,self.tag) ,-1)
    def get_diff(self, num_inputs):
        return [False]*num_inputs
    def __str__(self):
        return "Fill{%g}"%self.value
    def get_py_func(self, input_types):
        def f(reads, write):
            write[...] = self.value
        return f
    def pullback(self, inputs, output, goutput):
        raise NonDifferentiable
    def shp_apply(self, inputs):
        return inputs
    def typ_apply(self, input_types):
        assert all(map(_isintscalar, input_types)), "Fill Op should have integer scalars as arguments"
        return TensorType(self.dtype, len(input_types))
    def get_closure(self):
        typ = ctypes.c_long if self.value.dtype.kind=='i' else ctypes.c_double
        return [("value", typ, self.value.item())]
    def get_native_compile_info(self, _input_types, devtype):
        assert devtype == "cpu"
        outdtype = Dtype.canon(self.value.dtype)
        func_code=r"""
            CGT_EXPORT_C void $function($closure* cldata, cgtArray** reads, cgtArray* write) {
                long s = write->size();
                %(cdtype)s value = cldata->value;
                for (int i=0; i < s; ++i) write->at<%(cdtype)s>(i) = value;
            }"""%dict(cdtype = np2c[outdtype])
        return NativeCompileInfo(func_code=func_code, closure_triples=self.get_closure())

def _isintscalar(typ):
    return typ.dtype[0] == 'i' and typ.ndim == 0

def _list_is_valid_sli(input_types):
    return len(input_types)==3 and all(map(_isintscalar, input_types))

class Arange(Op):
    """
    (start,stop,step) -> 1D array, just like numpy
    """
    available_impls = ("python","native_cpu")        
    return_type="byval"
    def __init__(self, dtype='i8'):
        self.dtype = dtype
    def get_diff(self, num_inputs):
        return [False]*num_inputs
    def get_py_func(self, input_types):
        def f((start, stop, step)):
            return np.arange(start, stop, step, self.dtype)
        return f
    def pullback(self, inputs, output, goutput):
        raise NonDifferentiable
    def shp_apply(self, inputs):
        start,stop,step = inputs
        return [(stop - start)//step]
    def typ_apply(self, input_types):
        assert _list_is_valid_sli(input_types)
        return TensorType(self.dtype, 1)
    def get_native_compile_info(self, input_types, devtype):
        func_code=r"""
            CGT_EXPORT_C cgtArray* $function(void* cldata, cgtArray** reads) {
                long start=reads[0]->at<long>(0),
                       stop=reads[1]->at<long>(0),
                       step=reads[2]->at<long>(0);
                long size = (stop-start)/step;
                cgtArray* out = new cgtArray(1, &size, cgt_i8, cgtCPU);
                for (int i=0; i < size; ++i) out->at<long>(i) = start+i*step;
                return out;
            }"""
        return NativeCompileInfo(func_code=func_code)

class ScalarRng(Op):
    """
    (shape...) -> array filled with iid random numbers, from either uniform or normal distribution
    """
    available_impls = ("python",)        
    is_random_op = True
    def __init__(self, kind):
        assert kind in ("uniform","gaussian")
        self.kind = kind
    def get_diff(self, num_inputs):
        return [False]*num_inputs
    def __str__(self):
        return "Rng{%s}"%self.kind
    def get_py_func(self, input_types):
        def f(reads, write):
            if self.kind == "uniform": write[...] = np.random.rand(*reads)
            elif self.kind == "gaussian": write[...] = np.random.randn(*reads)
            else: raise RuntimeError
        return f
    def pullback(self, inputs, output, goutput):
        raise NonDifferentiable
    def shp_apply(self, inputs):
        return inputs
    def typ_apply(self, input_types):
        return TensorType(cgt.floatX, len(input_types))
    def get_native_compile_info(self, input_types, devtype):
        func_code=r"""
            CGT_EXPORT_C void $function(void* cldata, cgtArray** reads, cgtArray* write) {
                long start=reads[0]->at<long>(0),
                       stop=reads[1]->at<long>(0),
                       step=reads[2]->at<long>(0);
                long size = (stop-start)/step;
                cgtArray* out = new cgtArray(1, &size, cgt_i8, cgtCPU);
                for (int i=0; i < size; ++i) out->at<long>(i) = start+i*step;
                return out;
            }"""
        return NativeCompileInfo(func_code=func_code)

# Elementwise
# ----------------------------------------------------------------

def _no_grad():
    raise NonDifferentiable()

def _nu_sigmoid(x, out=None):
    return np.reciprocal(1+np.exp(-x), out=out)

def _nu_iceil(x,out=None):
    if out is None:
        return np.ceil(x)
    else:
        np.ceil(x, out=out, casting='unsafe')

def _nu_ifloor(x,out=None):
    if out is None:
        return np.floor(x)
    else:
        np.floor(x, out=out, casting='unsafe')

def _nu_divide(x, y, out=None):
    if x.dtype.kind != 'f': x = x.astype(cgt.floatX)
    if out is None:
        return np.divide(x,y)
    else:
        np.divide(x,y,out)

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

BinaryInfo = namedtuple("BinaryInfo", ("short", "pyfunc","commutes","diff","typeinfo","gradexpr", "cexpr"))


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
    "**"   : BinaryInfo("power",  np.power,      False,    (True,True), 'p',      lambda x, y, z, gz: [gz*y*cgt.power(x,y-1),gz*z*cgt.log(x)],"pow(x,y)"), 
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
    available_impls = ("python","native_cpu","native_gpu")    
    def __init__(self, opname, info=None):
        self.opname = opname
        self.info = UNARY_INFO[opname] if info is None else info
    def get_diff(self, _):
        return [self.info.diff]
    def __str__(self):
        return self.info.short
    def get_hash(self):
        return utils.hash_seq1(self.opname)
    def get_replacement(self, _newparents, _analysis):
        return None
    def pullback(self, (x,), y, gy): #pylint: disable=W0613
        return [self.info.gradexpr(x, y, gy)]
    def shp_apply(self, inputs):
        return cgt.shape(inputs[0])
    def typ_apply(self, input_types):
        typeinfo = self.info.typeinfo
        intype = input_types[0].dtype
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
        return TensorType(out_type, input_types[0].ndim)
    def get_py_func(self,_):
        def f(reads, write):
            self.info.pyfunc(reads[0], out=write)
        return f
    def get_native_compile_info(self, input_types, devtype):
        info = self.info
        out_dtype = self.typ_apply(input_types).dtype
        d = dict(cdtype0=np2c[input_types[0].dtype], cdtype1=np2c[out_dtype], cexpr=info.cexpr)
        if devtype == "cpu":
            code = r"""
                static inline %(cdtype1)s scalar_$function(%(cdtype0)s x) {return %(cexpr)s;}
                CGT_EXPORT_C void $function(void* cldata, cgtArray** reads, cgtArray* write) {
                    cgtArray* read = reads[0];
                    int s = read->size();
                    %(cdtype0)s* readdata = (%(cdtype0)s*)read->data();
                    %(cdtype1)s* writedata = (%(cdtype1)s*)write->data();
                    for (int i=0; i < s; ++i) {
                        writedata[i] = scalar_$function(readdata[i]);
                    }
                }"""%d
            return NativeCompileInfo(code, includes=["math.h"], link_flags="-lm")
        elif devtype == "gpu":
            cuda_code = r"""
                #include "cgt_cuda.h"
                __forceinline__ __device__ %(cdtype1)s $function(%(cdtype0)s x) {return %(cexpr)s;}        
                __global__ void ${function}_kernel(const long n, const %(cdtype0)s* in, %(cdtype1)s* out) {
                  CUDA_KERNEL_LOOP(i, n) {
                    out[i] = $function(in[i]);
                  }
                }
                void launchker_$function(long n, %(cdtype0)s* x, %(cdtype1)s* y) {
                    int num_blocks, num_threads;
                    cgt_get_bt(n, num_blocks, num_threads);
                    ${function}_kernel<<<num_blocks, num_threads>>>(n, x, y);                
                }
                """%d
            cpp_code = """
                extern void launchker_${function}(long, %(cdtype0)s*, %(cdtype1)s*);            
                CGT_EXPORT_C void $function(void* cldata, cgtArray** reads, cgtArray* write) {
                    cgtArray* read = reads[0];
                    long n = read->size();
                    launchker_$function(n, (%(cdtype0)s*)reads[0]->data(), (%(cdtype1)s*)write->data());
                }"""%d
            return NativeCompileInfo(cpp_code, includes=["math.h"], link_flags="-lm -lcudart",
                gpu_deref_mask=(True,), extra_srcs=[SrcFile("cuda",cuda_code)])
        else:
            raise Unreachable

class ElwiseBinary(Op):
    available_impls = ("python","native_cpu","native_gpu")        
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
        elif self.opname == "**":
            if r in node2sv and node2sv[r] == 1: out = l

        if out is not None:
            outtyp = self.typ_apply([p.typ for p in parents])
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
    def typ_apply(self, input_types):
        assert ((input_types[0].ndim==0) == self.scalar_mask[0]) and ((input_types[1].ndim==0) == self.scalar_mask[1])
        if self.scalar_mask==(False,False):
            assert input_types[0].ndim == input_types[1].ndim
            # assertequaln(cgt.shape(input_types[0]),cgt.shape(input_types[1]),"shape mismatch at elementwise binary operation")
        typeinfo = BINARY_INFO[self.opname].typeinfo
        if typeinfo == 'p':
            out_dtype = _promote(input_types[0].dtype, input_types[1].dtype)
        elif typeinfo == 'f':
            out_dtype = cgt.floatX
        else:
            out_dtype = typeinfo
        ind4shape = 1 if self.scalar_mask[0] else 0
        return TensorType(out_dtype, input_types[ind4shape].ndim)
    def get_py_func(self, input_types):
        def f(reads, write):
            x,y = reads
            if self.scalar_mask==(False,False):
                if x.shape != y.shape:
                    raise RuntimeError("mismatched shapes %s %s. Note that implicit broadcasting isn't allowed. Use the broadcast(...) function"%(x.shape, y.shape))
            self.info.pyfunc(x,y, out=write)
        return f
    def get_native_compile_info(self, input_types, devtype):
        typ2 = self.typ_apply(input_types)
        npdtype0 = input_types[0].dtype
        npdtype1 = input_types[1].dtype
        npdtype2 = typ2.dtype
        ind4shape = 1 if self.scalar_mask[0] else 0
        index0 = "0" if self.scalar_mask[0] else "i"
        index1 = "0" if self.scalar_mask[1] else "i"
        d = dict(cdtype0=np2c[npdtype0],cdtype1=np2c[npdtype1],cdtype2=np2c[npdtype2],
            cexpr=self.info.cexpr,index0=index0,index1=index1,ind4shape=ind4shape)
        if devtype == "cpu":
            code = r"""
                static inline %(cdtype2)s scalar_$function(%(cdtype0)s x, %(cdtype1)s y) {return %(cexpr)s;}
                CGT_EXPORT_C void $function(void* cldata, cgtArray** reads, cgtArray* write) {
                    int s = reads[%(ind4shape)s]->size();
                    %(cdtype0)s* in0 = (%(cdtype0)s*)reads[0]->data();
                    %(cdtype1)s* in1 = (%(cdtype1)s*)reads[1]->data();
                    %(cdtype2)s* out = (%(cdtype2)s*)write->data();
                    cgt_check(write->size() == s, "Shape error in elementwise binary operation. You might be missing a call to cgt.broadcast(...)");
                    for (int i=0; i < s; ++i) {
                        out[i] = scalar_$function(in0[%(index0)s], in1[%(index1)s]);
                    }
                }"""%d
            return NativeCompileInfo(func_code=code, includes=["math.h"])

        elif devtype == "gpu":
            cuda_code = r"""
                #include "cgt_cuda.h"
                __forceinline__ __device__ %(cdtype2)s $function(%(cdtype0)s x, %(cdtype1)s y) {return %(cexpr)s;}
                __global__ void ${function}_kernel(const long n, const %(cdtype0)s* x, const %(cdtype1)s* y, %(cdtype2)s* z) {
                  CUDA_KERNEL_LOOP(i, n) {
                    z[i] = $function(x[%(index0)s], y[%(index1)s]);
                  }
                }
                void launchker_$function(long n, %(cdtype0)s* x, %(cdtype1)s* y, %(cdtype2)s* z) {
                    int num_blocks,num_threads;                    
                    cgt_get_bt(n, num_blocks, num_threads);
                    ${function}_kernel<<<num_blocks, num_threads>>>(n, x, y, z);                
                }
            """%d
            cpp_code = """
                extern void launchker_${function}(long, %(cdtype0)s*, %(cdtype1)s*, %(cdtype2)s*);
                CGT_EXPORT_C void $function(void* cldata, cgtArray** reads, cgtArray* write) {
                    long n = reads[%(ind4shape)s]->size();
                    launchker_${function}(n, (%(cdtype0)s*)reads[0]->data(), (%(cdtype1)s*)reads[1]->data(), (%(cdtype2)s*)write->data());
                }"""%d
            return NativeCompileInfo(func_code=cpp_code, includes=["math.h"], link_flags="-lm -lcudart", gpu_deref_mask=(True,True),
                extra_srcs=[SrcFile("cuda",cuda_code)])

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
    return_type = "byval"
    available_impls = ("python","native_cpu")        
    def __init__(self, axis):
        self.axis = axis
    def get_diff(self, _):
        return [False]
    def __str__(self):
        return "Size{%i}"%self.axis
    def get_py_func(self, input_types):
        def f(reads):
            return np.array(reads[0].shape[self.axis],'i8')
        return f
    def pullback(self, inputs, output, goutput):
        raise NonDifferentiable
    def shp_apply(self, _inputs):
        return []
    def typ_apply(self, _):
        return TensorType('i8',0)
    def get_replacement(self, inputs, _analysis):
        x = inputs[0]
        if x.is_input():
            fixed_shape = x.get_fixed_shape()
            if fixed_shape[self.axis] is not None:
                return cgt.constant(fixed_shape[self.axis])
    def get_closure(self):
        return [("ax",ctypes.c_int,self.axis)]
    def get_native_compile_info(self, input_types, devtype):
        code = r"""
            CGT_EXPORT_C cgtArray* $function(void* cl0, cgtArray** reads) {
                $closure* cl = ($closure*)cl0;
                cgtArray* in = reads[0];
                cgtArray* out = new cgtArray(0, NULL, cgt_i8, cgtCPU);
                out->at<long>(0) = in->shape()[cl->ax];
                return out;
            }"""
        return NativeCompileInfo(code,closure_triples = self.get_closure())

class Reshape(Op):
    available_impls = ("python","native_cpu")        
    return_type = "byval"
    def get_diff(self, num_inputs):
        return [True] + [False]*(num_inputs-1)
    def get_py_func(self, input_types):
        def f(reads):
            out = reads[0].reshape(reads[1:])
            if not out.flags.c_contiguous: out = out.copy()
            return out
        return f
    def pullback(self, inputs, _out, gout):
        return [cgt.reshape(gout, cgt.shape(inputs[0]))] + [None]*(len(inputs)-1)
    def shp_apply(self, inputs):
        return inputs[1:]
    def typ_apply(self, input_types):
        return TensorType(input_types[0].dtype, len(input_types)-1)
    def get_closure(self, n_parents):
        return [("ndim", ctypes.c_int,n_parents-1)]
    def get_native_compile_info(self, input_types, devtype):
        code = r"""
            CGT_EXPORT_C cgtArray* $function($closure* cldata, cgtArray** reads) {
                cgtArray* in = reads[0];
                long* newshape = new long[cldata->ndim];
                for (int i=0; i < cldata->ndim; ++i) {
                    long s = reads[i+1]->at<long>(0);
                    newshape[i] = static_cast<long*>(reads[i+1]->data())[0];
                    cgt_assert((newshape[i] >= 0) && "negative size in reshape not supported");
                }
                cgtArray* out = new cgtArray(cldata->ndim, newshape, in->dtype(), in->devtype(), in->data(), false);
                return out;
            }
            """
        return NativeCompileInfo(code, closure_triples=self.get_closure(len(input_types)))

class Concatenate(Op):
    available_impls = ("python","native_cpu")        
    def __init__(self, axis):
        self.axis = axis
    def get_diff(self, num_inputs):
        return [True]*num_inputs
    def get_py_func(self, input_types):
        def f(reads, write): write[...] = np.concatenate(reads,axis=self.axis)
        return f
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
    def typ_apply(self, input_types):
        return TensorType(_promote_multi([x.dtype for x in input_types]), input_types[0].ndim)
    def get_native_compile_info(self, input_types, devtype):
        x = input_types[0]
        openloops = " ".join(["for (int i%(ax)s=0; i%(ax)s < in->shape()[%(ax)s]; ++i%(ax)s) {"%dict(ax=ax) for ax in xrange(x.ndim)])
        closeloops = "}"*x.ndim
        inidxexpr =  ",".join(["i%i"%ax for ax in xrange(x.ndim)])
        outidxexpr =  ",".join([("i%i+n" if ax == self.axis else "i%i")%ax for ax in xrange(x.ndim)])
        code = r"""
            CGT_EXPORT_C void $function(void* cldata, cgtArray** reads, cgtArray* write) {
                long n=0; // value along concat axis
                for (int i=0; i < %(n_in)s; ++i) {
                    cgtArray* in = reads[i];
                    %(openloops)s
                        write->at<%(cdtype)s>(%(outidxexpr)s) = in->at<%(cdtype)s>(%(inidxexpr)s);
                    %(closeloops)s
                    n += in->shape()[%(axis)s];
                }
            }
            """%dict(openloops=openloops, closeloops=closeloops, inidxexpr=inidxexpr, outidxexpr=outidxexpr, 
                n_in=len(input_types), cdtype=np2c[input_types[0].dtype],axis=self.axis)
        return NativeCompileInfo(code)

class Repeat(Op):
    available_impls = ("python","native_cpu")        
    def __init__(self, axes):
        self.axes = axes
    def get_diff(self, num_inputs):
        return [True] + [False for _ in xrange(num_inputs-1)]
    def get_py_func(self, input_types):
        def f(reads, write):
            arr = reads[0]
            numreps = reads[1:]
            shp = arr.shape
            assert all(shp[i] == 1 for i in self.axes)
            for (ax,numrep) in utils.safezip(self.axes, numreps):
                arr = np.repeat(arr, numrep, ax)
            np.copyto(write, arr)
        return f
    def get_native_compile_info(self, input_types, devtype):
        x = input_types[0]
        openloops = " ".join(["for (int i%(ax)s=0; i%(ax)s < write->shape()[%(ax)s]; ++i%(ax)s) {"%dict(ax=ax) for ax in xrange(x.ndim)])
        closeloops = "}"*x.ndim
        outidxexpr = ",".join(["i%(ax)s"%dict(ax=ax) for ax in xrange(x.ndim)])
        inidxexpr = ",".join(["0" if ax in self.axes else "i%(ax)s"%dict(ax=ax) for ax in xrange(x.ndim)])
        code = r"""
            CGT_EXPORT_C void $function(void* cldata, cgtArray** reads, cgtArray* write) {
                cgtArray *read=reads[0];
                %(openloops)s
                    write->at<%(cdtype)s>(%(outidxexpr)s) = read->at<%(cdtype)s>(%(inidxexpr)s);
                %(closeloops)s
            }
            """%dict(openloops=openloops, outidxexpr=outidxexpr, inidxexpr=inidxexpr, closeloops=closeloops,
                cdtype=np2c[input_types[0].dtype])
        return NativeCompileInfo(code)
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
    def typ_apply(self, input_types):
        assert all(x.dtype == "i8" for x in input_types[1:])
        return input_types[0]

class Transpose(Op):
    available_impls = ("python","native_cpu")        
    def __init__(self, axes):
        self.axes = axes
    def get_diff(self, _):
        return [True]
    def get_py_func(self, input_types):
        def f(reads, write):
            np.copyto(write, reads[0].transpose(self.axes))
        return f
    def pullback(self, inputs, output, goutput):
        return [cgt.transpose(goutput, utils.invert_perm(self.axes))]
    def shp_apply(self, inputs):
        inshape = cgt.shape(inputs[0])
        return [inshape[ax] for ax in self.axes]
    def typ_apply(self, input_types):
        return input_types[0]
    def __str__(self):
        return "Transpose{%s}"%",".join(map(str, self.axes))
    def get_native_compile_info(self, input_types, devtype):
        x = input_types[0]
        d = {}
        d["openloops"] = " ".join(["for (int i%(ax)s=0; i%(ax)s < write->shape()[%(ax)s]; ++i%(ax)s) {"%dict(ax=ax) for ax in xrange(x.ndim)])
        d["closeloops"] = "}"*x.ndim
        d["outidxexpr"] = ",".join(["i"+str(i) for i in xrange(x.ndim)])
        d["inidxexpr"] = ",".join(["i"+str(i) for i in utils.invert_perm(self.axes)])
        d["cdtype"] = np2c[x.dtype]
        code = r"""
            CGT_EXPORT_C void $function(void* cldata, cgtArray** reads, cgtArray* write) {
                cgtArray *read = reads[0];
                %(cdtype)s* indata = (%(cdtype)s*)read->data(), *outdata = (%(cdtype)s*)write->data();
                %(openloops)s
                    write->at<%(cdtype)s>(%(outidxexpr)s) = read->at<%(cdtype)s>(%(inidxexpr)s);
                %(closeloops)s
            }"""%d
        return NativeCompileInfo(code)

class Transport(Op):
    available_impls = ("native_cpu","native_gpu")
    def __init__(self, dev):
        self.dev = dev
    def typ_apply(self, input_types):
        return input_types[0]
    def shp_apply(self, inputs):
        return cgt.shape(inputs[0])
    def get_native_compile_info(self, _inputs, _devtype):
        # This C code should only be run if the input and output devices differ.
        # There should never be any no-op transports.
        code = r"""
            CGT_EXPORT_C void $function(void* cldata, cgtObject** reads, cgtObject* write) {
                cgt_copy_object(write, reads[0]);
            }
            """
        return NativeCompileInfo(code)

# TODO save computation by removing negative freq components
class RFFT(Op):
    available_impls = ("python",)        
    def __init__(self, axes):
        self.axes = axes
    def get_diff(self, num_inputs):
        return [True] + [False]*(num_inputs-1)
    def get_py_func(self, input_types):
        def f(reads, write):
            x = reads[0]
            shp = map(int,reads[1:])
            np.copyto(write, np.fft.fftn(x,shp,self.axes))
        return f
    def pullback(self, inputs, _outputs, goutput):
        return cgt.real(Result(RFFT(self.axes),[goutput]+inputs[1:]))
    def shp_apply(self, inputs):
        out = cgt.shape(inputs[0])
        for (ax,sz) in utils.safezip(self.axes, inputs[1:]):
            out[ax]=sz
        return out
    def typ_apply(self, input_types):
        x = input_types[0]
        assert x.dtype==cgt.floatX
        return TensorType(cgt.complexX,x.ndim)

class IRFFT(Op):
    available_impls = ("python",)        
    def __init__(self, axes):
        self.axes = axes
    def get_diff(self, _):
        return [True]
    def get_py_func(self, input_types):
        def f(reads, write):
            x = reads[0]
            shp = map(int,reads[1:])
            slis = [slice(0,None) for _ in xrange(x.ndim)]
            for (ax,s) in zip(self.axes,shp): slis[ax] = slice(0, s)
            np.copyto(write, np.real(np.fft.ifftn(x,axes=self.axes)[slis]))
        return f
    def pullback(self, inputs, _outputs, goutput):
        return Result(IRFFT(self.axes),[goutput]) # XXX is this right?
    def shp_apply(self, inputs):
        return cgt.shape(inputs[0])
    def typ_apply(self, inputs):
        return TensorType(cgt.floatX,inputs[0].ndim)

# Reductions
# ----------------------------------------------------------------

def gen_reduction_code(dtype, axes, ndim, reduction_expr, initval):
    openloops = " ".join(["for (int i%(ax)s=0; i%(ax)s < read->shape()[%(ax)s]; ++i%(ax)s) {"%dict(ax=ax) for ax in xrange(ndim)])
    closeloops = "}"*ndim
    inidxexpr = ",".join(["i"+str(i) for i in xrange(ndim)])
    outidxexpr = ",".join(["0" if i in axes else  "i"+str(i) for i in xrange(ndim)])
    d = dict(openloops=openloops, outidxexpr=outidxexpr, inidxexpr=inidxexpr, closeloops=closeloops,
            cdtype=np2c[dtype])
    reduction_expr %= d
    initval %= d
    d["reduction_expr"] = reduction_expr
    d["initval"] = initval
    return r"""
        static inline %(cdtype)s reduction_$function(%(cdtype)s x, %(cdtype)s y) {return %(reduction_expr)s;}
        CGT_EXPORT_C void $function(void* cldata, cgtArray** reads, cgtArray* write) {
            cgtArray *read=reads[0];
            for (int i=0; i < write->size(); ++i) write->at<%(cdtype)s>(i) = %(initval)s;
            %(openloops)s
                %(cdtype)s x = write->at<%(cdtype)s>(%(outidxexpr)s); 
                %(cdtype)s y = read->at<%(cdtype)s>(%(inidxexpr)s) ;
                write->at<%(cdtype)s>(%(outidxexpr)s) = reduction_$function(x, y);
            %(closeloops)s
        }
        """%d

class Sum(Op):
    available_impls = ("python","native_cpu")
    def __init__(self, axes):
        self.axes = tuple(axes)
    def get_diff(self, _):
        return [True]
    def __str__(self):
        return "Sum{%s}"%(",".join(map(str,self.axes)))
    def get_py_func(self, input_types):
        def f(reads, write):
            reads[0].sum(axis = self.axes or None, out=write, keepdims=True)
        return f
    def pullback(self, inputs, output, goutput):
        return [Result(Repeat(self.axes), [goutput] + [cgt.size(inputs[0],ax) for ax in self.axes])]
    def shp_apply(self, inputs):
        x = inputs[0]
        s = cgt.shape(x)
        return [(cgt.constant(1) if i in self.axes else s[i]) for i in xrange(x.ndim)]
    def typ_apply(self, input_types):
        return input_types[0]
    def get_native_compile_info(self, input_types, devtype):        
        code = gen_reduction_code(input_types[0].dtype, self.axes, input_types[0].ndim, "x+y","0")
        return NativeCompileInfo(code, includes=["string.h"])

class Max(Op):
    available_impls = ("python","native_cpu")    
    def __init__(self, axes):
        self.axes = tuple(axes)
    def get_diff(self, _):
        return [True]
    def __str__(self):
        return "Max{%s}"%(",".join(map(str,self.axes)))
    def get_py_func(self, input_types):
        def f(reads, write):
            reads[0].max(axis=self.axes or None,keepdims=True, out=write)
        return f
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
    def typ_apply(self, input_types):
        return input_types[0]
    def get_native_compile_info(self, input_types, devtype):
        code = gen_reduction_code(input_types[0].dtype, self.axes, input_types[0].ndim, "fmax(x,y)", "-std::numeric_limits<%(cdtype)s>::max()")
        return NativeCompileInfo(code, includes=["string.h","limits","math.h"])

class Argmax(Op):
    available_impls = ("python",)    
    def __init__(self, axis):
        self.axis = axis
    def get_diff(self, _):
        return [False]
    def __str__(self):
        return "Argmax{%s}"%self.axis
    def get_py_func(self, input_types):
        def f(reads, write):
            write.flat[:] = reads[0].argmax(axis=self.axis)
        return f
    def shp_apply(self, inputs):
        x = inputs[0]
        s = cgt.shape(x)
        return [(cgt.constant(1) if i == self.axis else s[i]) for i in xrange(x.ndim)]
    def typ_apply(self, inputs):
        return TensorType('i8', inputs[0].ndim)
    # re: native impl, this is a tricky one, since it requires some scratch space
    # to store the max values. probably just do a alloc/dealloc


# Slicing
# ----------------------------------------------------------------



class GetSli(Op):
    available_impls = ("python","native_cpu")
    def __init__(self, axis):
        self.axis = axis
    def get_diff(self, _):
        return [True,False,False,False]
    def get_py_func(self, input_types):
        def f(reads, write):
            x,start,stop,step=reads
            if step<0 and stop==-1: stop=None
            slices = [slice(None,None,None) for _ in xrange(x.ndim)]
            slices[self.axis] = slice(start,stop,step)
            write[:] = x[slices]
        return f
    def pullback(self, inputs, output, goutput):
        z = cgt.zeros_like(inputs[0])
        z.op.tag = id(output) # @TAG_HACK
        return [Result(IncSli(self.axis), [z] + inputs[1:] + [goutput])] + [None]*3
    def shp_apply(self, inputs):
        arr, start, stop, step = inputs
        s = cgt.shape(arr) #pylint: disable=W0621
        s[self.axis] = cgt.ceil_divide(stop - start, step)
        return s
    def typ_apply(self, input_types):        
        assert _list_is_valid_sli(input_types[1:])
        return input_types[0]
    def get_native_compile_info(self, input_types, devtype):
        x = input_types[0]
        openloops = " ".join(["for (int i%(ax)s=0; i%(ax)s < write->shape()[%(ax)s]; ++i%(ax)s) {"%dict(ax=ax) for ax in xrange(x.ndim)])
        closeloops = "}"*x.ndim

        outidxexpr = ",".join(["i%(ax)s"%dict(ax=ax) for ax in xrange(x.ndim)])
        inidxexpr = ",".join([("start + step*i%(ax)s" if ax==self.axis else "i%(ax)s")%dict(ax=ax) for ax in xrange(x.ndim)])

        code = r"""
            CGT_EXPORT_C void $function(void* cldata, cgtArray** reads, cgtArray* write) {
                cgtArray *in=reads[0];
                long start = reads[1]->at<long>(0);
                long step = reads[3]->at<long>(0);
                %(openloops)s
                    write->at<%(cdtype)s>(%(outidxexpr)s) = in->at<%(cdtype)s>(%(inidxexpr)s);
                %(closeloops)s
            }
            """%dict(openloops=openloops, outidxexpr=outidxexpr, inidxexpr=inidxexpr, closeloops=closeloops,
    cdtype=np2c[input_types[0].dtype])
        return NativeCompileInfo(code)

class IncSli(Op):
    available_impls = ("python","native_cpu")
    writes_to_input = 0
    def __init__(self, axis):
        self.axis = axis
    def get_diff(self, _):
        return [True,False,False,False,True]
    def get_py_func(self, input_types):
        def f(reads, write):
            x, start, stop, step, y=reads
            if step<0 and stop==-1: stop=None            
            slices = [slice(None,None,None) for _ in xrange(x.ndim)]
            slices[self.axis] = slice(start,stop,step)          
            if x.data != write.data:
                utils.warn("incsli not inplace!")
                np.copyto(write, x)
            write[slices] += y
        return f
    def pullback(self, inputs, output, goutput):
        _x, start,stop,step, _y = inputs
        return [goutput, None, None,None,Result(GetSli(self.axis), [goutput, start,stop,step])]        
    def shp_apply(self, inputs):
        return cgt.shape(inputs[0])
    def typ_apply(self, input_types):
        return input_types[0]
    def get_native_compile_info(self, input_types, devtype):
        x = input_types[0]
        openloops = " ".join(
            ["for (int i%(ax)s=0; i%(ax)s < inc->shape()[%(ax)s]; ++i%(ax)s) {"%dict(ax=ax) for ax in xrange(x.ndim)])
        closeloops = "}"*x.ndim

        incidxexpr = ",".join(["i%(ax)s"%dict(ax=ax) for ax in xrange(x.ndim)])
        outidxexpr = ",".join([("start + step*i%(ax)s" if ax==self.axis else "i%(ax)s")%dict(ax=ax) for ax in xrange(x.ndim)])

        code = r"""
            CGT_EXPORT_C void $function(void* cldata, cgtArray** reads, cgtArray* write) {
                cgtArray *in=reads[0], *inc = reads[4];
                long start = reads[1]->at<long>(0);
                long step = reads[3]->at<long>(0);
                cgt_assert(in->size() == write->size());
                if (write->data() != in->data()) cgt_copy_array(write, in);
                %(openloops)s
                    write->at<%(cdtype)s>(%(outidxexpr)s) += inc->at<%(cdtype)s>(%(incidxexpr)s);
                %(closeloops)s
            }
            """%dict(openloops=openloops, outidxexpr=outidxexpr, closeloops=closeloops,
    cdtype=np2c[input_types[0].dtype], incidxexpr=incidxexpr)
        return NativeCompileInfo(code)


class GetFancySli(Op):
    available_impls = ("python","native_cpu")
    def __init__(self, axis):
        self.axis = axis
    def get_diff(self, _):
        return [True,False]
    def get_py_func(self, input_types):
        def f(reads, write):
            x,inds=reads
            slices = [slice(None,None,None) for _ in xrange(x.ndim)]
            slices[self.axis] = inds
            write[:] = x[slices]
        return f
    def pullback(self, inputs, output, goutput):
        z = cgt.zeros_like(inputs[0])
        z.op.tag = id(output) # @TAG_HACK
        return [Result(IncFancySli(self.axis), [z, inputs[1], goutput]), None]
    def shp_apply(self, inputs):
        arr, inds = inputs
        s = cgt.shape(arr) #pylint: disable=W0621
        newshape = copy.copy(s)
        newshape[self.axis] = cgt.size(inds,0)
        return newshape
    def typ_apply(self, input_types):
        assert input_types[1] == TensorType('i8', 1)
        return input_types[0]
    def get_native_compile_info(self, input_types, devtype):
        x = input_types[0]
        openloops = " ".join(["for (int i%(ax)s=0; i%(ax)s < write->shape()[%(ax)s]; ++i%(ax)s) {"%dict(ax=ax) for ax in xrange(x.ndim)])
        closeloops = "}"*x.ndim

        outidxexpr = ",".join(["i%(ax)s"%dict(ax=ax) for ax in xrange(x.ndim)])
        inidxexpr = ",".join([("inds->at<long>(i%(ax)s)" if ax==self.axis else "i%(ax)s")%dict(ax=ax) for ax in xrange(x.ndim)])

        code = r"""
            CGT_EXPORT_C void $function(void* cldata, cgtArray** reads, cgtArray* write) {
                cgtArray *x=reads[0], *inds=reads[1];
                long start = reads[1]->at<long>(0);
                long step = reads[3]->at<long>(0);
                %(openloops)s
                    write->at<%(cdtype)s>(%(outidxexpr)s) = x->at<%(cdtype)s>(%(inidxexpr)s);
                %(closeloops)s
            }
            """%dict(openloops=openloops, outidxexpr=outidxexpr, inidxexpr=inidxexpr, closeloops=closeloops,
    cdtype=np2c[input_types[0].dtype])
        return NativeCompileInfo(code)

class IncFancySli(Op):
    available_impls = ("python","native_cpu")
    writes_to_input = 0
    def __init__(self, axis):
        self.axis = axis
    def get_diff(self, _):
        return [True,False,True,True]
    def get_py_func(self, input_types):
        def f(reads, write):
            x, inds, y=reads
            slices = [slice(None,None,None) for _ in xrange(x.ndim)]
            slices2 = [slice(None,None,None) for _ in xrange(x.ndim)]
            if x.data != write.data:
                utils.warn("incsli not inplace!")
                np.copyto(write, x)
            for (i,ind) in enumerate(inds):
                slices[self.axis]=ind
                slices2[self.axis]=i
                write[slices] += y[slices2]
        return f
    def pullback(self, inputs, output, goutput):
        raise NotImplementedError
    def shp_apply(self, inputs):
        return cgt.shape(inputs[0])
    def typ_apply(self, input_types):
        return input_types[0]
    def get_native_compile_info(self, input_types, devtype):
        x = input_types[0]
        openloops = " ".join(
            ["for (int i%(ax)s=0; i%(ax)s < inc->shape()[%(ax)s]; ++i%(ax)s) {"%dict(ax=ax) for ax in xrange(x.ndim)])
        closeloops = "}"*x.ndim

        incidxexpr = ",".join(["i%(ax)s"%dict(ax=ax) for ax in xrange(x.ndim)])
        outidxexpr = ",".join([("inds->at<long>(i%(ax)s)" if ax==self.axis else "i%(ax)s")%dict(ax=ax) for ax in xrange(x.ndim)])

        code = r"""
            CGT_EXPORT_C void $function(void* cldata, cgtArray** reads, cgtArray* write) {
                cgtArray *x=reads[0], *inds=reads[1], *inc = reads[2];
                cgt_assert(x->size() == write->size());
                if (write->data() != x->data()) cgt_copy_array(write, x);
                %(openloops)s
                    write->at<%(cdtype)s>(%(outidxexpr)s) += inc->at<%(cdtype)s>(%(incidxexpr)s);
                %(closeloops)s
            }
            """%dict(openloops=openloops, outidxexpr=outidxexpr, closeloops=closeloops,
    cdtype=np2c[input_types[0].dtype], incidxexpr=incidxexpr)
        return NativeCompileInfo(code)


class GetFlatIndices(Op):
    available_impls = ("python","native_cpu")        
    def get_diff(self, _):
        return [True,False]
    def get_py_func(self, input_types):
        def f(reads, write):
            np.copyto(write, reads[0].flat[reads[1]])
        return f
    def pullback(self, inputs, output, goutput):
        x,inds = inputs
        ginput = cgt.zeros_like(x)
        return [Result(IncFlatIndices(), [ginput, inds, goutput]), None]
    def shp_apply(self, inputs):
        return cgt.shape(inputs[1])
    def typ_apply(self, inputs):
        assert inputs[1].ndim == 1 and dtype_kind(inputs[1].dtype) == 'i'
        return TensorType(inputs[0].dtype,1)
    def get_native_compile_info(self, input_types, devtype):
        npdtype = input_types[0].dtype
        code = r"""
            CGT_EXPORT_C void $function(void**, cgtArray** xk, cgtArray* z) {
                cgtArray *x=xk[0], *k=xk[1];
                for (int i=0; i < z->size(); ++i) {
                    z->at<%(cdtype)s>(i) = x->at<%(cdtype)s>(k->at<long>(i));
                }
            }
            """%dict(cdtype = np2c[npdtype])    
        return NativeCompileInfo(code)

class IncFlatIndices(Op):
    available_impls = ("python","native_cpu")        
    writes_to_input = 0
    def get_diff(self, _):
        return [True,False,True]
    def get_py_func(self, input_types):
        def f(reads, write):
            x,inds,y = reads
            if x.data != write.data:
                utils.warn("incsli not inplace!")
                np.copyto(write, x)
            for (i,ind) in enumerate(inds):
                write.flat[ind] += y[i] 
            # This is unvectorized so it gives the right answer when inds are non-unique
            # faster vectorized version: write[inds] += y
        return f
    def pullback(self, inputs, output, goutput):
        x, inds, y = inputs
        return [goutput, None, Result(GetFlatIndices(), [goutput, inds])]        
    def shp_apply(self, inputs):
        return cgt.shape(inputs[0])
    def typ_apply(self, input_types):
        return input_types[0]
    def get_native_compile_info(self, input_types, devtype):
        npdtype = input_types[0].dtype
        code = r"""
            CGT_EXPORT_C void $function(void**, cgtArray** xkp, cgtArray* write) {
                cgtArray *x=xkp[0], *k=xkp[1], *p=xkp[2];
                if (write->data() != x->data()) cgt_memcpy(cgtCPU, cgtCPU, write, x, write->nbytes());            
                for (int i=0; i < p->size(); ++i) {
                    write->at<%(cdtype)s>(k->at<long>(i)) += p->at<%(cdtype)s>(i);
                }
            }
            """%dict(cdtype = np2c[npdtype])    
        return NativeCompileInfo(code)

class Flip(Op):
    available_impls = ("python","native_cpu")        
    def __init__(self, axes):
        self.axes = axes
    def get_diff(self, _):
        return [True]
    def get_py_func(self, input_types):
        def f(reads, write):
            x = reads[0]
            slices = [slice(0,None,None) for _ in xrange(x.ndim)]
            for ax in self.axes: slices[ax] = slice(None,None,-1)
            np.copyto(write, x[slices])
        return f
    def pullback(self, inputs, output, goutput):
        return [cgt.flip(goutput, self.axes)]
    def shp_apply(self, inputs):
        return cgt.shape(inputs[0])
    def typ_apply(self, input_types):
        return input_types[0]
    def get_native_compile_info(self, input_types, devtype):
        x = input_types[0]
        openloops = " ".join(["for (int i%(ax)s=0; i%(ax)s < shape[%(ax)s]; ++i%(ax)s) {"%dict(ax=ax) for ax in xrange(x.ndim)])
        closeloops = "}"*x.ndim
        inidxexpr =  ",".join(["i%i"%ax for ax in xrange(x.ndim)])
        outidxexpr =  ",".join([("shape[%(ax)s] - 1 - i%(ax)s" if ax in self.axes else "i%(ax)s")%dict(ax=ax) for ax in xrange(x.ndim)])
        code = r"""
            CGT_EXPORT_C void $function(void* cldata, cgtArray** reads, cgtArray* write) {
                cgtArray *in=reads[0], *out=write;
                cgt_assert(in->size() == out->size());
                const long* shape = in->shape();
                %(openloops)s
                    out->at<%(cdtype)s>(%(outidxexpr)s) = in->at<%(cdtype)s>(%(inidxexpr)s);
                %(closeloops)s
            }
            """%dict(openloops=openloops, outidxexpr=outidxexpr, closeloops=closeloops, 
    inidxexpr=inidxexpr, cdtype=np2c[input_types[0].dtype])
        return NativeCompileInfo(code)



# Linalg
# ----------------------------------------------------------------


class Mul21(Op):
    available_impls = ("python","native_cpu")        
    def __init__(self, tA):
        self.tA = tA
    def get_py_func(self, input_types):
        def f(reads, write):
            x,y = reads
            if self.tA: x = x.T
            x.dot(y, out=write)
        return f
    def get_replacement(self, inputs, analysis):
        if inputs[1] in analysis["node2sv"]:
            return cgt.sum(inputs[0],0 if self.tA else 1) * analysis["node2sv"][inputs[1]]
    def pullback(self, inputs, _output, goutput):
        return [cgt.outer(goutput,inputs[1]), Result(Mul21(not self.tA), [inputs[0],goutput])]
    def shp_apply(self, inputs):
        assertequal1(cgt.size(inputs[0],0 if self.tA else 1),cgt.size(inputs[1],0),
            "shape mismatch at matrix-vector multiplication")
        return [cgt.size(inputs[0], 1 if self.tA else 0)]
    def typ_apply(self, input_types):
        return TensorType(input_types[0].dtype, 1)
    def get_closure(self):
        return [("tA",ctypes.c_bool, self.tA),("handle", ctypes.c_void_p, 0)]
    # gemv docs: https://software.intel.com/en-us/node/520750
    def get_native_compile_info(self, input_types, devtype):
        npdtype = input_types[0].dtype
        try:
            letter = {"f4":"s","f8":"d","c8":"c","c16":"z"}[npdtype]
        except KeyError:
            raise MethodNotDefined("Dtype %s not supported by this BLAS. Falling back to numpy"%npdtype)
        if devtype == "cpu":            
            code = r"""
                CGT_EXPORT_C void $function($closure* cl, cgtArray** Ax, cgtArray* y) {
                    cgtArray *A=Ax[0], *x=Ax[1];
                    int lda = A->shape()[1];
                    int M = A->shape()[0];
                    int N = A->shape()[1];
                    const %(cdtype)s alpha=1, beta=0;
                    int incx = 1, incy = 1;
                  cblas_%(letter)sgemv(CblasRowMajor, (CBLAS_TRANSPOSE)(cl->tA + 111), M, N, alpha, (%(cdtype)s*)A->data(), lda, (%(cdtype)s*)x->data(),
                      incx, beta, (%(cdtype)s*)y->data(), incy);
                }
                """%dict(letter=letter, cdtype = np2c[npdtype])
        elif devtype == "gpu":
            code = r"""
                CGT_EXPORT_C void $function($closure* cl, cgtArray** Ax, cgtArray* y) {
                    if (!cl->handle) cublasCreate_v2((cublasHandle_t*)&cl->handle);                                    
                    cgtArray *A=Ax[0], *x=Ax[1];
                    int lda = A->shape()[1];
                    int M = A->shape()[0];
                    int N = A->shape()[1];
                    const %(cdtype)s alpha=1, beta=0;
                    int incx = 1, incy = 1;
                  cublas_%(letter)sgemv((cublasHandle_t)cl->handle, (cublasOperation_t)(!cl->tA), N, M, alpha, (%(cdtype)s*)A->data(), lda, (%(cdtype)s*)x->data(),
                      incx, beta, (%(cdtype)s*)y->data(), incy);
                }"""%dict(letter=letter, cdtype = np2c[npdtype])         
        return NativeCompileInfo(code, includes=["cblas.h"], link_flags="-lopenblas", closure_triples = self.get_closure())
    def get_expr(self, (xexpr,yexpr)):
        return u"%s%s \u00D7 %s"%(xexpr, u"\u1d57" if self.tA else "", yexpr)

class Mul22(Op):
    @property
    def available_impls(self):
        return ("python",) if cgt.get_precision() == "quad" else ("python","native_cpu","native_gpu")
    def __init__(self, tA, tB):
        self.tA = tA
        self.tB = tB
    def get_py_func(self, input_types):
        def f(reads, write):
            x,y = reads
            if self.tA: x = x.T
            if self.tB: y = y.T
            x.dot(y, out=write)
        return f
    def pullback(self, inputs, output, goutput):
        """
        mul(F,F) Aij Bjk -> Cik
        g[0]: GAij = mul(F,T) GCik Bjk
        g[1]: GBjk = mul(T,F) Aij GCik 

        mul(F,T) Aij Bkj -> Cik
        g[0]: GAij = mul(F,F) GCik Bkj
        g[1]: GBkj = mul(T,F) GCik Aij

        mul(T,F) Aji Bjk -> Cik
        g[0]: GAji = mul(F,T) Bjk GCik
        g[1]: GBjk = mul(F,F) Aji GCik 

        mul(T,T) Aji Bkj -> Cik
        g[0]: GAji = mul(T,T) Bkj GCik
        g[1]: GBkj = mul(T,T) GCik Aji

        """
        A,B = inputs
        GC = goutput
        if (self.tA, self.tB) == (False,False):
            return [Result(Mul22(False,True), [GC, B]),
                    Result(Mul22(True,False), [A, GC])]
        elif (self.tA, self.tB) == (False,True):
            return [Result(Mul22(False,False), [GC, B]),
                    Result(Mul22(True,False), [GC, A])]
        elif (self.tA, self.tB) == (True,False):
            return [Result(Mul22(False,True), [B, GC]),
                    Result(Mul22(False,False), [A, GC])]
        elif (self.tA, self.tB) == (True,True):
            return [Result(Mul22(True,True), [B, GC]),
                    Result(Mul22(True,True), [GC, A])]

    def shp_apply(self, inputs):
        return [cgt.size(inputs[0], 1 if self.tA else 0),cgt.size(inputs[1],0 if self.tB else 1)]
    def typ_apply(self, input_types):
        # assertequal1(cgt.size(inputs[0],0 if self.tA else 1),cgt.size(inputs[1],1 if self.tB else 0), 
        #     "shape mismatch at matrix-matrix multiplication")         
        # TODO put shape check somewhere
        assert input_types[0].dtype==cgt.floatX and input_types[1].dtype==cgt.floatX
        return input_types[0]
    def get_closure(self):
        return [("tA",ctypes.c_bool, self.tA), ("tB",ctypes.c_bool, self.tB), ("handle",ctypes.c_void_p, 0)]
    # best gemm docs: https://software.intel.com/en-us/node/520775
    def get_native_compile_info(self, input_types, devtype):
        npdtype = input_types[0].dtype
        try:
            letter = {"f4":"s","f8":"d","c8":"c","c16":"z"}[npdtype]
        except KeyError:
            raise MethodNotDefined("Dtype %s not supported by this BLAS. Falling back to numpy"%npdtype)
        if devtype == "cpu":
            code = r"""
                CGT_EXPORT_C void $function($closure* cl, cgtArray** AB, cgtArray* C) {
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
            return NativeCompileInfo(code, includes=["cblas.h"], link_flags="-lopenblas", closure_triples=self.get_closure())
        elif devtype == "gpu":
            letter = letter.upper()
            code = r"""
                CGT_EXPORT_C void $function($closure* cl, cgtArray** AB, cgtArray* C) {
                    if (!cl->handle) cublasCreate_v2((cublasHandle_t*)&cl->handle);
                    cgtArray *A=AB[0], *B=AB[1];
                    int lda = A->shape()[1], ldb = B->shape()[1], ldc = C->shape()[1];
                    int M = C->shape()[0];
                    int N = C->shape()[1];
                    int K = A->shape()[cl->tA ? 0 : 1];
                    const %(cdtype)s alpha=1, beta=0;
                    CUBLAS_CHECK(cublas%(letter)sgemm_v2((cublasHandle_t)cl->handle, (cublasOperation_t)cl->tB, (cublasOperation_t)cl->tA, N, M, K, &alpha, (%(cdtype)s*)B->data(), ldb, (%(cdtype)s*)A->data(),
                      lda, &beta, (%(cdtype)s*)C->data(), ldc));

                }
                """%dict(letter=letter, cdtype = np2c[npdtype])
            return NativeCompileInfo(code, includes=["cublas_v2.h","cgt_cuda.h"], link_flags="-lcublas -lcudart", closure_triples=self.get_closure())

    def get_expr(self, (xexpr,yexpr)):
        return u"%s%s \u00D7 %s%s"%(xexpr, u"\u1d57" if self.tA else "", yexpr, u"\u1d57" if self.tB else "")
    def __repr__(self):
        return "Mul22{%s,%s}"%("T" if self.tA else "N", "T" if self.tB else "N")

class BatchedMul22(Op):
    available_impls = ("python","native_cpu")        
    def __init__(self, tA, tB):
        self.tA = tA
        self.tB = tB
    def get_py_func(self, input_types):
        def f((x,y), z):
            for (xmat, ymat, zmat) in zip(x,y, z):
                if self.tA: xmat = xmat.T
                if self.tB: ymat = ymat.T            
                xmat.dot(ymat, out=zmat)
        return f
    def pullback(self, inputs, output, goutput):
        A,B = inputs
        GC = goutput
        if (self.tA, self.tB) == (False,False):
            return [Result(BatchedMul22(False,True), [GC, B]),
                    Result(BatchedMul22(True,False), [A, GC])]
        elif (self.tA, self.tB) == (False,True):
            return [Result(BatchedMul22(False,False), [GC, B]),
                    Result(BatchedMul22(True,False), [GC, A])]
        elif (self.tA, self.tB) == (True,False):
            return [Result(BatchedMul22(False,True), [B, GC]),
                    Result(BatchedMul22(False,False), [A, GC])]
        elif (self.tA, self.tB) == (True,True):
            return [Result(BatchedMul22(True,True), [B, GC]),
                    Result(BatchedMul22(True,True), [GC, A])]    
    def shp_apply(self, inputs):
        return [cgt.size(inputs[0],0), cgt.size(inputs[0], 2 if self.tA else 1),cgt.size(inputs[1],1 if self.tB else 2)]
    def typ_apply(self, input_types):
        # assert inputs[0].dtype==cgt.floatX and inputs[1].dtype==cgt.floatX
        return input_types[0]
    def get_closure(self):
        return [("tA",ctypes.c_bool, self.tA), ("tB",ctypes.c_bool, self.tB)]        
    # <COPIED FROM Mul22> but incremented all dimensions
    def get_native_compile_info(self, input_types, devtype):
        npdtype = input_types[0].dtype
        try:
            letter = {"f4":"s","f8":"d","c8":"c","c16":"z"}[npdtype]
        except KeyError:
            raise MethodNotDefined("Dtype %s not supported by this BLAS. Falling back to numpy"%npdtype)
        code = r"""
            CGT_EXPORT_C void $function($closure* cl, cgtArray** AB, cgtArray* C) {
                cgtArray *A=AB[0], *B=AB[1];
                int P = A->shape()[0];
                int lda = A->shape()[1+1], ldb = B->shape()[1+1], ldc = C->shape()[1+1];
                int M = C->shape()[1+0];
                int N = C->shape()[1+1];
                int K = A->shape()[1+(cl->tA ? 0 : 1)];
                const %(cdtype)s alpha=1, beta=0;
              for (int i=0; i < P; ++i) {
                  cblas_%(letter)sgemm(CblasRowMajor, (CBLAS_TRANSPOSE)(cl->tA + 111), (CBLAS_TRANSPOSE)(cl->tB + 111), M, N, K, alpha, (%(cdtype)s*)A->data()+i*A->stride(0), lda, 
                    (%(cdtype)s*)B->data()+i*B->stride(0), ldb, beta, (%(cdtype)s*)C->data()+ i*C->stride(0), ldc);  
              }
            }
            """%dict(letter=letter, cdtype = np2c[npdtype])
        return NativeCompileInfo(code, includes=["cblas.h"], link_flags="-lopenblas", closure_triples=self.get_closure())
    # </COPIED>

class Outer(Op):
    available_impls = ("python","native_cpu")        
    def get_py_func(self, input_types):
        def f(reads, write):
            write[:] = np.outer(reads[0], reads[1])
        return f
    def pullback(self, inputs, _output, goutput):
        return [goutput.dot(inputs[0]), inputs[1].dot(goutput)]
    def shp_apply(self, inputs):
        return [cgt.size(inputs[0],0), cgt.size(inputs[1],0)]
    def typ_apply(self, input_types):
        assert input_types[0] == input_types[1]
        return TensorType(input_types[0].dtype, 2)
    def get_native_compile_info(self, input_types, devtype):
        npdtype = input_types[0].dtype
        code = r"""
            CGT_EXPORT_C void $function(void**, cgtArray** xy, cgtArray* z) {
                cgtArray *x=xy[0], *y=xy[1];                
                for (int i=0; i < x->size(); ++i) {
                    for (int j=0; j < y->size(); ++j) {
                        z->at<%(cdtype)s>(i,j) = x->at<%(cdtype)s>(i) * y->at<%(cdtype)s>(j);
                    }
                }
            }
            """%dict(cdtype = np2c[npdtype])    
        return NativeCompileInfo(code)

# BLAS 1
# ----------------------------------------------------------------

class Dot(Op):
    available_impls = ("python","native_cpu")    
    return_type = "byref"
    def get_py_func(self, input_types):
        def f(reads,write):
            write[...] = np.dot(reads[0], reads[1])
        return f
    def pullback(self, inputs, _output, goutput):
        x, y = inputs
        return [y*goutput, x*goutput]
    def shp_apply(self, _):
        return []
    def typ_apply(self, input_types):
        assert input_types[0] == input_types[1]
        return TensorType(cgt.floatX, 0)
    def get_native_compile_info(self, input_types, devtype):
        npdtype = input_types[0].dtype
        code = r"""
            CGT_EXPORT_C void $function(void**, cgtArray** xy, cgtArray* z) {
                cgtArray *x=xy[0], *y=xy[1];
                %(cdtype)s out = 0;
                for (int i=0; i < x->size(); ++i) {
                    out += x->at<%(cdtype)s>(i) * y->at<%(cdtype)s>(i);
                }
                z->at<%(cdtype)s>(0) = out;
            }
            """%dict(cdtype = np2c[npdtype])    
        return NativeCompileInfo(code)

# Composition
# ----------------------------------------------------------------

class Composition(Op):
    available_impls = ("python",)        
    return_type = "byval"
    def __init__(self, inputs, outputs):
        self._inputs = inputs
        self._outputs = outputs
        analysis = analyze(outputs)
        node2shape = analysis["node2shape"]
        self._shp = tuple(node2shape[x] for x in outputs)
        assert [x.is_input() for x in inputs]
        self._nodes = list(topsorted(outputs))

        self._needs_compute_pullback = True

    def _compute_pullback(self):
        inputs = self._inputs
        outputs = self._outputs
        dio = set(differentiably_influences(outputs))
        wrt = [x for x in inputs if x in dio]

        self._goutput = [Argument(x.typ) for x in outputs]
        gwrt = pullback(self._outputs, self._goutput, wrt)
        
        wrtidx = 0
        self._gin = []
        for x in inputs:
            if x in dio:
                self._gin.append(gwrt[wrtidx])
                wrtidx += 1
            self._gin.append(None)

        self._diff = [node in dio for node in self._inputs]
        self._out_typs = [x.typ for x in outputs]
        self._needs_compute_pullback = False

    def get_diff(self, _):
        return self._diff
    def get_py_func(self, input_types):
        # TODO testme
        f = cgt.compilation.function(self._inputs, self._outputs)
        def py_impl(num_inputs):
            return tuple(f(num_inputs))
        return py_impl
    def pullback(self, inputs, output, goutput):
        # repl = {}
        # repl.update(utils.safezip(self._inputs, inputs))
        # repl.update(utils.safezip(self._outputs, output))
        # repl.update(utils.safezip(self._goutput, goutput))
        # return clone(self._gin, replace=repl)
        if self._needs_compute_pullback:
            self._compute_pullback()
        gwrt = pullback([output], [goutput], inputs)
    def shp_apply(self, inputs):
        out = clone(self._shp, replace=dict(utils.safezip(self._inputs, inputs)))
        return out
    def typ_apply(self, input_types):
        assert input_types == [x.typ for x in self._inputs]
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
    available_impls = ("python","native_cpu","native_gpu")        
    return_type="byval"
    def __init__(self, idx):
        self.idx = idx
    def get_py_func(self, input_types):
        def f(reads):
            return reads[0][self.idx]
        return f
    def shp_apply(self, inputs):
        return cgt.shape(inputs[0])[self.idx]
    def typ_apply(self, input_types):
        intype = input_types[0]
        assert isinstance(intype, TupleType)
        return intype[self.idx]
    def get_closure(self, _inputs):
        return [("idx",ctypes.c_int, self.idx)]
    def get_native_compile_info(self, input_types, devtype):
        code=r"""
            CGT_EXPORT_C cgtObject* $function($closure* cldata, cgtTuple** reads) {
                return reads[0]->getitem(cldata->idx);
            }"""
        return NativeCompileInfo(code, closure_triples=self.get_closure(input_types))



class MakeTuple(Op):
    available_impls = ("python",)        
    return_type="byval"
    def get_py_func(self, input_types):
        def f(inputs):
            return tuple(inputs)
        return f
    def shp_apply(self, inputs):
        return tuple(cgt.shape(x) for x in inputs)
    def typ_apply(self, input_types):
        assert all(isinstance(t, TensorType) for t in input_types), "Can only create tuples of tensors" # @TUPLES_OF_TENSORS
        return TupleType(*input_types)
    
def unpack(tup):
    return [Result(TupleIndex(i),[tup]) for i in xrange(len(tup.typ))]

# Assertion and debug operations
# ----------------------------------------------------------------

# XXX currently not being used / tested anywhere

class Assertion(Op):
    """
    Assertion gets evaluated when the graph is executed, and it prints out a stack trace on failure
    """
    available_impls = ("python",)        
    def __init__(self, msg):
        self.stack = traceback.extract_stack()[:-2]
        self.msg = msg
    def typ_apply(self, input_types):
        x, = input_types
        assert x.ndim==0 and x.dtype=='i1'
        return TensorType('i8',0)
    def shp_apply(self, _):
        return []
    def get_py_func(self, input_types):
        def f(reads, _):
            x = reads[0]
            if not x.item():
                self.display_error()
        return f
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
    available_impls = ("python",)    
    def __init__(self, yourfunc):
        self.yourfunc = yourfunc
    def typ_apply(self, _):
        return TensorType('i8',0)
    def shp_apply(self, _):
        return []
    def get_py_func(self, input_types):
        def f(_, __):
            def fn(*reads):
                self.yourfunc(*reads)
        return f

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
    return [repl[node] for node in outputs], analysis, repl

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
        newnode = node.clone(newparents)
        newnewnode = maybe_replace(newnode, analysis, repl)
        if newnewnode is None:
            return (orig,newnode)
        else:
            assert newnewnode.typ == orig.typ
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
    Moreover, analysis contains 

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
    assert len(node2shape[node]) == node.ndim or isinstance(node.typ,TupleType)
    # -- SCALAR VALUE --
    if not node.is_input():
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
        newnode = analysis["hash2node"][h]
        if VERBOSE_OPTIMIZATION: print "Did CSE", node
        assert newnode in repl and newnode.op.__class__ == node.op.__class__
        return newnode
    parents = node.parents
    # -- CONSTANT PROP --
    # ASSUMPTION: the only type of nullary ops that we can propagate this way
    # are subclasses of Constant
    if len(parents) > 0 and all(isinstance(par.op, Constant) for par in parents):
        c = cgt.compilation.get_callable(node.op, [par.typ for par in parents], "cpu", True)
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
    """
    xs : a variable or list of variables
    Compute equivalent expression(s) in which simplifications have been applied
    """
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
    """
    Given a list of output nodes, compute the number of ancestors
    """
    if isinstance(outputs, Node): outputs = [outputs]
    return len(list(topsorted(outputs)))

def clone(nodes, replace=None):
    if isinstance(nodes, Node): return _clone_list([nodes], replace)[0]
    else: return _clone_list(list(nodes), replace)

def _clone_list(nodes, replace):
    assert isinstance(nodes, list)
    if replace is None: replace = {}
    else:
        assert isinstance(replace, dict)
        replace = replace.copy()
        for (k,v) in replace.iteritems():
            if not isinstance(v, Node):
                replace[k] = as_node(v)
    for node in topsorted(nodes):
        if node in replace:
            pass
        elif node.is_input():
            replace[node] = node
        else:
            replace[node] = node.clone([replace[p] for p in node.parents])
    return [replace[node] for node in nodes]

def alloc_from_shp(shp, typ):
    if isinstance(shp, tuple):
        return tuple([alloc_from_shp(shpel,typel) for (shpel,typel) in utils.safezip(shp,typ)])
    else:
        return np.empty(shp,typ.dtype)

def alloc_output(node, vals):
    typ = node.typ
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
    args = [make_argument(p.typ) for p in node.parents]
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
        callable = cgt.compilation.get_callable(node.op, [par.typ for par in node.parents],"cpu", True)
    except MethodNotDefined:
        print 'Op %s has no Python implementation' % repr(node.op)
        raise

    if node.op.return_type == "byval":
        out = callable.call(vals)
    else:
        out = alloc_output(node,vals)
        callable.call(vals, out)
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

class Unreachable(Exception):
    pass

def get_cgt_src_root():
    return osp.dirname(osp.dirname(osp.realpath(__file__)))

# ================================================================
# Global config
# ================================================================
 
_CONFIG = None
def get_config(force_reload = False):
    """
    Return the global configuration, which is loaded from your rcfile
    and the environment variables CGT_FLAGS
    """
    global _CONFIG
    if _CONFIG is None or force_reload:
        _CONFIG = _load_config()
    return _CONFIG

def _load_config():
    from thirdparty.configobj import ConfigObj
    from thirdparty.validate import Validator
    rcfileloc = osp.join(osp.expanduser("~/.cgtrc"))
    specfilename = osp.join(get_cgt_src_root(), "cgtrc_spec.ini")
    config = ConfigObj(rcfileloc, configspec=specfilename)
    val = Validator()
    test = config.validate(val,preserve_errors=True)
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
            assert lhs in config, "Unrecognized config option %s provided"%lhs
            oldrhs = config[lhs]
            config[lhs] = rhs
            assert isinstance(rhs, (str,bool,int,float,list)), "You set %s=%s but rhs is invalid"%(lhs, rhs)
            if isinstance(oldrhs, str): pass
            elif isinstance(oldrhs, bool): config[lhs] = config.as_bool(lhs)
            elif isinstance(oldrhs, int): config[lhs] = config.as_int(lhs)
            elif isinstance(oldrhs, float): config[lhs] = config.as_float(lhs)
            elif isinstance(oldrhs, list): config[lhs] = config.as_list(lhs)
    config["default_device"] = Device()
    cgt.set_precision(config["precision"])
    return config

def reset_config():
    """
    Reload config from CGT_FLAGS and cgtrc
    I.e., discard values set at runtime, e.g. through update_config and set_precision
    """
    get_config(True)

def update_config(**kws):
    """
    Globally update the provided configuration variables
    """
    config = get_config()
    for (name,val) in kws.iteritems():
        if name not in config:
            raise ValueError("%s is not a valid config option"%name)
        config[name] = val

class scoped_update_config(object):
    """
    example usage: 

    with scoped_update_config(precision='single',backend='native', parallel=True)
        ...

    Changes relevant config variables in the scope of the `with` statements, and change them
    back when we leave this scope
    """
    def __init__(self, **kw):
        self.kw = kw
        config = get_config()
        self.prevsettings = {}
        for k in kw.iterkeys(): 
            if k in config: 
                self.prevsettings[k] = config[k]
            else:
                raise ValueError("%s is not a valid config option"%k)

    def __enter__(self):
        config = get_config()
        config.update(self.kw)
        cgt.set_precision(config["precision"])
    def __exit__(self, *args):
        config = get_config()
        config.update(self.prevsettings)


# TAGS
# Just a few labels in the code for assumptions we're making now
# which we might change later.
# @TUPLES_OF_TENSORS : assumes all elements of TupleType object are TensorType    
# @TAG_HACK : a non-local interaction between inplace optimization and other optimizations. 
#   Certain operations created by pullback should be performed in place, but other optimizations 
#   like CSE make that impossible. So we add an extra field that associates arrays of zeros with the node that
#   they represent the gradient for, to prevent CSE from cutting out these nodes
# @SHAPE_CHECK : eventually we should check the shape while building up the graph, but this functionality isn't set up in a fully coherent way yet
