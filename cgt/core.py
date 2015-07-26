import sys, numpy as np, hashlib, copy, cPickle, re, operator, ctypes, warnings, os.path as osp, subprocess, StringIO, os
from collections import defaultdict,namedtuple
import traceback
import __builtin__

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
            return floatX
        elif dt.char in 'iulBb':
            return 'i'+str(dt.itemsize)
        elif dt.char in 'FDG':
            return 'c'+str(dt.itemsize)
        else:
            raise ValueError("Invalid dtype %s"%dt)

class Type(object):
    """
    Represents a datatype for Nodes
    """
    pass

class Tensor(Type):
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

class Tuple(Type):
    """
    Currently unused, will represent a compound type
    """
    def __init__(self, *eltypes):
        assert all(isinstance(eltype, Type) for eltype in eltypes)
        self.eltypes = eltypes
        self.dtype = 'O'
    def __len__(self):
        return len(self.eltypes)
    def __getitem__(self, i):
        return self.eltypes[i]
    def __iter__(self):
        return iter(self.eltypes)
    def __repr__(self):
        return "(" + "".join(map(str,self.eltypes))+")"

class Vector(Type):
    """
    Currently unused, will represent a variable length vector of a given type
    """
    def __init__(self, eltype):
        self.eltype = eltype

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
    def __repr__(self):
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
        return complexX
    elif d1 == 'f' or d2 == 'f': 
        return floatX
    elif d1 == 'i' and d2 == 'i':
        assert d1 == d2
        return d1 + __builtin__.max(s1,s2)
    else:
        raise ValueError("Don't know what to do with dtypes %s,%s"%(typ1, typ2))

def _promote_multi(xtypes):
    return reduce(_promote, xtypes)

def _dtype_kind(dtype):
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
    d1 = _dtype_kind(typ1)
    if d1 == 'f' or d1 == 'c':
        return 'i8'
    else:
        return typ1

floatX = "f4"
complexX = "c8"

def set_precision(prec):
    """
    prec in {"single", "double"}
    globally set floating point precision for float and complex types
    """    
    assert prec in ("half","single", "double","quad")
    global floatX, complexX
    if prec == "half":
        floatX = 'f2'
        complexX = None
        warn("half precision not yet supported")
    elif prec == "single":
        floatX = 'f4'
        complexX = 'c8'
    elif prec == "double":
        floatX = 'f8'
        complexX = 'c16'
    elif prec == "quad":
        floatX = 'f16'
        complexX = 'c32'

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
        return isinstance(self, Input)
    def is_argument(self):
        return isinstance(self, Argument)
    def is_data(self):
        return isinstance(self, Data)
    def get_dtype(self):
        return self.typ.dtype
    def get_ndim(self):
        return self.typ.ndim if isinstance(self.typ, Tensor) else 0
    def get_type(self):
        return self.typ
    def get_name(self):
        raise NotImplementedError
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
        return elwise_binary("^", self, other)
    def __floordiv__(self, other):
        return floor_divide(self, other)
    def __radd__(self, other):
        return self.__add__(other)
    def __rmul__(self, other):
        return self.__mul__(other)
    def __rdiv__(self, other):
        return constant(other).__div__(self)
    def __rtruediv__(self, other):
        return constant(other).__rtruediv__(self)
    def __rsub__(self, other):
        return constant(other).__sub__(self)
    def __rfloordiv__(self, other):
        return constant(other).__floordiv__(self)
    @property
    def shape(self):
        return shape(self)
    @property
    def ndim(self):
        return self.get_ndim()
    @property
    def dtype(self):
        return self.get_dtype()
    @property
    def T(self):
        return transpose(self)
    
        
    def __getitem__(self, slis):
        return getitem(self, slis)
    def __iter__(self):
        raise TypeError("Node is not iterable")
    def __repr__(self):
        return self.get_name()

    def reshape(self, shp):
        assert isinstance(shp, (list,tuple))
        return reshape(self, shp)
    def dot(self, other):
        return dot(self, other)
    def sum(self, axis=None, keepdims=False):
        return sum(self, axis=axis, keepdims=keepdims)
    def max(self, axis=None, keepdims=False):
        return max(self, axis=axis, keepdims=keepdims)
    def argmax(self, axis=None, keepdims=False):
        return argmax(self, axis=axis, keepdims=keepdims)
    def mean(self, axis=None, keepdims=False):
        return mean(self, axis=axis, keepdims=keepdims)
    def transpose(self, axes=None):
        return transpose(self, axes=axes)
    def flatten(self):
        return flatten(self)

def _ndarray_type(value):
    assert isinstance(value, np.ndarray)
    return Tensor(value.dtype, value.ndim)

def _get_value_device(value):
    warnings.warn("todo: properly implement")
    return Device()

def _put_on_device(value, device):
    warnings.warn("not putting value on device")
    return value

def num_components(node):
    return len(node.get_type()) if isinstance(node.get_type(), Tuple) else 1

class MethodNotDefined(Exception):
    pass

class Op(object):
    """
    Describes an operation that will be performed on some data.
    """

    # attributes that can be overwritten in subclasses
    needs_alloc = True # memory must be allocated for the output
    dynamic_data = False # op contains to data that might be changed externally
    gpu_uses_c = False # even if output on GPU, use C implementation, not CUDA
    c_extra_includes = []
    c_extra_link_flags = ""
    cuda_extra_includes = []
    cuda_extra_link_flags = ""

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
        return "%s(%s)"%(self.get_name(), ",".join(parent_exprs))
    def get_hash(self):
        """
        Return a string that uniquely identifies the value of this Op.
        Should ideally be fixed across program runs
        """
        return cPickle.dumps(self.__dict__)
    def get_name(self):
        """
        Get a human-readable description of the Op, including its attributes
        """
        return type(self).__name__.lower()
    def get_numeric_py(self):
        """
        Get a numeric implementation of the Op, which takes the input arrays as arguments
        and returns an output array.
        This method is optional: if you define c_code or cuda_code, the graph execution will 
        fall back on these methods (BUT DOESN'T CURRENTLY DO SO -- TODO)
        """
        raise MethodNotDefined
    def get_closure(self, _inputs):
        """
        Return a ctypes Structure which will be passed as the first argument to the C/CUDA function.
        """
        return None
    def c_code(self, inputs):
        """
        Return C code implementing this function, with the name of the function replaced by CGT_FUNCNAME.
        """
        raise MethodNotDefined
    def cuda_code(self, inputs):
        """
        See c_code
        This code will be compiled with nvcc
        """
        raise MethodNotDefined
    def get_replacement(self, _newparents, _analysis):
        """
        Return the name of this node
        """
        return None
    def numeric_apply(self, inputs):
        # TODO: use C/CUDA when python func not defined
        return self.get_numeric_py()(*inputs)
    def pullback(self, inputs, output, goutput): #pylint: disable=W0613
        """
        Compute symbolic expressions for derivatives obtained by backpropagation on this Op
        Given a function y = f(x_1, x_2, ..., x_k), let J_k denote the Jacobian dy/dx_k
        pullback(...) computes gradx_k = J_k^T grady
        """
        raise MethodNotDefined
    def pushforward(self, inputs, output, goutput):
        """
        Compute symbolic expressions for derivatives obtained by "tangent propagation" on this Op
        Given a function y = f(x_1, x_2, ..., x_k), let J_k denote the Jacobian dy/dx_k
        pullback([x_1, ..., x_k], y, grady) := \sum_k J_k gradx_k
        """
        raise MethodNotDefined

    def __repr__(self):
        return self.get_name()

def _as_node(val_or_node):
    """
    If numeric data received, convert to a constant node
    """
    if isinstance(val_or_node, Node):
        return val_or_node
    elif isinstance(val_or_node, (int, float, np.ndarray)):
        return constant(val_or_node)
    else:
        raise ValueError("expected numeric data or Node, got object of type %s"%type(val_or_node))

class Result(Node):
    """
    Node representing an intermediate computational result, which depends on its parents in the graph
    (TODO: be more precise about semantics. Does it depend ONLY on the parents, or can it depend on 
    some exogenous input too? What about random variables?)
    """
    def __init__(self, op, parents, typ = None):
        parents = map(_as_node, parents)
        typ = op.typ_apply(parents) if typ is None else typ
        assert op is not None
        # self.stackinfo = traceback.extract_stack()
        Node.__init__(self, typ, op, parents)
    def get_expr(self, parent_exprs):
        return "%s(%s)"%(self.get_name(), ",".join(parent_exprs))
    def get_hash(self, node2hash):
        hashobj = hashlib.md5(self.op.get_hash())
        for p in self.parents: hashobj.update(node2hash[p])
        return hashobj.hexdigest()
    def get_name(self):
        return "Res{%s}"%self.op.get_name()
    def __repr__(self):
        return self.get_name()

class Input(Node):
    """
    Abstract class representing an input to the graph -- a node with no parents, which does not
    correspond to a computation.
    """
    def __init__(self, typ, name=None):
        self.name = "" if name is None else name
        assert isinstance(self.name, (str,unicode))
        Node.__init__(self, typ, None, [])
    def get_hash(self, _node2hash):
        hashobj = hashlib.md5(str(id(self)))
        # XXX
        return hashobj.hexdigest()
    def get_name(self):
        return self.name
    def get_fixed_shape(self):
        raise NotImplementedError

class Argument(Input):
    """
    Input to the graph that is an argument to a function call
    """
    def __init__(self, typ, name=None, fixed_shape=None):        
        Input.__init__(self, typ, name)
        if fixed_shape is not None:
            assert len(fixed_shape) == self.ndim
            self.fixed_shape = fixed_shape
        else:
            self.fixed_shape = (None,)*self.ndim
    def __repr__(self):
        return "Arg{%s,%s}"%(self.get_dtype(), self.get_ndim())
    def get_fixed_shape(self):
        return self.fixed_shape


# Just here as a temporary  hack so node.op does the right thing in cython
class GetData(Op):
    def __init__(self, datanode):
        self.datanode = datanode        
    def get_numeric_py(self):
        def fn():
            return self.datanode.get_value()
        return fn

class Data(Input):
    """
    An input to the graph, which is associated with a value and implicitly provided
    during function calls.
    Data is similar to global variables in standard programming languages.
    """
    def __init__(self, value,name=None,device=None, fixed_shape_mask=None):
        value = _as_valid_array(value)
        if device is None:
            self.value = value
            self.device = _get_value_device(value)
        else:
            self.value = _put_on_device(value, device)
            self.device = device
        self.name = "unnamed" if name is None else name
        assert self.value.dtype != object
        self.fixed_shape_mask = (False,)*self.value.ndim if fixed_shape_mask is None else fixed_shape_mask
        Input.__init__(self, _ndarray_type(value), name)
        self.op = GetData(self)
    def get_name(self):
        return self.name
    def get_device(self):
        return self.device
    def get_fixed_shape(self):
        shp = self.value.shape
        return [s if bfixed else None for (bfixed,s) in _safezip(self.fixed_shape_mask, shp)]
    def get_value(self):
        return self.value
    # TODO: remove external accesses to .value


# ================================================================
# Variable Constructors
# ================================================================

def scalar(name=None, dtype=None):
    return Argument(Tensor(floatX if dtype is None else dtype, 0), name)
def vector(name=None, dtype=None):
    return Argument(Tensor(floatX if dtype is None else dtype, 1), name)
def matrix(name=None, dtype=None):
    return Argument(Tensor(floatX if dtype is None else dtype, 2), name)
def tensor3(name=None, dtype=None):
    return Argument(Tensor(floatX if dtype is None else dtype, 3), name)
def tensor4(name=None, dtype=None):
    return Argument(Tensor(floatX if dtype is None else dtype, 4), name)

def tensor(dtype, ndim, name=None):
    return Argument(Tensor(floatX if dtype is None else dtype, ndim), name)

def _singleton_ones(dtype, ndim):
    return constant(np.ones((1,)*ndim, dtype))

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
            for (p,d) in _safezip(node.parents, node.get_diff()):
                if d: diset.add(p)
    return diset

def differentiably_influenced_by(wrt, outputs=None, nodelist=None):
    assert (outputs is None) != (nodelist is None) # one of these are provided
    if nodelist is None: nodelist = list(topsorted(outputs))
    dibset = set(wrt)
    for node in nodelist:
        if any(p in dibset and d for (p,d) in _safezip(node.parents, node.get_diff())):
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
    # XXX get rid of first args
    # assert all(node.is_input() for node in wrt)    
    # TODO Maybe get rid of inputs argument

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
    for (node, gnode) in _safezip(outputs, goutputs):
        var2gs[node] = [gnode]

    active = dio.intersection(dibw)

    for node in reversed(nodelist):
        if node not in active: continue
        # once we reach a node, we have already backpropagated from all parents
        # so now we can sum up the gradients
        if len(var2gs[node]) > 1:
            var2gs[node] = [add_multi(var2gs[node])]
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
                for (par,gpar) in _safezip(node.parents, gpars):
                    var2gs[par].append(gpar)
                    if gpar is not None:
                        assert (par.get_ndim() == gpar.get_ndim())
                        for (s0,s1) in zip(shape(par), shape(gpar)):
                            assert_(equal(s0,s1),"node: "+str(node))

    # only we already summed up the gradients for the input nodes, so just take
    # 0th element
    return [var2gs[node][0] for node in wrt]

def grad(cost, wrt):    
    """
    Compute the gradient of scalar-valued `cost` with respect to a list of variables `wrt`
    """
    # TODO: be more clear on what types wrt are allowed to be. Do they need to be inputs?
    assert cost.get_ndim() == 0
    gout = _singleton_ones(cost.get_dtype(), 0)
    return pullback([cost], [gout], wrt)



# ================================================================
# Execution
# ================================================================

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
        return Tensor('i8',0)
    def shp_apply(self, _):
        return []
    def get_numeric_py(self):
        def fn(x):
            if not x.item():
                self.display_error()
            return x
        return fn
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
        return Tensor('i8',0)
    def shp_apply(self, _):
        return []
    def get_numeric_py(self):
        def fn(*xs):
            self.yourfunc(*xs)
        return fn

def assert_(x,msg=None):
    add_debug_node(Result(Assertion(msg or "(empty)"), [x]))

def dbg_call(yourfunc, *args):
    add_debug_node(Result(DebugFunc(yourfunc), list(args)))

def add_debug_node(x):
    if debug_context.global_context is not None:
        debug_context.global_context.nodes.append(x)

class debug_context(object):
    global_context = None
    def __init__(self):
        self.nodes = []
    def __enter__(self):
        assert debug_context.global_context is None, "can only be in one debug context at a time"
        debug_context.global_context = self
        return self
    def __exit__(self, type, value, traceback):
        debug_context.global_context = None

def make_function(inputs, outputs, dbg = None, fixed_sizes=False, backend=None):
    config = load_config()
    backend = backend or config["backend"]

    single = isinstance(outputs, Node)
    if single: outputs = [outputs]
    if dbg: 
        if backend == "python":            
            outputs = dbg.nodes + outputs
        else:
            warn("Debugging nodes can currently only be used with the python backend, but %s was selected. Ignoring"%backend)
    
    if backend == "python":
        def fn(*invals):
            out = numeric_eval(outputs, {innode:inval for (innode,inval) in _safezip(inputs, invals)})
            if dbg and len(dbg.nodes)>0: out = out[len(dbg.nodes):]
            if single: out = out[0]
            return out
        return fn
    elif backend == "cython":
        if fixed_sizes: fn = FixedSizeFunc(inputs, outputs)
        else: fn = VarSizeFunc(inputs, outputs)
        if single: return lambda *invals : fn(*invals)[0]
        else: return fn        
    else:
        raise NotImplementedError("invalid backend %s"%backend)
    return fn

def get_cgt_src_root():
    return osp.dirname(osp.dirname(osp.realpath(__file__)))

_CONFIG = None
def load_config():
    global _CONFIG
    if _CONFIG is None:
        from configobj import ConfigObj
        from validate import Validator
        rcfileloc = osp.join(osp.expanduser("~/.cgtrc"))
        specfilename = osp.join(get_cgt_src_root(), "cgtrc_spec.ini")
        _CONFIG = ConfigObj(rcfileloc, configspec=specfilename)
        val = Validator()
        test = _CONFIG.validate(val,preserve_errors=True)
        if test is not True:
            for (k,v) in test.items():
                if v is not True:
                    error("%s: %s in %s"%(k,v.message,rcfileloc))
            raise ValueError
        envflags = os.getenv("CGT_FLAGS")
        if envflags:
            pairs = envflags.split(",")
            for pair in pairs:
                lhs,rhs = pair.split("=")
                assert lhs in _CONFIG
                _CONFIG[lhs] = rhs
        print "CONFIG",_CONFIG
    return _CONFIG


def numeric_eval(outputs, arg2val):
    """
    Evaluate outputs numerically. arg2val is a dictionary mapping arguments to numerical values
    """
    single = isinstance(outputs, Node)

    if single: outputs = [outputs]
    node2val = {}
    for node in topsorted(outputs):
        if node.is_argument():
            node2val[node] = arg2val[node]
        elif node.is_data():
            node2val[node] = node.get_value()
        else:
            node2val[node] = node.op.numeric_apply([node2val[par] for par in node.parents])
        # assert node.get_ndim() == np.array(node2val[node]).ndim
    numeric_outputs = [node2val[node] for node in outputs]
    if single:
        return numeric_outputs[0]
    else:
        return numeric_outputs

# UNUSED, but gives a much simpler implementation of CallSequence implemented in cython
class CallSequence(object):
    def __init__(self, inputs, outputs):
        fncalls = []
        var2idx = {}
        nodes = self.nodes = list(topsorted(outputs))
        for (i,node) in enumerate(nodes):
            var2idx[node] = i
            if not node.is_input():
                args = [var2idx[var] for var in node.parents]
                args.append(i)
                fn = node.op.get_numeric_py()
                fncalls.append( ( fn, args ) )

        self.varstore = [np.array(666, node.dtype) for node in nodes]
        self.fncalls = fncalls
        self.ininds = _indexin(inputs, nodes)
        self.outinds = _indexin(outputs, nodes)

    def execute(self):
        for (fn, varinds) in self.fncalls:
            self.varstore[varinds[-1]][...] = fn(*(self.varstore[i] for i in varinds[:-1])) # XXX


class VarSizeFunc(object):
    def __init__(self, inputs, tmpoutputs, devfn = None):
        import cycgt #pylint: disable=F0401
        self.inputs = inputs        
        tmpoutputs = simplify(tmpoutputs)
        self.outputs, self.node2device = assign_devices(tmpoutputs, devfn)
        self.analysis = analyze(self.outputs)
        self.nodes = list(topsorted(self.outputs))  
        node2shape = self.analysis["node2shape"]
        self.shapevars = simplify([constant(np.array([],'i8')) 
            if node.get_ndim()==0 or node.get_ndim() is None 
            else stack(node2shape[node]) for node in self.nodes])
        self.callseq = cycgt.CallSequence(self.inputs, self.outputs, self.nodes, 
            node2device = self.node2device, check = load_config()["backend_check_values"])
    def __call__(self, *invals):
        shapes = numeric_eval(self.shapevars, dict(_safezip(self.inputs, invals)))
        self.callseq.set_shapes(shapes)
        self.callseq.set_inputs(invals)
        self.callseq.execute()
        return self.callseq.get_outputs_numpy()

class FixedSizeFunc(object):
    def __init__(self, inputs, tmpoutputs, devfn = None):
        import cycgt #pylint: disable=F0401
        self.inputs = inputs        
        tmpoutputs = simplify(tmpoutputs)
        self.outputs, self.node2device = assign_devices(tmpoutputs, devfn)
        self.analysis = analyze(self.outputs)
        self.nodes = list(topsorted(self.outputs))  
        node2shape = self.analysis["node2shape"]
        self.shapevars = simplify([constant(np.array([],'i8')) 
            if node.get_ndim()==0 or node.get_ndim() is None 
            else stack(node2shape[node]) for node in self.nodes])
        self.callseq = cycgt.CallSequence(self.inputs, self.outputs, self.nodes, 
            node2device = self.node2device, check = load_config()["backend_check_values"])        
        self.first_time = True

    def __call__(self, *invals):
        if self.first_time:
            shapes = numeric_eval(self.shapevars, dict(_safezip(self.inputs, invals)))
            self.callseq.set_shapes(shapes)
            self.first_time=False
        self.callseq.set_inputs(invals)
        self.callseq.execute()
        return self.callseq.get_outputs_numpy()

# ================================================================
# Printing 
# ================================================================

def print_tree(outputs, o=sys.stdout, nodefn=None):
    """
    Print out a representation of the computation graph as a tree
    nodefn is called after printing the result for every node, as
    nodefn(node, o)
    So you can print more attributes of the node
    """
    if isinstance(outputs, Node):
        outputs = [outputs]
    node2name = {}
    expands = []
    for node in outputs:
        _print_tree(node, 0, node2name, expands, o, nodefn)
    assert expands == []
    return node2name

def _print_tree(node, depth, node2name, expands, o, nodefn):
    o.write("| "*depth)
    if node in node2name: 
        varname = node2name[node]
        new = False
    else:
        varname = node.get_name() + "@%i"%len(node2name)
        node2name[node] = varname
        new = True

    color = Color.GREEN if node.is_input() else Color.RED
    colorprint(color, varname, o)

    if new:
        if nodefn is not None: nodefn(node, o)
        o.write("\n")
        for p in node.parents:
            _print_tree(p, depth+1, node2name, expands, o, nodefn)
    else:
        if not node.is_input(): o.write(" (see above)")
        o.write("\n")

def print_expr(x, o=sys.stdout):
    """
    Returns a string that represents a computation graph
    """
    node2s = {}
    o.write(_get_expr(x, node2s))
    o.write("\n")

def _get_expr(node, node2s):
    if node in node2s:
        return node2s[node]
    else:
        if node.is_input():
            name = node2s[node] = "@%i"%len(node2s)
            return name
        else:
            parent_exprs = [_get_expr(parent, node2s) 
                for parent in node.parents]
            return node.op.get_expr(parent_exprs)

def print_text(outputs, o=sys.stdout):
    """
    Print computation graph in single-statement assignment form,
    inspired by LLVM IR. (needs work)
    """
    if isinstance(outputs, Node):
        outputs = [outputs]
    node2name = {}
    for node in topsorted(outputs):
        thisname = node2name[node] = "@%i"%len(node2name)
        if node.is_input():
            o.write("Input: %s\n"%thisname)
        else:
            o.write("%s = %s %s\n"%(thisname, node.get_name(), " ".join(node2name[parent]
                for parent in node.parents)))

# ================================================================
# Ops 
# ================================================================


# Constants
# ----------------------------------------------------------------

class Constant(Op):
    needs_alloc=False
    def __init__(self, value):
        self.value = _as_valid_array(value)
        assert self.value.dtype != object
    def get_expr(self, parent_exprs):
        return self.get_name()
    def get_name(self):
        return "const{%s}"%self.value
    def get_numeric_py(self):
        def fn():
            return self.value
        return fn
    def pullback(self, _inps, _out, _gout):
        return []
    def shp_apply(self, _inputs):
        return [constant(x) for x in self.value.shape]
    def typ_apply(self, inputs):
        assert len(inputs)==0
        return _ndarray_type(self.value)
    def get_hash(self):
        return str(id(self))
    def get_closure(self, _):
        return [(ctypes.c_void_p, self.value.ctypes.data)] # XXX wont work for gpu
    def c_code(self, inputs):
        return """
typedef struct constcl {void* ptr} constcl; 
void CGT_FUNCNAME(void* cldata, cgt_array** io) {
    io[0]->data = ((constcl*)cldata)->ptr;
    io[0]->ownsdata = false;
}
"""

class Fill(Op):
    """
    (value, shape...) -> array filled with `value`, with shape `shape`
    """
    def __init__(self, value):
        self.value = np.array(value)
        self.dtype = self.value.dtype
        assert self.value.ndim==0
    def get_diff(self, num_inputs):
        return [False]*num_inputs
    def get_name(self):
        return "fill{%g}"%self.value
    def get_numeric_py(self):
        def fn(*shp):
            if len(shp) == 0: return np.array(self.value)
            else:
                out = np.empty(shp, self.dtype)
                out[:] = self.value
                return out
        return fn
    def pullback(self, inputs, output, goutput):
        raise NonDifferentiable
    def shp_apply(self, inputs):
        return inputs
    def typ_apply(self, inputs):
        assert all(x.get_dtype() == 'i8' for x in inputs)
        return Tensor(self.dtype, len(inputs))

class Arange(Op):
    """
    (start,stop,step) -> 1D array, just like numpy
    """
    def __init__(self, dtype='i8'):
        self.dtype = dtype
    def get_diff(self, num_inputs):
        return [False]*num_inputs
    def get_numeric_py(self):
        def fn(start,stop,step):
            return np.arange(start,stop,step, self.dtype)
        return fn
    def pullback(self, inputs, output, goutput):
        raise NonDifferentiable
    def shp_apply(self, inputs):
        start,stop,step = inputs
        return [(stop - start)//step]
    def typ_apply(self, inputs):
        assert _list_is_valid_shape(inputs)
        return Tensor(self.dtype, 1)

class ScalarRng(Op):
    """
    (shape...) -> array filled with iid random numbers, from either uniform or normal distribution
    """
    dynamic_data = True
    def __init__(self, kind):
        assert kind in ("uniform","gaussian")
        self.kind = kind
    def get_diff(self, num_inputs):
        return [False]*num_inputs
    def get_name(self):
        return "rng{%s}"%self.kind
    def get_numeric_py(self):
        if self.kind == "uniform": return lambda *shp: np.random.rand(*shp).astype(floatX)
        elif self.kind == "gaussian": return lambda *shp: np.random.randn(*shp).astype(floatX)
    def pullback(self, inputs, output, goutput):
        raise NonDifferentiable
    def shp_apply(self, inputs):
        return inputs
    def typ_apply(self, inputs):
        assert _list_is_valid_shape(inputs)
        return Tensor(floatX, len(inputs))

# Elementwise
# ----------------------------------------------------------------

def _no_grad():
    raise NonDifferentiable()

def _nu_sigmoid(x):
    return np.reciprocal(1+np.exp(-x))

def _nu_iceil(x):
    return np.ceil(x).astype(_type_to_int(Dtype.canon(x.dtype)))

def _nu_ifloor(x):
    return np.floor(x).astype(_type_to_int(Dtype.canon(x.dtype)))


UnaryInfo = namedtuple("UnaryInfo", ("short","pyfunc","diff","typeinfo", "gradexpr", "cexpr"))

UNARY_INFO = {
    "abs" : UnaryInfo(   "abs", np.abs,  True,   's', "gy*sign(x)", "fabs(x)"),
    "ceil" : UnaryInfo(  "ceil", np.ceil, False,  'i',  "_no_grad()", "ceil(x)"),
    "cos" : UnaryInfo(   "cos", np.cos,  True,   'f',   "-gy*sin(x)", "cos(x)"),
    "exp" : UnaryInfo(   "exp", np.exp,  True,   'f',   "gy*exp(x)", "exp(x)"),
    "iceil" : UnaryInfo( "iceil", _nu_iceil, False, 'i',   "_no_grad()", "(int)ceil(x)"),
    "ifloor" : UnaryInfo( "ifloor", _nu_ifloor, False, 'i',   "_no_grad()", "(int)floor(x)"),
    "log" : UnaryInfo(   "log", np.log,  True,   'f', "gy/x", "log(x)"),
    "neg" : UnaryInfo(   "negative", np.negative, True, 's', "-gy", "(-x)"),
    "sign" : UnaryInfo(   "sign", np.sign, False,   's',  "_no_grad()", "2*(x>0)-1"),
    "sin" : UnaryInfo(    "sin", np.sin,    True, 'f',  "gy*cos(x)", "sin(x)"),
    "square" : UnaryInfo( "square", np.square, True, 's',  "2.0*gy*x", "x*x"),
    "sqrt" : UnaryInfo( "sqrt", np.sqrt, True, 'f', "gy/(2.0*y)", "sqrt(x)"),
    "tanh" : UnaryInfo(   "tanh", np.tanh, True,   'f', "gy*(1-square(y))", "tanh(x)"),
    "sigmoid" : UnaryInfo( "sigmoid", _nu_sigmoid, True, 'f', "gy*y*(1-y)", "1.0/(1.0+exp(-x))"),
    "conj" : UnaryInfo( "conj", np.conj, True, 'c', "conj(gy)", "conj(x)")
}

BinaryInfo = namedtuple("BinaryInfo", ("short","pyfunc","commutes","diff","typeinfo","gradexpr", "cexpr"))


BINARY_INFO = {
    #infix             short      pyfunc    commutes     diff        typeinfo
    "*"   : BinaryInfo("multiply",  np.multiply, True,    (True,True),    'p',        "[y*gz,x*gz]", "x*y"),
    "+"   : BinaryInfo("add",  np.add,   True,    (True,True),    'p',        "[gz,gz]", "x+y"),
    "-"   : BinaryInfo("subtract",  np.subtract, False,    (True,True),   'p',       "[gz,-gz]", "x-y"),
    "/"   : BinaryInfo("divide",  np.divide,  False,    (True,True),    'p',       "[gz/y,-gz*z/y]", "x/y"),
    "<"   : BinaryInfo("less",   np.less,    False,    (False,False),  'i1',     "_no_grad()", "x<y"),
    "**"   : BinaryInfo("power",  np.power,      False,    (True,True), 'p',      "[gz*y*z/x,z*log(x)]","pow(x,y)"),
    "=="  : BinaryInfo("equal", lambda x,y : np.equal(x,y).astype('i'),      True,      (False, False), 'i1',  "_no_grad()", "x==y"),
}

class ElwiseUnary(Op):
    """
    Elementwise unary operation
    """
    c_extra_includes =  ["math.h"]

    def __init__(self, opname, info=None):
        self.opname = opname
        self.info = UNARY_INFO[opname] if info is None else info
    def get_diff(self, _):
        return [self.info.diff]
    def get_name(self):
        return self.opname
    def get_hash(self):
        return _hash_seq1(self.opname)
    def get_numeric_py(self):
        return self.info.pyfunc
    def get_replacement(self, _newparents, _analysis):
        return None
    def pullback(self, (x,), y, gy): #pylint: disable=W0613
        return [eval(self.info.gradexpr)]
    def shp_apply(self, inputs):
        return shape(inputs[0])
    def typ_apply(self, inputs):
        typeinfo = self.info.typeinfo
        intype = inputs[0].get_dtype()
        if typeinfo == 's':
            out_type = intype
        elif typeinfo == 'i':
            out_type = _type_to_int(intype)
        elif typeinfo == 'f':
            out_type = floatX
        elif typeinfo == 'c':
            out_type = complexX
        else:
            assert typeinfo in (floatX, complexX, 'i1','i2','i4','i8')
            out_type = typeinfo
        return Tensor(out_type, inputs[0].get_ndim())
    def c_code(self, inputs):
        info = self.info
        return """
static inline %(cdtype)s scalar_CGT_FUNCNAME(%(cdtype)s x) {return %(cexpr)s;}
void CGT_FUNCNAME(void* cldata, cgt_array** io) {
    int s = cgt_size(io[0]);
    %(cdtype)s* in = (%(cdtype)s*)io[0]->data;
    %(cdtype)s* out = (%(cdtype)s*)io[1]->data;
    for (int i=0; i < s; ++i) {
        out[i] = scalar_CGT_FUNCNAME(in[i]);
    }
}
"""%dict(cdtype=np2c[inputs[0].dtype],cexpr=info.cexpr)
    def cuda_code(self, inputs):
        info = self.info
        npdtype = inputs[0].dtype
        return """
__forceinline__ __device__ %(cdtype)s CGT_FUNCNAME(%(cdtype)s x) {return %(cexpr)s;}        
__global__ void CGT_FUNCNAME_kernel(const size_t n, const %(cdtype)s* in, %(cdtype)s* out) {
  CUDA_KERNEL_LOOP(i, n) {
    out[i] = CGT_FUNCNAME(in[i]);
  }
}
extern "C" void CGT_FUNCNAME(void* cldata, cgt_array** io) {
    size_t n = cgt_size(io[0]);
    int num_blocks, num_threads;
    cgt_get_bt(n, &num_blocks, &num_threads);
    CGT_FUNCNAME_kernel<<<num_blocks, num_threads>>>(n, (%(cdtype)s*)io[0]->data, (%(cdtype)s*)io[1]->data);
}
"""%dict(cdtype=np2c[npdtype],cexpr=info.cexpr)
    def cuda_includes(self):
        return ["cgt_cuda.h","cgt_utils.h"]

class ElwiseBinary(Op):
    c_extra_includes =  ["math.h"]    
    # +, -, *, /, <, ^, //
    def __init__(self, opname, scalar_mask, info=None):
        assert opname in BINARY_INFO        
        self.opname = opname
        self.info = BINARY_INFO[opname] if info is None else info
        self.scalar_mask = scalar_mask
    def get_diff(self, _):
        return BINARY_INFO[self.opname].diff
    def get_hash(self):
        return _hash_seq1(self.opname)        
    def get_expr(self, parent_exprs):
        return "(%s %s %s)"%(parent_exprs[0], self.opname, parent_exprs[1])
    def get_name(self):
        return BINARY_INFO[self.opname].short
    def get_numeric_py(self):
        return self.info.pyfunc
    def get_replacement(self, parents, analysis):
        node2sv = analysis["node2sv"]
        ind4shape = 1 if self.scalar_mask[0] else 0
        l,r = parents
        if l in node2sv and r in node2sv:
            return Result(Fill(self.numeric_apply([node2sv[l], node2sv[r]])), analysis["node2shape"][parents[ind4shape]])
        elif l in node2sv and isinstance(r.op, Constant) and r.ndim > 0:
            return Result( Constant(self.numeric_apply([node2sv[l], r.op.val])), [l,r])
        elif r in node2sv and isinstance(l.op, Constant) and l.ndim > 0:
            return Result(Constant(self.numeric_apply([l.op.val, node2sv[r]])), [l,r])

        if self.opname == "*":
            if r in node2sv and not l in node2sv: l,r = r,l
            if l in node2sv:
                if node2sv[l] == 1: return r
                elif node2sv[l] == -1: return -r
    def pullback(self, (x, y), z, gz): #pylint: disable=W0613
        gin = eval(BINARY_INFO[self.opname].gradexpr)
        return [sum(gv) if (v.ndim==0 and gv.ndim > 0) else gv for (v,gv) in _safezip([x,y],gin)]
    def shp_apply(self, inputs):
        ind4shape = 1 if self.scalar_mask[0] else 0
        return shape(inputs[ind4shape])
    def typ_apply(self, inputs):
        assert ((inputs[0].ndim==0) == self.scalar_mask[0]) and ((inputs[1].ndim==0) == self.scalar_mask[1])
        typeinfo = BINARY_INFO[self.opname].typeinfo
        if typeinfo == 'p':
            out_type = _promote(inputs[0].get_dtype(), inputs[1].get_dtype())
        else:
            out_type = typeinfo
        ind4shape = 1 if self.scalar_mask[0] else 0
        return Tensor(out_type, inputs[ind4shape].ndim)
    def c_code(self, inputs):
        typ2 = self.typ_apply(inputs)
        npdtype0 = inputs[0].dtype
        npdtype1 = inputs[1].dtype
        npdtype2 = typ2.dtype
        ind4shape = 1 if self.scalar_mask[0] else 0
        index0 = "0" if self.scalar_mask[0] else "i"
        index1 = "0" if self.scalar_mask[1] else "i"
        return """
static inline %(cdtype2)s scalar_CGT_FUNCNAME(%(cdtype0)s x, %(cdtype1)s y) {return %(cexpr)s;}
void CGT_FUNCNAME(void* cldata, cgt_array** io) {
    int s = cgt_size(io[%(ind4shape)s]);
    %(cdtype0)s* in0 = (%(cdtype0)s*)io[0]->data;
    %(cdtype1)s* in1 = (%(cdtype1)s*)io[1]->data;
    %(cdtype2)s* out = (%(cdtype2)s*)io[2]->data;
    for (int i=0; i < s; ++i) {
        out[i] = scalar_CGT_FUNCNAME(in0[%(index0)s], in1[%(index1)s]);
    }
}
"""%dict(cdtype0=np2c[npdtype0],cdtype1=np2c[npdtype1],cdtype2=np2c[npdtype2],
    cexpr=self.info.cexpr, index0=index0, index1=index1, ind4shape=ind4shape)  
    def c_includes(self):
        return ["cgt_utils.h","math.h"]
    def cuda_code(self, inputs):
        typ2 = self.typ_apply(inputs)
        npdtype0 = inputs[0].dtype
        npdtype1 = inputs[1].dtype
        npdtype2 = typ2.dtype
        return """
__forceinline__ __device__ %(cdtype2)s CGT_FUNCNAME(%(cdtype0)s x, %(cdtype1)s) {return %(cexpr)s;}
__global__ void CGT_FUNCNAME_kernel(const size_t n, const %(cdtype0)s* x, %(cdtype1)s* y, %(cdtype2)s z) { \
  CUDA_KERNEL_LOOP(i, n) {
    z[i] = CGT_FUNCNAME(x[%(index0)s], y[%(index1)s]);
  }
}
extern "C" void CGT_FUNCNAME(void* cldata, cgt_array** io) {
    size_t n = cgt_size(io[1]);
    int num_blocks,num_threads;
    cgt_get_bt(n, &num_blocks, &num_threads);
    TODO
    CGT_FUNCNAME_kernel<<<num_blocks, num_threads>>>(n, (%(cdtype0)s*)io[0]->data, (%(cdtype1)s*)io[1]->data, (%(cdtype2)s*)io[2]->data);
}
"""%dict(cdtype0=np2c[npdtype0],cdtype1=np2c[npdtype1],cdtype2=np2c[npdtype2],cexpr=self.info.cexpr,funcname=funcname, index="0" if self.kind==ElwiseBinary.ST else "i")  
    def cuda_includes(self):
        return ["cgt_cuda.h","cgt_utils.h"]


# Shape manip
# ----------------------------------------------------------------

class Size(Op):
    """
    Return an element of the shape of a tensor
    """
    def __init__(self, axis):
        self.axis = axis
    def get_diff(self, _):
        return [False]
    def get_name(self):
        return "size{%i}"%self.axis
    def get_numeric_py(self):
        def fn(arr):
            return arr.shape[self.axis]
        return fn
    def pullback(self, inputs, output, goutput):
        raise NonDifferentiable
    def shp_apply(self, _inputs):
        return []
    def typ_apply(self, _inputs):
        return Tensor('i8',0)
    def get_replacement(self, inputs, _analysis):
        x = inputs[0]
        if x.is_input():
            fixed_shape = x.get_fixed_shape()
            if fixed_shape[self.axis] is not None:
                return constant(fixed_shape[self.axis])
    def get_closure(self, inputs):
        return [(ctypes.c_int, self.axis)]
    def c_code(self, _):
        return r"""
        typedef struct axcl {int ax} axcl;
        void CGT_FUNCNAME(void* cldata, cgt_array** io) {
            ((long*)io[1]->data)[0] = io[0]->shape[((axcl*)cldata)->ax];
        }"""

class Reshape(Op):
    needs_alloc = False
    # XXX restore after we're sure the right thing happens with python impls
    gpu_uses_c = True
    def get_diff(self, num_inputs):
        return [True] + [False]*(num_inputs-1)
    def get_numeric_py(self):
        def fn(arr,*shp):
            return arr.reshape(shp)
        return fn
    def pullback(self, inputs, _out, gout):
        return [reshape(gout, shape(inputs[0]))] + [None]*(len(inputs)-1)
    def shp_apply(self, inputs):
        return inputs[1:]
    def typ_apply(self, inputs):
        return Tensor(inputs[0].get_dtype(), len(inputs)-1)
    def get_closure(self, parents):
        return [(ctypes.c_int,len(parents))]
    def c_code(self, _inputs):
        return "void CGT_FUNCNAME(void* cldata, cgt_array** io) {io[*((int*)cldata)]->data = io[0]->data; io[0]->ownsdata=false;}"

class Concatenate(Op):
    def __init__(self, axis):
        self.axis = axis
    def get_diff(self, num_inputs):
        return [True]*num_inputs
    def get_numeric_py(self):
        def fn(*arrs):
            return np.concatenate(arrs)
        return fn
    def pullback(self, inputs, _output, gout):
        start = 0
        out = []
        for x in inputs:
            end = start + size(x, self.axis)
            out.append(Result(GetSli(gout), [start,end]))
            start = end
        return out
    def shp_apply(self, inputs):
        out = shape(inputs[0])
        out[self.axis] = add_multi([size(x,self.axis) for x in inputs])
        return out
    def typ_apply(self, inputs):
        return Tensor(_promote_multi([x.dtype for x in inputs]), inputs[0].ndim)

class Stack(Op):
    def get_diff(self, num_inputs):
        return [True for _ in xrange(num_inputs)]
    def get_numeric_py(self):
        def fn(*vals):
            return np.array(vals)
        return fn
    def pullback(self, inputs, output, goutput):
        return [goutput[i] for i in xrange(len(inputs))]
    def shp_apply(self, inputs):
        return [constant(len(inputs))] + shape(inputs[0])
    def typ_apply(self, inputs):
        assert _allsame([x.get_type() for x in inputs])
        return Tensor(inputs[0].get_dtype(), inputs[0].get_ndim()+1)

class Repeat(Op):
    def __init__(self, axes):
        self.axes = axes
    def get_diff(self, num_inputs):
        return [True] + [False for _ in xrange(num_inputs-1)]
    def get_numeric_py(self):
        def fn(arr, *numreps):
            shp = arr.shape
            assert all(shp[i] == 1 for i in self.axes)
            for (ax,numrep) in _safezip(self.axes, numreps):
                arr = np.repeat(arr, numrep, ax)
            return arr
        return fn
    def get_replacement(self, parents, analysis):
        if parents[0] in analysis["node2sv"]:
            value = analysis["node2sv"][parents[0]]
            shp = self.shp_apply(parents)
            return Result(Fill(value), shp)
    def pullback(self, inputs, output, goutput):
        return [sum(goutput, self.axes, keepdims=True)] + [None]*(len(inputs)-1)
    def shp_apply(self, inputs):
        out = shape(inputs[0])
        for (ax,rep) in _safezip(self.axes, inputs[1:]):
            out[ax] = rep
        return out
    def typ_apply(self, inputs):
        assert all(x.get_dtype() == "i8" for x in inputs[1:])
        return inputs[0].get_type()

class Transpose(Op):
    def __init__(self, axes):
        self.axes = axes
    def get_diff(self, _):
        return [True]
    def get_numeric_py(self):
        def fn(arr):
            return arr.transpose(self.axes)
        return fn
    def pullback(self, inputs, output, goutput):
        return [transpose(goutput, _invert_perm(self.axes))] # XXX is this right?
    def shp_apply(self, inputs):
        inshape = shape(inputs[0])
        return [inshape[ax] for ax in self.axes]
    def typ_apply(self, inputs):
        return inputs[0].get_type()
    def c_code(self, inputs):
        return r"""
        void CGT_FUNCNAME(void* cldata, cgt_array** io) {
            cgt_array *in = io[0], *out = io[1];
            %(cdtype)s* indata = in->data, *outdata = out->data;
            for (int i=0; i < in->shape[0]; ++i) {
                for (int j=0; j < in->shape[1]; ++j) {
                    outdata[j*in->shape[0] + i] = indata[i*in->shape[1] + j];
                }
            }
        }"""%dict(cdtype=np2c[inputs[0].dtype])

class Transport(Op):
    gpu_uses_c = True
    def __init__(self, src, targ):
        assert isinstance(src, Device)
        self.src = src
        self.targ = targ
    def typ_apply(self, inputs):
        return inputs[0].get_type()
    def get_numeric_py(self):
        def fn(x):
            return x
        return fn
    def shp_apply(self, inputs):
        return shape(inputs[0])
    def c_code(self, _inputs):
        return """
void CGT_FUNCNAME(void* cldata, cgt_array** io) {
    cgt_memcpy(io[1]->devtype, io[0]->devtype, io[1]->data, io[0]->data, cgt_nbytes(io[0]));    
}
"""

# TODO save computation by removing negative freq components
class RFFT(Op):
    def __init__(self, axes):
        self.axes = axes
    def get_diff(self, num_inputs):
        return [True] + [False]*(num_inputs-1)
    def get_numeric_py(self):
        def fn(x,*shp):
            return np.fft.fftn(x,shp,self.axes)
        return fn
    def pullback(self, inputs, _outputs, goutput):
        return real(Result(RFFT(self.axes),[goutput]+inputs[1:]))
    def shp_apply(self, inputs):
        out = shape(inputs[0])
        for (ax,sz) in _safezip(self.axes, inputs[1:]):
            out[ax]=sz
        return out
    def typ_apply(self, inputs):
        assert inputs[0].dtype==floatX
        return Tensor(complexX,inputs[0].ndim)

class IRFFT(Op):
    def __init__(self, axes):
        self.axes = axes
    def get_diff(self, _):
        return [True]
    def get_numeric_py(self):
        def fn(x,*shp):
            slis = [slice(0,None) for _ in xrange(x.ndim)]
            for (ax,s) in zip(self.axes,shp): slis[ax] = slice(0, s)
            return np.real(np.fft.ifftn(x,axes=self.axes)[slis])
        return fn
    def pullback(self, inputs, _outputs, goutput):
        return Result(IRFFT(self.axes),[goutput]) # XXX is this right?
    def shp_apply(self, inputs):
        return shape(inputs[0])
    def typ_apply(self, inputs):
        return Tensor(floatX,inputs[0].ndim)

def rfft(x, periods, axes):
    return Result(RFFT(axes),[x]+list(periods))

def irfft(x, axes):
    return Result(IRFFT(axes),[x])

# Reductions
# ----------------------------------------------------------------

class Sum(Op):
    def __init__(self, axes):
        self.axes = tuple(axes)
    def get_diff(self, _):
        return [True]
    def get_name(self):
        return "sum{%s}"%(",".join(map(str,self.axes)))
    def get_numeric_py(self):
        return lambda x : x.sum(axis=self.axes or None,keepdims=True)
    def pullback(self, inputs, output, goutput):
        return [Result(Repeat(self.axes), [goutput] + [size(inputs[0],ax) for ax in self.axes])]
    def shp_apply(self, inputs):
        x = inputs[0]
        s = shape(x)
        return [(constant(1) if i in self.axes else s[i]) for i in xrange(x.get_ndim())]
    def typ_apply(self, inputs):
        return inputs[0].get_type()
    def c_code(self, inputs):
        x = inputs[0]
        openloops = " ".join(["for (int i%(ax)s=0; i%(ax)s < in->shape[%(ax)s]; ++i%(ax)s) {"%dict(ax=ax) for ax in xrange(x.ndim)])
        closeloops = "}"*x.ndim
        outdims = [i for i in xrange(x.ndim) if i not in self.axes]
        inidxexpr = " + ".join(["i%(ax)s * instrides[%(ax)s]"%dict(ax=ax) for ax in xrange(x.ndim)])
        outidxexpr = " + ".join(["i%(ax)s * outstrides[%(ax)s]"%dict(ax=ax) for ax in outdims])\
            if len(outdims)>0 else "0"
        return r"""
void CGT_FUNCNAME(void* cldata, cgt_array** io) {
    cgt_array *in=io[0], *out=io[1];
    size_t instrides[in->ndim];
    cgt_get_strides(in, instrides);
    size_t outstrides[out->ndim];
    cgt_get_strides(out, outstrides);
    memset(out->data, 0, cgt_nbytes(out));
    %(openloops)s
        ((%(cdtype)s*)out->data)[%(outidxexpr)s] += ((%(cdtype)s*)in->data)[%(inidxexpr)s];
    %(closeloops)s
}
"""%dict(openloops=openloops, outidxexpr=outidxexpr, inidxexpr=inidxexpr, closeloops=closeloops,
    cdtype=np2c[inputs[0].dtype])
    c_extra_includes = ["string.h"]

class Max(Op):
    def __init__(self, axes):
        self.axes = tuple(axes)
    def get_diff(self, _):
        return [True]
    def get_name(self):
        return "max{%s}"%(",".join(map(str,self.axes)))
    def get_numeric_py(self):
        return lambda x : x.max(axis=self.axes or None,keepdims=True)
    def pullback(self, inputs, output, goutput):
        x = inputs[0]
        inputpat = "x"*x.ndim
        singpat = "".join(["1" if i in self.axes else "x" for i in xrange(x.ndim)])
        return [broadcast("==", output, x, singpat+","+inputpat) * goutput]
        # XXX doesn't deal well with corner case
    def shp_apply(self, inputs):
        x = inputs[0]
        s = shape(x)
        return [(constant(1) if i in self.axes else s[i]) for i in xrange(x.get_ndim())]
    def typ_apply(self, inputs):
        return inputs[0].get_type()

class Argmax(Op):
    def __init__(self, axis):
        self.axis = axis
    def get_diff(self, _):
        return [False]
    def get_name(self):
        return "argmax{%s}"%self.axis
    def get_numeric_py(self):
        return lambda x : x.argmax(axis=self.axis,keepdims=True)
    def shp_apply(self, inputs):
        x = inputs[0]
        s = shape(x)
        return [(constant(1) if i == self.axis else s[i]) for i in xrange(x.get_ndim())]
    def typ_apply(self, inputs):
        return Tensor('i8', inputs[0].ndim)

# TODO consider reducing code duplication for e.g. shp_apply

# Casting / data movement
# ----------------------------------------------------------------

# class Copy(Op):
#     pass # TODO

# Slicing
# ----------------------------------------------------------------

class GetSli(Op):
    def __init__(self, axis):
        self.axis = axis
    def get_diff(self, _):
        return [True,False,False,False]
    def get_numeric_py(self):
        def fn(x, start, stop, step):
            assert step > 0
            slices = [slice(None,None,None) for _ in xrange(x.ndim)]
            slices[self.axis] = slice(start,stop,step)
            if step < 0:
                raise NotImplementedError
            return x[slices]
        return fn
    def pullback(self, inputs, output, goutput):
        ginput = zeros_like(inputs[0])
        return [Result(IncSli(self.axis), [ginput] + inputs[1:] + [goutput])] + [None]*3
    def shp_apply(self, inputs):
        arr, start, stop, step = inputs
        s = shape(arr) #pylint: disable=W0621
        newshape = copy.copy(s)
        newshape[self.axis] = ceil_divide(stop - start, step)
        return newshape
    def typ_apply(self, inputs):
        assert inputs[1].dtype == inputs[2].dtype == inputs[3].dtype == 'i8'
        return inputs[0].get_type()
    def c_code(self, inputs):
        x = inputs[0]
        openloops = " ".join(["for (int i%(ax)s=0; i%(ax)s < out->shape[%(ax)s]; i%(ax)s += %(step)s) {"%dict(ax=ax, step="step" if ax==self.axis else "1") for ax in xrange(x.ndim)])
        closeloops = "}"*x.ndim
        inidxexpr =  " + ".join([("(start+i%i*step)"%ax if ax==self.axis else "i%i"%ax) + "*instrides[%i]"%ax for ax in xrange(x.ndim)])
        outidxexpr = " + ".join(["i%i"%ax + "*outstrides[%i]"%ax for ax in xrange(x.ndim)])
        return r"""
void CGT_FUNCNAME(void* cldata, cgt_array** io) {
    cgt_array *in=io[0], *out=io[4];
    long start = ((long*)io[1]->data)[0];
    //long stop = ((long*)io[2]->data)[0];
    long step = ((long*)io[3]->data)[0];
    size_t instrides[in->ndim];
    cgt_get_strides(in, instrides);
    size_t outstrides[out->ndim];
    cgt_get_strides(out, outstrides);
    %(openloops)s
        ((%(cdtype)s*)out->data)[%(outidxexpr)s] = ((%(cdtype)s*)in->data)[%(inidxexpr)s];
    %(closeloops)s
}
"""%dict(openloops=openloops, outidxexpr=outidxexpr, inidxexpr=inidxexpr, closeloops=closeloops,
    cdtype=np2c[inputs[0].dtype])

class IncSli(Op):
    def __init__(self, axis):
        self.axis = axis
    def get_diff(self, _):
        return [True,False,True,True]
    def get_numeric_py(self):
        def fn(x, start, stop, step, y):
            newx = x.copy()
            slices = [slice(None,None,None) for _ in xrange(x.ndim)]
            slices[self.axis] = slice(start,stop,step)
            newx[slices] += y
            return newx
        return fn
    def pullback(self, inputs, output, goutput):
        _x, start, stop, step, _y = inputs
        gx = goutput
        gy = Result(GetSli(self.axis), [goutput, start, stop, step])
        return [gx, None, None, None, gy]
    def shp_apply(self, inputs):
        return shape(inputs[0])
    def typ_apply(self, inputs):
        return inputs[0].get_type()
    def c_code(self, inputs):
        x = inputs[0]
        openloops = " ".join(
            ["for (int i%(ax)s=0; i%(ax)s < out->shape[%(ax)s]; i%(ax)s += %(step)s) {"%dict(ax=ax, step="step" 
            if ax==self.axis else "1") for ax in xrange(x.ndim)])
        closeloops = "}"*x.ndim
        inidxexpr =  " + ".join([("(start+i%i*step)"%ax if ax==self.axis else "i%i"%ax)
         + "*instrides[%i]"%ax for ax in xrange(x.ndim)])
        outidxexpr = " + ".join(["i%i"%ax + "*outstrides[%i]"%ax for ax in xrange(x.ndim)])
        return r"""
void CGT_FUNCNAME(void* cldata, cgt_array** io) {
    cgt_array *in=io[0], *out=io[4];
    long start = ((long*)io[1]->data)[0];
    //long stop = ((long*)io[2]->data)[0];
    long step = ((long*)io[3]->data)[0];
    size_t instrides[in->ndim];
    cgt_get_strides(in, instrides);
    size_t outstrides[out->ndim];
    cgt_get_strides(out, outstrides);
    %(openloops)s
        ((%(cdtype)s*)out->data)[%(outidxexpr)s] = ((%(cdtype)s*)in->data)[%(inidxexpr)s];
    %(closeloops)s
}
"""%dict(openloops=openloops, outidxexpr=outidxexpr, inidxexpr=inidxexpr, closeloops=closeloops,
    cdtype=np2c[inputs[0].dtype])

class GetFlatIndices(Op):
    def get_diff(self, _):
        return [True,False]
    def get_numeric_py(self):
        def fn(x, inds):
            return x.flat[inds]
        return fn
    def pullback(self, inputs, output, goutput):
        x,inds = inputs
        ginput = zeros_like(x)
        return [Result(IncFlatIndices(), [ginput, inds, goutput]), None]
    def shp_apply(self, inputs):
        return shape(inputs[1])
    def typ_apply(self, inputs):
        assert inputs[1].ndim == 1 and _dtype_kind(inputs[1].dtype) == 'i'
        return Tensor(inputs[0].dtype,1)


class IncFlatIndices(Op):
    def get_diff(self, _):
        return [True,False,True]
    def get_numeric_py(self):
        def fn(x, inds, y):
            newx = x.copy()
            newx.flat[inds] += y
            return newx
        return fn
    def pullback(self, inputs, output, goutput):
        _x, inds, _y = inputs 
        gx = goutput
        gy = Result(GetFlatIndices(), [goutput, inds])
        return [gx, None, gy]
    def shp_apply(self, inputs):
        return shape(inputs[0])
    def typ_apply(self, inputs):
        return inputs[0].get_type()


# Linalg
# ----------------------------------------------------------------


class Mul21(Op):
    c_extra_includes = ["cblas.h"]

    def __init__(self, tA):
        self.tA = tA
    def get_numeric_py(self):
        def fn(x,y):
            if self.tA: x = x.T
            return x.dot(y)
        return fn
    def get_replacement(self, inputs, analysis):
        if inputs[1] in analysis["node2sv"]:
            return sum(inputs[0],0 if self.tA else 1) * analysis["node2sv"][inputs[1]]
    def pullback(self, inputs, _output, goutput):
        return [outer(goutput,inputs[1]), Result(Mul21(not self.tA), [inputs[0],goutput])]
    def shp_apply(self, inputs):
        return [size(inputs[0], 1 if self.tA else 0)]
    def typ_apply(self, inputs):
        return Tensor(inputs[0].get_dtype(), 1)

class Mul22(Op):
    c_extra_includes = ["cblas.h"]
    c_extra_link_flags = "-lblas"
    def __init__(self, tA, tB):
        self.tA = tA
        self.tB = tB
    def get_numeric_py(self):
        def fn(x,y):
            if self.tA: x = x.T
            if self.tB: y = y.T
            return x.dot(y)
        return fn
    def pullback(self, inputs, output, goutput):
        return [Result(Mul22(False, not self.tB), [goutput, inputs[1]]),
                Result(Mul22(not self.tA, False), [inputs[0], goutput])]
    def shp_apply(self, inputs):
        return [size(inputs[0], 1 if self.tA else 0),size(inputs[1],0 if self.tB else 1)]
    def typ_apply(self, inputs):
        assert inputs[0].get_dtype()==floatX and inputs[1].get_dtype()==floatX
        return inputs[0].get_type()
    def get_closure(self, inputs):
        return [(ctypes.c_bool, self.tA), (ctypes.c_bool, self.tB)]
    def c_name(self, npdtype):
        return "mul22_%s"%(npdtype)
    # best gemm docs: https://software.intel.com/en-us/node/520775
    def c_code(self, inputs):
        npdtype = inputs[0].dtype
        try:
            letter = {"f4":"s","f8":"d","c8":"c","c16":"z"}[npdtype]
        except KeyError:
            raise MethodNotDefined("Dtype %s not supported by this BLAS. Falling back to numpy"%npdtype)
        return """
typedef struct gemm_closure {
    bool tA, tB;
} gemm_closure;        
void CGT_FUNCNAME(void* cldata, cgt_array** ABC) {
    gemm_closure* cl = (gemm_closure*)cldata;
    cgt_array *A=ABC[0], *B=ABC[1], *C=ABC[2];
    int lda = A->shape[1], ldb = B->shape[1], ldc = C->shape[1];
    int M = C->shape[0];
    int N = C->shape[1];  
    int K = A->shape[cl->tA ? 0 : 1];
    const %(cdtype)s alpha=1, beta=0;
  cblas_%(letter)sgemm(CblasRowMajor, cl->tA + 111, cl->tB + 111, M, N, K, alpha, (%(cdtype)s*)A->data, lda, (%(cdtype)s*)B->data,
      ldb, beta, (%(cdtype)s*)C->data, ldc);
}
"""%dict(letter=letter, cdtype = np2c[npdtype])

class BatchedMul22(Op):
    def __init__(self, tA, tB):
        self.tA = tA
        self.tB = tB
    def get_numeric_py(self):
        def fn(x,y):
            z = np.empty((x.shape[0],x.shape[1],y.shape[2]),(x.flat[0]*y.flat[0]).dtype)
            for (xmat, ymat, zmat) in zip(x,y, z):
                xmat.dot(ymat, zmat)
            return z
        return fn
    def pullback(self, inputs, output, goutput):
        return [Result(BatchedMul22(False, not self.tB), [goutput, inputs[1]]),
                Result(BatchedMul22(not self.tA, False), [inputs[0], goutput])]
    def shp_apply(self, inputs):
        return [size(inputs[0], 2 if self.tA else 1),size(inputs[1],1 if self.tB else 2)]
    def typ_apply(self, inputs):
        # assert inputs[0].get_dtype()==floatX and inputs[1].get_dtype()==floatX
        return inputs[0].get_type()

class Outer(Op):
    def get_numeric_py(self):
        return np.outer
    def pullback(self, inputs, _output, goutput):
        return [goutput.dot(inputs[0]), inputs[1].dot(goutput)]
    def shp_apply(self, inputs):
        return [size(inputs[0],0), size(inputs[1],0)]
    def typ_apply(self, _inputs):
        return Tensor(floatX, 2)



# BLAS 1
# ----------------------------------------------------------------

class Dot(Op):
    def get_numeric_py(self):
        return np.dot
    def pullback(self, inputs, _output, goutput):
        x, y = inputs
        return [y*goutput, x*goutput]
    def shp_apply(self, _inputs):
        return []
    def typ_apply(self, _inputs):
        return Tensor(floatX, 0)

# Composition
# ----------------------------------------------------------------

class Composition(Op):
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
    def get_numeric_py(self):
        f = make_function(self._inputs, self._outputs)
        return lambda num_inputs: tuple(f(num_inputs))
    def pullback(self, inputs, output, goutput):
        # repl = {}
        # repl.update(_safezip(self._inputs, inputs))
        # repl.update(_safezip(self._outputs, output))
        # repl.update(_safezip(self._goutput, goutput))
        # return clone(self._gin, replace=repl)
        gwrt = pullback([output], [goutput], inputs)
    def shp_apply(self, inputs):
        out = clone(self._shp, replace=dict(_safezip(self._inputs, inputs)))
        return out
    def typ_apply(self, inputs):
        assert [x.get_type() for x in inputs] == [x.get_type() for x in self._inputs]
        return Tuple(*self._out_typs)
    @property
    def n_out(self):
        return len(self._outputs)
    def shapes(self):
        return self._shp
    def expand(self, inputs):
        return clone(self._outputs, replace=dict(_safezip(self._inputs, inputs)))
    def get_nodes(self):
        return self._nodes

class TupleIndex(Op):
    def __init__(self, idx):
        self.idx = idx
    def get_numeric_py(self):
        def fn(x):
            return x[self.idx]
        return fn
    def shp_apply(self, inputs):
        raise NotImplementedError
    def typ_apply(self, inputs):
        intype = inputs[0].get_type()
        assert isinstance(intype, Tuple)
        return inputs[0].get_type()[self.idx]
        
# TODO what is this shapes() method and which ops need to have it?

def unpack(tup):
    return [Result(TupleIndex(i),[tup]) for i in xrange(len(tup.get_type()))]

class EasyPyOp(Op):

    def numeric_forward(self, *args):
        raise NotImplementedError
    def numeric_backward(self, *args):
        raise NotImplementedError
    def shp_apply(self, inputs):
        raise NotImplementedError
    def input_types(self):
        raise NotImplementedError
    def output_type(self):
        raise NotImplementedError

    def typ_apply(self, inputs):
        assert (x.get_type() == reqtype for (x,reqtype) in _safezip(inputs, self.input_types()))
        return self.output_type()
    def get_diff(self, inputs):
        return [True]*len(inputs)
    def get_numeric_py(self):
        return self.numeric_forward
    def pullback(self, inputs, output, goutput):
        class PBOp(Op):
            def get_numeric_py(self1): #pylint: disable=E0213
                def fn(*vals):
                    return self.numeric_backward(*vals)
                return fn
            def typ_apply(self1, inputs): #pylint: disable=E0213
                return Tuple(x.get_type() for x in inputs[:-2])
            def shp_apply(self1, inputs): #pylint: disable=E0213
                return [shape(x) for x in inputs[:-2]]
            def shapes(self):
                return map(shape, inputs)
        return unpack(Result(PBOp(), inputs + [output, goutput]))


class EasyCodeOp(Op):
    def __init__(self, lang):
        assert lang in ("c","cuda")
        self.lang = lang
    def forward_code(self):
        raise NotImplementedError
    def pullback_code(self):
        raise NotImplementedError
    def shp_apply(self):
        raise NotImplementedError
    def input_types(self):
        raise NotImplementedError
    def output_type(self):
        raise NotImplementedError

    def c_includes(self):
        return []
    def cuda_includes(self):
        return []

    def typ_apply(self, inputs):
        assert (x.get_type() == reqtype for (x,reqtype) in _safezip(inputs, self.input_types()))
        return self.output_type()
    def get_diff(self, inputs):
        return [True]*len(inputs)
    def get_numeric_py(self):
        raise MethodNotDefined

    def c_code(self, _inputs):
        return self.code("c")
    def cuda_code(self, _inputs):
        return self.code("cuda", funcname)
    def code(self, lang, funcname):
        if lang != self.lang:
            raise MethodNotDefined
        forward_name = "FUNCTION"
        orig_code = self.forward_code()
        if forward_name not in orig_code:
            raise ValueError("forward code should have function named %s"%forward_name)            
        return self.forward_code().replace(forward_name,funcname)
    def pullback(self, inputs, output, goutput):
        class PBOp(Op):
            def get_numeric_py(self1): #pylint: disable=E0213
                raise MethodNotDefined
            def typ_apply(self1, inputs): #pylint: disable=E0213
                return Tuple(x.get_type() for x in inputs[:-2])
            def shp_apply(self1, inputs): #pylint: disable=E0213
                return [shape(x) for x in inputs[:-2]]
            def shapes(self):
                return map(shape, inputs)
            def c_code(self1, _inputs): #pylint: disable=E0213
                return self.code("c"), 
            def cuda_code(self1, _inputs): #pylint: disable=E0213
                return self.code("cuda"), 
            def code(self1, lang, funcname): #pylint: disable=E0213
                if lang != self.lang:
                    raise MethodNotDefined
                pullback_name = "FUNCTION"
                orig_code = self.pullback_code()
                if pullback_name not in orig_code:
                    raise ValueError("pullback code should have function named %s"%pullback_name)        
                return orig_code.replace(pullback_name,funcname)

        return unpack(Result(PBOp(), inputs + [output, goutput]))




# Higher-order Operations
# ----------------------------------------------------------------

# class IdxMap(Op):
#     def __init__(self, innerop):
#         self.innerop = innerop
#     def get_diff(self, inputs):
#         return [False] + self.innerop.get_diff()
#     def get_numeric_py(self):
#         def fn(args):
#             n = args[0]
#             otherargs = args[1:]
#             return np.array([self.innerfn(*[otherarg[i] for otherarg in otherargs]) for i in xrange(n)])
#     # def pullback(self, inputs, _output, goutput):
#     #     return [goutput.dot(inputs[0]), inputs[1].dot(goutput)]
#     # def shp_apply(self, inputs):
#     #     return [size(inputs[0],0), size(inputs[1],0)]
#     # def typ_apply(self, _inputs):
#     #     return Tensor(floatX, 2)
#     # XXX unfinished




np2c = {"i1":"int8_t","i2":"int16_t","i4":"int32_t","i8":"int64_t",
        "f4":"float","f8":"double","f16":"long double",
        "c4" : "float complex", "c8" : "double complex", "c16" : "long double complex"}

COMPILE_CONFIG = None
def get_compile_info():
    global COMPILE_CONFIG
    
    if COMPILE_CONFIG is None:

        config = load_config()

        import cycgt #pylint: disable=F0401
        CGT_BUILD_ROOT = osp.dirname(osp.dirname(osp.realpath(cycgt.__file__)))

        cmake_info = {}
        with open(osp.join(CGT_BUILD_ROOT,"build_info.txt")) as fh:
            lines = fh.readlines()
        for line in lines:
            if ":=" not in line: print "skipping",line
            lhs,rhs = line.split(":=")
            lhs = lhs.strip()
            rhs = rhs.strip()
            cmake_info[lhs] = rhs

        CUDA_ROOT = cmake_info["CUDA_ROOT"]
        DEFINITIONS = " ".join("-D"+s for s in filter(None,cmake_info["CGT_DEFS"].split(",")))
        COMPILE_CONFIG = dict(        
            OPENBLAS_INCLUDE_DIR = osp.join(CGT_BUILD_ROOT,"OpenBLAS"),
            CGT_INCLUDE_DIR = cmake_info["CGT_INCLUDE_DIR"],
            CGT_LIBRARY_DIR = osp.join(CGT_BUILD_ROOT,"lib"),
            CUDA_LIBRARY_DIR = osp.join(CUDA_ROOT,"lib"),
            CUDA_INCLUDE_DIR = osp.join(CUDA_ROOT,"include"), 
            CUDA_LIBRARIES = cmake_info["CUDA_LIBRARIES"], 
            DEFINITIONS = DEFINITIONS,  
            CUDA_ROOT = CUDA_ROOT,
            CACHE_ROOT = osp.expanduser(config["cache_dir"]),
            CGT_ENABLE_CUDA = cmake_info["CGT_ENABLE_CUDA"] in ["1","ON"],
            # CGT_LIBRARY = cmake_info["CGT_LIBRARY"],
        )
    return COMPILE_CONFIG


def cap(cmd):
    print "\x1b[32m%s\x1b[0m"%cmd
    subprocess.check_call(cmd,shell=True)

def compile_file(fname, libpath, extra_link_flags = ""):
    info = get_compile_info()
    includes = "-I%(CGT_INCLUDE_DIR)s -I%(CUDA_INCLUDE_DIR)s -I%(OPENBLAS_INCLUDE_DIR)s"%info    
    d = dict(cacheroot = info["CACHE_ROOT"], srcpath = fname, includes = includes, defines = info["DEFINITIONS"], libname = osp.basename(libpath), libpath = libpath, cgtlibdir = info["CGT_LIBRARY_DIR"], extralink=extra_link_flags)            
    if fname.endswith(".cu"):
        if not info["CGT_ENABLE_CUDA"]:
            raise RuntimeError("Trying to compile a CUDA function but CUDA is disabled in your build. Rebuild with CGT_ENABLE_CUDA=ON")
        d.update(cudalibs = info["CUDA_LIBRARIES"], cudaroot = info["CUDA_ROOT"], cudalibdir = info["CUDA_LIBRARY_DIR"])

    if sys.platform == "darwin":
        if fname.endswith(".c"):
            cap(r'''
cd %(cacheroot)s && \
cc -fPIC -O3 -DNDEBUG %(srcpath)s -c -o %(srcpath)s.o %(includes)s %(defines)s && \
cc -fPIC -O3 -DNDEBUG %(srcpath)s.o -dynamiclib -Wl,-headerpad_max_install_names -install_name %(libname)s -o %(libpath)s -L%(cgtlibdir)s -lcgt %(extralink)s
            '''%d)
        elif fname.endswith(".cu"):
            cap(r'''
cd %(cacheroot)s && \
nvcc %(srcpath)s -c -o %(srcpath)s.o -ccbin cc -m64 -Xcompiler  -fPIC -Xcompiler -O3 -Xcompiler -arch -Xcompiler x86_64 %(includes)s %(defines)s && \
c++ -fPIC -O3 -DNDEBUG -fPIC -dynamiclib -Wl,-headerpad_max_install_names %(cudalibs)s -Wl,-rpath,%(cudalibdir)s -install_name %(libname)s -o %(libpath)s %(srcpath)s.o
            '''%d)
                # gpulinkflags = "-dynamiclib -Wl,-headerpad_max_install_names %(CUDA_LIBRARIES)s -Wl,-rpath,%(CUDA_LIBRARY_DIR)s"%d

    else:
        if fname.endswith(".c"):
            cap('''
cc -fPIC -O3 -DNDEBUG %(srcpath)s -std=c99 -c -o %(srcpath)s.o %(includes)s %(defines)s && \
cc -fPIC -O3 -DNDEBUG -shared -rdynamic -Wl,-soname,%(libname)s -o %(libpath)s %(srcpath)s.o -L%(cgtlibdir)s -lcgt
            '''%d)
        elif fname.endswith(".cu"):
            cap(r'''
cd %(cacheroot)s && 
nvcc %(srcpath)s -c -o %(srcpath)s.o -ccbin cc -m64 -Xcompiler -fPIC -Xcompiler -O3 -Xcompiler -DNDEBUG %(includes)s %(defines)s && \
c++  -fPIC -O3 -DNDEBUG -shared -rdynamic -Wl,-soname,%(libname)s -o %(libpath)s %(srcpath)s.o %(cudalibs)s -Wl,-rpath,%(cudaroot)s
            '''%d
            )



def get_impl(node, devtype):
    code_raw = (node.op.c_code if devtype=="cpu" else node.op.cuda_code)(node.parents)
    s = StringIO.StringIO()
    if devtype == "cpu":
        includes = ["cgt_utils.h","cgt_mem.h","stdint.h","stddef.h"] + node.op.c_extra_includes
    else:
        includes = ["cgt_utils.h","cgt_mem.h","cgt_cuda.h"] + node.op.cuda_extra_includes
    for filename in includes:
        s.write('#include "%s"\n'%filename)
    h = hashlib.md5(code_raw).hexdigest()[:10]
    funcname = devtype + node.op.__class__.__name__ + h
    ci = get_compile_info()
    CACHE_ROOT = ci["CACHE_ROOT"]
    libpath = osp.join(CACHE_ROOT, funcname + ".so")
    closure = node.op.get_closure(node.parents)

    if not osp.exists(libpath):
        if not osp.exists(CACHE_ROOT): os.makedirs(CACHE_ROOT)
        print "compiling %(libpath)s for node %(node)s"%locals()
        ext = "c" if devtype == "cpu" else "cu"
        srcpath = osp.join(CACHE_ROOT, funcname + "." + ext)
        # write c code to tmp file
        s = StringIO.StringIO()
        for filename in includes:
            s.write('#include "%s"\n'%filename)
        code = code_raw.replace("CGT_FUNCNAME",funcname)
        s.write(code)
        with open(srcpath,"w") as fh:
            fh.write(s.getvalue())

        compile_file(srcpath, osp.splitext(srcpath)[0]+".so", extra_link_flags = node.op.c_extra_link_flags)

    return (libpath,funcname,closure)


# ================================================================
# Funcs wrapping ops (numpy-like)
# ================================================================

def zeros(shape, dtype=None): #pylint: disable=W0621
    if dtype is None: dtype=floatX
    return Result(Fill(np.array(0,dtype)), shape)

def arange(start, stop=None, step=1, dtype=None):
    if stop is None:
        start, stop = 0, start
    if dtype is None: dtype = 'i8'
    return Result(Arange(dtype), [start,stop,step])

def elwise_binary(opname, x,y):
    x,y = map(_as_node, (x,y))
    scalar_mask = (x.ndim==0, y.ndim==0)
    op = ElwiseBinary(opname, scalar_mask)
    if scalar_mask == (False, False): assert x.ndim==y.ndim
    return Result(op, [x,y])

def zeros_like(x):
    return zeros(shape(x), x.get_dtype())

def shape(x):
    return [size(x, i) for i in xrange(x.ndim)]

def size(x, axis):
    return Result(Size(axis), [x])

def _red_axes(axis, ndim):
    if axis is None: return range(ndim)
    elif isinstance(axis, int): return [axis]
    elif isinstance(axis, (list, tuple)): return list(axis)
    else: raise ValueError("invalid argument 'axis'=%s"%axis)


def sum(x, axis=None, keepdims=False): #pylint: disable=W0622    
    axes = _red_axes(axis, x.ndim)
    if len(axes)==0: return x
    out = Result(Sum(axes), [x])
    if not keepdims: out = _dropdims(out, axes)
    return out

# copy pasted from sum
def max(x, axis=None, keepdims=False): #pylint: disable=W0622
    axes = _red_axes(axis, x.ndim)
    out = Result(Max(axes), [x])
    if not keepdims: out = _dropdims(out, axes)
    return out

def argmax(x, axis=None, keepdims=False):
    if axis is None:
        out = flatten(x).argmax(axis=0)
    else:
        assert isinstance(axis, int)
        out = Result(Argmax(axis), [x])
        if not keepdims: out = _dropdims(out, [axis])
    return out

def mean(x, axis=None, keepdims=False):
    axes = _red_axes(axis, x.ndim)
    return sum(x, axis=axes, keepdims=keepdims) / mul_multi([size(x, ax) for ax in axes])
    
def _dropdims(x, axes):
    # todo: add asserts
    return reshape(x, [size(x, i) for i in xrange(x.get_ndim()) if i not in axes])

def stack(scalars):
    assert len(scalars)>0
    # todo: add assert
    return Result(Stack(), scalars)

def constant(val):
    return Result(Constant(val), [])

def shared(val,name="",device=None,fixed_shape_mask=None):
    return Data(val,name=name,device=device,fixed_shape_mask=fixed_shape_mask)

def reshape(x, shape): #pylint: disable=W0621
    return Result(Reshape(), [x] + list(shape))

def dot(x, y):
    xdim = x.get_ndim()
    ydim = y.get_ndim()
    if xdim == 1:
        if ydim == 1:
            return Result(Dot(), [x, y])
        elif ydim == 2:
            return Result(Mul21(True), [y, x])
        else:
            raise NotImplementedError
    elif xdim == 2:
        if ydim == 1:
            return Result(Mul21(False), [x, y])
        elif ydim == 2:
            return Result(Mul22(False, False), [x, y])
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

def outer(x, y):
    assert x.get_ndim() == y.get_ndim() == 1
    return Result(Outer(), [x, y])

def broadcast(opname, x, y, bcpat):
    xpat, ypat = bcpat.split(",")
    xbcaxes,ybcaxes = [[i for (i,letter) in enumerate(pat) if letter=='1'] for pat in (xpat, ypat)]
    assert x.get_ndim() == y.get_ndim()
    if xbcaxes: x = Result(Repeat(xbcaxes), [x] + [size(y,ax) for ax in xbcaxes])
    if ybcaxes: y = Result(Repeat(ybcaxes), [y] + [size(x,ax) for ax in ybcaxes])
    return elwise_binary(opname, x, y)

def repeat(x, repeats, axis):
    return Result(Repeat([axis]), [x, constant(repeats)])

def concatenate(xs, axis=0):
    return Result(Concatenate(axis), xs)

def norm(x, axes=None, p=2):
    return pow(pow(x, p).sum(axes=axes),(1.0/p))


def floor_divide(x,y):
    return ifloor(x/y)

def ceil_divide(x,y):
    return iceil(x/y)

def add_multi(xs):
    return reduce(operator.add, xs)

def mul_multi(xs):
    return reduce(operator.mul, xs) if len(xs)>0 else constant(np.array(1,dtype='i8'))


def getitem(arr, slis):
    if not _is_sequence(slis):
        slis = [slis]
    if all(isinstance(sli, (int,slice)) or isinstance(sli,slice) and sli==COLON for sli in slis):
        return getitem_nonfancy(arr,slis)
    elif all(isinstance(sli, (np.ndarray, Node)) for sli in slis):
        return getitem_fancy(arr,slis)
    else:
        raise ValueError("Tried to index with slices %s. Either all should be in {slice,int,colon} or all must be a ndarray of intsMust either all be of type {slice,int} or of type {np.ndarray, Node}"%str(slis))

def sub2ind(subs, shp):
    ndim = len(shp)
    assert ndim >= 1
    strides = [None]*(ndim-1) + [1]
    for i in xrange(ndim-2, -1, -1):
        strides[i] = shp[i+1] * strides[i+1]
    return add_multi([stride*sub for (stride,sub) in _safezip(strides, subs)])

def getitem_fancy(arr, indarrs):
    assert all(indarr.ndim==1 for indarr in indarrs)
    indarrs = map(_as_node, indarrs)
    flatinds = sub2ind(indarrs, shape(arr))
    return Result(GetFlatIndices(), [arr, flatinds])

COLON = slice(None,None,None)


def getitem_nonfancy(arr, slis):
    out = arr
    ax = 0
    shapedesc = []
    if not _is_sequence(slis):
        slis = [slis]
    for sli in slis:
        if sli == COLON:
            shapedesc.append(ax)
        else:
            if sli is None:
                shapedesc.append('+')
                ax -= 1
            elif isinstance(sli, bool):
                raise ValueError("tried to index with a bool")
            else:
                if isinstance(sli,slice):
                    shapedesc.append('k')
                elif isinstance(sli, int):
                    sli = slice(sli,sli+1,1)
                    shapedesc.append('-')
                else:
                    raise NotImplementedError
                start = sli.start or 0
                stop = sli.stop or size(arr, ax)
                step = sli.step or 1
                if isinstance(stop, int) and stop < 0: stop = size(arr, ax) - stop
                out = Result(GetSli(ax), [out, start, stop, step])
        ax += 1

    if all(x=='k' for x in shapedesc):
        return out
    else:
        axidx = 0
        newshape = []
        for d in shapedesc:
            if d == '+':
                newshape.append(d)                    
            elif d == '-':
                axidx += 1
            else: # 'x'
                newshape.append(size(out, axidx))
                axidx += 1
        for axidx in xrange(axidx, out.ndim):
            newshape.append(size(out, axidx))
        out = reshape(out, newshape)
    return out

def einsum(desc, x, y):
    pat = r'(\w+),(\w+)->(\w+)'
    match = re.match(pat, desc)
    if match is None:
        raise ValueError("einsum error: desc should match regexp %s"%pat)
    xdesc,ydesc,zdesc = match.groups()
    if not _is_unique(xdesc) and _is_unique(ydesc) and _is_unique(zdesc):
        raise ValueError("Invalid tensor description %s passed into einsum. Tensor indices should be unique"%desc)
    if not _is_subset_of(xdesc+ydesc,zdesc):
        raise ValueError("Invalid tensor description %s passed into einsum. Unrecognized index in output."%desc)
    loop = []
    justx = []
    contr = []
    justy = []
    for c in xdesc:
        if c in ydesc:
            if c in zdesc:
                loop.append(c)
            else:
                contr.append(c)
        else:
            justx.append(c)
    for c in ydesc:
        if not c in xdesc:
            justy.append(c)
    ixloop, ijustx, ixcontr = [[xdesc.index(c) for c in chars] for chars in [loop,justx,contr]]
    iyloop, ijusty, iycontr = [[ydesc.index(c) for c in chars] for chars in [loop,justy,contr]]
    xshp = shape(x)
    yshp = shape(y)
    xt = transpose(x, ixloop + ijustx + ixcontr).reshape([mul_multi([xshp[i] for i in icol]) for icol in [ixloop, ijustx, ixcontr]])
    yt = transpose(y, iyloop + iycontr + ijusty).reshape([mul_multi([yshp[i] for i in icol]) for icol in [iyloop, iycontr, ijusty]])
    zt = batched_matmul(xt, yt)

    return transpose(
           zt.reshape( [size(x, xdesc.index(c)) for c in loop]
                     + [size(x, xdesc.index(c)) for c in justx]
                     + [size(y, ydesc.index(c)) for c in justy]),
           _invert_perm([zdesc.index(c) for c in loop + justx + justy]))

def flatten(x):
    return reshape(x, [mul_multi(shape(x))])

def flatcat(xs):
    return concatenate([flatten(x) for x in xs])

def transport(x, src, targ):
    return Result(Transport(src,targ),[x])

def cast(x, dtype):
    dtype = Dtype.canon(dtype)
    if x.dtype == dtype: 
        return x
    else:
        diff = _dtype_kind(dtype) in "cf"
        opname = "castto%s"%dtype
        ui = UnaryInfo(opname, lambda x: x.astype(dtype),diff, dtype, "cast(gy,%s)"%x.dtype if diff else "_no_grad()", "((%s)x)"%np2c[dtype])
        return Result(ElwiseUnary(opname, ui), [x])

def real(x):
    assert _dtype_kind(x.dtype) == 'c'
    return cast(x, floatX)

def rand(shp):
    shp = map(_as_node, _to_list(shp))
    return Result(ScalarRng("uniform"), shp)

def randn(shp):
    shp = map(_as_node, _to_list(shp))
    return Result(ScalarRng("gaussian"), shp)

def rand_bernoulli(x):
    return rand(x)


# ================================================================
# Graph Optimization
# ================================================================

def _init_analysis():
    return {"node2hash":{},"hash2node":{},"node2shape":{},"node2sv":{},"repl":{}}


def analyze(outputs):
    analysis = _init_analysis()
    for node in topsorted(outputs):
        do_analysis(node, analysis)
    return analysis


def simplify_and_analyze(outputs):
    assert _is_sequence(outputs)
    analysis = _init_analysis()
    repl = {}
    for output in outputs: _s_and_a(output, analysis, repl)
    return [repl[node] for node in outputs], analysis

def _stackupdate(stack, analysis, repl): #pylint: disable=W0621
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
        newnewnode = _try_repl(newnode, analysis, repl)
        if newnewnode is None:
            return (orig,newnode)
        else:
            if newnewnode in repl:
                return (orig, newnewnode)
            else:
                stack.append((orig, newnewnode))

def _s_and_a(node, analysis, repl):
    stack = [(node,node)] #pylint: disable=W0621
    while stack:
        maybe_pair = _stackupdate(stack, analysis, repl)
        if maybe_pair:
            (orig,node) = maybe_pair #pylint: disable=W0633
            # if not node.is_input():
            #     for shpcmp in node.op.shp_apply(node.parents): 
            #         _s_and_a(shpcmp, analysis, repl, True)
            do_analysis(node, analysis)
            repl[orig] = node
            repl[node] = node

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
        node2shape[node] = [size(node, i) for i in xrange(node.get_ndim())]
    elif isinstance(node.op, TupleIndex):
        node2shape[node] = node2shape[node.parents[0]][node.op.idx]
    else:
        newparents = node.parents
        node2shape[node] = node.op.shp_apply(newparents)
        # assert all([s.get_dtype() == "i8" for s in node2shape[node]])
    assert len(node2shape[node]) == node.ndim or isinstance(node.get_type(),Tuple)
    # -- SCALAR VALUE --
    if isinstance(node, Result):
        op = node.op
        if isinstance(op, Fill):
            node2sv[node] = op.value
        elif isinstance(op, Constant) and _is_singleton(op.value):
            node2sv[node] = op.value.flat[0]
        elif isinstance(op, Repeat) and newparents[0] in node2sv:
            node2sv[node] = node2sv[newparents[0]]
        elif isinstance(op, (ElwiseUnary, ElwiseBinary)) and all(p in node2sv for p in newparents):
            node2sv[node] = node.op.numeric_apply([node2sv[p] for p in newparents])


def _try_repl(node, analysis, repl):
    if node.is_input(): return
    if isinstance(node.op, Constant): return
    # -- CSE --
    node2hash = analysis["node2hash"]
    h = node.get_hash(node2hash)
    if h in analysis["hash2node"]:
        assert analysis["hash2node"][h] in repl
        return analysis["hash2node"][h]
    parents = node.parents
    # -- CONSTANT PROP --
    if all(isinstance(par.op, Constant) for par in parents) and not node.op.dynamic_data:
        return constant(node.op.numeric_apply([p.op.value for p in parents]))
    # -- SIZE --
    if isinstance(node.op, Size):
        s = analysis["node2shape"][parents[0]][node.op.axis]
        if not (isinstance(s.op, Size) and s.parents[0] == node.parents[0]): return s
    # -- OP IDENTITY --
    maybe_repl = node.op.get_replacement(parents, analysis)
    if maybe_repl is not None: return maybe_repl

    return None

def simplify(outputs):
    single = isinstance(outputs, Node)
    if single: outputs = [outputs]
    result = simplify_and_analyze(outputs)[0]
    return result[0] if single else result

def determine_device(node, node2dev, devtype=None, machine=None, idx = None):
    op = node.op
    parents = node.parents
    parent_devices = [node2dev[par] for par in parents]
    if isinstance(op,Transport):
        assert parent_devices[0].devtype==op.src
        devtype = op.targ   
    elif any(pardev.devtype == "gpu" for pardev in parent_devices):
        devtype = "gpu"
    else:
        devtype = "cpu"
    if devtype == "gpu":
        try:
            get_impl(node, "gpu")
        except MethodNotDefined:
            print "couldn't get gpu func for ", node
            devtype = "cpu"


    # devtype = "cpu" if devtype is None else ("gpu" if any(pardev.devtype == "gpu" for pardev in parent_devices) else "cpu")
    idx = 0 if idx is None else idx
    machine = "default" if machine is None else machine
    return Device(machine, devtype, idx)


def assign_devices(outputs, devfn=None):
    # First assign each node to a device
    node2dev={}
    for node in topsorted(outputs):        
        maybedev = None if devfn is None else devfn(node)
        if maybedev: 
            node2dev[node] = maybedev
        elif node.is_argument():
            node2dev[node] = Device(devtype="cpu")
        elif node.is_data():
            node2dev[node] = node.get_device()
        else:
            node2dev[node] = determine_device(node, node2dev)

    # Now make a new computation graph with 
    replace = {}
    newnode2dev = {}
    for node in topsorted(outputs):
        parents = node.parents
        dev = node2dev[node]
        if node.is_input():
            replace[node] = node
        else:
            newparents = []
            for par in parents:
                if node2dev[par] == dev:
                    newparents.append(replace[par])
                else:
                    newparents.append(transport(replace[par], node2dev[par], dev))
                    newnode2dev[newparents[-1]] = dev
            replace[node] = Result(node.op, newparents, typ=node.get_type())
        newnode2dev[replace[node]] = dev

    return [replace[node] for node in outputs], newnode2dev

# ================================================================
# Graph Traversal
# ================================================================

def topsorted(outputs):
    marks = {}
    out = []
    stack = [] #pylint: disable=W0621
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


def topsorted1(outputs):
    visited = set()
    for output in outputs:
        for node in _topsorted(output, visited):
            yield node

def _topsorted(node, visited):
    if node in visited:
        return
    else:
        for par in node.parents:
            for pred in _topsorted(par, visited):
                yield pred
        visited.add(node)
        yield node

def topsorted_active(outputs, wrt):
    # node is active iff there's a chain of differentiable edges from wrt
    # XXX change so it has same style as node2dev
    n2a = {n:True for n in wrt} # node to bool indicating if it's active
    for node in topsorted(outputs):
        if node not in n2a:
            if node.is_input():
                n2a[node] = False        
            else:
                parents = node.parents
                n2a[node] = any(n2a[par] and d for (par,d) in _safezip(parents, node.op.get_diff(len(parents))))
        if n2a[node]: yield node

def count_nodes(outputs):
    if isinstance(outputs, Node): outputs = [outputs]
    return len(list(topsorted(outputs)))

def clone(nodes, replace=None):
    assert _is_sequence(nodes)
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

def transpose(arr, axes=None):
    if axes is None: 
        assert arr.ndim == 2
        axes = [1,0]
    else:
        assert _is_sequence(axes) and len(axes) == arr.get_ndim()
        axes = list(axes)
    if axes == range(arr.get_ndim()):
        return arr
    else:
        return Result(Transpose(axes), [arr])

def batched_matmul(x, y):
    return Result(BatchedMul22(False,False), [x,y])

# def idxmap(n, fn):
#     idxvar = cgt.scalar(dtype='i8')
#     innerop = Composition([idxvar], [fn(idxvar)])
#     return IdxMap(innerop)(idxvar)


# ================================================================
# Utils
# ================================================================

class Color: #pylint: disable=W0232
    GRAY=30,
    RED=31,
    GREEN=32,
    YELLOW=33,
    BLUE=34,
    MAGENTA=35,
    CYAN=36,
    WHITE=37,
    CRIMSON=38    

def colorprint(colorcode, text, o=sys.stdout):
    o.write("\x1b[%im"%colorcode)
    o.write(text)
    o.write("\x1b[0m")

def warn(msg):
    colorprint(Color.YELLOW, msg)
    sys.stdout.write("\n")
def error(msg):
    colorprint(Color.RED, msg)
    sys.stdout.write("\n")

def _is_sequence(x):
    return hasattr(x, "__iter__") and not isinstance(x, Node)

class NonDifferentiable(Exception):
    pass

class Disconnected(Exception):
    pass

def _is_singleton(x):
    return np.prod(x.shape)==1

def _get_scalar_value(x):
    return x.flat[0]

def _canon_order(x, y):
    return (x,y) if x.get_name() < y.get_name() else (y,x)

def _safezip(x,y):
    assert len(x) == len(y)
    return zip(x,y)

def _flatten(lis):
    out = []
    for li in lis:
        out.extend(li)
    return out

def _indexin(xs, ys):
    y2i = {y:i for (i,y) in enumerate(ys)}
    return [y2i.get(x,None) for x in xs]

def _allsame(xs):
    out = True
    if len(xs)>0:
        x0 = xs[0]
        for x in xs[1:]:
            out &= x==x0
    return out

def _is_unique(col):
    return len(set(col))==len(col)

def _is_subset_of(maybesub, bigset):
    return len(set(bigset).difference(set(maybesub)))==0

def _invert_perm(x):
    return list(np.argsort(x))

class Todo(Exception):
    pass

def _as_valid_array(x):
    x = np.asarray(x)
    x = x.astype(Dtype.canon(x.dtype))
    return x

def _hash_seq(args):
    hashobj = hashlib.md5()
    for a in args: hashobj.update(a)
    return hashobj.hexdigest()

def _hash_seq1(*args):
    return _hash_seq(args)

def _to_list(x):
    if _is_sequence(x): return list(x)
    else: return [x]

def _is_int(node):
    return _dtype_kind(node.dtype)=='i'

def _list_is_valid_shape(inputs):
    return len(inputs)==3 and all(x.ndim==0 and _is_int(x) for x in inputs)


if sys.argv[0] != "gen_py.py":
    from api_autogen import * #pylint: disable=F0401
