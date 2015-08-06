from cgt.core import *
import impls
from collections import defaultdict
from time import time


def function(inputs, outputs, dbg = None, updates=None):
    assert isinstance(inputs, list), "Inputs must be a list"
    assert all(isinstance(el, Node) for el in inputs), "Invalid input: should all be symbolic variables"
    assert isinstance(outputs, list), "Outputs must be a list. For a single output use function1"
    assert all(isinstance(el, Node) for el in inputs), "Invalid output: should all be symbolic variables"
    if updates is None: updates = []
    else: assert all(isinstance(before, Data) for (before,_) in updates), "rhs of updates must be Data instances"

    if dbg: raise Todo("debug functionality is broken")
    
    outputs = [cgt.make_tuple(*x) if isinstance(x, tuple) else x for x in outputs]

    interp = run_compilation_pipeline(inputs, outputs, updates)
    return interp

def function1(inputs, output, dbg=None, updates=None):
    f_output_list = function(inputs, [output], dbg = dbg, updates=updates)
    return lambda *args : f_output_list(*args)[0]


# ================================================================
# Execution
# ================================================================

def determine_devices(nodes):
    return {}


# def determine_device(node, node2dev, devtype=None, machine=None, idx = None):
#     op = node.op
#     parents = node.parents
#     parent_devices = [node2dev[par] for par in parents]
#     if isinstance(op,Transport):
#         assert parent_devices[0].devtype==op.src
#         devtype = op.targ   
#     elif any(pardev.devtype == "gpu" for pardev in parent_devices):
#         devtype = "gpu"
#     else:
#         devtype = "cpu"
#     if devtype == "gpu":
#         try:
#             get_impl(node, "gpu")
#         except MethodNotDefined:
#             print "couldn't get gpu func for ", node
#             devtype = "cpu"


#     # devtype = "cpu" if devtype is None else ("gpu" if any(pardev.devtype == "gpu" for pardev in parent_devices) else "cpu")
#     idx = 0 if idx is None else idx
#     machine = "default" if machine is None else machine
#     return Device(machine, devtype, idx)


# def assign_devices(outputs, devfn=None):
#     # First assign each node to a device
#     node2dev={}
#     for node in topsorted(outputs):        
#         maybedev = None if devfn is None else devfn(node)
#         if maybedev: 
#             node2dev[node] = maybedev
#         elif node.is_argument():
#             node2dev[node] = Device(devtype="cpu")
#         elif node.is_data():
#             node2dev[node] = node.get_device()
#         else:
#             node2dev[node] = determine_device(node, node2dev)

#     # Now make a new computation graph with 
#     replace = {}
#     newnode2dev = {}
#     for node in topsorted(outputs):
#         parents = node.parents
#         dev = node2dev[node]
#         if node.is_input():
#             replace[node] = node
#         else:
#             newparents = []
#             for par in parents:
#                 if node2dev[par] == dev:
#                     newparents.append(replace[par])
#                 else:
#                     newparents.append(transport(replace[par], node2dev[par], dev))
#                     newnode2dev[newparents[-1]] = dev
#             replace[node] = Result(node.op, newparents, typ=node.get_type())
#         newnode2dev[replace[node]] = dev

#     return [replace[node] for node in outputs], newnode2dev


def is_tensor(x):
    return isinstance(x.typ, TensorType)
def is_tuple(x):
    return isinstance(x.typ, TupleType)

def make_interpreter(inputs, outputs, eg, node2memloc, node2dev):
    assert isinstance(eg, ExecutionGraph)
    input_types = [input.get_type() for input in inputs]
    output_types = [output.get_type() for output in outputs]
    output_locs = [node2memloc[node] for node in outputs]

    backend = load_config()["backend"]
    if backend == "python":
        oplib = impls.OpLibrary(force_python_impl=True)
        return SequentialInterpreter(eg, oplib, output_locs, input_types)
    elif backend == "cython":
        oplib = impls.OpLibrary(force_python_impl=False)
        import cycgt2
        return cycgt2.CppInterpreterWrapper(eg, oplib, input_types, output_locs)
    else:
        raise NotImplementedError("invalid backend %s"%backend)

def topsorted_shapes_first(outputs, node2shape):
    # Almost identical to topsorted(...) function
    # But we also need to visit the shape elements of an in-place node
    # before visiting that node
    marks = {}
    out = []
    stack = [] 
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
            ###### Changed part ######
            if i.ndim > 0 and not i.is_input() and i.op.call_type=="inplace":
                if i in node2shape:
                    shpels = node2shape[i]
                else:
                    utils.warn("odd...")
                    shpels = i.op.shp_apply(i.parents)
                ps = ps + shpels
            elif is_tuple(i):
                for arrshp in node2shape[i]:
                    ps = ps + arrshp
            ##########################
            if jidx == len(ps):
                marks[i] = 2
                out.append(i)
            else:
                stack.append((i,jidx+1))
                j = ps[jidx]
                stack.append((j,0))
    return out

def determine_memowner(nodes_sorted, updates, node2dev):
    # TODO use node2loc

    # First determine how many "child" nodes each node has
    node2child = defaultdict(list)
    for node in nodes_sorted:
        for parent in node.parents:
            node2child[parent].append(node)

    # Now traverse graph again and see where we can use the same memory
    node2memowner = {} # mapping node x -> the node that owns its memory
    
    # For updates, memlocation(RHS) = memlocation(LHS)
    after2before = {after:before for (before,after) in updates}

    enable_inplace_opt = load_config()["enable_inplace_opt"]

    for node in nodes_sorted:

        base = node # by default, 
        if node.is_argument():
            pass
        elif node.op.writes_to_input >= 0:
            base = node2memowner[node.parents[node.op.writes_to_input]]
        elif node in after2before:
            base = after2before[node]
        elif enable_inplace_opt and node.op.call_type == "inplace" \
            and node.op.inplace_alias_ok:
            nodeshape = node.op.shp_apply(node.parents)
            for parent in node.parents:
                if (len(node2child[parent])==1
                        and nodeshape==cgt.shape(parent) # XXX not a very robust way to check
                        and node.dtype == parent.dtype
                        and is_data_mutable(parent)):
                    base = parent
                    break
        # TODO: add optimization for in-place incrementing
        node2memowner[node] = base

    return node2memowner

class MemCounter(object):
    """
    returns `MemLocation`s with indices 0,1,...
    `count` member indicates how many have been returned thus far
    """
    def __init__(self):
        self.count=0
    def new_memloc(self):
        out = MemLocation(self.count)
        self.count += 1
        return out


def create_execution_graph(inputs, outputs, nodes_sorted, node2shape, node2memowner, node2dev):
    instrs = []
    counter = MemCounter()
    node2memloc = {}

    for node in nodes_sorted:
        if node.is_argument():
            write_loc = counter.new_memloc()
            node2memloc[node] = write_loc
            i = inputs.index(node)
            instrs.append(LoadArgument(i, write_loc))
        else:
            read_locs = [node2memloc[p] for p in node.parents]
            if node.op.call_type == "inplace":
                if node2memowner[node] == node:
                    if is_tensor(node): # just make one memory location for output
                        nodeshape = node2shape[node] if node.ndim > 0 else []
                        shape_locs = [node2memloc[shpel] for shpel in nodeshape]
                        # XXX can probably get rid of condition
                        write_loc = counter.new_memloc()                    
                        instrs.append(Alloc(node.dtype, shape_locs, write_loc))
                    else: # if it's a tuple, we need to allocate all of the components, then build tuple
                        nodeshape = node2shape[node]
                        assert isinstance(nodeshape, tuple)
                        arr_locs = []
                        for (arrshp, arrtyp) in utils.safezip(nodeshape, node.get_type()):
                            arr_loc = counter.new_memloc()
                            shape_locs = [node2memloc[shpel] for shpel in arrshp]
                            instrs.append(Alloc(arrtyp.dtype, shape_locs, arr_loc))
                            arr_locs.append(arr_loc)
                        write_loc = counter.new_memloc()   
                        instrs.append(BuildTup(node.get_type(), arr_locs, write_loc))

                else:
                    write_loc = node2memloc[node2memowner[node]]
                instrs.append(ReturnByRef(node, read_locs, write_loc))
            else:
                assert node.op.call_type == "valret"
                write_loc = counter.new_memloc()
                instrs.append(ReturnByVal(node, read_locs, write_loc))
        node2memloc[node] = write_loc
    return ExecutionGraph(instrs, len(inputs), counter.count), node2memloc


def run_compilation_pipeline(inputs, outputs, updates):
    """
    Compiles the expression graph into an execution graph. 
    """
    # Phase 1: simplification and analysis of expression graph
    # ------------------------------------------------------
    # Add add update targets to outputs
    outputs_updatetargs = outputs + [after for (_before, after) in updates]
    # Do simplification + analysis pass on expression graph
    outputs_updatetargs_simple, analysis = \
        simplify_and_analyze(outputs_updatetargs) if load_config()["enable_simplification"] \
        else (outputs_updatetargs, analyze(outputs_updatetargs))
    # Determine location and device of nodes
    node2dev = determine_devices(outputs_updatetargs_simple)

    # Phase 2: build execution graph
    # ------------------------------------------------------
    # Sort nodes so that shape elements appear before a given node
    nodes_sorted = topsorted_shapes_first(outputs_updatetargs_simple, analysis["node2shape"])
    # For each node, figure out if its output should be written to a previous node's memory
    # (memowner : "memory owner")
    updatetargs_simple = outputs_updatetargs_simple[len(outputs):]
    updatesrcs = [before for (before, _) in updates]
    node2memowner = determine_memowner(nodes_sorted, zip(updatesrcs, updatetargs_simple), node2dev)
    # Find the outputs we want to return
    outputs_simple = outputs_updatetargs_simple[:len(outputs)] # get rid
    # Generate execution graph
    eg, node2memloc = create_execution_graph(inputs, outputs_simple, nodes_sorted, analysis["node2shape"], node2memowner, node2dev)

    # Phase 3: create C or Python interpreter for graph
    # ------------------------------------------------------
    interp = make_interpreter(inputs, outputs_simple, eg, node2memloc, node2dev)

    # Done!
    return interp



################################################################
### Simple numeric eval via traversal 
################################################################
 
def numeric_eval(outputs, arg2val):
    """
    Evaluate outputs numerically. arg2val is a dictionary mapping arguments to numerical values
    """
    assert isinstance(outputs, list)
    assert isinstance(arg2val, dict)

    nodes = list(topsorted(outputs))

    node2val = {}
    for node in nodes:
        if node.is_argument():
            node2val[node] = arg2val[node]
        elif node.is_data():
            node2val[node] = node.get_value()
        else:
            parentvals = [node2val[par] for par in node.parents]
            node2val[node] = py_numeric_apply(node, parentvals)
        # assert node.get_ndim() == np.array(node2val[node]).ndim
    numeric_outputs = [node2val[node] for node in outputs]

    return numeric_outputs

def numeric_eval1(output, arg2val):
    return numeric_eval([output], arg2val)[0]

################################################################
### Execution graph 
################################################################
MemInfo = namedtuple("MemInfo",["loc","access"])
MEM_OVERWRITE = "overwrite"
MEM_INCREMENT = "increment"

class ExecutionGraph(object):
    def __init__(self, instrs, n_args, n_locs):
        self.instrs = instrs
        self.n_args = n_args
        self.n_locs = n_locs
    def to_json(self):
        return {
            "instrs" : _list_to_json(self.instrs),
        }

class MemLocation(object):
    def __init__(self, idx):
        assert isinstance(idx, int)
        self.index = idx
    def to_json(self):
        return self.index

class Interpreter(object):
    def __call__(self, args):
        raise NotImplementedError
    def get(self, mem):
        raise NotImplementedError
    def set(self, mem, val):
        raise NotImplementedError
    def getarg(self, i):
        raise NotImplementedError
    def apply_inplace(self, node, reads, write):
        raise NotImplementedError
    def apply_valret(self, node, reads):
        raise NotImplementedError

def as_valid_arg(x):
    if isinstance(x, tuple):
        return tuple(as_valid_arg(el) for el in x)
    else:
        return as_valid_array(x)

class _Profiler(object):
    def __init__(self):
        self.instr2stats = {}
        self.on = False
        self.t_total = 0.0
    def start(self): self.on = True
    def stop(self): self.on = False
    def update(self, instr, elapsed):
        (prevcount, prevtime) = self.instr2stats.get(instr, (0,0.0))
        self.instr2stats[instr] = (prevcount+1, prevtime+elapsed)
        self.t_total += elapsed
    def print_stats(self):
        op2stats = {}
        # Collapse by Op, rather than instruction
        for (instr,(count,time)) in self.instr2stats.iteritems():
            opkey = str(instr) # XXX
            (prevcount, prevtime) = op2stats.get(opkey, (0, 0.0))
            op2stats[opkey] = (prevcount+count, prevtime+time)

        print "Total time elapsed: %.3g seconds"%self.t_total
        _print_heading("By instruction")
        _print_stats(self.instr2stats, self.t_total)
        _print_heading("By Op")
        _print_stats(op2stats, self.t_total)
    def clear_stats(self):
        self.instr2stats = {}
        self.t_total = 0.0

profiler = _Profiler()

def _print_heading(heading):
    heading = "  " + heading + "  "
    width = 60
    assert len(heading) < width-10
    print
    print "*"*width
    padleft = (width-len(heading))//2
    padright = width-len(heading)-padleft
    print "*"*padleft + heading + "*"*padright
    print "*"*width

def _print_stats(key2stats, t_total):
    rows = []
    for (key, (count,time)) in key2stats.iteritems():
        rows.append([str(key), count, time, time/t_total])
    rows = sorted(rows, key=lambda row: row[2], reverse=True)
    cumsum = 0
    for row in rows:
        cumsum += row[3]
        row.append(cumsum)
    from thirdparty.tabulate import tabulate
    print tabulate(rows, headers=["Instruction","Count","Time","Frac","Frac cumsum"])

def _copy(x):
    if isinstance(x, np.ndarray): return x.copy()
    elif isinstance(x, tuple): return tuple(el.copy() for el in x)
    elif np.isscalar(x): return x # xxx is this case ok?
    else: raise NotImplementedError


def typecheck_args(numargs, types):
    assert len(numargs)==len(types), "wrong number of arguments. got %i, expected %i"%(len(numargs),len(types))
    for (numarg,typ) in zip(numargs,types):
        if isinstance(typ, TensorType):
            assert numarg.dtype==typ.dtype and numarg.ndim==typ.ndim
    
class SequentialInterpreter(Interpreter):
    """
    Runs an execution graph
    """
    def __init__(self, eg, oplib, output_locs, input_types, copy_outputs=True):
        self.eg = eg
        self.oplib = oplib
        self.input_types = input_types
        self.output_locs = output_locs
        self.storage = [None for _ in xrange(self.eg.n_locs)]
        self.args = None
        self.copy_outputs = copy_outputs
    def __call__(self, *args):
        self.args = tuple(as_valid_arg(arg) for arg in args)
        typecheck_args(self.args, self.input_types)
        for instr in self.eg.instrs:
            if profiler.on: tstart = time()
            instr.fire(self)
            if profiler.on: profiler.update(instr, time()-tstart)
        outputs = [self.get(loc) for loc in self.output_locs]
        if self.copy_outputs: outputs = map(_copy, outputs)
        return outputs
        # need to copy because otherwise we might mess up the data when we call func again
        # todo: add option that prevents this behavior
    def get(self, mem):
        return self.storage[mem.index]
    def set(self, mem, val):
        self.storage[mem.index] = val
    def getarg(self, i):
        return self.args[i]
    def apply_inplace(self, node, reads, write):
        self.oplib.py_apply_inplace(node, reads, write)
    def apply_valret(self, node, reads):
        return self.oplib.py_apply_valret(node, reads)

def is_data_mutable(node):
    if isinstance(node, Result):
        return not isinstance(node.op, Constant) 
    elif isinstance(node, Input):
        return False
    else:
        raise RuntimeError

def _list_to_json(xs):
    return [x.to_json() for x in xs]

class Instr(object):
    def fire(self, interp):
        raise NotImplementedError
    def to_json(self):
        raise NotImplementedError

class LoadArgument(Instr):
    def __init__(self, ind, write_loc):
        self.ind = ind
        self.write_loc = write_loc
    def fire(self, interp):
        interp.set(self.write_loc, interp.getarg(self.ind))
    def __repr__(self):
        return "LoadArg:%i"%self.ind
    def to_json(self):
        return {"type" : "LoadArgument", "write_loc" : self.write_loc.to_json(), "ind" : self.ind}

class Alloc(Instr):
    def __init__(self, dtype, read_locs, write_loc):
        self.dtype = dtype
        self.read_locs = read_locs
        self.write_loc = write_loc
    def fire(self, interp):
        shp = tuple(interp.get(mem) for mem in self.read_locs)
        prevarr = interp.get(self.write_loc)
        if prevarr is None or prevarr.shape != shp: 
            interp.set(self.write_loc, np.ones(shp, self.dtype))
    def __repr__(self):
        return "Alloc:%s"%self.dtype
    def to_json(self):
        return {"type" : "Alloc", "read_locs" : _list_to_json(self.read_locs), "write_loc" : self.write_loc.to_json()}

class BuildTup(Instr):
    def __init__(self, typ, read_locs, write_loc):
        self.typ = typ
        self.read_locs = read_locs
        self.write_loc = write_loc
    def fire(self, interp):
        interp.set(self.write_loc, tuple(interp.get(loc) for loc in self.read_locs))
    def __repr__(self):
        return "BuildTup:%s"%self.typ
    def to_json(self):
        return {"type" : "BuildTup"} # XXX

class ReturnByRef(Instr):
    def __init__(self, node, read_locs, write_loc):
        self.node = node # XXX shouldn't need to store node here.
        self.read_locs = read_locs
        self.write_loc = write_loc
    def fire(self, interp):
        interp.apply_inplace(self.node,
            [interp.get(mem) for mem in self.read_locs],
            interp.get(self.write_loc))
    def __repr__(self):
        return "ReturnByRef:%s"%self.node.op.get_name()
    def to_json(self):
        return {"type" : "ReturnByRef", "read_locs" : _list_to_json(self.read_locs), "write_loc" : self.write_loc.to_json(), "op" : str(self.node.op)}

class ReturnByVal(Instr):
    def __init__(self, node, read_locs, write_loc):
        self.node = node
        self.read_locs = read_locs
        self.write_loc = write_loc
    def fire(self, interp):
        interp.set(self.write_loc, interp.apply_valret(self.node, [interp.get(mem) for mem in self.read_locs]))
    def __repr__(self):
        return "ByVal:%s"%self.node.op.get_name()
    def to_json(self):
        return {"type" : "ReturnByVal", "read_locs" : _list_to_json(self.read_locs), "write_loc" : self.write_loc.to_json(), "op" : str(self.node.op)}
