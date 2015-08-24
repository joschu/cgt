from . import core, utils
import cgt
import ctypes, os.path as osp, hashlib, numpy as np, sys, subprocess, string, os, time, traceback
from collections import defaultdict, namedtuple
from StringIO import StringIO
import logging

def function(inputs, outputs, dbg=None, updates=None, givens=None):
    assert isinstance(inputs, list), "Inputs must be a list"
    assert all(isinstance(el, core.Argument) for el in inputs), "Invalid input: should be a list of Argument nodes"

    if isinstance(outputs, list): 
        assert all(isinstance(el, core.Node) for el in outputs), "Invalid output: should all be symbolic variables"
        return _function_listout(inputs, outputs, dbg, updates, givens)
    elif isinstance(outputs, core.Node):         
        f_listout = _function_listout(inputs, [outputs], dbg, updates, givens)
        return lambda *args : f_listout(*args)[0]
    else:
        raise ValueError("Expected `outputs` to be a Node or a list of Nodes. Got an object of type %s"%type(outputs))

def _function_listout(inputs, outputs, dbg = None, updates=None, givens=None):
    if updates is None:  updates = []
    else: assert (isinstance(updates, list) and 
                all(isinstance(a,tuple) and len(a)==2 
                    and isinstance(a[0], core.Node) and isinstance(a[1], core.Node) 
                    for a in updates)), "updates should be a list of pairs (before, after)"
    if givens is None: givens = []
    else: assert all(isinstance(before, core.Data) for (before,_) in updates), "lhs of updates must be Data instances"

    if dbg: raise core.Todo("debug functionality is broken")
    
    outputs = [cgt.make_tuple(*x) if isinstance(x, tuple) else x for x in outputs]
    
    interp = run_compilation_pipeline(inputs, outputs, updates, givens)
    return interp

# ================================================================
# Execution
# ================================================================

def determine_devices(nodes_sorted, updatetarg2src):
    # Op definitions (available impls, inplace-ness, etc) define constraints
    # on possible devices for a node

    # (1) Get available devices for nodes, determined by which impls are available and node types
    compile_info = get_compile_info()

    cuda_enabled = compile_info["CGT_ENABLE_CUDA"]

    node2dev = {}
    home_device = core.Device(devtype="cpu", idx=0)

    for node in nodes_sorted:

        default_device = node.props.get("default_device", home_device)
        if node in updatetarg2src:
            device = node2dev[updatetarg2src[node]]
        elif node.is_data():
            device = node.device
        elif node.is_argument():
            device = home_device
        else:

            if "native_gpu" in node.op.available_impls and (default_device.devtype == "gpu" or "native_cpu" not in node.op.available_impls):                
                assert cuda_enabled, "trying to put op on gpu but cuda is disabled"
                device = core.Device("gpu", default_device.idx)
            else:
                device = core.Device(devtype="cpu", idx=default_device.idx)
        node2dev[node] = device

    return node2dev

def is_tensor(x):
    return isinstance(x.typ, core.TensorType)
def is_tuple(x):
    return isinstance(x.typ, core.TupleType)

def create_interpreter(inputs, outputs, eg, node2memloc):
    assert isinstance(eg, ExecutionGraph)
    input_types = [input.typ for input in inputs] #pylint: disable=W0622
    output_locs = [node2memloc[node] for node in outputs]

    config = cgt.get_config()
    backend = config["backend"]
    parallel = config["parallel"]
    if backend == "python":
        if parallel:
            raise NotImplementedError("For parallel=True, set backend=native")
            # return ParallelInterpreter(eg, output_locs, input_types)
        else:
            return SequentialInterpreter(eg, output_locs, input_types)
    elif backend == "native":
        if parallel:
            return cgt.cycgt.CppInterpreterWrapper(eg, input_types, output_locs, config["num_threads"])
        else:
            return cgt.cycgt.CppInterpreterWrapper(eg, input_types, output_locs, 0)
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
            if i.ndim > 0 and not i.is_input() and i.op.return_type=="byref":
                if i in node2shape:
                    shpels = node2shape[i]
                else:
                    raise core.Unreachable
                    # shpels = i.op.shp_apply(i.parents)
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
    # First determine how many "child" nodes each node has
    node2child = defaultdict(list)
    for node in nodes_sorted:
        for parent in node.parents:
            node2child[parent].append(node)

    # Now traverse graph again and see where we can use the same memory
    node2memowner = {} # mapping node x -> the node that owns its memory
    
    # For updates, memlocation(RHS) = memlocation(LHS)
    after2before = {after:before for (before,after) in updates}

    enable_inplace_opt = core.get_config()["enable_inplace_opt"]

    for node in nodes_sorted:

        base = node # by default, 
        if node.is_argument():
            pass
        elif node.op.writes_to_input >= 0:
            base = node2memowner[node.parents[node.op.writes_to_input]]
        elif node in after2before:
            base = after2before[node]
        elif enable_inplace_opt and node.op.return_type == "byref": # TODO think about if we need any other conditions
            nodeshape = node.op.shp_apply(node.parents)
            for parent in node.parents:
                if (len(node2child[parent])==1
                        and nodeshape==cgt.shape(parent) # XXX not a very robust way to check
                        and node.dtype == parent.dtype
                        and _is_data_mutable(parent)):
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
    def new_memloc(self, devtype):
        out = MemLocation(self.count, devtype)
        self.count += 1
        return out


def create_execution_graph(inputs, nodes_sorted, node2shape, node2memowner, node2dev):
    # node2impltype = copy.copy(node2impltype) # we'll insert transport ops
    instrs = []
    counter = MemCounter()
    node2memloc = {}

    for node in nodes_sorted:
        if node not in node2dev: node2dev[node] = core.Device(devtype="cpu",idx=node2dev[node.parents[0]].idx if len(node.parents)>0 else 0)
        if node.is_argument():
            write_loc = counter.new_memloc(node2dev[node].devtype)
            node2memloc[node] = write_loc
            i = inputs.index(node)
            instrs.append(LoadArgument(i, write_loc))
        else:
            read_locs = [node2memloc[parent] for parent in node.parents]
            if node.op.return_type == "byref":
                if node2memowner[node] is node:
                    if is_tensor(node): # just make one memory location for output
                        nodeshape = node2shape[node] if node.ndim > 0 else []
                        shape_locs = [node2memloc[shpel] for shpel in nodeshape]
                        write_loc = counter.new_memloc(node2dev[node].devtype)                    
                        instrs.append(Alloc(node.dtype, shape_locs, write_loc))
                    else: # if it's a tuple, we need to allocate all of the components, then build tuple
                        nodeshape = node2shape[node]
                        assert isinstance(nodeshape, tuple)
                        arr_locs = []
                        for (arrshp, arrtyp) in utils.safezip(nodeshape, node.typ):
                            arr_loc = counter.new_memloc(node2dev[node].devtype)
                            shape_locs = [node2memloc[shpel] for shpel in arrshp]
                            instrs.append(Alloc(arrtyp.dtype, shape_locs, arr_loc))
                            arr_locs.append(arr_loc)
                        write_loc = counter.new_memloc(node2dev[node].devtype)
                        instrs.append(BuildTup(node.typ, arr_locs, write_loc))

                else:
                    # If this node writes to another node's memory, the devices must be the same
                    # this should have been enforced in determine_devices()
                    assert node2dev[node] == node2dev[node2memowner[node]]
                    write_loc = node2memloc[node2memowner[node]]                
                instrs.append(ReturnByRef(node.op, [par.typ for par in node.parents], read_locs, write_loc, node_props=node.props))
            else:
                assert node.op.return_type == "byval"
                write_loc = counter.new_memloc(node2dev[node].devtype)
                instrs.append(ReturnByVal(node.op, [par.typ for par in node.parents], read_locs, write_loc, node_props=node.props))
        node2memloc[node] = write_loc
    return ExecutionGraph(instrs, len(inputs), counter.count), node2memloc


def get_callable(op, input_types, devtype, prefer_python=False):
    assert op.available_impls, "need to set op.available_impls"
    config = core.get_config()
    if (prefer_python or config["force_python_impl"]) and "python" in op.available_impls:
        return op.get_py_callable(input_types)        
    elif config["backend"] == "python":
        if "python" in op.available_impls:
            return op.get_py_callable(input_types)
        else:
            assert devtype=="cpu", "can't use devtype=gpu with python backend"
            if "native_cpu" in op.available_impls:
                return get_native_callable(op, input_types, "cpu")
            else:
                raise RuntimeError("Can't find an implementation of %s suitable for python backend. Just have available_impls=%s"%(op,op.available_impls))
    else: # backend = native
        if devtype == "cpu":
            if "native_cpu" in op.available_impls:
                return get_native_callable(op, input_types, "cpu")
            else:
                print "using python impl for",op
                return op.get_py_callable(input_types)
        else:
            if "native_gpu" in op.available_impls:
                return get_native_callable(op, input_types, "gpu")
            else:
                raise RuntimeError("Tried to put Op %s on the GPU but I only have a python impl :("%op)


def get_native_callable(op, input_types, devtype):
    nci = op.get_native_compile_info(input_types, devtype)
    nci.op_str = str(op)
    nci.return_type = op.return_type
    nci.n_in = len(input_types)
    return nci2callable(nci)

def add_transports(nodelist, node2dev, node2shape):

    node2child = defaultdict(list)
    for node in nodelist:
        for par in node.parents:
            node2child[par].append(node)

    # XXX look at native compilation info, gpu deref mask
    for node in nodelist:
        dev = node2dev[node]
        dev2copy = {}
        for child in node2child[node]:
            childdev = node2dev[child]
            if not childdev == dev:
                if childdev not in dev2copy:
                    nodecopy = core.Result(core.Transport(childdev), [node])
                    node2dev[nodecopy] = childdev
                    dev2copy[childdev] = nodecopy
                    node2shape[nodecopy] = node2shape[node]
                replace_parents(child, node, dev2copy[childdev])


def replace_parents(node, before, after):
    for (i,p) in enumerate(node.parents):
        if p is before:
            node.parents[i] = after

def run_compilation_pipeline(inputs, outputs, updates, givens):
    """
    Compiles the expression graph into an execution graph. 
    """
    config = core.get_config()

    # Phase 1: simplification and analysis of expression graph
    # ------------------------------------------------------
    # Add add update targets to outputs
    logging.info("Simplification")
    outputs_updatetargs = outputs + [after for (_before, after) in updates]
    if givens: outputs_updatetargs = core.clone(outputs_updatetargs, dict(givens))
    # Do simplification + analysis pass on expression graph
    outputs_updatetargs_simple, analysis, _ = \
        core.simplify_and_analyze(outputs_updatetargs) if config["enable_simplification"] \
        else (outputs_updatetargs, core.analyze(outputs_updatetargs), {})


    # Phase 2: device targeting
    # ------------------------------------------------------
    logging.info("Device targeting")
    outputs_updatetargs_simple = cgt.core.clone(outputs_updatetargs_simple)
    analysis = core.analyze(outputs_updatetargs_simple) 
    # XXX inefficient to just copy the graph and redo analysis
    nodelist = core.topsorted(outputs_updatetargs_simple)
    updatesrcs = [before for (before, _) in updates]    
    updatetargs_simple = outputs_updatetargs_simple[len(outputs):]
    node2dev = determine_devices(nodelist, {targ:src for (src,targ) in zip(updatesrcs, updatetargs_simple)})
    add_transports(nodelist, node2dev, analysis["node2shape"])

    # Phase 3: build execution graph
    # ------------------------------------------------------
    # Sort nodes so that shape elements appear before a given node
    logging.info("Build execution graph")
    nodes_sorted = topsorted_shapes_first(outputs_updatetargs_simple, analysis["node2shape"])
    # For each node, figure out if its output should be written to a previous node's memory
    # (memowner : "memory owner")
    updatetargs_simple = outputs_updatetargs_simple[len(outputs):]
    node2memowner = determine_memowner(nodes_sorted, zip(updatesrcs, updatetargs_simple), node2dev)
    # Find the outputs we want to return
    outputs_simple = outputs_updatetargs_simple[:len(outputs)] # get rid
    # Generate execution graph
    eg, node2memloc = create_execution_graph(
        inputs, nodes_sorted, analysis["node2shape"], node2memowner, node2dev)

    # print execution graph
    if config["verbose"]:
        print 'begin'
        print '\n'.join(str(i)+'.) \t'+repr(instr) for (i,instr) in enumerate(eg.instrs))
        print 'end'

    # Phase 3: create C or Python interpreter for graph
    # ------------------------------------------------------
    interp = create_interpreter(inputs, outputs_simple, eg, node2memloc)

    # Done!
    return interp


# ================================================================
# Simple numeric eval via traversal
# ================================================================

def numeric_eval(output, arg2val):
    """
    Numerically evaluates symbolic variable without any compilation,
    by associating each argument with a value (via `arg2val`) and traversing the
    computation graph


    Inputs
    ------
    output: symbolic variable or list of variables we would like to evaluate
    arg2val: dictionary assigning each argument that output depends on to a numerical value

    Returns
    -------
    Numeric value or list of numeric values of variables corresponding to output
    """
    if isinstance(output, list):
        assert all(isinstance(x, core.Node) for x in output), "expected a list of Nodes"
        return _numeric_eval_listout(output, arg2val)
    elif isinstance(output, core.Node):
        return _numeric_eval_listout([output],arg2val)[0]
    else:
        raise ValueError("expected `output` to be a Node or a list of Nodes. Got an object of type %s"%type(output))

def _numeric_eval_listout(outputs, arg2val):
    """
    Evaluate outputs numerically. arg2val is a dictionary mapping arguments to numerical values
    """
    assert isinstance(outputs, list)
    assert isinstance(arg2val, dict)

    nodes = list(core.topsorted(outputs))

    node2val = {}
    for node in nodes:
        if node.is_argument():
            node2val[node] = arg2val[node]
        elif node.is_data():
            node2val[node] = node.get_value()
        else:
            parentvals = [node2val[par] for par in node.parents]
            node2val[node] = core.py_numeric_apply(node, parentvals)
        # assert node.get_ndim() == np.array(node2val[node]).ndim
    numeric_outputs = [node2val[node] for node in outputs]

    return numeric_outputs

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

class MemLocation(object):
    def __init__(self, idx, devtype):
        assert isinstance(idx, int) and devtype in ["cpu", "gpu"]
        self.index = idx
        self.devtype = devtype
        # TODO: dtype
    def __repr__(self):
        return "%%%i/%s" % (self.index, self.devtype)


# ================================================================
# Instructions
# ================================================================

class Instr(object):
    def fire(self, interp):
        raise NotImplementedError

class LoadArgument(Instr):
    def __init__(self, ind, write_loc):
        self.ind = ind
        self.write_loc = write_loc
    def fire(self, interp):
        interp.set(self.write_loc, interp.getarg(self.ind))
    def __repr__(self):
        return "%s = LoadArg ind:%i" % (self.write_loc, self.ind)

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
        return "%s = Alloc shp:%s dtype:%s" % (self.write_loc, str(self.read_locs), self.dtype)

class BuildTup(Instr):
    def __init__(self, typ, read_locs, write_loc):
        self.typ = typ
        self.read_locs = read_locs
        self.write_loc = write_loc
    def fire(self, interp):
        interp.set(self.write_loc, tuple(interp.get(loc) for loc in self.read_locs))
    def __repr__(self):
        return "%s = BuildTup args:%s" % (self.write_loc, str(self.read_locs))

class ReturnByRef(Instr):
    def __init__(self, op, input_types, read_locs, write_loc, node_props=None):
        self.op = op
        self.input_types = input_types
        self.read_locs = read_locs
        self.write_loc = write_loc
        self._callable = None
        self.node_props=node_props
    def fire(self, interp):
        if self._callable is None: self._callable = self.get_callable()
        self._callable.call(
            [interp.get(mem) for mem in self.read_locs],
            interp.get(self.write_loc))
    def __repr__(self):
        return "%s = ReturnByRef op:%s args:%s" % (self.write_loc, str(self.op), str(self.read_locs))
    def get_callable(self):
        return get_callable(self.op, self.input_types, self.write_loc.devtype)

class ReturnByVal(Instr):
    def __init__(self, op, input_types, read_locs, write_loc, node_props=None):
        self.op = op
        self.input_types = input_types
        self.read_locs = read_locs
        self.write_loc = write_loc
        self._callable = None
        self.node_props=node_props        
    def fire(self, interp):
        if self._callable is None: self._callable = self.get_callable()        
        interp.set(self.write_loc, self._callable.call([interp.get(mem) for mem in self.read_locs]))
    def get_callable(self):
        return get_callable(self.op, self.input_types, self.write_loc.devtype)
    def __repr__(self):
        return "%s = ReturnByVal op:%s args:%s" % (self.write_loc, str(self.op), str(self.read_locs))


# ================================================================
# Compiling native code
# ================================================================

def nci2callable(nci):
    template_code = gen_templated_code(nci.includes, nci.closure_triples, nci.func_code)
    compile_info = get_compile_info()    
    prefix = utils.hash_seq1(template_code, compile_info["CPP_FLAGS"], *(src.code for src in nci.extra_srcs))
    d = dict(function=_funcname(prefix), closure=_closurename(prefix),setup=_setupname(prefix),teardown=_teardownname(prefix))
    
    fn_srcfile = core.SrcFile("c++",string.Template(template_code).substitute(d))
    srcfiles = [fn_srcfile]
    srcfiles.extend(core.SrcFile(sf.lang, string.Template(sf.code).substitute(d)) for sf in nci.extra_srcs)
    
    CACHE_ROOT = compile_info["CACHE_ROOT"]
    libpath = osp.join(CACHE_ROOT, prefix+".so")
    if not osp.exists(libpath):
        tu = TranslationUnit(srcfiles, nci.link_flags)
        tu.compile(prefix, libpath)

    lib = get_or_load_lib(libpath)
    fptr = getattr(lib, _funcname(prefix))
    setup_fptr = getattr(lib, _setupname(prefix)) if nci.setup else None
    teardown_fptr = getattr(lib, _teardownname(prefix)) if nci.teardown else None
    cldata = _build_closure(nci.closure_triples)
    return core.NativeCallable(nci.n_in, nci.return_type, nci.op_str, fptr, cldata=cldata, setup_fptr=setup_fptr, teardown_fptr=teardown_fptr,
        store_objects=nci.store_objects)

def _funcname(prefix):
    return "call_"+prefix
def _setupname(prefix):
    return "setup_"+prefix
def _teardownname(prefix):
    return "teardown_"+prefix
def _closurename(prefix):
    return "closure_"+prefix


def gen_templated_code(includes, closure_info, func_code):
    s = StringIO()
    includes = ["cgt_common.h"] + includes
    for fname in includes:
        s.write('#include "%s"\n'%fname)
    gen_struct_code(closure_info, s)
    s.write(func_code)
    return s.getvalue()

def gen_struct_code(triples, outstream):
    if triples is None:
        return
    outstream.write("typedef struct $closure {\n")
    for (fieldname,fieldtype,_val) in triples:
        outstream.write(_ctypes2str[fieldtype])
        outstream.write(" ")
        outstream.write(fieldname)
        outstream.write(";\n")
    outstream.write("} $closure;\n")

_LIBRARIES = {}
def get_or_load_lib(libname):
    if libname in _LIBRARIES:
        return _LIBRARIES[libname]
    else:
        out = ctypes.cdll.LoadLibrary(libname)
        _LIBRARIES[libname] = out
        return out


class TranslationUnit(object):
    """All the input that goes into building a native binary for one or more ops"""

    def __init__(self, srcfiles, link_flags):
        self.srcfiles = srcfiles
        self.link_flags = link_flags

    def compile(self, prefix, libpath):
        """
        Compiles all of the files, places them in the cache directory
        Then links them creating prefix.so
        """
        CACHE_ROOT = get_compile_info()["CACHE_ROOT"]
        cmds = ["cd %s"%CACHE_ROOT]
        objs = []
        for (i,(lang,code)) in enumerate(self.srcfiles):
            if lang=="c++":
                srcpath = osp.join(CACHE_ROOT, prefix+"_%i.cpp"%i)
                cmds.append(_make_cpp_compile_cmd(srcpath))
            elif lang=="cuda":
                srcpath = osp.join(CACHE_ROOT, prefix+"_%i.cu"%i)
                cmds.append(_make_cuda_compile_cmd(srcpath))
            else:
                raise NotImplementedError 
            with open(srcpath,"w") as fh: fh.write(code)
            objs.append(srcpath+".o")
        cmds.append(_make_link_cmd(objs, self.link_flags, libpath))
        bigcmd = " && ".join(cmds)
        call_and_print(bigcmd)


_COMPILE_CONFIG = None
def get_compile_info():
    global _COMPILE_CONFIG
    
    if _COMPILE_CONFIG is None:

        config = core.get_config()

        CGT_BUILD_ROOT = cgt.cycgt.cgt_build_root() #pylint: disable=E1101

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
        CGT_ENABLE_CUDA = cmake_info["CGT_ENABLE_CUDA"] in ["1","ON"]
        CGT_ENABLE_CUDNN = cmake_info["CGT_ENABLE_CUDNN"] in ["1","ON"]        
        DEFINITIONS = "-DENABLE_CUDA" if CGT_ENABLE_CUDA else ""
        CUDNN_ROOT = cmake_info["CUDNN_ROOT"]


        _COMPILE_CONFIG = dict(        
            OPENBLAS_INCLUDE_DIR = osp.join(CGT_BUILD_ROOT,"OpenBLAS"),
            CGT_INCLUDE_DIR = cmake_info["CGT_INCLUDE_DIR"],
            CGT_LIBRARY_DIR = osp.join(CGT_BUILD_ROOT,"lib"),
            CUDA_LIBRARY_DIR = osp.join(CUDA_ROOT,"lib"),
            CUDA_INCLUDE_DIR = osp.join(CUDA_ROOT,"include"), 
            CUDA_LIBRARIES = cmake_info["CUDA_LIBRARIES"], 
            DEFINITIONS = DEFINITIONS,  
            CUDA_ROOT = CUDA_ROOT,
            CUDNN_ROOT = CUDNN_ROOT,
            CACHE_ROOT = osp.expanduser(config["cache_dir"]),
            CGT_ENABLE_CUDA = CGT_ENABLE_CUDA,
            CGT_ENABLE_CUDNN = CGT_ENABLE_CUDNN,
            # CGT_LIBRARY = cmake_info["CGT_LIBRARY"],
        )

        includes  = "-I"+_COMPILE_CONFIG["CGT_INCLUDE_DIR"]
        includes += " -I"+_COMPILE_CONFIG["OPENBLAS_INCLUDE_DIR"]
        link_flags = ""
        if _COMPILE_CONFIG["CGT_ENABLE_CUDA"]:  includes += " -I"+_COMPILE_CONFIG["CUDA_INCLUDE_DIR"]
        if _COMPILE_CONFIG["CGT_ENABLE_CUDNN"]: includes += " -I"+_COMPILE_CONFIG["CUDNN_ROOT"]
        _COMPILE_CONFIG["INCLUDES"] = includes

        link_flags = "-lcgt -L"+_COMPILE_CONFIG["CGT_LIBRARY_DIR"]
        if _COMPILE_CONFIG["CGT_ENABLE_CUDA"]: link_flags += " -L"+_COMPILE_CONFIG["CUDA_LIBRARY_DIR"]
        if _COMPILE_CONFIG["CGT_ENABLE_CUDNN"]:
            link_flags += " -L"+_COMPILE_CONFIG["CUDNN_ROOT"]
            link_flags += " -Wl,-rpath,"+_COMPILE_CONFIG["CUDNN_ROOT"]            

        if sys.platform == "darwin":
            link_flags += " -dynamiclib -Wl,-headerpad_max_install_names"
        else:
            link_flags += " -shared -rdynamic"
        _COMPILE_CONFIG["LINK_FLAGS"] = link_flags

        cpp_flags = "-fvisibility=hidden -std=c++11 -fPIC" + (" -O0 -g" if config["debug_cpp"] else " -O3 -DNDEBUG")
        if sys.platform == "darwin": cpp_flags += " -stdlib=libc++"
        _COMPILE_CONFIG["CPP_FLAGS"] = cpp_flags

        CACHE_ROOT = _COMPILE_CONFIG["CACHE_ROOT"]
        if not osp.exists(CACHE_ROOT):
            os.makedirs(CACHE_ROOT)
    return _COMPILE_CONFIG

def _make_cpp_compile_cmd(srcpath):
    d = get_compile_info()
    return "c++ %(cpp_flags)s %(srcpath)s -c -o %(srcpath)s.o %(includes)s %(definitions)s"%dict(
        srcpath = srcpath, includes=d["INCLUDES"], definitions=d["DEFINITIONS"], 
        cpp_flags=d["CPP_FLAGS"], cacheroot=d["CACHE_ROOT"])

def _make_cuda_compile_cmd(srcpath):
    d = get_compile_info()
    return "nvcc %(srcpath)s -c -o %(srcpath)s.o -ccbin cc -m64 -Xcompiler  -fPIC -Xcompiler -O3 -Xcompiler -arch -Xcompiler x86_64 %(includes)s %(definitions)s"%dict(
        srcpath = srcpath, includes=d["INCLUDES"], definitions=d["DEFINITIONS"])

def _make_link_cmd(objs, extra_link_flags, libpath):
    d = get_compile_info()
    iname = "-install_name %s"%osp.basename(libpath) if sys.platform=="darwin" else ""
    return r"c++ %(cpp_flags)s %(objnames)s  %(link_flags)s %(iname)s -o %(libpath)s"%dict(
        objnames=" ".join(objs), includes=d["INCLUDES"], cpp_flags=d["CPP_FLAGS"], libpath=libpath, 
        link_flags=d["LINK_FLAGS"]+" "+extra_link_flags, cacheroot=d["CACHE_ROOT"], iname=iname)

def call_and_print(cmd):
    print "\x1b[32m%s\x1b[0m"%cmd
    subprocess.check_call(cmd,shell=True)

_ctypes2str = {
    ctypes.c_byte : "uint8_t",
    ctypes.c_bool : "bool",
    ctypes.c_char : "char",
    ctypes.c_int : "int",
    ctypes.c_long : "long",
    ctypes.c_void_p : "void*",
    ctypes.c_double : "double",
    ctypes.c_float : "float"
}

def _build_closure(triples):
    if triples is None:
        return ctypes.c_void_p(0)
    vals = []
    fields = []
    for (fieldname,fieldtype,val) in triples:
        vals.append(val)
        fields.append((fieldname,fieldtype))
    class S(ctypes.Structure):
        _fields_ = fields
    closure = S(*vals)    
    return closure


################################################################
### Interpreters 
################################################################
 
class Interpreter(object):
    def __call__(self, args):
        raise NotImplementedError
    def get(self, mem):
        raise NotImplementedError
    def set(self, mem, val):
        raise NotImplementedError
    def getarg(self, i):
        raise NotImplementedError


class SequentialInterpreter(Interpreter):
    """
    Runs an execution graph
    """
    def __init__(self, eg, output_locs, input_types, copy_outputs=True):
        self.eg = eg
        self.input_types = input_types
        self.output_locs = output_locs
        self.storage = [None for _ in xrange(self.eg.n_locs)]
        self.args = None
        self.copy_outputs = copy_outputs
    def __call__(self, *args):
        assert len(args) == len(self.input_types), "Wrong number of inputs provided"
        self.args = tuple(core.as_valid_array(arg, intype) for (arg, intype) in zip(args, self.input_types))
        for instr in self.eg.instrs:
            if profiler.on: tstart = time.time()
            try:
                instr.fire(self)
            except Exception as e:
                traceback.print_exc()
                if isinstance(instr, (ReturnByRef,ReturnByVal)):
                    if core.get_config()["debug"]:
                        assert "stack" in instr.node_props
                        utils.colorprint(utils.Color.MAGENTA, "HERE'S THE STACK WHEN THE OFFENDING NODE WAS CREATED\n",o=sys.stderr)
                        print>>sys.stderr, ">>>>>>>>>>>>>>>>>>>>>>>>>>"        
                        traceback.print_list(instr.node_props["stack"])
                        print>>sys.stderr, "<<<<<<<<<<<<<<<<<<<<<<<<<<"        
                        raise e
                    else:
                        utils.error("Didn't save the stack so I can't give you a nice traceback :(. Try running with CGT_FLAGS=debug=True")
                        raise e
                else:
                    utils.error("Oy vey, an exception occurred in a %s Instruction. I don't know how to help you debug this one right now :(."%type(instr))
                    raise e

            if profiler.on: profiler.update(instr, time.time()-tstart)
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


# ================================================================
# Profiler
# ================================================================


class _Profiler(object):
    """
    Profiler for Python backend, i.e. Interpreter
    """
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
        for (instr,(count,t)) in self.instr2stats.iteritems():
            if isinstance(instr, (ReturnByRef, ReturnByVal)):
                opkey = str(instr.op)
            elif isinstance(instr, Alloc):
                opkey = "Alloc{dtype=%s,ndim=%i}"%(instr.dtype, len(instr.read_locs))
            else:
                opkey = instr.__class__.__name__

            (prevcount, prevtime) = op2stats.get(opkey, (0, 0.0))
            op2stats[opkey] = (prevcount+count, prevtime+t)

        print "Total time elapsed: %.3g seconds"%self.t_total
        # _print_heading("By instruction")
        # _print_stats(self.instr2stats, self.t_total)
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
    for (key, (count,t)) in key2stats.iteritems():
        rows.append([str(key), count, t, t/t_total])
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
        if isinstance(typ, core.TensorType):
            assert numarg.dtype==typ.dtype and numarg.ndim==typ.ndim
    

# ================================================================
# Utils
# ================================================================

def _list_to_json(xs):
    return [x.to_json() for x in xs]

def _is_data_mutable(node):
    if isinstance(node, core.Result):
        return not isinstance(node.op, core.Constant) 
    elif isinstance(node, core.Input):
        return False
    else:
        raise RuntimeError


