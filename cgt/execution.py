from core import *
from collections import defaultdict
import os
import os.path as osp
from StringIO import StringIO
# ================================================================
# Execution
# ================================================================


def _get_cgt_src_root():
    return osp.dirname(osp.dirname(osp.realpath(__file__)))

_CONFIG = None
def load_config():
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG

    from configobj import ConfigObj
    from validate import Validator
    rcfileloc = osp.join(osp.expanduser("~/.cgtrc"))
    specfilename = osp.join(_get_cgt_src_root(), "cgtrc_spec.ini")
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


_COMPILE_CONFIG = None
def get_compile_info():
    global _COMPILE_CONFIG
    if _COMPILE_CONFIG is None:

        config = load_config()

        import cycgt2 as cycgt #pylint: disable=F0401
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
        CGT_ENABLE_CUDA = cmake_info["CGT_ENABLE_CUDA"] in ["1","ON"]
        DEFINITIONS = "-DENABLE_CUDA" if CGT_ENABLE_CUDA else ""


        _COMPILE_CONFIG = dict(        
            OPENBLAS_INCLUDE_DIR = osp.join(CGT_BUILD_ROOT,"OpenBLAS"),
            CGT_INCLUDE_DIR = cmake_info["CGT_INCLUDE_DIR"],
            CGT_LIBRARY_DIR = osp.join(CGT_BUILD_ROOT,"lib"),
            CUDA_LIBRARY_DIR = osp.join(CUDA_ROOT,"lib"),
            CUDA_INCLUDE_DIR = osp.join(CUDA_ROOT,"include"), 
            CUDA_LIBRARIES = cmake_info["CUDA_LIBRARIES"], 
            DEFINITIONS = DEFINITIONS,  
            CUDA_ROOT = CUDA_ROOT,
            CACHE_ROOT = osp.expanduser(config["cache_dir"]),
            CGT_ENABLE_CUDA = CGT_ENABLE_CUDA
            # CGT_LIBRARY = cmake_info["CGT_LIBRARY"],
        )
    return _COMPILE_CONFIG

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
        if fname.endswith(".cc"):
            cap(r'''
cd %(cacheroot)s && \
c++ -fPIC -O3 -DNDEBUG %(srcpath)s -std=c++11 -c -o %(srcpath)s.o %(includes)s %(defines)s && \
c++ -fPIC -O3 -DNDEBUG %(srcpath)s.o -dynamiclib -Wl,-headerpad_max_install_names -install_name %(libname)s -o %(libpath)s -L%(cgtlibdir)s -lcgt %(extralink)s
            '''%d)
        # TODO set up way to switch to -O0 -g
        elif fname.endswith(".cu"):
            cap(r'''
cd %(cacheroot)s && \
nvcc %(srcpath)s -c -o %(srcpath)s.o -ccbin cc -m64 -Xcompiler  -fPIC -Xcompiler -O3 -Xcompiler -arch -Xcompiler x86_64 %(includes)s %(defines)s && \
c++ -fPIC -O3 -DNDEBUG -fPIC -dynamiclib -Wl,-headerpad_max_install_names %(cudalibs)s -Wl,-rpath,%(cudalibdir)s -install_name %(libname)s -o %(libpath)s %(srcpath)s.o
            '''%d)
                # gpulinkflags = "-dynamiclib -Wl,-headerpad_max_install_names %(CUDA_LIBRARIES)s -Wl,-rpath,%(CUDA_LIBRARY_DIR)s"%d

    else:
        if fname.endswith(".cc"):
            cap('''
c++ -fPIC -O3 -DNDEBUG %(srcpath)s -std=c++11 -c -o %(srcpath)s.o %(includes)s %(defines)s && \
c++ -fPIC -O3 -DNDEBUG -shared -rdynamic -Wl,-soname,%(libname)s -o %(libpath)s %(srcpath)s.o -L%(cgtlibdir)s -lcgt
            '''%d)
        elif fname.endswith(".cu"):
            cap(r'''
cd %(cacheroot)s && 
nvcc %(srcpath)s -c -o %(srcpath)s.o -ccbin cc -m64 -Xcompiler -fPIC -Xcompiler -O3 -Xcompiler -DNDEBUG %(includes)s %(defines)s && \
c++  -fPIC -O3 -DNDEBUG -shared -rdynamic -Wl,-soname,%(libname)s -o %(libpath)s %(srcpath)s.o %(cudalibs)s -Wl,-rpath,%(cudaroot)s
            '''%d
            )

ctypes2str = {
    ctypes.c_int : "int",
    ctypes.c_long : "long",
    ctypes.c_void_p : "void*",
    ctypes.c_double : "double",
    ctypes.c_float : "float"
}

def get_impl(node, devtype):

    # TODO: includes should be in cache, as well as info about definitions like
    # CGT_ENABLE_CUDA

    compile_info = get_compile_info()    
    if devtype == "gpu" and not compile_info["CGT_ENABLE_CUDA"]:
        raise RuntimeError("tried to get CUDA implementation but CUDA is disabled (set CGT_ENABLE_CUDA and recompile)")

    code_raw = (node.op.c_code if devtype=="cpu" else node.op.cuda_code)(node.parents)
    if devtype == "cpu":
        includes = ["cgt_common.h","stdint.h","stddef.h"] + node.op.c_extra_includes
    else:
        includes = ["cgt_common.h","cgt_cuda.h"] + node.op.cuda_extra_includes

    struct_code = StringIO()
    vals = []
    fields = []
    triples = node.op.get_closure(node.parents)
    if triples is None:
        closure = ctypes.c_void_p(0)
    else:
        struct_code.write("typedef struct CGT_FUNCNAME_closure {\n")
        for (fieldname,fieldtype,val) in triples:
            vals.append(val)
            struct_code.write(ctypes2str[fieldtype])
            struct_code.write(" ")
            struct_code.write(fieldname)
            struct_code.write(";\n")
            fields.append((fieldname,fieldtype))
        struct_code.write("} CGT_FUNCNAME_closure;\n")

        class S(ctypes.Structure):
            _fields_ = fields
        closure = S(*vals)


    h = hashlib.md5(code_raw).hexdigest()[:10]
    funcname = devtype + node.op.__class__.__name__ + h
    ci = get_compile_info()
    CACHE_ROOT = ci["CACHE_ROOT"]
    libpath = osp.join(CACHE_ROOT, funcname + ".so")

    if not osp.exists(libpath):
        s = StringIO()        
        if not osp.exists(CACHE_ROOT): os.makedirs(CACHE_ROOT)
        print "compiling %(libpath)s for node %(node)s"%locals()
        ext = "cc" if devtype == "cpu" else "cu"
        srcpath = osp.join(CACHE_ROOT, funcname + "." + ext)
        # write c code to tmp file
        s = StringIO()
        for filename in includes:
            s.write('#include "%s"\n'%filename)
        s.write(struct_code.getvalue().replace("CGT_FUNCNAME",funcname))
        code = code_raw.replace("CGT_FUNCNAME",funcname)
        s.write(code)
        with open(srcpath,"w") as fh:
            fh.write(s.getvalue())

        compile_file(srcpath, osp.splitext(srcpath)[0]+".so", extra_link_flags = node.op.c_extra_link_flags)

    return (libpath,funcname,closure)


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

def make_function(inputs, outputs, dbg = None, fixed_sizes=False, backend=None):
    config = load_config()
    backend = backend or config["backend"]

    if isinstance(outputs, tuple):
        outputs = tuplify(outputs)
    elif isinstance(outputs, list):
        outputs = map(tuplify, outputs)

    single = isinstance(outputs, Node)
    if single: outputs = [outputs]

    if dbg: 
        if backend == "python":            
            outputs = dbg.nodes + outputs
        else:
            utils.warn("Debugging nodes can currently only be used with the python backend, but %s was selected. Ignoring"%backend)

    if backend == "python":
        outputs = simplify(outputs)
        eg = make_execution_graph(inputs, outputs)
        oplib = OpLibrary()
        vm = SequentialInterpreter(eg, oplib)
        def fn(*invals):
            out = vm(invals)
            if dbg and len(dbg.nodes)>0: out = out[len(dbg.nodes):]
            if single: out = out[0]
            return out
        return fn
    elif backend == "cython":
        raise OopsThisCodeIsBrokenException
        if fixed_sizes: fn = FixedSizeFunc(inputs, outputs)
        else: fn = VarSizeFunc(inputs, outputs)
        if single: return lambda *invals : fn(*invals)[0]
        else: return fn        
    else:
        raise NotImplementedError("invalid backend %s"%backend)
    return fn


################################################################
### Simple numeric eval via traversal 
################################################################
 
def numeric_eval(outputs, arg2val):
    """
    Evaluate outputs numerically. arg2val is a dictionary mapping arguments to numerical values
    """
    single = isinstance(outputs, Node)
    if single: outputs = [outputs]

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

    if single:
        return numeric_outputs[0]
    else:
        return numeric_outputs


################################################################
### Execution graph 
################################################################
class OpLibrary(object):
    """
    Stores compiled Op implementations. Performs just-in-time compiling.
    """

    def __init__(self):
        self.op2implhash = {}
        self.implhash2binary = {}

    def _get_op_impl(self, op):
        """
        Grabs the preferred implementation of an op, falling back to
        Python if necessary
        """
        # TODO: C and CUDA, with fallback to Python
        # This implementation just uses the Python Impl
        try:
            impl = op.get_py_impl()
            return impl
        except MethodNotDefined:
            print 'Op %s has no Python implementation' % repr(op)
            raise

    def _compile_impl(self, op, impl):
        # "Compile" refers to extracting and processing an implementation
        # so "compiling" a Python implementation just means to form the Python
        # function that computes the Op
        # a Python "binary" is just this function

        binary = None

        # Only works with Python impls now
        assert isinstance(impl, PyImpl)
        if op.call_type == "inplace":
            assert impl.inplace_func is not None
            binary = impl.inplace_func
        else:
            assert impl.valret_func is not None
            binary = impl.valret_func

        return binary

    def _fetch_binary(self, op):
        """
        Gets the binary for an Op, compiling its Impl if necessary.
        Assumes that if get_*_impl() is called on an Op twice, then the two Impls have the same hashes.
        """

        # If Op has never been seen, check if its impl has been compiled,
        # and compile if necessary
        if op not in self.op2implhash:
            impl = self._get_op_impl(op)
            implhash = self.op2implhash[op] = impl.hash()
            if implhash in self.implhash2binary:
                return self.implhash2binary[implhash]
            binary = self.implhash2binary[implhash] = self._compile_impl(op, impl)
            return binary
        # We saw the Op before, just return its compiled impl directly
        implhash = self.op2implhash[op]
        assert implhash in self.implhash2binary
        return self.implhash2binary[implhash]

    def apply_inplace(self, op, reads, write):
        self._fetch_binary(op)(reads, write)
    def apply_valret(self, op, reads):
        return self._fetch_binary(op)(reads)



MemInfo = namedtuple("MemInfo",["loc","access"])
MEM_OVERWRITE = "overwrite"
MEM_INCREMENT = "increment"

class ExecutionGraph:
    def __init__(self, n_args):
        self.n_args = n_args
        self.locs = []
        self.instrs = []
        self.output_locs = []
        self.cur_idx = 0
    def new_loc(self):
        loc = MemLocation(self.cur_idx)
        self.cur_idx += 1
        self.locs.append(loc)
        return loc
    def add_instr(self, instr):
        self.instrs.append(instr)
    def set_outputs(self, locs):
        self.output_locs = locs
    def n_locs(self):
        return self.cur_idx
    def to_json(self):
        import json
        return json.dumps({
            "instrs" : _list_to_json(self.instrs),
            "output_locs" : _list_to_json(self.output_locs),
            "n_mem" : self.cur_idx
        }, indent=4, sort_keys=True)

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
    def apply_op_inplace(self, op, reads, write):
        raise NotImplementedError
    def apply_op_valret(self, op, reads):
        raise NotImplementedError

class SequentialInterpreter(Interpreter):
    """
    Executes an execution graph
    """
    def __init__(self, eg, oplib):
        self.eg = eg
        self.oplib = oplib
        self.storage = [None for _ in xrange(self.eg.n_locs())]
        self.args = None
    def __call__(self, args):
        self.args = args
        for instr in self.eg.instrs:
            instr.fire(self)
        return [self.get(loc) for loc in self.eg.output_locs]
    def get(self, mem):
        return self.storage[mem.index]
    def set(self, mem, val):
        self.storage[mem.index] = val
    def getarg(self, i):
        return self.args[i]
    def apply_op_inplace(self, op, reads, write):
        self.oplib.apply_inplace(op, reads, write)
    def apply_op_valret(self, op, reads):
        return self.oplib.apply_valret(op, reads)

def make_execution_graph(inputs, outputs):
    G = ExecutionGraph(len(inputs))
    nodes = list(topsorted(outputs))
    node2mem = {}

    node2child = defaultdict(list)
    for node in nodes:
        for parent in node.parents:
            node2child[parent].append(node)

    analysis = analyze(outputs)
    node2shape = analysis["node2shape"]

    tupnode2shpnode = {}
    # XXX won't work if we have nested tuples. or will it?

    augoutputs = []
    for node in nodes:
        shp = node2shape[node]
        if isinstance(shp, list):
            augoutputs.extend(shp)
        else:
            newnode = make_tuple(*shp)
            tupnode2shpnode[node] = newnode
            augoutputs.append(newnode)
    augoutputs.extend(outputs)

    nodes2 = topsorted(augoutputs)
    # XXX this only works because of details of topsort implementation
    # in general we're not guaranteed that the shape components will be computed before the nodes

    # TODO: incremental versions of stuff
    for node in nodes2:        
        if node.is_argument():
            write_loc = G.new_loc()
            node2mem[node] = MemInfo(write_loc, MEM_OVERWRITE)
            i = inputs.index(node)
            G.add_instr(LoadArgument(i, write_loc))
        else:
            read_locs = [node2mem[parent].loc for parent in node.parents]
            if node.op.call_type == "inplace":
                needs_alloc=True
                if node.op.writes_to_input >= 0:
                    write_loc = node2mem[node.parents[node.op.writes_to_input]].loc
                    needs_alloc=False
                else:
                    nodeshape = node.op.shp_apply(node.parents)
                    for parent in node.parents:
                        if len(node2child[parent])==1 and nodeshape==shape(parent) and node.dtype == parent.dtype and _is_data_mutable(parent):
                            write_loc = node2mem[parent].loc
                            needs_alloc = False
                            break                          
                if needs_alloc:
                    write_loc = G.new_loc()
                    if isinstance(node.typ, Tensor):
                        shape_locs = [node2mem[shpel].loc for shpel in node2shape[node]] if node.ndim>0 else []
                        G.add_instr(Alloc(node.dtype, shape_locs, write_loc))
                    else:
                        shpnode = tupnode2shpnode[node]
                        G.add_instr(AllocTup(node.get_type(), node2mem[shpnode].loc, write_loc))
                G.add_instr(InPlace(node, read_locs, write_loc))
            else:                
                write_loc = G.new_loc()
                G.add_instr(ValReturning(node, read_locs, write_loc))
        node2mem[node] = MemInfo(write_loc, MEM_OVERWRITE)
    G.set_outputs([node2mem[node].loc for node in outputs])
    return G

def _is_data_mutable(node):
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
    def to_json(self):
        return {"type" : "LoadArgument", "write_loc" : self.write_loc.to_json(), "ind" : self.ind}

class Alloc(Instr):
    def __init__(self, dtype, read_locs, write_loc):
        self.dtype = dtype
        self.read_locs = read_locs
        self.write_loc = write_loc
    def fire(self, interp):
        shp = tuple(interp.get(mem) for mem in self.read_locs)
        interp.set(self.write_loc, np.zeros(shp, self.dtype))
    def to_json(self):
        return {"type" : "Alloc", "read_locs" : _list_to_json(self.read_locs), "write_loc" : self.write_loc.to_json()}

class AllocTup(Instr):
    def __init__(self, typ, read_loc, write_loc):
        self.typ = typ
        self.read_loc = read_loc
        self.write_loc = write_loc
    def fire(self, interp):
        shp = interp.get(self.read_loc)
        interp.set(self.write_loc, alloc_from_shp(shp, self.typ))
    def to_json(self):
        return {"type" : "AllocTup"} # XXX

class InPlace(Instr):
    def __init__(self, node, read_locs, write_loc):
        self.node = node # XXX shouldn't need to store node here.
        self.read_locs = read_locs
        self.write_loc = write_loc
    def fire(self, interp):
        interp.apply_op_inplace(self.node.op,
            [interp.get(mem) for mem in self.read_locs],
            interp.get(self.write_loc))
    def to_json(self):
        return {"type" : "InPlace", "read_locs" : _list_to_json(self.read_locs), "write_loc" : self.write_loc.to_json(), "op" : str(self.node.op)}

class ValReturning(Instr):
    def __init__(self, node, read_locs, write_loc):
        self.node = node
        self.read_locs = read_locs
        self.write_loc = write_loc
    def fire(self, interp):
        interp.set(self.write_loc, interp.apply_op_valret(self.node.op, [interp.get(mem) for mem in self.read_locs]))
    def to_json(self):
        return {"type" : "ValReturning", "read_locs" : _list_to_json(self.read_locs), "write_loc" : self.write_loc.to_json(), "op" : str(self.node.op)}
