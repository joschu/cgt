from core import *
from collections import defaultdict
import os
import os.path as osp

# ================================================================
# Execution
# ================================================================


def _get_cgt_src_root():
    return osp.dirname(osp.dirname(osp.realpath(__file__)))

_CONFIG = None
def load_config():
    global _CONFIG
    if _CONFIG is None:
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
        print "CONFIG",_CONFIG
    return _CONFIG


_COMPILE_CONFIG = None
def get_compile_info():
    global _COMPILE_CONFIG
    
    if _COMPILE_CONFIG is None:

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
    h = hashlib.md5(code_raw).hexdigest()[:10]
    funcname = devtype + node.op.__class__.__name__ + h
    ci = get_compile_info()
    CACHE_ROOT = ci["CACHE_ROOT"]
    libpath = osp.join(CACHE_ROOT, funcname + ".so")
    closure = node.op.get_closure(node.parents)

    if not osp.exists(libpath):
        s = StringIO.StringIO()        
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
            shapes = numeric_eval(self.shapevars, dict(utils.safezip(self.inputs, invals)))
            self.callseq.set_shapes(shapes)
            self.first_time=False
        self.callseq.set_inputs(invals)
        self.callseq.execute()
        return self.callseq.get_outputs_numpy()


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
        shapes = numeric_eval(self.shapevars, dict(utils.safezip(self.inputs, invals)))
        self.callseq.set_shapes(shapes)
        self.callseq.set_inputs(invals)
        self.callseq.execute()
        return self.callseq.get_outputs_numpy()


def make_function(inputs, outputs, dbg = None, fixed_sizes=False, backend=None):
    config = load_config()
    backend = backend or config["backend"]

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
        def fn(*invals):
            out = eg.run(invals)
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

MemInfo = namedtuple("MemInfo",["loc","access"])
MEM_OVERWRITE = 'overwrite'
MEM_INCREMENT = 'increment'

class ExecutionGraph(object):
    """
    Abstract interface for execution graph, which might be sequential or parallel
    """
    def new_loc(self):
        raise NotImplementedError
    def add_instr(self, instr):
        raise NotImplementedError
    def set_outputs(self, locs):
        raise NotImplementedError


class ExecutionContext(object):
    """
    Run-time information for the current functional call
    """
    def __init__(self, inputs):
        self.inputs = inputs
    def get_input(self, i):
        return self.inputs[i]

class MemLocation(object):
    def __init__(self):
        self.val = None
    def set_to(self, val):
        self.val = val
    def get_ref(self):
        return self.val

class SequentialExecution(ExecutionGraph):
    def __init__(self):
        self.locs = []
        self.instrs = []
        self.output_locs = []
    def new_loc(self):
        loc = MemLocation()
        self.locs.append(loc)
        return loc
    def add_instr(self, instr):
        self.instrs.append(instr)
    def run(self, args):
        c = ExecutionContext(args)
        for instr in self.instrs:
            instr.fire(c)
        return [loc.get_ref() for loc in self.output_locs]
    def set_outputs(self, locs):
        self.output_locs = locs


def make_execution_graph(inputs, outputs):
    G = SequentialExecution()
    nodes = list(topsorted(outputs))
    node2mem = {}

    node2child = defaultdict(list)
    for node in nodes:
        for parent in node.parents:
            node2child[parent].append(node)

    # TODO: incremental versions of stuff

    for node in nodes:        
        if node.is_argument():
            write_loc = G.new_loc()
            node2mem[node] = MemInfo(write_loc, MEM_OVERWRITE)
            i = inputs.index(node)
            G.add_instr(LoadInput(node, i, write_loc))
        else:
            read_locs = [node2mem[parent].loc for parent in node.parents]
            if node.op.call_type == "inplace":
                needs_alloc=True
                skip_op=False
                if node.op.writes_to_input >= 0:
                        write_loc = node2mem[node.parents[node.op.writes_to_input]].loc
                        needs_alloc=False
                else:
                    nodeshape = node.op.shp_apply(node.parents)
                    for parent in node.parents:
                        if len(node2child[parent])==1 and nodeshape==shape(parent) and data_mutable(parent):
                            write_loc = node2mem[parent].loc
                            needs_alloc=False
                            print "yay, inplace opt" 
                            break                          
                if needs_alloc:
                    write_loc = G.new_loc()
                    G.add_instr(Alloc(node, read_locs, write_loc))
                G.add_instr(InPlace(node, read_locs, write_loc))
            else:                
                write_loc = G.new_loc()
                G.add_instr(ValReturning(node, read_locs, write_loc))
        node2mem[node] = MemInfo(write_loc, MEM_OVERWRITE)
    G.set_outputs([node2mem[node].loc for node in outputs])
    return G

def data_mutable(node):
    if isinstance(node, Result):
        return not isinstance(node.op, Constant) 
    elif isinstance(node, Input):
        return False

class Instr(object):
    def fire(self, c):
        raise NotImplementedError

class LoadInput(Instr):
    def __init__(self, node, ind, write_loc):
        self.node = node
        self.ind = ind
        self.write_loc = write_loc
    def fire(self, c):
        self.write_loc.set_to(c.get_input(self.ind))

class Alloc(Instr):
    def __init__(self, node, read_locs, write_loc):
        self.node = node
        self.shape_fun = get_numeric_shape_fun(self.node)
        self.read_locs = read_locs
        self.write_loc = write_loc
    def fire(self, c):
        shp = self.shape_fun([mem.get_ref() for mem in self.read_locs])
        self.write_loc.set_to(np.zeros(shp, self.node.dtype))

class InPlace(Instr):
    def __init__(self, node, read_locs, write_loc):
        self.node = node
        self.read_locs = read_locs
        self.write_loc = write_loc
    def fire(self, c):
        self.node.op.py_apply_inplace(
            [mem.get_ref() for mem in self.read_locs], 
            self.write_loc.get_ref())

class ValReturning(Instr):
    def __init__(self, node, read_locs, write_loc):
        self.node = node
        self.read_locs = read_locs
        self.write_loc = write_loc
    def fire(self,c):
        self.write_loc.set_to(self.node.op.py_apply_valret([mem.get_ref() for mem in self.read_locs]))