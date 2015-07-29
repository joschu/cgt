from libcpp.vector cimport vector
cimport numpy as cnp
import numpy as np
import ctypes
import os.path as osp
import cgt

################################################################
### CGT common datatypes 
################################################################
 
cdef extern from "cgt_common.h":

    cdef enum cgt_dtype:
        cgt_i1
        cgt_i2
        cgt_i4
        cgt_i8
        cgt_f2
        cgt_f4
        cgt_f8
        cgt_f16
        cgt_c8
        cgt_c16
        cgt_c32
        cgt_O

    ctypedef void (*cgt_fun)(void*, cgt_array**);

    cdef enum cgt_devtype:
        cgt_cpu
        cgt_gpu

    cdef enum cgt_typetag:
        cgt_tupletype
        cgt_arraytype
    
    struct cgt_array:
        int ndim
        cgt_dtype dtype
        cgt_devtype devtype
        size_t* shape
        void* data
        size_t stride
        bint ownsdata    

    struct cgt_tuple:
        pass # TODO

    struct cgt_object:
        cgt_typetag typetag

    cgt_typetag cgt_get_tag(cgt_object*)

################################################################
### Dynamic loading 
################################################################
 

cdef extern from "dlfcn.h":
    void *dlopen(const char *filename, int flag)
    char *dlerror()
    void *dlsym(void *handle, const char *symbol)
    int dlclose(void *handle) 
    int RTLD_GLOBAL
    int RTLD_LAZY
    int RTLD_NOW

LIB_DIRS = None
LIB_HANDLES = {}

def initialize_lib_dirs():
    global LIB_DIRS
    if LIB_DIRS is None:
        import cycgt
        LIB_DIRS = [".cgt/build/lib"]

cdef void* get_or_load_lib(libname):
    cdef void* handle
    initialize_lib_dirs()
    if libname in LIB_HANDLES:
        return <void*><size_t>LIB_HANDLES[libname]
    else:
        for ld in LIB_DIRS:
            libpath = osp.join(ld,libname)
            if osp.exists(libpath):
                handle = dlopen(libpath, RTLD_NOW | RTLD_GLOBAL)
        if handle == NULL:
            raise ValueError("couldn't load library named %s: %s"%(libname, <bytes>dlerror()))
        else:
            LIB_HANDLES[libname] = <object><size_t>handle
        return handle


################################################################
### Execution graph 
################################################################
 
cdef extern from "execution.h" namespace "cgt":
    cppclass FuncClosure:
        FuncClosure(cgt_fun*, void*)
        FuncClosure()
    cppclass MemLocation:
        MemLocation(size_t)
        MemLocation()
    cppclass Instruction:
        pass
    cppclass ExecutionGraph:
        void add_instr(Instruction*)
        cgt_object* get(MemLocation)
        ExecutionGraph(int, vector[MemLocation])        
    cppclass LoadArgument(Instruction):
        LoadArgument(int, MemLocation)
    cppclass Alloc(Instruction):
        Alloc(cgt_dtype, vector[MemLocation], MemLocation)
    cppclass InPlace(Instruction):
        InPlace(vector[MemLocation], MemLocation, FuncClosure)
    cppclass ValReturning(Instruction):
        ValReturning(vector[MemLocation], MemLocation, FuncClosure)


# Conversion funcs
# ----------------------------------------------------------------

cdef vector[int] _tovectorint(object xs):
    cdef vector[int] out = vector[int]()
    for x in xs: out.push_back(<int>x)
    return out

cdef object _cgtarray2py(cgt_array* a):
    raise NotImplementedError

cdef object _cgttuple2py(cgt_tuple* t):
    raise NotImplementedError

cdef object _cgtobj2py(cgt_object* o):
    tag = cgt_get_tag(o)
    if tag == cgt_tupletype:
        return _cgttuple2py(<cgt_tuple*>o)
    elif tag == cgt_arraytype:
        return _cgtarray2py(<cgt_array*>o)
    else:
        raise RuntimeError("invalid type tag. something's very wrong")

cdef void* _ctypesstructptr(object o):
    return <void*><size_t>ctypes.cast(ctypes.pointer(o), ctypes.c_voidp).value    

cdef FuncClosure _node2closure(node):
    import cgt
    libname, funcname, cldata = cgt.get_impl(node, "cpu") # TODO
    cdef void* lib_handle = get_or_load_lib(libname)
    cdef cgt_fun* cfun = <cgt_fun*>dlsym(lib_handle, funcname)
    return FuncClosure(cfun, _ctypesstructptr(cldata))

cdef MemLocation _tocppmem(object pymem):
    return MemLocation(<size_t>pymem.index)

cdef vector[MemLocation] _tocppmemvec(object pymemlist):
    cdef vector[MemLocation] out = vector[MemLocation]()
    for pymem in pymemlist:
        out.push_back(_tocppmem(pymem))
    return out

cdef Instruction* _tocppinstr(object pyinstr):
    t = type(pyinstr)
    cdef Instruction* out
    if t == cgt.LoadArgument:
        out = new LoadArgument(pyinstr.ind, _tocppmem(pyinstr.write_loc))
    elif t == cgt.Alloc:
        out = new Alloc(pyinstr.dtype, _tocppmemvec(pyinstr.read_locs), _tocppmem(pyinstr.write_loc))
    elif t == cgt.InPlace:
        out = new InPlace(_tocppmemvec(pyinstr.read_locs), _tocppmem(pyinstr.write_loc), _node2closure(pyinstr.node))
    elif t == cgt.ValReturning:
        out = new ValReturning(_tocppmemvec(pyinstr.read_locs), _tocppmem(pyinstr.write_loc),_node2closure(pyinstr.node))
    else:
        raise RuntimeError("expected instance of type Instruction. got type %s"%t)
    return out

cdef class _CppExecutionGraph:
    cdef ExecutionGraph* eg
    def __init__(self, pyeg):
        self.eg = new ExecutionGraph(pyeg.n_locs(), _tocppmemvec(pyeg.output_locs))
        for instr in pyeg.instrs:
            self.eg.add_instr(_tocppinstr(instr))
    def __call__(self):
        raise cgt.Todo
        # # TODO create cgt arrays for numpy arrays
        # return [_cgtobj2py(eg.get(loc)) for loc in self.eg.output_locs]

################################################################
### Wrapper classes
################################################################
 
cdef class CgtArray:
    cdef cgt_array* array
    def to_numpy(self):
        pass # TODO

def _make_execution_graph(pyeg):
    "create an execution graph object"
    raise cgt.Todo

def make_interpreter(pyeg):
    "create an interpreter object"
    raise cgt.Todo

