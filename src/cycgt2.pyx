from libcpp.vector cimport vector
cimport numpy as cnp
import numpy as np
import ctypes
import os.path as osp
import cgt
cimport cpython
import cgt


cnp.import_array()

################################################################
### CGT common datatypes 
################################################################
 
cdef extern from "cgt_common.h" namespace "cgt":

    cdef enum Dtype:
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

    cppclass IRC[T]:
        pass

    cppclass Object:
        pass

    ctypedef void (*Inplacefun)(void*, Object**);
    ctypedef Object* (*Valretfun)(void*, Object**);

    enum Devtype:
        DevCPU
        DevGPU

    cppclass Array(Object):
        Array(int, size_t*, Dtype, Devtype)
        int ndim
        Dtype dtype
        Devtype devtype
        size_t* shape
        void* data
        size_t stride
        bint ownsdata    

    cppclass Tuple(Object):
        Tuple(size_t)
        void setitem(int, Object*)
        Object* getitem(int)
        int len
        Object** members        

    size_t cgt_size(const Array* a)
    int cgt_itemsize(Dtype dtype)
    size_t cgt_nbytes(const Array* a)

    bint cgt_is_array(Object*)
    bint cgt_is_tuple(Object*)

    void* cgt_alloc(Devtype devtype, size_t)    
    void cgt_free(Devtype devtype, void* ptr)
    void cgt_memcpy(Devtype dest_type, Devtype src_type, void* dest_ptr, void* src_ptr, size_t nbytes)


# Conversion funcs
# ----------------------------------------------------------------

ctypedef cnp.Py_intptr_t npy_intp_t


cdef object cgt2py_object(Object* o):
    if cgt_is_array(o): return cgt2py_array(<Array*>o)
    elif cgt_is_tuple(o): return cgt2py_tuple(<Tuple*>o)
    else: raise RuntimeError("cgt object seems to be invalid")

cdef object cgt2py_array(Array* a):
    cdef cnp.ndarray nparr = cnp.PyArray_SimpleNew(a.ndim, <npy_intp_t*>a.shape, a.dtype) # XXX DANGEROUS CAST
    cgt_memcpy(DevCPU, a.devtype, cnp.PyArray_DATA(nparr), a.data, cnp.PyArray_NBYTES(nparr))
    return nparr

cdef object cgt2py_tuple(Tuple* t):
    cdef int i
    out = cpython.PyTuple_New(t.len)
    for i in xrange(t.len):
        cpython.PyTuple_SetItem(out, i, cgt2py_object(t.getitem(i)))
    return out

cdef Object* py2cgt_object(object o):
    if isinstance(o, np.ndarray):
        return py2Array(o)
    elif isinstance(o, tuple):
        return py2Tuple(o)
    else:
        raise RuntimeError("only can convert ndarray and tuple to cgt datatype")

cdef Array* py2Array(cnp.ndarray arr):
    cdef Array* out = new Array(arr.ndim, <size_t*>arr.shape, dtype_fromstr(arr.dtype), DevCPU)
    cgt_memcpy(out.devtype, DevCPU, out.data, cnp.PyArray_DATA(arr), cgt_nbytes(out))
    return out

cdef Tuple* py2Tuple(object o):
    cdef Tuple* out = new Tuple(len(o))
    cdef int i
    for i in xrange(len(o)):
        out.setitem(i, py2cgt_object(o[i]))
    return out


cdef Dtype dtype_fromstr(s):
    if s=='i1':
        return cgt_i1
    elif s=='i2':
        return cgt_i2
    elif s=='i4':
        return cgt_i4
    elif s=='i8':
        return cgt_i8
    elif s=='f2':
        return cgt_f2
    elif s=='f4':
        return cgt_f4
    elif s=='f8':
        return cgt_f8
    elif s=='f16':
        return cgt_f16
    elif s=='c8':
        return cgt_c8
    elif s=='c16':
        return cgt_c16
    elif s=='c32':
        return cgt_c32
    elif s == 'O':
        return cgt_O
    else:
        raise ValueError("unrecognized dtype %s"%s)

cdef object dtype_tostr(Dtype d):
    if d == cgt_i1:
        return 'i1'
    elif d == cgt_i2:
        return 'i2'
    elif d == cgt_i4:
        return 'i4'
    elif d == cgt_i8:
        return 'i8'
    elif d == cgt_f4:
        return 'f4'
    elif d == cgt_f8:
        return 'f8'
    elif d == cgt_f16:
        return 'f16'
    elif d == cgt_c8:
        return 'c8'
    elif d == cgt_c16:
        return 'c16'
    elif d == cgt_c32:
        return 'c32'
    elif d == cgt_O:
        return 'obj'
    else:
        raise ValueError("invalid Dtype")

cdef object devtype_tostr(Devtype d):
    if d == DevCPU:
        return "cpu"
    elif d == DevGPU:
        return "gpu"
    else:
        raise RuntimeError

cdef Devtype devtype_fromstr(object s):
    if s == "cpu":
        return DevCPU
    elif s == "gpu":
        return DevGPU
    else:
        raise ValueError("unrecognized devtype %s"%s)


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
        LIB_DIRS = [".cgt/build/lib"]

cdef void* get_or_load_lib(libname) except *:
    cdef void* handle
    initialize_lib_dirs()
    if libname in LIB_HANDLES:
        # print "already loaded",libname
        return <void*><size_t>LIB_HANDLES[libname]
    else:
        # print "loading",libname
        for ld in LIB_DIRS:
            libpath = osp.join(ld,libname)
            if osp.exists(libpath):
                handle = dlopen(libpath, RTLD_NOW | RTLD_GLOBAL)
            else:
                raise IOError("tried to load non-existent library %s"%libpath)
        if handle == NULL:
            raise ValueError("couldn't load library named %s: %s"%(libname, <bytes>dlerror()))
        else:
            LIB_HANDLES[libname] = <object><size_t>handle
        return handle


################################################################
### Execution graph 
################################################################
 
cdef extern from "execution.h" namespace "cgt":
    cppclass InPlaceFun:
        InPlaceFun(Inplacefun, void*)
        InPlaceFun()
    cppclass ValRetFun:
        ValRetFun(Valretfun, void*)
        ValRetFun()
    cppclass MemLocation:
        MemLocation(size_t)
        MemLocation()
    cppclass Instruction:
        pass
    cppclass ExecutionGraph:
        void add_instr(Instruction*)
        Object* get(MemLocation)
        ExecutionGraph(int, int, vector[MemLocation])        
    cppclass LoadArgument(Instruction):
        LoadArgument(int, MemLocation)
    cppclass Alloc(Instruction):
        Alloc(Dtype, vector[MemLocation], MemLocation)
    cppclass InPlace(Instruction):
        InPlace(vector[MemLocation], MemLocation, InPlaceFun)
    cppclass ValReturning(Instruction):
        ValReturning(vector[MemLocation], MemLocation, ValRetFun)

    cppclass Interpreter:
        Tuple* run(Tuple*)

    Interpreter* create_interpreter(ExecutionGraph*)

# Conversion funcs
# ----------------------------------------------------------------

cdef vector[size_t] _tovectorlong(object xs):
    cdef vector[size_t] out = vector[size_t]()
    for x in xs: out.push_back(<size_t>x)
    return out

cdef object _cgtarray2py(Array* a):
    raise NotImplementedError

cdef object _cgttuple2py(Tuple* t):
    raise NotImplementedError

cdef object _cgtobj2py(Object* o):
    if cgt_is_array(o):
        return _cgtarray2py(<Array*>o)
    else:
        return _cgttuple2py(<Tuple*>o)

cdef void* _ctypesstructptr(object o) except *:
    if o is None: return NULL
    else: return <void*><size_t>ctypes.cast(ctypes.pointer(o), ctypes.c_voidp).value    

asdf = []

cdef void* _getfun(libname, funcname) except *:
    cdef void* lib_handle = get_or_load_lib(libname)
    cdef void* out = dlsym(lib_handle, funcname)
    if out == NULL:
        raise RuntimeError("couldn't load function %s from %s. maybe you forgot extern C"%(libname, funcname))
    return out


cdef InPlaceFun _node2inplaceclosure(node) except *:
    libname, funcname, cldata = cgt.get_impl(node, "cpu") # TODO
    cfun = _getfun(libname, funcname)
    asdf.append(cldata)  # XXX
    return InPlaceFun(<Inplacefun>cfun, _ctypesstructptr(cldata))

cdef ValRetFun _node2valretclosure(node) except *:
    libname, funcname, cldata = cgt.get_impl(node, "cpu") # TODO
    cfun = _getfun(libname, funcname)
    asdf.append(cldata)  # XXX
    return ValRetFun(<Valretfun>cfun, _ctypesstructptr(cldata))

cdef MemLocation _tocppmem(object pymem):
    return MemLocation(<size_t>pymem.index)

cdef vector[MemLocation] _tocppmemvec(object pymemlist) except *:
    cdef vector[MemLocation] out = vector[MemLocation]()
    for pymem in pymemlist:
        out.push_back(_tocppmem(pymem))
    return out

cdef Instruction* _tocppinstr(object pyinstr) except *:
    t = type(pyinstr)
    cdef Instruction* out
    if t == cgt.LoadArgument:
        out = new LoadArgument(pyinstr.ind, _tocppmem(pyinstr.write_loc))
    elif t == cgt.Alloc:
        out = new Alloc(dtype_fromstr(pyinstr.dtype), _tocppmemvec(pyinstr.read_locs), _tocppmem(pyinstr.write_loc))
    elif t == cgt.InPlace:
        out = new InPlace(_tocppmemvec(pyinstr.read_locs), _tocppmem(pyinstr.write_loc), _node2inplaceclosure(pyinstr.node))
    elif t == cgt.ValReturning:
        out = new ValReturning(_tocppmemvec(pyinstr.read_locs), _tocppmem(pyinstr.write_loc),_node2valretclosure(pyinstr.node))
    else:
        raise RuntimeError("expected instance of type Instruction. got type %s"%t)
    return out

################################################################
### Wrapper classes
################################################################

cdef ExecutionGraph* create_execution_graph(pyeg) except *:
    "create an execution graph object"
    cdef ExecutionGraph* eg = new ExecutionGraph(pyeg.n_args, pyeg.n_locs(), _tocppmemvec(pyeg.output_locs))
    for instr in pyeg.instrs:
        eg.add_instr(_tocppinstr(instr))
    return eg

cdef class cInterpreter:
    cdef ExecutionGraph* eg
    cdef Interpreter* interp
    def __init__(self, pyeg):
        self.eg = create_execution_graph(pyeg)
        self.interp = create_interpreter(self.eg)
    def __dealloc__(self):
        if self.interp != NULL: del self.interp
        if self.eg != NULL: del self.eg
    def __call__(self, pyargs):
        cdef Tuple* cargs = new Tuple(len(pyargs))
        cdef vector[size_t] shp
        for (i,pyarg) in enumerate(pyargs):
            cargs.setitem(i, py2cgt_object(pyarg))
        cdef Tuple* ret = self.interp.run(cargs)
        del cargs
        return cgt2py_object(ret.getitem(0)) # XXX


