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

    cppclass IRC[T]:
        pass

    ctypedef void (*cgt_inplacefun)(void*, cgt_object**);
    ctypedef cgt_object* (*cgt_valretfun)(void*, cgt_object**);

    enum cgt_devtype:
        cgt_cpu
        cgt_gpu

    enum cgt_typetag:
        cgt_undef
        cgt_tupletype
        cgt_arraytype
    
    cppclass cgt_object:
        pass

    cppclass cgt_array(cgt_object):
        cgt_array(int, size_t*, cgt_dtype, cgt_devtype)
        int ndim
        cgt_dtype dtype
        cgt_devtype devtype
        size_t* shape
        void* data
        size_t stride
        bint ownsdata    

    cppclass cgt_tuple(cgt_object):
        cgt_tuple(size_t)
        void setitem(int, cgt_object*)
        cgt_object* getitem(int)
        cgt_typetag typetag
        int len
        cgt_object** members        

    struct cgt_object:
        cgt_typetag typetag

    size_t cgt_size(const cgt_array* a)
    int cgt_itemsize(cgt_dtype dtype)
    size_t cgt_nbytes(const cgt_array* a)

    cgt_typetag cgt_type(cgt_object*)
    bint cgt_is_array(cgt_object*)
    bint cgt_is_tuple(cgt_object*)

    void* cgt_alloc(cgt_devtype devtype, size_t size)    
    void cgt_free(cgt_devtype devtype, void* ptr)
    void cgt_memcpy(cgt_devtype dest_type, cgt_devtype src_type, void* dest_ptr, void* src_ptr, size_t nbytes)


# Conversion funcs
# ----------------------------------------------------------------

ctypedef cnp.Py_intptr_t npy_intp_t


cdef object cgt2py_object(cgt_object* o):
    if cgt_is_array(o): return cgt2py_array(<cgt_array*>o)
    elif cgt_is_tuple(o): return cgt2py_tuple(<cgt_tuple*>o)
    else: raise RuntimeError("cgt object seems to be invalid")

cdef object cgt2py_array(cgt_array* a):
    cdef cnp.ndarray nparr = cnp.PyArray_SimpleNew(a.ndim, <npy_intp_t*>a.shape, a.dtype) # XXX DANGEROUS CAST
    cgt_memcpy(cgt_cpu, a.devtype, cnp.PyArray_DATA(nparr), a.data, cnp.PyArray_NBYTES(nparr))
    return nparr

cdef object cgt2py_tuple(cgt_tuple* t):
    cdef int i
    out = cpython.PyTuple_New(t.len)
    for i in xrange(t.len):
        cpython.PyTuple_SetItem(out, i, cgt2py_object(t.getitem(i)))
    return out

cdef cgt_object* py2cgt_object(object o):
    if isinstance(o, np.ndarray):
        return py2cgt_array(o)
    elif isinstance(o, tuple):
        return py2cgt_tuple(o)
    else:
        raise RuntimeError("only can convert ndarray and tuple to cgt datatype")

cdef cgt_array* py2cgt_array(cnp.ndarray arr):
    cdef cgt_array* out = new cgt_array(arr.ndim, <size_t*>arr.shape, dtype_fromstr(arr.dtype), cgt_cpu)
    cgt_memcpy(out.devtype, cgt_cpu, out.data, cnp.PyArray_DATA(arr), cgt_nbytes(out))
    return out

cdef cgt_tuple* py2cgt_tuple(object o):
    cdef cgt_tuple* out = new cgt_tuple(len(o))
    cdef int i
    for i in xrange(len(o)):
        out.setitem(i, py2cgt_object(o[i]))
    return out


cdef cgt_dtype dtype_fromstr(s):
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

cdef object dtype_tostr(cgt_dtype d):
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
        raise ValueError("invalid cgt_dtype")

cdef object devtype_tostr(cgt_devtype d):
    if d == cgt_cpu:
        return "cpu"
    elif d == cgt_gpu:
        return "gpu"
    else:
        raise RuntimeError

cdef cgt_devtype devtype_fromstr(object s):
    if s == "cpu":
        return cgt_cpu
    elif s == "gpu":
        return cgt_gpu
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

cdef void* get_or_load_lib(libname):
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
        InPlaceFun(cgt_inplacefun, void*)
        InPlaceFun()
    cppclass ValRetFun:
        ValRetFun(cgt_valretfun, void*)
        ValRetFun()
    cppclass MemLocation:
        MemLocation(size_t)
        MemLocation()
    cppclass Instruction:
        pass
    cppclass ExecutionGraph:
        void add_instr(Instruction*)
        cgt_object* get(MemLocation)
        ExecutionGraph(int, int, vector[MemLocation])        
    cppclass LoadArgument(Instruction):
        LoadArgument(int, MemLocation)
    cppclass Alloc(Instruction):
        Alloc(cgt_dtype, vector[MemLocation], MemLocation)
    cppclass InPlace(Instruction):
        InPlace(vector[MemLocation], MemLocation, InPlaceFun)
    cppclass ValReturning(Instruction):
        ValReturning(vector[MemLocation], MemLocation, ValRetFun)

    cppclass Interpreter:
        cgt_tuple* run(cgt_tuple*)

    Interpreter* create_interpreter(ExecutionGraph*)

# Conversion funcs
# ----------------------------------------------------------------

cdef vector[size_t] _tovectorlong(object xs):
    cdef vector[size_t] out = vector[size_t]()
    for x in xs: out.push_back(<size_t>x)
    return out

cdef object _cgtarray2py(cgt_array* a):
    raise NotImplementedError

cdef object _cgttuple2py(cgt_tuple* t):
    raise NotImplementedError

cdef object _cgtobj2py(cgt_object* o):
    if cgt_is_array(o):
        return _cgtarray2py(<cgt_array*>o)
    else:
        return _cgttuple2py(<cgt_tuple*>o)

cdef void* _ctypesstructptr(object o) except *:
    if o is None: return NULL
    else: return <void*><size_t>ctypes.cast(ctypes.pointer(o), ctypes.c_voidp).value    

asdf = []

cdef void* _getfun(libname, funcname) except *:
    cdef void* lib_handle = get_or_load_lib(libname)
    cdef void* out = dlsym(lib_handle, funcname)
    assert out != NULL, "couldn't load function. maybe you forgot extern C"
    return out


cdef InPlaceFun _node2inplaceclosure(node) except *:
    libname, funcname, cldata = cgt.get_impl(node, "cpu") # TODO
    cfun = _getfun(libname, funcname)
    asdf.append(cldata)  # XXX
    return InPlaceFun(<cgt_inplacefun>cfun, _ctypesstructptr(cldata))

cdef ValRetFun _node2valretclosure(node) except *:
    libname, funcname, cldata = cgt.get_impl(node, "cpu") # TODO
    cfun = _getfun(libname, funcname)
    asdf.append(cldata)  # XXX
    return ValRetFun(<cgt_valretfun>cfun, _ctypesstructptr(cldata))

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
 
cdef class cArray:
    cdef cgt_array* array
    def to_numpy(self):
        pass # TODO

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
        cdef cgt_tuple* cargs = new cgt_tuple(len(pyargs))
        cdef vector[size_t] shp
        for (i,pyarg) in enumerate(pyargs):
            cargs.setitem(i, py2cgt_object(pyarg))
        cdef cgt_tuple* ret = self.interp.run(cargs)
        del cargs
        return cgt2py_object(ret.getitem(0)) # XXX


