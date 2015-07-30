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

    ctypedef void (*cgt_fun)(void*, cgt_array**);

    cdef enum cgt_devtype:
        cgt_cpu
        cgt_gpu

    cdef enum cgt_typetag:
        cgt_tupletype
        cgt_arraytype
    
    struct cgt_object:
        cgt_typetag typetag

    struct cgt_array:
        cgt_typetag typetag
        int ndim
        cgt_dtype dtype
        cgt_devtype devtype
        size_t* shape
        void* data
        size_t stride
        bint ownsdata    

    struct cgt_tuple:
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

    cgt_array* new_cgt_array(int ndim, size_t* shape, cgt_dtype, cgt_devtype);
    void delete_cgt_array(cgt_array*);

    cgt_tuple* new_cgt_tuple(int len)
    void delete_cgt_tuple(cgt_tuple*)

    void* cgt_alloc(cgt_devtype devtype, size_t size)    
    void cgt_free(cgt_devtype devtype, void* ptr)
    void cgt_memcpy(cgt_devtype dest_type, cgt_devtype src_type, void* dest_ptr, void* src_ptr, size_t nbytes)


# Conversion funcs
# ----------------------------------------------------------------

ctypedef cnp.Py_intptr_t npy_intp_t


cdef object cgt_object_to_python(cgt_object* o):
    if cgt_is_array(o): return cgt_array_to_python(<cgt_array*>o)
    elif cgt_is_tuple(o): return cgt_tuple_to_python(<cgt_tuple*>o)
    else: raise RuntimeError


cdef object cgt_array_to_python(cgt_array* a):
    cdef cnp.ndarray nparr = cnp.PyArray_SimpleNew(a.ndim, <npy_intp_t*>a.shape, a.dtype) # XXX DANGEROUS CAST
    cgt_memcpy(cgt_cpu, a.devtype, cnp.PyArray_DATA(nparr), a.data, cnp.PyArray_NBYTES(nparr))
    return nparr

cdef object cgt_tuple_to_python(cgt_tuple* t):
    cdef int i
    out = cpython.PyTuple_New(t.len)
    for i in xrange(t.len):
        cpython.PyTuple_SetItem(out, i, cgt_object_to_python(t.members[i]))
    return out

cdef void nparr_to_cgt_array(cgt_array* a, cnp.ndarray nparr):
    assert nparr.ndim == a.ndim and nparr.dtype.num == a.dtype and nparr.flags.c_contiguous
    if nparr.dtype.num != a.dtype:
        nparr = nparr.astype(dtype_tostr(a.dtype))
    # TODO: break this down
    cdef int i
    assert a.ndim == nparr.ndim, "ndimsame"
    for i in xrange(a.ndim): assert a.shape[i] == nparr.shape[i],"shapesame%i,%s,%s"%(i,(<object>nparr).shape, (a.shape[0], a.shape[1]))
    
    assert a.data != NULL or cgt_nbytes(a) == 0,"notnull"
    cgt_memcpy(a.devtype, cgt_cpu, a.data, cnp.PyArray_DATA(nparr), cgt_nbytes(a))

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
        import cycgt
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
    cppclass FuncClosure:
        FuncClosure(cgt_fun, void*)
        FuncClosure()
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
        InPlace(vector[MemLocation], MemLocation, FuncClosure)
    cppclass ValReturning(Instruction):
        ValReturning(vector[MemLocation], MemLocation, FuncClosure)

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
    tag = cgt_type(o)
    if tag == cgt_tupletype:
        return _cgttuple2py(<cgt_tuple*>o)
    elif tag == cgt_arraytype:
        return _cgtarray2py(<cgt_array*>o)
    else:
        raise RuntimeError("invalid type tag. something's very wrong")

cdef void* _ctypesstructptr(object o) except *:
    if o is None: return NULL
    else: return <void*><size_t>ctypes.cast(ctypes.pointer(o), ctypes.c_voidp).value    

asdf = []

cdef FuncClosure _node2closure(node) except *:
    libname, funcname, cldata = cgt.get_impl(node, "cpu") # TODO
    asdf.append(cldata)  # XXX
    cdef void* lib_handle = get_or_load_lib(libname)
    cdef cgt_fun cfun = <cgt_fun>dlsym(lib_handle, funcname)
    return FuncClosure(cfun, _ctypesstructptr(cldata))

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
        out = new InPlace(_tocppmemvec(pyinstr.read_locs), _tocppmem(pyinstr.write_loc), _node2closure(pyinstr.node))
    elif t == cgt.ValReturning:
        out = new ValReturning(_tocppmemvec(pyinstr.read_locs), _tocppmem(pyinstr.write_loc),_node2closure(pyinstr.node))
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
    def __call__(self, arrs):
        cdef cgt_tuple* args = new_cgt_tuple(len(arrs))
        cdef vector[size_t] shp
        for (i,arr) in enumerate(arrs):
            shp = _tovectorlong(arr.shape)
            args.members[i] = <cgt_object*>new_cgt_array(arr.ndim, shp.data(), dtype_fromstr(arr.dtype), cgt_cpu)
        cdef cgt_tuple* ret = self.interp.run(args)
        delete_cgt_tuple(args)
        out = cgt_array_to_python(<cgt_array*>ret.members[0])
        return out


