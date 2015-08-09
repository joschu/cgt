from cpython.ref cimport PyObject
cimport numpy as cnp
cimport cpython
from libc.stdlib cimport abort
from libcpp.string cimport string
from libcpp.vector cimport vector

import numpy as np
import ctypes
import os.path as osp
import traceback


import cgt
from cgt import core, execution, impls

cnp.import_array()

################################################################
### CGT common datatypes 
################################################################


cdef extern from "cgt_common.h":

    cppclass IRC[T]:
        pass

    cppclass cgtObject:
        pass

    ctypedef void (*cgtByRefFun)(void*, cgtObject**, cgtObject*)
    ctypedef cgtObject* (*cgtByValFun)(void*, cgtObject**)

    enum cgtDevtype:
        cgtCPU
        cgtGPU

    cppclass cgtArray(cgtObject):
        cgtArray(size_t, const size_t*, cgtDtype, cgtDevtype)
        cgtArray(size_t, const size_t*, cgtDtype, cgtDevtype, void* fromdata, bint copy)
        size_t ndim() const
        const size_t* shape() const
        size_t size()
        size_t nbytes() const
        size_t stride(size_t)
        cgtDtype dtype() const
        cgtDevtype devtype() const
        bint ownsdata() const
        void* data()

    cppclass cgtTuple(cgtObject):
        cgtTuple(size_t)
        void setitem(int, cgtObject*)
        cgtObject* getitem(int)
        size_t size()
        size_t len
        cgtObject** members        


    cdef enum cgtDtype:
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

    bint cgt_is_array(cgtObject*)
    bint cgt_is_tuple(cgtObject*)

    void* cgt_alloc(cgtDevtype devtype, size_t)    
    void cgt_free(cgtDevtype devtype, void* ptr)
    void cgt_memcpy(cgtDevtype dest_type, cgtDevtype src_type, void* dest_ptr, void* src_ptr, size_t nbytes)


# Conversion funcs
# ----------------------------------------------------------------

ctypedef cnp.Py_intptr_t npy_intp_t

cdef object cgt2py_object(cgtObject* o):
    if cgt_is_array(o): return cgt2py_array(<cgtArray*>o)
    elif cgt_is_tuple(o): return cgt2py_tuple(<cgtTuple*>o)
    else: raise RuntimeError("cgt object seems to be invalid")

cdef object cgt2py_array(cgtArray* a):
    cdef cnp.ndarray nparr = cnp.PyArray_SimpleNew(a.ndim(), <npy_intp_t*>a.shape(), a.dtype()) # XXX DANGEROUS CAST
    cgt_memcpy(cgtCPU, a.devtype(), cnp.PyArray_DATA(nparr), a.data(), cnp.PyArray_NBYTES(nparr))
    return nparr

cdef object cgt2py_tuple(cgtTuple* t):
    cdef int i
    return tuple(cgt2py_object(t.getitem(i)) for i in xrange(t.len))
    # why doesn't the following work:
    # out = cpython.PyTuple_New(t.len)
    # for i in xrange(t.len):
    #     cpython.PyTuple_SetItem(out, i, cgt2py_object(t.getitem(i)))
    # return out

cdef cnp.ndarray _to_valid_array(object arr):
    cdef cnp.ndarray out = np.asarray(arr, order='C')
    if not out.flags.c_contiguous: 
        out = out.copy()
    return out

cdef cgtObject* py2cgt_object(object o) except *:
    if isinstance(o, tuple):
        return py2cgt_tuple(o)
    else:
        o = _to_valid_array(o)
        return py2cgt_array(o, cgtCPU)

cdef cgtArray* py2cgt_array(cnp.ndarray arr, cgtDevtype devtype):
    cdef cgtArray* out = new cgtArray(arr.ndim, <size_t*>arr.shape, arr.dtype.num, devtype)
    if not arr.flags.c_contiguous: arr = np.ascontiguousarray(arr)
    cgt_memcpy(out.devtype(), cgtCPU, out.data(), cnp.PyArray_DATA(arr), out.nbytes())
    return out

cdef cgtArray* py2cgt_arrayview(cnp.ndarray arr):
    cdef cgtArray* out = new cgtArray(arr.ndim, <size_t*>arr.shape, arr.dtype.num, cgtCPU, cnp.PyArray_DATA(arr), False)
    assert arr.flags.c_contiguous
    return out

cdef cgtTuple* py2cgt_tuple(object o):
    cdef cgtTuple* out = new cgtTuple(len(o))
    cdef int i
    for i in xrange(len(o)):
        out.setitem(i, py2cgt_object(o[i]))
    return out


cdef cgtDtype dtype_fromstr(s):
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

cdef object dtype_tostr(cgtDtype d):
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
        raise ValueError("invalid cgtDtype")

cdef object devtype_tostr(cgtDevtype d):
    if d == cgtCPU:
        return "cpu"
    elif d == cgtGPU:
        return "gpu"
    else:
        raise RuntimeError

cdef cgtDevtype devtype_fromstr(object s):
    if s == "cpu":
        return cgtCPU
    elif s == "gpu":
        return cgtGPU
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

cdef void* get_or_load_lib(libname) except NULL:
    cdef void* handle
    initialize_lib_dirs()
    if libname in LIB_HANDLES:
        return <void*><size_t>LIB_HANDLES[libname]
    else:
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
    cppclass ByRefFunCl:
        ByRefFunCl(cgtByRefFun, void*)
        ByRefFunCl()
    cppclass ByValFunCl:
        ByValFunCl(cgtByValFun, void*)
        ByValFunCl()
    cppclass MemLocation:
        MemLocation(size_t)
        MemLocation()
    cppclass Instruction:
        pass
    cppclass ExecutionGraph:
        ExecutionGraph(vector[Instruction*], int, int)        
        int n_args()
    cppclass LoadArgument(Instruction):
        LoadArgument(string, int, MemLocation)
    cppclass Alloc(Instruction):
        Alloc(string, cgtDtype, vector[MemLocation], MemLocation)
    cppclass BuildTup(Instruction):
        BuildTup(string, vector[MemLocation], MemLocation)
    cppclass ReturnByRef(Instruction):
        ReturnByRef(const string&, vector[MemLocation], MemLocation, ByRefFunCl)
    cppclass ReturnByVal(Instruction):
        ReturnByVal(const string&, vector[MemLocation], MemLocation, ByValFunCl)

    cppclass Interpreter:
        cgtTuple* run(cgtTuple*)

    Interpreter* create_interpreter(ExecutionGraph*, vector[MemLocation], bint)

cdef vector[size_t] _tovectorlong(object xs):
    cdef vector[size_t] out = vector[size_t]()
    for x in xs: out.push_back(<size_t>x)
    return out

cdef void* _ctypesstructptr(object o) except NULL:
    if o is None: return NULL
    else: return <void*><size_t>ctypes.cast(ctypes.pointer(o), ctypes.c_voidp).value    

# TODO inplace op can operate on views of data
cdef void _pyfunc_inplace(void* cldata, cgtObject** reads, cgtObject* write):
    (pyfun, nin, nout) = <object>cldata
    pyread = [cgt2py_object(reads[i]) for i in xrange(nin)]
    pywrite = cgt2py_object(write)
    try:
        pyfun(pyread, pywrite)
    except Exception:
        traceback.print_exc()
        abort()
    cdef cgtTuple* tup
    cdef cgtArray* a
    if cgt_is_array(write):
        npout = <cnp.ndarray>pywrite
        cgt_memcpy(cgtCPU, cgtCPU, (<cgtArray*>write).data(), npout.data, (<cgtArray*>write).nbytes())
    else:
        tup = <cgtTuple*> write
        for i in xrange(tup.size()):
            npout = <cnp.ndarray>pywrite[i]
            a = <cgtArray*>tup.getitem(i)
            assert cgt_is_array(a)
            cgt_memcpy(cgtCPU, cgtCPU, a.data(), npout.data, a.nbytes())


cdef cgtObject* _pyfunc_valret(void* cldata, cgtObject** args):
    (pyfun, nin, nout) = <object>cldata
    pyread = [cgt2py_object(args[i]) for i in xrange(nin)]
    try:
        pyout = pyfun(pyread)
    except Exception:
        traceback.print_exc()
        abort()
    return py2cgt_object(pyout)


shit2 = [] # XXX this is a memory leak, will fix later

cdef void* _getfun(libname, funcname) except *:
    cdef void* lib_handle = get_or_load_lib(libname)
    cdef void* out = dlsym(lib_handle, funcname)
    if out == NULL:
        raise RuntimeError("couldn't load function %s from %s. maybe you forgot extern C"%(libname, funcname))
    return out


cdef ByRefFunCl _node2inplaceclosure(oplib, node) except *:
    binary = oplib.fetch_binary(node, devtype="cpu")
    if binary.pyimpl is not None:
        assert binary.pyimpl.valret_func is None
        pyfun = binary.pyimpl.inplace_func        
        py_closure = (pyfun, len(node.parents), 1)
        shit2.append(py_closure)
        return ByRefFunCl(&_pyfunc_inplace, <PyObject*>py_closure)
    else:
        cfun = _getfun(binary.c_libpath, binary.c_funcname)
        closure = oplib.fetch_closure(node)
        shit2.append(closure)  # XXX
        return ByRefFunCl(<cgtByRefFun>cfun, _ctypesstructptr(closure))

cdef ByValFunCl _node2valretclosure(oplib, node) except *:
    binary = oplib.fetch_binary(node, devtype="cpu")
    if binary.pyimpl is not None:
        assert binary.pyimpl.inplace_func is None
        pyfun = binary.pyimpl.valret_func        
        py_closure = (pyfun, len(node.parents), 1)
        shit2.append(py_closure)
        return ByValFunCl(&_pyfunc_valret, <PyObject*>py_closure)
    else:
        cfun = _getfun(binary.c_libpath, binary.c_funcname)
        closure = oplib.fetch_closure(node)
        shit2.append(closure)  # XXX
        return ByValFunCl(<cgtByValFun>cfun, _ctypesstructptr(closure))

cdef MemLocation _tocppmem(object pymem):
    return MemLocation(<size_t>pymem.index)

cdef vector[MemLocation] _tocppmemvec(object pymemlist) except *:
    cdef vector[MemLocation] out = vector[MemLocation]()
    for pymem in pymemlist:
        out.push_back(_tocppmem(pymem))
    return out

cdef Instruction* _tocppinstr(object oplib, object pyinstr) except *:
    t = type(pyinstr)
    cdef Instruction* out
    if t == execution.LoadArgument:
        out = new LoadArgument(repr(pyinstr), pyinstr.ind, _tocppmem(pyinstr.write_loc))
    elif t == execution.Alloc:
        out = new Alloc(repr(pyinstr), dtype_fromstr(pyinstr.dtype), _tocppmemvec(pyinstr.read_locs), _tocppmem(pyinstr.write_loc))
    elif t == execution.BuildTup:
        out = new BuildTup(repr(pyinstr), _tocppmemvec(pyinstr.read_locs), _tocppmem(pyinstr.write_loc))
    elif t == execution.ReturnByRef:
        out = new ReturnByRef(repr(pyinstr), _tocppmemvec(pyinstr.read_locs), _tocppmem(pyinstr.write_loc), _node2inplaceclosure(oplib, pyinstr.node))
    elif t == execution.ReturnByVal:
        out = new ReturnByVal(repr(pyinstr), _tocppmemvec(pyinstr.read_locs), _tocppmem(pyinstr.write_loc),_node2valretclosure(oplib, pyinstr.node))
    else:
        raise RuntimeError("expected instance of type Instruction. got type %s"%t)
    return out

################################################################
### Wrapper classes
################################################################

cdef ExecutionGraph* make_cpp_execution_graph(pyeg, oplib) except *:
    "make an execution graph object"
    cdef vector[Instruction*] instrs
    for instr in pyeg.instrs:
        instrs.push_back(_tocppinstr(oplib, instr))
    return new ExecutionGraph(instrs,pyeg.n_args, pyeg.n_locs)

cdef class CppArrayWrapper:
    cdef cgtArray* arr
    def to_numpy(self):
        return cnp.PyArray_SimpleNewFromData(self.arr.ndim(), <cnp.npy_intp*>self.arr.shape(), self.arr.dtype(), self.arr.data())
    @staticmethod
    def from_numpy(cnp.ndarray nparr, object pydevtype="cpu"):
        cdef cgtDevtype devtype = devtype_fromstr(pydevtype)
        out = CppArrayWrapper()
        out.arr = py2cgt_array(nparr, devtype)
        return out
    @property
    def ndim(self):
        return self.arr.ndim()
    @property
    def shape(self):
        return [self.arr.shape()[i] for i in range(self.arr.ndim())]
    @property
    def dtype(self):
        return dtype_tostr(self.arr.dtype())
    def get_pointer(self):
        return <size_t>self.arr
    def __dealloc__(self):
        del self.arr


cdef class CppInterpreterWrapper:
    """
    Convert python inputs to C++
    Run interpreter on execution graph
    Then grab the outputs
    """
    cdef ExecutionGraph* eg # owned
    cdef Interpreter* interp # owned
    cdef object input_types
    def __init__(self, pyeg, oplib, input_types, output_locs, parallel):
        self.eg = make_cpp_execution_graph(pyeg, oplib)
        cdef vector[MemLocation] cpp_output_locs = _tocppmemvec(output_locs)
        self.interp = create_interpreter(self.eg, cpp_output_locs, parallel)
        self.input_types = input_types
    def __dealloc__(self):
        if self.interp != NULL: del self.interp
        if self.eg != NULL: del self.eg
    def __call__(self, *pyargs):
        pyargs = tuple(execution.as_valid_arg(arg) for arg in pyargs)
        execution.typecheck_args(pyargs, self.input_types)

        # TODO: much better type checking on inputs
        cdef cgtTuple* cargs = new cgtTuple(len(pyargs))
        for (i,pyarg) in enumerate(pyargs):
            cargs.setitem(i, py2cgt_object(pyarg))
        cdef cgtTuple* ret = self.interp.run(cargs)
        del cargs
        return list(cgt2py_object(ret))

