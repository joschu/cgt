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
from cgt import core, compilation

cnp.import_array()

# TODO use smart pointers instead of returning cgtObject*

################################################################
### CGT common datatypes 
################################################################


cdef extern from "cgt_common.h":

    cppclass IRC[T]:
        IRC()
        IRC(T*)
        T* get()

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
        int ndim() const
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

cdef object cgt2py_object(cgtObject* o, bint view):
    if cgt_is_array(o): return cgt2py_array(<cgtArray*>o, view)
    elif cgt_is_tuple(o): return cgt2py_tuple(<cgtTuple*>o, view)
    else: raise RuntimeError("cgt object seems to be invalid")

cdef object cgt2py_array(cgtArray* a, bint view):
    cdef cnp.ndarray nparr
    if view:
        return cnp.PyArray_SimpleNewFromData(a.ndim(), <cnp.npy_intp*>a.shape(), a.dtype(), a.data())
    else:
        nparr = cnp.PyArray_SimpleNew(a.ndim(), <npy_intp_t*>a.shape(), a.dtype())
        cgt_memcpy(cgtCPU, a.devtype(), cnp.PyArray_DATA(nparr), a.data(), cnp.PyArray_NBYTES(nparr))
        return nparr

cdef object cgt2py_tuple(cgtTuple* t, bint view):
    cdef int i
    return tuple(cgt2py_object(t.getitem(i), view) for i in xrange(t.len))
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

cdef bint _is_valid_array(cnp.ndarray arr):
    return arr.flags.c_contiguous


cdef cgtObject* py2cgt_object(object o, bint view) except *:
    if isinstance(o, tuple):
        return py2cgt_tuple(o, view)
    else:
        if view and o.flags.c_contiguous:  
        # TODO add a warning if not contiguous
        # Doing a copy here could cause wrong behavior for inplace operation
            return py2cgt_arrayview(o)
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

cdef cgtTuple* py2cgt_tuple(object o, bint view):
    cdef cgtTuple* out = new cgtTuple(len(o))
    cdef int i
    for i in xrange(len(o)):
        out.setitem(i, py2cgt_object(o[i], view))
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
### Execution graph 
################################################################
 
cdef extern from "execution.h" namespace "cgt":
    cppclass ByRefCallable:
        ByRefCallable(cgtByRefFun, void*)
        ByRefCallable()
    cppclass ByValCallable:
        ByValCallable(cgtByValFun, void*)
        ByValCallable()
    cppclass MemLocation:
        MemLocation()
        MemLocation(size_t, cgtDevtype)
        size_t index()
        cgtDevtype devtype()
    cppclass Instruction:
        pass
    cppclass ExecutionGraph:
        ExecutionGraph(vector[Instruction*], int, int)        
        int n_args()
    cppclass LoadArgument(Instruction):
        LoadArgument(const string&, int, const MemLocation&)
    cppclass Alloc(Instruction):
        Alloc(const string&, cgtDtype, vector[MemLocation], const MemLocation&)
    cppclass BuildTup(Instruction):
        BuildTup(const string&, vector[MemLocation], const MemLocation&)
    cppclass ReturnByRef(Instruction):
        ReturnByRef(const string&, vector[MemLocation], const MemLocation&, ByRefCallable)
    cppclass ReturnByVal(Instruction):
        ReturnByVal(const string&, vector[MemLocation], const MemLocation&, ByValCallable)

    cppclass Interpreter:
        cgtTuple* run(cgtTuple*)

    Interpreter* create_interpreter(ExecutionGraph*, vector[MemLocation], bint)

cdef vector[size_t] _tovectorlong(object xs):
    cdef vector[size_t] out = vector[size_t]()
    for x in xs: out.push_back(<size_t>x)
    return out

cdef void* _getstructptr(object o) except NULL: # XXX except NULL causes unnecessary exception check
    if o is None: 
        return NULL
    else: 
        return <void*><size_t>ctypes.cast(ctypes.pointer(o), ctypes.c_voidp).value    
    # XXX be more careful about what's coming into this fn

cdef void* _getfuncptr(object o) except NULL:
    cdef void* out= <void*><size_t>ctypes.cast(o, ctypes.c_void_p).value
    assert out != NULL
    return out



# TODO inplace op can operate on views of data
cdef void _pyfunc_byref(void* cldata, cgtObject** reads, cgtObject* write):
    (pyfun, nin) = <object>cldata
    pyread = [cgt2py_object(reads[i], True) for i in xrange(nin)]
    pywrite = cgt2py_object(write, True)
    try:
        pyfun(pyread, pywrite)
    except Exception:
        traceback.print_exc()
        abort()
    # cdef cgtTuple* tup
    # cdef cgtArray* a
    # if cgt_is_array(write):
    #     npout = <cnp.ndarray>pywrite
    #     cgt_memcpy(cgtCPU, cgtCPU, (<cgtArray*>write).data(), npout.data, (<cgtArray*>write).nbytes())
    # else:
    #     tup = <cgtTuple*> write
    #     for i in xrange(tup.size()):
    #         npout = <cnp.ndarray>pywrite[i]
    #         a = <cgtArray*>tup.getitem(i)
    #         assert cgt_is_array(a)
    #         cgt_memcpy(cgtCPU, cgtCPU, a.data(), npout.data, a.nbytes())

cdef cgtObject* _pyfunc_byval(void* cldata, cgtObject** args):
    (pyfun, nin) = <object>cldata
    pyread = [cgt2py_object(args[i], True) for i in xrange(nin)]
    try:
        pyout = pyfun(pyread)
    except Exception:
        traceback.print_exc()
        abort()
    return py2cgt_object(pyout, False)

cdef ByRefCallable _tocppbyrefcallable(callable, storage) except *:
    storage.append(callable)
    if callable.kind == "native":
        return ByRefCallable(<cgtByRefFun>_getfuncptr(callable.fptr), _getstructptr(callable.cldata))
    else:
        py_cldata = (callable.func, callable.n_in)
        storage.append(py_cldata)
        return ByRefCallable(&_pyfunc_byref, <PyObject*>py_cldata)

cdef ByValCallable _tocppbyvalcallable(callable, storage) except *:
    storage.append(callable)
    if callable.kind == "native":
        return ByValCallable(<cgtByValFun>_getfuncptr(callable.fptr), _getstructptr(callable.cldata))
    else:
        py_cldata = (callable.func, callable.n_in)
        storage.append(py_cldata)
        return ByValCallable(&_pyfunc_byval, <PyObject*>py_cldata)

cdef MemLocation _tocppmem(object pymem):
    return MemLocation(<size_t>pymem.index, devtype_fromstr(pymem.devtype))

cdef vector[MemLocation] _tocppmemvec(object pymemlist) except *:
    cdef vector[MemLocation] out = vector[MemLocation]()
    for pymem in pymemlist:
        out.push_back(_tocppmem(pymem))
    return out

cdef Instruction* _tocppinstr(object pyinstr, object storage) except *:
    t = type(pyinstr)
    cdef Instruction* out
    cdef MemLocation wloc = _tocppmem(pyinstr.write_loc)
    if t == compilation.LoadArgument:
        out = new LoadArgument(repr(pyinstr), pyinstr.ind, wloc)
    elif t == compilation.Alloc:
        out = new Alloc(repr(pyinstr), dtype_fromstr(pyinstr.dtype), _tocppmemvec(pyinstr.read_locs), wloc)
    elif t == compilation.BuildTup:
        out = new BuildTup(repr(pyinstr), _tocppmemvec(pyinstr.read_locs), wloc)
    elif t == compilation.ReturnByRef:
        out = new ReturnByRef(repr(pyinstr), _tocppmemvec(pyinstr.read_locs), wloc, _tocppbyrefcallable(pyinstr.get_callable(), storage))
    elif t == compilation.ReturnByVal:
        out = new ReturnByVal(repr(pyinstr), _tocppmemvec(pyinstr.read_locs), wloc, _tocppbyvalcallable(pyinstr.get_callable(), storage))
    else:
        raise RuntimeError("expected instance of type Instruction. got type %s"%t)
    return out


################################################################
### Wrapper classes
################################################################

cdef ExecutionGraph* make_cpp_execution_graph(pyeg, storage) except *:
    "make an execution graph object"
    cdef vector[Instruction*] instrs
    for instr in pyeg.instrs:
        instrs.push_back(_tocppinstr(instr, storage))
    return new ExecutionGraph(instrs,pyeg.n_args, pyeg.n_locs)

cdef class CppArrayWrapper:
    cdef IRC[cgtArray] arr
    def to_numpy(self):
        return cgt2py_array(self.arr.get(),True)
    @staticmethod
    def from_numpy(cnp.ndarray nparr, object pydevtype="cpu", view=True):
        cdef cgtDevtype devtype = devtype_fromstr(pydevtype)
        out = CppArrayWrapper()
        if view:
            out.arr = IRC[cgtArray](py2cgt_arrayview(nparr))
        else:
            out.arr = IRC[cgtArray](py2cgt_array(nparr, devtype))
        return out
    @property
    def ndim(self):
        return self.arr.get().ndim()
    @property
    def shape(self):
        return [self.arr.get().shape()[i] for i in range(self.arr.get().ndim())]
    @property
    def size(self):
        return self.arr.get().size()
    @property
    def dtype(self):
        return dtype_tostr(self.arr.get().dtype())
    @property
    def ptr(self):
        return <size_t>self.arr.get()
    @property
    def data(self):
        return <size_t>self.arr.get().data()


cdef class CppInterpreterWrapper:
    """
    Convert python inputs to C++
    Run interpreter on execution graph
    Then grab the outputs
    """
    cdef ExecutionGraph* eg # owned
    cdef Interpreter* interp # owned
    cdef object input_types
    cdef object storage
    def __init__(self, pyeg, input_types, output_locs, parallel):
        self.storage = []
        self.eg = make_cpp_execution_graph(pyeg, self.storage)
        cdef vector[MemLocation] cpp_output_locs = _tocppmemvec(output_locs)
        self.interp = create_interpreter(self.eg, cpp_output_locs, parallel)
        self.input_types = input_types
    def __dealloc__(self):
        if self.interp != NULL: del self.interp
        if self.eg != NULL: del self.eg
    def __call__(self, *pyargs):
        assert len(pyargs) == len(self.input_types)
        pyargs = tuple(core.as_valid_array(arg,typ) for (arg,typ) in zip(pyargs,self.input_types)) 
        # compilation.typecheck_args(pyargs, self.input_types)

        # TODO: much better type checking on inputs
        cdef cgtTuple* cargs = new cgtTuple(len(pyargs))
        for (i,pyarg) in enumerate(pyargs):
            cargs.setitem(i, py2cgt_object(pyarg, True))
        cdef IRC[cgtTuple] ret = IRC[cgtTuple](self.interp.run(cargs))
        del cargs
        return list(cgt2py_object(ret.get(), False)) # TODO maybe allow returning view?

def cgt_build_root():
    return osp.dirname(osp.dirname(osp.abspath(__file__)))

def apply_byref(fptr, cldata, inputs, output):
    cdef vector[IRC[cgtObject]] cgtinputs
    for x in inputs:
        cgtinputs.push_back(IRC[cgtObject](py2cgt_object(x,True)))
    cdef IRC[cgtObject] cgtoutput = IRC[cgtObject](py2cgt_object(output,True))
    cdef cgtByRefFun f = <cgtByRefFun>_getfuncptr(fptr)
    cdef void* cldataptr = _getstructptr(cldata)
    # Note: for IRC it is possible to cast the smart pointer into a regular pointer
    # since there are no extra fields (see IRC.h)
    # Yes, this is atrocious
    f(cldataptr, <cgtObject**>cgtinputs.data(), cgtoutput.get())

    # for (i,x) in enumerate(inputs): reads[i] = 
    # <cgtByRefFun>_getfuncptr(fptr), _getstructptr(cldata), [py2cgt_object(x) for x in inputs]
    # del[] reads