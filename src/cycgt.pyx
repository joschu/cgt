from cpython.ref cimport PyObject
from libc.stdlib cimport malloc, free
import cgt
import ctypes
cimport numpy as cnp
import numpy as np
# see https://github.com/cython/cython/blob/master/Cython/Includes/numpy/__init__.pxd
import os.path as osp


cdef extern from "cgt_common.h":

    struct cgt_array:
        int ndim
        cgt_dtype dtype
        cgt_devtype devtype
        size_t* shape
        void* data
        size_t stride
        bint ownsdata

    size_t cgt_size(const cgt_array* a)

    int cgt_itemsize(cgt_dtype dtype)

    size_t cgt_nbytes(const cgt_array* a)

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

    cdef enum cgt_devtype:
        cgt_cpu
        cgt_gpu

    ctypedef int cgt_error

    ctypedef void (*cgt_fun)(void*, cgt_array**);

    void* cgt_alloc(cgt_devtype devtype, size_t size)
    void cgt_free(cgt_devtype devtype, void* ptr)
    void cgt_memcpy(cgt_devtype dest_type, cgt_devtype src_type, void* dest_ptr, void* src_ptr, size_t nbytes)


    cdef enum cgt_status:
        cgt_ok
        cgt_err

    void cgt_clear_error()
    char* cgt_global_errmsg
    cgt_status cgt_global_status


ctypedef cnp.Py_intptr_t npy_intp_t
# assert sizeof(npy_intp_t) == sizeof(size_t)



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

cdef struct cgt_funcall:
    cgt_fun fptr
    void* cldata
    int nargs
    cgt_array** args


cdef struct cgt_dag:
    int nnodes
    cgt_funcall* funcalls
    int* npreds
    int** predecessors
    int* nsuccs
    int** successors

cdef struct cgt_seq:
    int nfuncalls
    cgt_funcall* funcalls


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

cnp.import_array()

cdef class CallSequence(object):
    cdef:
        int n_nodes
        int n_inputs
        int* input_inds
        int n_outputs
        int* output_inds
        cgt_array* cgtarrs
        cgt_funcall* funcalls
        object objstore
        object nodes
        bint check

    def __init__(self, inputs, outputs, nodes, node2device=None, check=False):
        var2idx = {}
        if node2device is None: node2device = {}
        self.n_nodes = len(nodes)
        self.n_inputs = len(inputs)
        self.input_inds = <int*>malloc(self.n_inputs*sizeof(int))
        self.n_outputs = len(outputs)
        self.output_inds = <int*>malloc(self.n_outputs*sizeof(int))
        self.cgtarrs = <cgt_array*>malloc(len(nodes)*sizeof(cgt_array))
        self.funcalls = <cgt_funcall*>malloc(len(nodes)*sizeof(cgt_funcall))
        self.objstore = []
        self.nodes = nodes
        self.check = check

        cdef cgt_funcall* funcall
        cdef cgt_array* cgtarr

        for (inode,node) in enumerate(nodes):            
            var2idx[node] = inode
            cgtarr = &self.cgtarrs[inode]
            cgtarr.ndim = node.ndim
            cgtarr.dtype = dtype_fromstr(node.dtype)
            maybe_dev = node2device.get(node)
            cgtarr.devtype = cgt_cpu if maybe_dev is None else devtype_fromstr(maybe_dev.devtype)
            cgtarr.shape = <size_t*>malloc(sizeof(size_t)*cgtarr.ndim)
            cgtarr.data = NULL
            cgtarr.ownsdata = False
            funcall = &self.funcalls[inode]
            parents = node.parents
            if node.is_argument():
                funcall.fptr = NULL
                funcall.cldata = NULL
                funcall.nargs = 0
                funcall.args = NULL
            elif isinstance(node.op, cgt.TupleIndex):
                funcall.fptr = NULL
                funcall.cldata = NULL
                funcall.nargs = 0
                funcall.args = NULL
                parentidx = var2idx[node.parents[0]]
                parentnin = len(node.parents[0].parents)
                self.funcalls[parentidx].args[parentnin+node.op.idx] = &self.cgtarrs[inode]
            else:
                try:
                    libname, funcname, clstuff = cgt.get_impl(node, devtype_tostr(cgtarr.devtype))
                    lib_handle = get_or_load_lib(libname)
                    cfun = dlsym(lib_handle, funcname)
                    if cfun == NULL:
                        raise RuntimeError("Could not load function %s from %s"%(funcname, libname))
                    funcall.fptr = <cgt_fun>cfun
                    if clstuff is None:
                        funcall.cldata = NULL     
                    else:
                        class S(ctypes.Structure): _fields_ = [("x%i"%i,cdtype) for (i,(cdtype,_)) in enumerate(clstuff)]
                        s = S(*(val for (_,val) in clstuff))                        
                        self.objstore.append(s)
                        funcall.cldata = <void*><size_t>ctypes.cast(ctypes.pointer(s), ctypes.c_voidp).value
                except cgt.MethodNotDefined:                
                    funcall.fptr = &call_pyfunc
                    pyfun = node.op.get_numeric_py()
                    cldata = (pyfun, len(parents), cgt.num_components(node))
                    self.objstore.append(cldata)
                    funcall.cldata = <PyObject*>cldata

                funcall.nargs = len(parents) + cgt.num_components(node)
                funcall.args = <cgt_array**>malloc(funcall.nargs*sizeof(cgt_array*))
                for (i,p) in enumerate(parents):
                    funcall.args[i] = &self.cgtarrs[var2idx[p]]
                if isinstance(node.get_type(), cgt.Tensor):
                    funcall.args[funcall.nargs-1] = &self.cgtarrs[inode]
                # OTHERWISE we'll use the TupleIndex nodes



        for (inode, node) in enumerate(inputs):
            self.input_inds[inode] = var2idx.get(node,-1)

        for (inode, node) in enumerate(outputs):
            self.output_inds[inode] = var2idx[node]


    def set_shapes(self, shapes):
        cdef cgt_array* cgtarr
        assert len(shapes) == self.n_nodes
        for (i,shape) in enumerate(shapes):
            assert len(shape) == self.cgtarrs[i].ndim
            cgtarr = &self.cgtarrs[i]
            for (j,shape_elt) in enumerate(shape):
                cgtarr.shape[j] = <size_t>shape_elt
            if (cgtarr.data != NULL and cgtarr.ownsdata):
                cgt_free(cgtarr.devtype, cgtarr.data)
                cgtarr.data = NULL
                cgtarr.ownsdata = False
            if (self.nodes[i].is_argument() or self.nodes[i].op.needs_alloc):
                # XXX probably dont need to alloc for inputs
                cgtarr.data = cgt_alloc(cgtarr.devtype, cgt_nbytes(cgtarr))
                cgtarr.ownsdata = True

    def set_inputs(self, inputs):
        assert len(inputs) == self.n_inputs
        cdef cgt_array* cgtarr
        for (i, arr) in enumerate(inputs):
            cgtarr = &self.cgtarrs[self.input_inds[i]]
            if self.input_inds[i] >= 0:
                to_cgt_array(np.asarray(arr, dtype=dtype_tostr(cgtarr.dtype), order='C'), cgtarr)
    def execute(self):
        cdef int i
        cdef cgt_funcall* funcall
        for i in xrange(self.n_nodes):
            funcall = &self.funcalls[i]
            if funcall.fptr != NULL:
                funcall.fptr(funcall.cldata, funcall.args)
                if cgt_global_status != cgt_ok:
                    msg = <object>cgt_global_errmsg
                    raise RuntimeError("Error encountered during execution: %s.\nRun your function with the python backend for easier debugging. (CGT_FLAGS=backend=python)"%msg)
        if self.check: self.dbg_check_values()

    def get_outputs_numpy(self):
        return [to_numpy_array(&self.cgtarrs[self.output_inds[i]]) for i in xrange(self.n_outputs)]

    def dbg_check_values(self):
        in2val = {}
        node2val = {}
        for i in xrange(self.n_inputs):
            nodeind = self.input_inds[i]
            if nodeind >= 0: 
                in2val[self.nodes[nodeind]] = to_numpy_array(&self.cgtarrs[nodeind])
        for (inode,node) in enumerate(self.nodes):
            if node.is_argument():
                if node in in2val:
                    node2val[node] = in2val[node]
            elif isinstance(node.get_type(), cgt.Tensor):
                node2val[node] = node.op.numeric_apply([node2val[par] for par in node.parents])
                if not np.allclose(node2val[node],to_numpy_array(&self.cgtarrs[inode]),atol=1e-5):
                    print "error at",node
                    print node2val[node].flat[:10]
                    print to_numpy_array(&self.cgtarrs[inode]).flat[:10]
                    raise RuntimeError

    def __dealloc__(self):
        cdef cgt_array* cgtarr
        cdef cgt_funcall* funcall

        for i in xrange(self.n_nodes):
            cgtarr = &self.cgtarrs[i]
            if cgtarr.data != NULL and cgtarr.ownsdata:
                cgt_free(cgtarr.devtype, cgtarr.data)
                free(cgtarr.shape)
                cgtarr.data = NULL
                cgtarr.ownsdata = False
            funcall = &self.funcalls[i]
            if funcall.args != NULL: free(funcall.args)

        free(self.input_inds)
        free(self.output_inds)
        free(self.cgtarrs)
        free(self.funcalls)



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

cdef object to_numpy_array(cgt_array* a):
    # cdef npy_intp_t* newshape = <npy_intp_t*>malloc(sizeof(npy_intp_t*)*a.ndim)
    # for i in xrange(a.ndim): newshape[i] = a.shape[i]
    # cdef np.ndarray nparr = np.PyArray_SimpleNew(a.ndim, newshape, a.dtype)
    cdef cnp.ndarray nparr = cnp.PyArray_SimpleNew(a.ndim, <npy_intp_t*>a.shape, a.dtype) # XXX DANGEROUS CAST
    # nparr = np.empty(shape=tuple([a.shape[i] for i in xrange(a.ndim)]), dtype=dtype_tostr(a.dtype))    
    cgt_memcpy(cgt_cpu, a.devtype, cnp.PyArray_DATA(nparr), a.data, cnp.PyArray_NBYTES(nparr))
    return nparr

cdef void to_cgt_array(cnp.ndarray nparr, cgt_array* a):
    # assert nparr.ndim == a.ndim and nparr.dtype.num == a.dtype and nparr.flags.c_contiguous
    if nparr.dtype.num != a.dtype:
        nparr = nparr.astype(dtype_tostr(a.dtype))
    # TODO: break this down
    cdef int i
    assert a.ndim == nparr.ndim, "ndimsame"
    for i in xrange(a.ndim): assert a.shape[i] == nparr.shape[i],"shapesame%i,%s,%s"%(i,(<object>nparr).shape, (a.shape[0], a.shape[1]))
    
    assert a.data != NULL or cgt_nbytes(a) == 0,"notnull"
    cgt_memcpy(a.devtype, cgt_cpu, a.data, cnp.PyArray_DATA(nparr), cgt_nbytes(a))

cdef _to_valid_array(arr):
    cdef cnp.ndarray out = np.asarray(arr, order='C')
    if not out.flags.c_contiguous: 
        out = out.copy()
    return out


cdef void call_pyfunc(void* cldata, cgt_array** args):
    (pyfun, nin, nout) = <object>cldata
    pyargs = tuple([to_numpy_array(args[i]) for i in xrange(nin)])
    if nout == 1:
        arr = _to_valid_array(pyfun(*pyargs))
        to_cgt_array(arr, args[nin])
    else:
        for (i,arr) in enumerate(pyfun(*pyargs)):
            to_cgt_array(_to_valid_array(arr), args[nin+i])
    # npin = to_numpy_array(args[nin])
    # assert np.allclose(npout,npin)

    # XXX why is pyarray_Ensurearray wrong?!

    # TODO: less copying of nparray data.
