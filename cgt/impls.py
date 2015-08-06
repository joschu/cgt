from cgt.core import MethodNotDefined
from collections import namedtuple
import os, subprocess
import os.path as osp
from StringIO import StringIO
from cgt import core
import sys, ctypes, hashlib

_COMPILE_CONFIG = None
def get_compile_info():
    global _COMPILE_CONFIG
    
    if _COMPILE_CONFIG is None:

        config = core.load_config()

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
        if fname.endswith(".cpp"):
            cap(r'''
cd %(cacheroot)s && \
c++ -fPIC -O3 -DNDEBUG -ffast-math %(srcpath)s -std=c++11 -c -o %(srcpath)s.o %(includes)s %(defines)s && \
c++ -fPIC -O3 -DNDEBUG -ffast-math %(srcpath)s.o -dynamiclib -Wl,-headerpad_max_install_names -install_name %(libname)s -o %(libpath)s -L%(cgtlibdir)s -lcgt %(extralink)s
            '''%d)
        elif fname.endswith(".cu"):
            cap(r'''
cd %(cacheroot)s && \
nvcc %(srcpath)s -c -o %(srcpath)s.o -ccbin cc -m64 -Xcompiler  -fPIC -Xcompiler -O3 -Xcompiler -arch -Xcompiler x86_64 %(includes)s %(defines)s && \
c++ -fPIC -O3 -DNDEBUG -ffast-math -fPIC -dynamiclib -Wl,-headerpad_max_install_names %(cudalibs)s -Wl,-rpath,%(cudalibdir)s -install_name %(libname)s -o %(libpath)s %(srcpath)s.o
            '''%d)
                # gpulinkflags = "-dynamiclib -Wl,-headerpad_max_install_names %(CUDA_LIBRARIES)s -Wl,-rpath,%(CUDA_LIBRARY_DIR)s"%d

    else:
        if fname.endswith(".cpp"):
            cap('''
c++ -fPIC -O3 -DNDEBUG -ffast-math %(srcpath)s -std=c++11 -c -o %(srcpath)s.o %(includes)s %(defines)s && \
c++ -fPIC -O3 -DNDEBUG -ffast-math -shared -rdynamic -Wl,-soname,%(libname)s -o %(libpath)s %(srcpath)s.o -L%(cgtlibdir)s -lcgt
            '''%d)
        elif fname.endswith(".cu"):
            cap(r'''
cd %(cacheroot)s && 
nvcc %(srcpath)s -c -o %(srcpath)s.o -ccbin cc -m64 -Xcompiler -fPIC -Xcompiler -O3 -Xcompiler -DNDEBUG %(includes)s %(defines)s && \
c++  -fPIC -O3 -DNDEBUG -ffast-math -shared -rdynamic -Wl,-soname,%(libname)s -o %(libpath)s %(srcpath)s.o %(cudalibs)s -Wl,-rpath,%(cudaroot)s
            '''%d
            )

ctypes2str = {
    ctypes.c_byte : "uint8_t",
    ctypes.c_bool : "bool",
    ctypes.c_char : "char",
    ctypes.c_int : "int",
    ctypes.c_long : "long",
    ctypes.c_void_p : "void*",
    ctypes.c_double : "double",
    ctypes.c_float : "float"
}

_Binary = namedtuple('Binary', ['pyimpl', 'c_libpath', 'c_funcname'])
def Binary(impl, c_libpath=None, c_funcname=None):
    if impl.is_py():
        return _Binary(impl, None, None)
    assert not (impl.is_c() or impl.is_cuda()) or (c_libpath is not None and c_funcname is not None)
    return _Binary(None, c_libpath, c_funcname)


def _build_struct_code(triples):
    if triples is None:
        return ""
    struct_code = StringIO()
    struct_code.write("typedef struct CGT_FUNCNAME_closure {\n")
    for (fieldname,fieldtype,val) in triples:
        struct_code.write(ctypes2str[fieldtype])
        struct_code.write(" ")
        struct_code.write(fieldname)
        struct_code.write(";\n")
    struct_code.write("} CGT_FUNCNAME_closure;\n")
    return struct_code.getvalue()

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


class OpLibrary(object):
    """
    Stores compiled Op implementations. Performs just-in-time compiling.
    """

    def __init__(self, force_python_impl=False):
        self.node2implhash = {} # maps node -> (impl hash, closure)
        self.implhash2binary = {}
        self.force_python_impl = force_python_impl

    def num_impls_compiled(self):
        return len(self.implhash2binary)

    def _get_op_impl(self, node, devtype=None):
        """
        Grabs the preferred implementation of an op, falling back to
        Python if necessary

        devtype ignored if Python impls are forced
        """

        config = core.load_config()
        force_python_impl = config["force_python_impl"]
        disallow_python_impl = config["disallow_python_impl"]
        compile_info = get_compile_info()

        if not force_python_impl and not self.force_python_impl:
            assert devtype is not None
            if devtype == "gpu":
                if not compile_info["CGT_ENABLE_CUDA"]:
                    raise RuntimeError("tried to get CUDA implementation but CUDA is disabled (set CGT_ENABLE_CUDA and recompile)")
                try:
                    impl = node.op.get_cuda_impl(node.parents)
                except MethodNotDefined:
                    raise RuntimeError('Op %s has no CUDA implementation, but GPU mode is requested' % repr(node.op))
                return impl

            assert devtype == "cpu"
            try:
                impl = node.op.get_c_impl(node.parents)
            except MethodNotDefined:
                if disallow_python_impl:
                    raise RuntimeError("Op %s has no C implementation, but Python fallback is disabled" % repr(node.op))
                else:
                    print "Op %s has no C implementation, falling back to Python" % repr(node.op)
            else:
                return impl

        if disallow_python_impl:
            raise RuntimeError("Requested Python implementation for op %s, but Python implementations are disabled" % repr(node.op))
        try:
            impl = node.op.get_py_impl()
        except MethodNotDefined:
            raise RuntimeError("Op %s has no Python implementation" % repr(node.op))
        return impl


    @staticmethod
    def _compile_impl(node, impl):
        if impl.is_py():
            # A Python "binary" is just the function provided in the
            # Op implementation
            return Binary(impl)

        if impl.is_c() or impl.is_cuda():
            common_includes = ["cgt_common.h","stdint.h","stddef.h"] if impl.is_c() else ["cgt_common.h","cgt_cuda.h"]
            includes = common_includes + impl.includes

            struct_code = _build_struct_code(node.op.get_closure(node.parents))

            devtype = "cpu" if impl.is_c() else "gpu"
            funcname = "%s_%s_%s" % (devtype, node.op.__class__.__name__, impl.hash())
            ci = get_compile_info()
            CACHE_ROOT = ci["CACHE_ROOT"]
            libpath = osp.join(CACHE_ROOT, funcname + ".so")

            if not osp.exists(libpath):
                s = StringIO()
                if not osp.exists(CACHE_ROOT): os.makedirs(CACHE_ROOT)
                ext = "cpp" if impl.is_c() else "cu"
                srcpath = osp.join(CACHE_ROOT, funcname + "." + ext)
                # write c code to tmp file
                s = StringIO()
                for filename in includes:
                    s.write('#include "%s"\n'%filename)
                s.write(struct_code.replace("CGT_FUNCNAME",funcname))
                code = impl.code.replace("CGT_FUNCNAME",funcname)
                s.write(code)
                with open(srcpath,"w") as fh:
                    fh.write(s.getvalue())
                print "compiling %(srcpath)s to %(libpath)s for node %(node)s"%locals()
                compile_file(srcpath, osp.splitext(srcpath)[0]+".so", extra_link_flags=impl.link_flags)

            return Binary(impl, c_libpath=libpath, c_funcname=funcname)

        raise NotImplementedError

    def fetch_binary(self, node, devtype=None):
        """
        Gets the binary for node's Op, compiling its Impl if necessary.
        Assumes that if get_*_impl() is called on node.op twice,
        then the two Impls have the same hashes.
        """
        # If node has never been seen, check if its impl has been compiled,
        # and compile if necessary
        if node not in self.node2implhash:
            impl = self._get_op_impl(node, devtype)
            ihash = self.node2implhash[node] = impl.hash()
            if ihash in self.implhash2binary:
                # Already compiled
                return self.implhash2binary[ihash]
            # Compile and store
            binary = self.implhash2binary[ihash] = self._compile_impl(node, impl)
            return binary
        # We saw the node before, just return its compiled impl directly
        ihash = self.node2implhash[node]
        assert ihash in self.implhash2binary
        return self.implhash2binary[ihash]

    def fetch_closure(self, node):
        # TODO: cache these
        return _build_closure(node.op.get_closure(node.parents))

    # These functions are only for the python SequentialInterpreter
    # and only work when the nodes use (forced) python impls
    def py_apply_inplace(self, node, reads, write):
        self.fetch_binary(node).pyimpl.inplace_func(reads, write)

    def py_apply_valret(self, node, reads):
        return self.fetch_binary(node).pyimpl.valret_func(reads)
