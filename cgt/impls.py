import subprocess
import os.path as osp, os
from StringIO import StringIO
import cgt
from cgt import core
import sys, ctypes, hashlib
import string #pylint: disable=W0402
    
def nci2callable(nci):
    common_includes =["cgt_common.h","cgt_cuda.h"] if nci.involves_gpu() else ["cgt_common.h"]
    code = _gen_code(common_includes + nci.includes, nci.closure_triples, nci.func_code)
    tu = TranslationUnit(nci.lang, code, nci.link_flags)
    compile_info = get_compile_info()
    CACHE_ROOT = compile_info["CACHE_ROOT"]
    if not osp.exists(CACHE_ROOT): os.makedirs(CACHE_ROOT)
    prefix = tu.hash()
    ext = {"c++" : ".cpp", "cuda" : ".cu"}[nci.lang]
    srcpath = osp.join(CACHE_ROOT, prefix+ext)
    libpath = osp.join(CACHE_ROOT, prefix+".so")
    if not osp.exists(libpath):
        tu.compile(prefix, srcpath, libpath)
    lib = _get_or_load_lib(libpath)
    fptr = getattr(lib, _funcname(prefix))
    setup_fptr = getattr(lib, _setupname(prefix)) if nci.setup else None
    teardown_fptr = getattr(lib, _teardownname(prefix)) if nci.teardown else None
    cldata = _build_closure(nci.closure_triples)
    if nci.lang == "cuda": assert nci.gpu_deref_mask is not None
    return core.NativeCallable(nci.n_in, nci.call_type, nci.op_str, fptr, cldata=cldata, setup_fptr=setup_fptr, teardown_fptr=teardown_fptr,
        store_objects=nci.store_objects)

def _funcname(prefix):
    return "call_"+prefix
def _setupname(prefix):
    return "setup_"+prefix
def _teardownname(prefix):
    return "teardown_"+prefix
def _closurename(prefix):
    return "closure_"+prefix

_LIBRARIES = {}
def _get_or_load_lib(libname):
    if libname in _LIBRARIES:
        return _LIBRARIES[libname]
    else:
        out = ctypes.cdll.LoadLibrary(libname)
        _LIBRARIES[libname] = out
        return out

def _gen_code(includes, closure_info, func_code):
    s = StringIO()
    for fname in includes:
        s.write('#include "%s"\n'%fname)
    _gen_struct_code(closure_info, s)
    s.write(func_code)
    return s.getvalue()


def _gen_struct_code(triples, outstream):
    if triples is None:
        return
    outstream.write("typedef struct $closure {\n")
    for (fieldname,fieldtype,_val) in triples:
        outstream.write(_ctypes2str[fieldtype])
        outstream.write(" ")
        outstream.write(fieldname)
        outstream.write(";\n")
    outstream.write("} $closure;\n")


class TranslationUnit(object):
    """All the input that goes into building a native binary for one or more ops"""

    def __init__(self, lang, template_code, link_flags, compile_flags=None):
        self.lang = lang
        self.template_code = template_code
        self.link_flags = link_flags
        if compile_flags is None:
            compile_flags = "-fPIC -O0 -g" if core.get_config()["debug_cpp"] else "-O3 -DNDEBUG -ffast-math"
        self.compile_flags = compile_flags

    def hash(self):
        h = hashlib.md5()
        for item in (self.template_code, self.link_flags, self.compile_flags):
            h.update(item)
        return h.hexdigest()

    def generate_code(self, prefix):
        d = dict(function=_funcname(prefix), closure=_closurename(prefix),setup=_setupname(prefix),teardown=_teardownname(prefix))
        return string.Template(self.template_code).substitute(d)

    def compile(self, prefix, srcpath, libpath):
        """Runs the compiler, placing code in srcpath and the output in libpath"""
        with open(srcpath,"w") as fh:
            fh.write(self.generate_code(prefix))
        call_and_print(_make_compile_command(srcpath, libpath, self.link_flags, self.compile_flags))


_COMPILE_CONFIG = None
def get_compile_info():
    global _COMPILE_CONFIG
    
    if _COMPILE_CONFIG is None:

        config = core.get_config()

        CGT_BUILD_ROOT = cgt.cycgt.cgt_build_root()

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
        CGT_ENABLE_CUDNN = cmake_info["CGT_ENABLE_CUDNN"] in ["1","ON"]        
        DEFINITIONS = "-DENABLE_CUDA" if CGT_ENABLE_CUDA else ""
        CUDNN_ROOT = cmake_info["CUDNN_ROOT"]


        _COMPILE_CONFIG = dict(        
            OPENBLAS_INCLUDE_DIR = osp.join(CGT_BUILD_ROOT,"OpenBLAS"),
            CGT_INCLUDE_DIR = cmake_info["CGT_INCLUDE_DIR"],
            CGT_LIBRARY_DIR = osp.join(CGT_BUILD_ROOT,"lib"),
            CUDA_LIBRARY_DIR = osp.join(CUDA_ROOT,"lib"),
            CUDA_INCLUDE_DIR = osp.join(CUDA_ROOT,"include"), 
            CUDA_LIBRARIES = cmake_info["CUDA_LIBRARIES"], 
            DEFINITIONS = DEFINITIONS,  
            CUDA_ROOT = CUDA_ROOT,
            CUDNN_ROOT = CUDNN_ROOT,
            CACHE_ROOT = osp.expanduser(config["cache_dir"]),
            CGT_ENABLE_CUDA = CGT_ENABLE_CUDA,
            CGT_ENABLE_CUDNN = CGT_ENABLE_CUDNN,
            # CGT_LIBRARY = cmake_info["CGT_LIBRARY"],
        )
    return _COMPILE_CONFIG

def _make_compile_command(fname, libpath, extra_link_flags, compile_flags):
    info = get_compile_info()
    includes  = "-I"+info["CGT_INCLUDE_DIR"]
    includes += " -I"+info["OPENBLAS_INCLUDE_DIR"]
    if info["CGT_ENABLE_CUDA"]:  includes += " -I"+info["CUDA_INCLUDE_DIR"]
    if info["CGT_ENABLE_CUDNN"]: includes += " -I"+info["CUDNN_ROOT"]


    linkdirs = "-L"+info["CGT_LIBRARY_DIR"]
    if info["CGT_ENABLE_CUDA"]: linkdirs += " -L"+info["CUDA_LIBRARY_DIR"]
    if info["CGT_ENABLE_CUDNN"]: linkdirs += " -L"+info["CUDNN_ROOT"]

    d = dict(
        cacheroot=info["CACHE_ROOT"],
        srcpath=fname,
        includes=includes,
        linkdirs =linkdirs,
        defines=info["DEFINITIONS"],
        libname=osp.basename(libpath),
        libpath=libpath,
        cgtlibdir=info["CGT_LIBRARY_DIR"],
        extralink=extra_link_flags,
        compileflags=compile_flags
    )
    if fname.endswith(".cu"):
        if not info["CGT_ENABLE_CUDA"]:
            raise RuntimeError("Trying to compile a CUDA function but CUDA is disabled in your build. Rebuild with CGT_ENABLE_CUDA=ON")
        d.update(cudalibs=info["CUDA_LIBRARIES"], cudaroot=info["CUDA_ROOT"], cudalibdir=info["CUDA_LIBRARY_DIR"])

    cmd = None
    if sys.platform == "darwin":
        if fname.endswith(".cpp"):
            cmd = r'''
cd %(cacheroot)s && \
c++ %(compileflags)s %(srcpath)s -fvisibility=hidden -std=c++11 -stdlib=libc++ -c -o %(srcpath)s.o %(includes)s %(defines)s && \
c++ %(compileflags)s %(srcpath)s.o -dynamiclib -Wl,-headerpad_max_install_names -install_name %(libname)s -o %(libpath)s %(linkdirs)s -lcgt %(extralink)s
            '''%d
        elif fname.endswith(".cu"):
            # TODO fix cuda compilation commands
            cmd = r'''
cd %(cacheroot)s && \
nvcc %(srcpath)s -c -o %(srcpath)s.o -ccbin cc -m64 -Xcompiler  -fPIC -Xcompiler -O3 -Xcompiler -arch -Xcompiler x86_64 %(includes)s %(defines)s && \
c++ %(compileflags)s -dynamiclib -Wl,-headerpad_max_install_names %(cudalibs)s -Wl,-rpath,%(cudalibdir)s -install_name %(libname)s -o %(libpath)s %(srcpath)s.o
            '''%d
                # gpulinkflags = "-dynamiclib -Wl,-headerpad_max_install_names %(CUDA_LIBRARIES)s -Wl,-rpath,%(CUDA_LIBRARY_DIR)s"%d
    else:
        if fname.endswith(".cpp"):
            cmd = '''
c++ %(compileflags)s %(srcpath)s -fvisibility=hidden -std=c++11 -stdlib=libc++ -c -o %(srcpath)s.o %(includes)s %(defines)s && \
c++ %(compileflags)s -shared -rdynamic -Wl,-soname,%(libname)s -o %(libpath)s %(srcpath)s.o %(linkdirs)s -lcgt
            '''%d
        elif fname.endswith(".cu"):
            cmd = r'''
cd %(cacheroot)s && 
nvcc %(srcpath)s -c -o %(srcpath)s.o -ccbin cc -m64 -Xcompiler -fPIC -Xcompiler -O3 -Xcompiler -DNDEBUG %(includes)s %(defines)s && \
c++  %(compileflags)s -shared -rdynamic -Wl,-soname,%(libname)s -o %(libpath)s %(srcpath)s.o %(cudalibs)s -Wl,-rpath,%(cudaroot)s
            '''%d

    assert cmd is not None
    return cmd


def call_and_print(cmd):
    print "\x1b[32m%s\x1b[0m"%cmd
    subprocess.check_call(cmd,shell=True)

_ctypes2str = {
    ctypes.c_byte : "uint8_t",
    ctypes.c_bool : "bool",
    ctypes.c_char : "char",
    ctypes.c_int : "int",
    ctypes.c_long : "long",
    ctypes.c_void_p : "void*",
    ctypes.c_double : "double",
    ctypes.c_float : "float"
}

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
