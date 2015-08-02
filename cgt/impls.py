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
c++ -fPIC -O3 -DNDEBUG %(srcpath)s -std=c++11 -c -o %(srcpath)s.o %(includes)s %(defines)s && \
c++ -fPIC -O3 -DNDEBUG %(srcpath)s.o -dynamiclib -Wl,-headerpad_max_install_names -install_name %(libname)s -o %(libpath)s -L%(cgtlibdir)s -lcgt %(extralink)s
            '''%d)
        elif fname.endswith(".cu"):
            cap(r'''
cd %(cacheroot)s && \
nvcc %(srcpath)s -c -o %(srcpath)s.o -ccbin cc -m64 -Xcompiler  -fPIC -Xcompiler -O3 -Xcompiler -arch -Xcompiler x86_64 %(includes)s %(defines)s && \
c++ -fPIC -O3 -DNDEBUG -fPIC -dynamiclib -Wl,-headerpad_max_install_names %(cudalibs)s -Wl,-rpath,%(cudalibdir)s -install_name %(libname)s -o %(libpath)s %(srcpath)s.o
            '''%d)
                # gpulinkflags = "-dynamiclib -Wl,-headerpad_max_install_names %(CUDA_LIBRARIES)s -Wl,-rpath,%(CUDA_LIBRARY_DIR)s"%d

    else:
        if fname.endswith(".cpp"):
            cap('''
c++ -fPIC -O3 -DNDEBUG %(srcpath)s -std=c++11 -c -o %(srcpath)s.o %(includes)s %(defines)s && \
c++ -fPIC -O3 -DNDEBUG -shared -rdynamic -Wl,-soname,%(libname)s -o %(libpath)s %(srcpath)s.o -L%(cgtlibdir)s -lcgt
            '''%d)
        elif fname.endswith(".cu"):
            cap(r'''
cd %(cacheroot)s && 
nvcc %(srcpath)s -c -o %(srcpath)s.o -ccbin cc -m64 -Xcompiler -fPIC -Xcompiler -O3 -Xcompiler -DNDEBUG %(includes)s %(defines)s && \
c++  -fPIC -O3 -DNDEBUG -shared -rdynamic -Wl,-soname,%(libname)s -o %(libpath)s %(srcpath)s.o %(cudalibs)s -Wl,-rpath,%(cudaroot)s
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

def get_impl(node, devtype):

    if core.load_config()["force_python_impl"]: raise core.MethodNotDefined # XXX hack

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

    struct_code = StringIO()
    vals = []
    fields = []
    triples = node.op.get_closure(node.parents)
    if triples is None:
        closure = ctypes.c_void_p(0)
    else:
        struct_code.write("typedef struct CGT_FUNCNAME_closure {\n")
        for (fieldname,fieldtype,val) in triples:
            vals.append(val)
            struct_code.write(ctypes2str[fieldtype])
            struct_code.write(" ")
            struct_code.write(fieldname)
            struct_code.write(";\n")
            fields.append((fieldname,fieldtype))
        struct_code.write("} CGT_FUNCNAME_closure;\n")

        class S(ctypes.Structure):
            _fields_ = fields
        closure = S(*vals)


    h = hashlib.md5(code_raw).hexdigest()[:10]
    funcname = devtype + node.op.__class__.__name__ + h
    ci = get_compile_info()
    CACHE_ROOT = ci["CACHE_ROOT"]
    libpath = osp.join(CACHE_ROOT, funcname + ".so")

    if not osp.exists(libpath):
        s = StringIO()        
        if not osp.exists(CACHE_ROOT): os.makedirs(CACHE_ROOT)
        print "compiling %(libpath)s for node %(node)s"%locals()
        ext = "cpp" if devtype == "cpu" else "cu"
        srcpath = osp.join(CACHE_ROOT, funcname + "." + ext)
        # write c code to tmp file
        s = StringIO()
        for filename in includes:
            s.write('#include "%s"\n'%filename)
        s.write(struct_code.getvalue().replace("CGT_FUNCNAME",funcname))
        code = code_raw.replace("CGT_FUNCNAME",funcname)
        s.write(code)
        with open(srcpath,"w") as fh:
            fh.write(s.getvalue())

        compile_file(srcpath, osp.splitext(srcpath)[0]+".so", extra_link_flags = node.op.c_extra_link_flags)

    return (libpath,funcname,closure)

