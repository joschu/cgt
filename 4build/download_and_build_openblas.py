#!/usr/bin/env python
import subprocess,sys,os,shutil,os.path as osp
import cgt, urllib

unpack_dir = sys.argv[1]
print "will download openblas and unpack to %s"%unpack_dir

def cap(cmd):
    "call and print"
    print "\x1b[32m%s\x1b[0m"%cmd
    subprocess.check_call(cmd,shell=True)

fname = "openblas.tar.gz"
url = "https://github.com/xianyi/OpenBLAS/archive/v0.2.14.tar.gz"

if not osp.exists(fname):
    urllib.urlretrieve(url, fname)
    shutil.move("{fname}.part".format(fname=fname),"{fname}".format(fname=fname))
cap("mkdir -p {unpack_dir} && tar -xf {fname} --directory {unpack_dir}  --strip-components=1".format(
    fname=fname,unpack_dir=unpack_dir))
os.chdir(unpack_dir)
cgt.utils.warn("Compiling OpenBLAS...this will take a minute or so")
cap("make -j ONLY_CBLAS=1 NO_LAPACK=1 NO_LAPACKE=1 USE_THREAD=0 USE_OPENMP=0 &> compile_output.txt")
