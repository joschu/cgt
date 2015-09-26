#!/usr/bin/env python
import subprocess,sys,os,shutil,os.path as osp
import urllib
import multiprocessing

unpack_dir = sys.argv[1]

def call_and_print(cmd):
    print "\x1b[32m%s\x1b[0m"%cmd
    subprocess.check_call(cmd,shell=True)

fname = "openblas.tar.gz"
url = "https://github.com/xianyi/OpenBLAS/archive/v0.2.14.tar.gz"

if osp.exists(fname):
    print "already downloaded openblas.tar.gz"
else:
    print "will download openblas and unpack to %s"%unpack_dir
    urllib.urlretrieve(url, fname+".part")
    shutil.move("{fname}.part".format(fname=fname),"{fname}".format(fname=fname))
call_and_print("mkdir -p {unpack_dir} && tar -xf {fname} --directory {unpack_dir}  --strip-components=1".format(
    fname=fname,unpack_dir=unpack_dir))
os.chdir(unpack_dir)
print "Compiling OpenBLAS...this will take a minute or so"
call_and_print("make -j ONLY_CBLAS=1 NO_LAPACK=1 NO_LAPACKE=1 USE_THREAD=0 USE_OPENMP=0 NUM_THREADS=%i &> compile_output.txt"%multiprocessing.cpu_count())
