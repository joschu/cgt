#!/usr/bin/env python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--files",nargs="+")
parser.add_argument("--patfile", type=argparse.FileType("r"))
args = parser.parse_args()


import subprocess, os, os.path as osp, cgt
from glob import glob

os.chdir(osp.dirname(osp.dirname(osp.realpath(cgt.__file__))))

if args.files is None and args.patfile is None: args.patfile=open("4dev/lintfiles.txt","r")

def cap(cmd):
    "call and print"
    print("\x1b[32m%s\x1b[0m"%cmd)
    subprocess.call(cmd,shell=True)

def filelist_from_patterns(pats, rootdir=None):
    if rootdir is None: rootdir = "."
    # filelist = []
    fileset = set([])
    lines = [line.strip() for line in pats]
    for line in lines:
        pat  = line[2:]
        newfiles = glob(osp.join(rootdir,pat))
        if line.startswith("+"):
            fileset.update(newfiles)
        elif line.startswith("-"):
            fileset.difference_update(newfiles)
        else:
            raise ValueError("line must start with + or -")
    filelist = list(fileset)
    return filelist

assert args.files is not None or args.patfile is not None
if args.files is not None:
    filelist = args.files
elif args.patfile is not None:
    filelist = filelist_from_patterns(args.patfile.readlines())
else:
    raise Exception("unreachable")

rcfile = "4dev/pylintrc"
lint = "pylint"
if filelist is not None:
    for fname in filelist:
        result = cap("%s -f colorized --rcfile %s -r n %s"%(lint, rcfile, fname))
else:
    result = cap("%s -f colorized  --rcfile %s -r n  *.py"%(lint,rcfile))

