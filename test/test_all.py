from glob import glob
import subprocess, os.path as osp, os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--cov",action="store_true")
args = parser.parse_args()

scripts = glob(osp.join(osp.dirname(__file__),"test_*.py"))
fails = []
successes = []

if osp.exists(".coverage"):
    os.unlink(".coverage")

for fname in scripts:
    if fname.endswith("test_all.py"): continue
    print "*** Running %s ***"%fname
    try:
        if args.cov:
            interp = "coverage run -a"
        else:
            interp = "python"
        cmd = "%s %s"%(interp,fname)
        subprocess.check_call(cmd, shell=True)
        successes.append(fname)
    except subprocess.CalledProcessError:
        fails.append(fname)
        print "FAIL"

print
print "======== Results ========="
print "passed:",",".join(successes)
print "failed:",",".join(fails)
print "=========================="

if args.cov: 
    subprocess.check_call("coverage html", shell=True)
    subprocess.check_call("open htmlcov/index.html",shell=True)