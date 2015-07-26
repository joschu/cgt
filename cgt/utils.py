import sys
import numpy as np
import hashlib

# ================================================================
# Utils
# ================================================================

class Color: #pylint: disable=W0232
    GRAY=30,
    RED=31,
    GREEN=32,
    YELLOW=33,
    BLUE=34,
    MAGENTA=35,
    CYAN=36,
    WHITE=37,
    CRIMSON=38    

def colorprint(colorcode, text, o=sys.stdout):
    o.write("\x1b[%im"%colorcode)
    o.write(text)
    o.write("\x1b[0m")

def warn(msg):
    colorprint(Color.YELLOW, msg)
    sys.stdout.write("\n")

def error(msg):
    colorprint(Color.RED, msg)
    sys.stdout.write("\n")

def is_singleton(x):
    return np.prod(x.shape)==1

def safezip(x,y):
    assert len(x) == len(y)
    return zip(x,y)

def allsame(xs):
    out = True
    if len(xs)>0:
        x0 = xs[0]
        for x in xs[1:]:
            out &= x==x0
    return out

def invert_perm(x):
    return list(np.argsort(x))

def _hash_seq(args):
    hashobj = hashlib.md5()
    for a in args: hashobj.update(a)
    return hashobj.hexdigest()

def hash_seq1(*args):
    return _hash_seq(args)

