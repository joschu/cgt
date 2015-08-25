import sys
import numpy as np
import hashlib
import time

# ================================================================
# Utils
# ================================================================

class Color: #pylint: disable=W0232
    GRAY=30
    RED=31
    GREEN=32
    YELLOW=33
    BLUE=34
    MAGENTA=35
    CYAN=36
    WHITE=37
    CRIMSON=38    


def colorize(num, string, bold=False, highlight = False):
    assert isinstance(num, int)
    attr = []
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)

def colorprint(colorcode, text, o=sys.stdout):
    o.write(colorize(colorcode, text))

def warn(msg):
    print colorize(Color.YELLOW, msg)

def error(msg):
    print colorize(Color.RED, msg)

def is_singleton(x):
    return isinstance(x, np.ndarray) and np.prod(x.shape)==1

def safezip(x,y):
    assert len(x) == len(y)
    return zip(x,y)

def safezip3(x,y,z):
    assert len(x) == len(y) == len(z)
    return zip(x,y,z)


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

MESSAGE_DEPTH = 0
class Message(object):
    def __init__(self, msg):
        self.msg = msg
    def __enter__(self):
        global MESSAGE_DEPTH #pylint: disable=W0603
        print colorize(Color.MAGENTA, '\t'*MESSAGE_DEPTH + '=: ' + self.msg)
        self.tstart = time.time()
        MESSAGE_DEPTH += 1
    def __exit__(self, etype, *args):
        global MESSAGE_DEPTH #pylint: disable=W0603
        MESSAGE_DEPTH -= 1
        maybe_exc = "" if etype is None else " (with exception)"
        print colorize(Color.MAGENTA, '\t'*MESSAGE_DEPTH + "done%s in %.3f seconds"%(maybe_exc, time.time() - self.tstart))
