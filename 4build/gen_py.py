import sys
from cgt.core import UNARY_INFO, BINARY_INFO
import cgt, os, os.path as osp
fh = sys.stdout

os.chdir(osp.dirname(osp.dirname(osp.realpath(cgt.__file__))))

with open("cgt/api_autogen.py","w") as fh:

    for (shortname,info) in sorted(UNARY_INFO.iteritems(), key = lambda x:x[1].short):    
        fh.write(
"""
def {npname}(x):
    return core.Result(core.ElwiseUnary("{shortname}"), [x])
    """.format(shortname=shortname,npname=info.short))

    for (infixname,info) in sorted(BINARY_INFO.iteritems(), key = lambda x:x[1].short):    
        fh.write(
"""
def {npname}(x, y):
    return core.elwise_binary("{infixname}", x,y)
    """.format(infixname = infixname, npname=info.short))
    
    fh.write("\nfrom . import core\n")
