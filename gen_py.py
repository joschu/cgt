import sys
from cgt import UNARY_INFO, BINARY_INFO
fh = sys.stdout

fh.write("from cgt import ElwiseUnary, ElwiseBinary, Result, elwise_binary\n")
for (shortname,info) in sorted(UNARY_INFO.iteritems(), key = lambda x:x[1].short):    
    fh.write(
"""def {npname}(x):
    return Result(ElwiseUnary("{shortname}"), [x])
""".format(shortname=shortname,npname=info.short))

for (infixname,info) in BINARY_INFO.iteritems():    
    fh.write(
"""def {npname}(x, y):
    return elwise_binary("{infixname}", x,y)
""".format(infixname = infixname, npname=info.short))
