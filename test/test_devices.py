import numpy as np
import cgt

cgt.set_precision('single')
N = 10
K = 3

Xval = np.random.randn(N,K)
wval = np.random.randn(K)
bval = np.random.randn()
yval = np.random.randn(N)

X_nk = cgt.shared(Xval, "X", device=cgt.Device(devtype='gpu'))
y_n = cgt.shared(yval, "y")
w_k = cgt.shared(wval, "w")
b = cgt.shared(bval, name="b")

print "bval",bval

ypred = cgt.dot(cgt.square(X_nk), w_k) + b

err = cgt.sum(cgt.square(ypred - y_n))
g = cgt.grad(err, [w_k, b])
outputs = [err]
def devfn(node):
    if isinstance(node, cgt.Result) and node.op == err.op: # XXX add fn for this
        return cgt.Device(devtype="cpu")

func=cgt.VarSizeFunc([], outputs, devfn = devfn)

def writedev(node,o):
    o.write(" | device: %s"%func.node2device[node])

cgt.print_tree(func.outputs, nodefn=writedev)



print "ready..."
numerr = func()
print "done"

assert np.allclose(numerr,cgt.numeric_eval([err],{}))