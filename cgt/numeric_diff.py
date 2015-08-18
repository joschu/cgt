import numpy as np

def numeric_grad(f,x,eps=1e-9,method="central"):
    if method == "central":
        xpert = x.copy()
        out = np.zeros_like(x)
        for i in xrange(x.size):
            xpert.flat[i] = x.flat[i] + eps
            yplus = f(xpert)
            xpert.flat[i] = x.flat[i] - eps
            yminus = f(xpert)
            xpert.flat[i] = x.flat[i]
            out.flat[i] = (yplus - yminus) / (2*eps)
            if (i+1)%1000 == 0: print "%i/%i components done"%(i+1,x.size)
        return out
    else:
        raise NotImplementedError("invalid method %s"%method)

def numeric_grad_multi(f, xs, eps=1e-9,method="central"):
    out = []
    for i in xrange(len(xs)):
        li = list(xs)        
        def f1(x):
            li[i] = x
            return f(*li)
        out.append(numeric_grad(f1, xs[i], eps, method=method))
    return out

