import numpy as np

def numeric_grad(f,x,eps=1e-9,method="central"):
    if method == "central":
        xpert = x.copy()
        out = np.zeros_like(x)
        for i in xrange(len(x)):
            xpert[i] = x[i] + eps
            yplus = f(xpert)
            xpert[i] = x[i] - eps
            yminus = f(xpert)
            xpert[i] = x[i]
            out[i] = (yplus - yminus) / (2*eps)
        return out
    else:
        raise NotImplementedError("invalid method %s"%method)