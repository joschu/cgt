import numpy as np, scipy.ndimage as ndi, scipy.signal
import cgt,nn
np.random.seed(0)
x = np.random.randn(2,2,5,17)
f = np.random.randn(3,2,4,7)

filtrows = f.shape[2]
filtcols = f.shape[3]

batchsize = x.shape[0]
outchans = f.shape[0]

out = np.zeros((batchsize,outchans,x.shape[2]+filtrows-1,x.shape[3]+filtcols-1))
for b in xrange(x.shape[0]):
    for inchan in xrange(x.shape[1]):
        for outchan in xrange(outchans):
            out[b,outchan] += scipy.signal.convolve2d(x[b,inchan],f[outchan,inchan],mode='full')

cgt.set_precision('double')
out1 = cgt.numeric_eval(nn.conv2d(cgt.constant(x), cgt.constant(f)), {})
assert np.allclose(out,out1)