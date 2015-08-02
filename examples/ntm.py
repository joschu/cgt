import cgt, numpy as np, scipy.signal as ss, numpy.random as nr
from cgt import floatX, core, nn
#    Subscript indicate dimensions of array, and what each dimension indexes over
#    b := batch size
#    h := number of heads
#    n := number of memory sites
#    m := dimension at each memory site
#    k := dimension of input
#    p := dimension of output

class Controller(object):
    def __init__(self, n_head):
        self.n_head = n_head
    def __call__(self):
        raise NotImplementedError


def make_ff_controller(b,h,m,p,k):

    H = 2*h
    in_size = k + h*m
    out_size = H*m + H + H + H*3 + H + h*m + h*m + p

    r_bhm = cgt.tensor3("r", fixed_shape = (b,h,m))
    X_bk = cgt.tensor3("x", fixed_shape = (b,k))
    r_b_hm = r_bhm.reshape([r_bhm.shape[0], r_bhm.shape[1]*r_bhm.shape[2]])
    inp_bq = cgt.concatenate([r_b_hm, X_bk], axis=0)

    hid_sizes = [50,50]
    activation = cgt.tanh

    layer_out_sizes = [in_size] + hid_sizes + [out_size]
    last_out = inp_bq

    for i in xrange(len(layer_out_sizes)-1):
        indim = layer_out_sizes[i]
        outdim = layer_out_sizes[i+1]
        W = cgt.shared(np.zeros(indim, outdim))
        b = cgt.shared(np.zeros((1, outdim)))
        last_out = activation(last_out.dot(W)+b)

    idx = 0
    beta_bH = last_out[idx:idx+H];      idx += H
    g_bH = last_out[idx:idx+H];         idx += H
    s_bH3 = last_out[idx:idx+3*H];      idx += 3*H
    gamma_bH = last_out[idx:idx+H];     idx += H
    e_bhm = last_out[idx:idx+h*m];      idx += h*m
    a_bhm = last_out[idx:idx+h*m];      idx += h*m
    y_bp = last_out[idx:idx+p];         idx += p

    return nn.Module([r_bhm, X_bk], [beta_bH, g_bH, s_bH3, gamma_bH, e_bhm, a_bhm, y_bp])

def make_ntm_initial_states(n,h,m,b):
    M_1nm = np.zeros((1,n,m))
    winit_1hn = np.zeros((1,n,m))
    rinit_1hm = np.zeros((1,h,m))
    return [cgt.repeat(arr, b, axis=0) for arr in (M_1nm, winit_1hn, rinit_1hm)]

def ntm_address(wprev_bhn, M_bnm, k_bhm, beta_bh, g_bh, s_bh3, gamma_bh):
    # Content addressing
    csim_bhn = cosine_similarity(k_bhm, M_bnm, axis=2)
    e_bhn = cgt.broadcast("*", beta_bh[:,:,None], csim_bhn, "xxx,xx1")
    wc_bhn = sum_normalize(cgt.exp( e_bhn ), axis=2)
    # Interpolation
    g_bh1 = g_bh[:,:,None]
    wg_bhn = cgt.broadcast("*", wprev_bhn, (1 - g_bh1), "xxx,xx1") \
            + cgt.broadcast("*", wc_bhn, g_bh1, "xxx,xx1")
    # Shift
    wtil_bhn = correlate1d(wg_bhn, s_bh3, axis=2)
    # Sharpening
    wfin_bhn = sum_normalize(cgt.broadcast("^", wtil_bhn, gamma_bh[:,:,None], "xxx,xx1"), axis=2)
    return wfin_bhn

def ntm_read(M_bnm, w_bhn):
    r_bhm = cgt.einsum('bhn,bnm->bhm', w_bhn, M_bnm)
    return r_bhm

def ntm_write(M_bnm, w_bhn, e_bhm, a_bhm):
    q_b1m = (1-e_bhm).prod(axis=1,keepdims=True)
    q_bnm = cgt.broadcast("*", w_bhn, q_b1m, 'xxx,x1x')
    M1_bnm = M_bnm * q_bnm
    a_b1m = a_bhm.sum(axis=1,keepdims=True)
    Mtil_bnm = cgt.broadcast("+", M1_bnm, a_b1m, 'xxx,x1x')
    return Mtil_bnm

def ntm_step(Mprev_bnm, X_bk, wprev_bHn, rprev_bhm, controller):
    n_heads = rprev_bhm.shape[1]
    k_bHm, beta_bH, g_bH, s_bH3, gamma_bH, e_bhm, a_bhm, y_bp = controller(rprev_bhm, X_bk)
    w_bHn = ntm_address(wprev_bHn, Mprev_bnm, k_bHm, beta_bH, g_bH, s_bH3, gamma_bH)
    wr_bhn = w_bHn[:,:n_heads,:]
    ww_bhn = w_bHn[:,n_heads:,:]    
    r_bhm = ntm_read(Mprev_bnm, wr_bhn)
    M_bnm = ntm_write(Mprev_bnm, ww_bhn, e_bhm, a_bhm)
    w_bHn = cgt.concatenate([wr_bhn, ww_bhn], axis=1)
    return M_bnm, w_bHn, r_bhm, y_bp

def sum_normalize(x, axis):
    return x / x.sum(axes=[axis],keepdims=True)

def cosine_similarity(x, y, axis):
    return (x*y) / (cgt.norm(x, axes=[axis]) * cgt.norm(y, axes=[axis]))

class Correlate1d(cgt.EasyCustomOp):
    def __init__(self, axis, ndim):
        self.axis = axis
        cgt.EasyCustomOp.__init__(self,
            input_types = [core.Tensor(floatX, ndim), core.Tensor(floatX, ndim)],
            output_type = core.Tensor(floatX, ndim),
            forward_impl = self.correlate_forward,
            pullback_impl = self.correlate_pullback)
    def correlate_forward(self, x, s):
        # assert s.shape[self.axis] == 1
        return ss.correlate(x, s, mode='same')
    def correlate_pullback(self, x, y, _z, gz):
        padshp = list(y.shape)
        padshp[self.axis] = y.shape[self.axis]//2
        padzeros = np.zeros(tuple(padshp),dtype=x.dtype)
        gzpad = np.concatenate([padzeros, gz, padzeros], axis=self.axis)
        return (ss.correlate(gzpad, y, mode='valid')[::-1], 
                ss.correlate(gzpad, x, mode='valid')[::-1])
    def shp_apply(self, inputs):
        return cgt.shape(inputs[0])
    def typ_apply(self, inputs):
        return inputs[0].typ

def correlate1d(x, s, axis):
    assert x.ndim == s.ndim
    return core.Result(Correlate1d(axis, x.ndim), [x,s])

def main():
    cgt.set_precision('double')
    x = cgt.vector('x')
    y = cgt.vector('y')
    xval = nr.randn(100)
    yval = nr.randn(3)
    assert np.allclose( cgt.numeric_eval1(correlate1d(x,y, 0), {x:xval,y:yval}),
        ss.correlate(xval, yval, mode='same'))
    import sys
    sys.path.append("../test")
    from test_affine import check_affine #pylint: disable=F0401
    def asdf(x,y):
        if isinstance(x, np.ndarray):
            return np.sum(ss.correlate(x,y,mode='same'))
        else:
            return cgt.sum(correlate1d(x,y,axis=0))
    check_affine(asdf, xval, yval)


    

if __name__ == "__main__":
    main()
