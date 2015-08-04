import cgt, numpy as np, scipy.signal as ss, numpy.random as nr
from cgt import floatX, core, nn
from collections import namedtuple
from cgt.core import infer_shape

# Subscript indicate dimensions of array, and what each dimension indexes over
NTMOpts = namedtuple("NTMOpts",[
    "b", # batch size
    "h", # number of heads
    "n", # number of memory sites
    "m", # dimension at each memory site
    "k", # dimension of input
    "p", # dimension of output
])


# Names of arrays:
# Mprev_bnm: previous memory state
# X_bk: inputs
# wprev_bHn: previous weights (read & write, concatenated along axis 1)
# rprev_bhm: previous vector read from memory

# k: key vector
# beta: key strength
# g: gating for weight update
# s: shift weighting
# gamma: sharpening

def make_ff_controller(opt):

    b, h, m, p, k = opt.b, opt.h, opt.m, opt.p, opt.k

    H = 2*h
    in_size = k + h*m
    out_size = H*m + H + H + H*3 + H + h*m + h*m + p

    r_bhm = cgt.tensor3("r", fixed_shape = (b,h,m))
    X_bk = cgt.matrix("x", fixed_shape = (b,k))
    r_b_hm = r_bhm.reshape([r_bhm.shape[0], r_bhm.shape[1]*r_bhm.shape[2]])
    inp_bq = cgt.concatenate([r_b_hm, X_bk], axis=1)

    hid_sizes = []
    activation = cgt.tanh

    layer_out_sizes = [in_size] + hid_sizes + [out_size]
    last_out = inp_bq

    # feedforward part. we could simplify a bit by using nn.Affine
    for i in xrange(len(layer_out_sizes)-1):
        indim = layer_out_sizes[i]
        outdim = layer_out_sizes[i+1]        
        W = cgt.shared(np.zeros((indim, outdim)), name="W%i"%i, fixed_shape_mask="all")
        bias = cgt.shared(np.zeros((1, outdim)), name="b%i"%i, fixed_shape_mask="all")
        last_out = cgt.broadcast("+",last_out.dot(W),bias,"xx,1x")
        if i != len(layer_out_sizes)-2: last_out = activation(last_out)


    idx = 0
    k_bHm = last_out[:,idx:idx+H*m];      idx += H*m;         k_bHm = k_bHm.reshape([b,H,m])
    beta_bH = last_out[:,idx:idx+H];      idx += H
    g_bH = last_out[:,idx:idx+H];         idx += H
    s_bH3 = last_out[:,idx:idx+3*H];      idx += 3*H;         s_bH3 = s_bH3.reshape([b,H,3])
    gamma_bH = last_out[:,idx:idx+H];     idx += H
    e_bhm = last_out[:,idx:idx+h*m];      idx += h*m;         e_bhm = e_bhm.reshape([b,h,m])
    a_bhm = last_out[:,idx:idx+h*m];      idx += h*m;         a_bhm = a_bhm.reshape([b,h,m])
    y_bp = last_out[:,idx:idx+p];         idx += p

    k_bHm = cgt.tanh(k_bHm)
    beta_bH = cgt.sigmoid(beta_bH)
    g_bH = cgt.sigmoid(g_bH)
    s_bH3 = cgt.sigmoid(s_bH3)
    gamma_bH = cgt.sigmoid(gamma_bH)+1
    e_bhm = cgt.sigmoid(e_bhm)
    a_bhm = cgt.tanh(a_bhm)

    assert infer_shape(k_bHm) == (b,H,m)
    assert infer_shape(beta_bH) == (b,H)
    assert infer_shape(g_bH) == (b,H)
    assert infer_shape(s_bH3) == (b,H,3)
    assert infer_shape(gamma_bH) == (b,H)
    assert infer_shape(e_bhm) == (b,h,m)
    assert infer_shape(a_bhm) == (b,h,m)
    assert infer_shape(y_bp) == (b,p)

    return nn.Module([r_bhm, X_bk], [k_bHm, beta_bH, g_bH, s_bH3, gamma_bH, e_bhm, a_bhm, y_bp])
# return M_bnm, w_bHn, r_bhm, y_bp
def make_ntm_initial_states(opt):
    n, m, h, b = opt.n, opt.m, opt.h, opt.b
    M_1nm = cgt.shared(nr.randn(1,n,m))
    winit_1Hn = cgt.shared(nr.rand(1,2*h,n))
    rinit_1hm = cgt.shared(nr.randn(1,h,m))
    return [cgt.repeat(arr, b, axis=0) for arr in (M_1nm, winit_1Hn, rinit_1hm)]

def ntm_address(opt, wprev_bhn, M_bnm, k_bhm, beta_bh, g_bh, s_bh3, gamma_bh):

    # Content addressing

    # Cosine similarity
    numer = cgt.einsum("bhm,bnm->bhn", k_bhm, M_bnm)
    denom = cgt.broadcast("*",
        cgt.norm(k_bhm, axis=2, keepdims=True),
        cgt.norm(M_bnm, axis=2, keepdims=True).transpose([0,2,1]),
        "xx1,x1x"
    )
    csim_bhn =  numer / denom
    assert infer_shape(csim_bhn) == (opt.b, 2*opt.h, opt.n)
    e_bhn = cgt.broadcast("*", beta_bh[:,:,None], csim_bhn, "xx1,xxx")
    wc_bhn = sum_normalize2(cgt.exp( e_bhn ))
    # Interpolation
    g_bh1 = g_bh[:,:,None]
    wg_bhn = cgt.broadcast("*", wprev_bhn, (1 - g_bh1), "xxx,xx1") \
            + cgt.broadcast("*", wc_bhn, g_bh1, "xxx,xx1")
    # Shift
    wtil_bhn = correlate1d(wg_bhn, s_bh3, axis=2)
    # Sharpening
    wfin_bhn = sum_normalize2(cgt.broadcast("**", wtil_bhn, gamma_bh.reshape([opt.b,2*opt.h,1]), "xxx,xx1"))

    b,h,n = opt.b, 2*opt.h, opt.n
    assert infer_shape(wtil_bhn) == (b,h,n)
    assert infer_shape(gamma_bh) == (b,h)
    assert infer_shape(gamma_bh[:,:,None]) == (b,h,1)
    return wfin_bhn

def ntm_read(M_bnm, w_bhn):
    r_bhm = cgt.einsum('bhn,bnm->bhm', w_bhn, M_bnm)
    return r_bhm

def ntm_write(M_bnm, w_bhn, e_bhm, a_bhm):
    e_b1m = (1-e_bhm).prod(axis=1,keepdims=True)
    M_bnm = cgt.broadcast("*", M_bnm, e_b1m, 'xxx,x1x')
    a_b1m = a_bhm.sum(axis=1,keepdims=True)
    M_bnm = cgt.broadcast("+", M_bnm, a_b1m, 'xxx,x1x')
    return M_bnm

def ntm_step(opt, Mprev_bnm, X_bk, wprev_bHn, rprev_bhm, controller):
    n_heads = rprev_bhm.shape[1]
    k_bHm, beta_bH, g_bH, s_bH3, gamma_bH, e_bhm, a_bhm, y_bp = controller.expand([rprev_bhm, X_bk])
    w_bHn = ntm_address(opt, wprev_bHn, Mprev_bnm, k_bHm, beta_bH, g_bH, s_bH3, gamma_bH)
    wr_bhn = w_bHn[:,:n_heads,:]
    ww_bhn = w_bHn[:,n_heads:,:]    
    r_bhm = ntm_read(Mprev_bnm, wr_bhn)
    M_bnm = ntm_write(Mprev_bnm, ww_bhn, e_bhm, a_bhm)
    w_bHn = cgt.concatenate([wr_bhn, ww_bhn], axis=1)
    return M_bnm, w_bHn, r_bhm, y_bp

def sum_normalize2(x):
    return cgt.broadcast("/", x, x.sum(axis=2,keepdims=True), "xxx,xx1")


def make_ntm(opt):
    Mprev_bnm = cgt.tensor3("M", fixed_shape=(opt.b, opt.n, opt.m))
    X_bk = cgt.matrix("X", fixed_shape=(opt.b, opt.k))
    wprev_bHn = cgt.tensor3("w", fixed_shape=(opt.b, opt.h*2, opt.n))
    rprev_bhm = cgt.tensor3("r", fixed_shape=(opt.b, opt.h, opt.m))
    controller = make_ff_controller(opt)
    M_bnm, w_bHn, r_bhm, y_bp = ntm_step(opt, Mprev_bnm, X_bk, wprev_bHn, rprev_bhm, controller)
    # in this form it looks like a standard seq-to-seq model
    # external input and output are first elements
    ntm = nn.Module([X_bk, Mprev_bnm, wprev_bHn, rprev_bhm], [y_bp, M_bnm, w_bHn, r_bhm])
    return ntm


def make_funcs(opt, ntm, total_time, loss_timesteps):
    x_tbk = cgt.tensor3("x", fixed_shape=(total_time, opt.b, opt.k))
    y_tbk = cgt.tensor3("y", fixed_shape=(total_time, opt.b, opt.k))
    loss_timesteps = set(loss_timesteps)

    loss = 0

    state_arrs = make_ntm_initial_states(opt)
    for t in xrange(total_time):
        tmp = ntm.expand([x_tbk[t]] + state_arrs)
        raw_pred = tmp[0]
        state_arrs = tmp[1:4]

        if t in loss_timesteps:
            p_pred = cgt.sigmoid(raw_pred)
            negloglik = - (y_tbk[t] * cgt.log(p_pred)).sum() # cross-entropy of bernoulli distribution
            loss = loss + negloglik

    params = ntm.get_parameters()
    gradloss = cgt.grad(loss, params)

    flatgrad = cgt.flatcat(gradloss)

    f_loss = cgt.function1([x_tbk, y_tbk], loss)
    f_loss_and_grad = cgt.function([x_tbk, y_tbk], [loss, flatgrad])

    return f_loss, f_loss_and_grad


class Correlate1d(cgt.EasyCustomOp):
    def __init__(self, axis, ndim):
        self.axis = axis
        cgt.EasyCustomOp.__init__(self,
            input_types = [core.Tensor(floatX, ndim), core.Tensor(floatX, ndim)],
            output_type = core.Tensor(floatX, ndim),
            forward_impl = self.correlate_forward,
            pullback_impl = self.correlate_pullback)
    def correlate_forward(self, x, s):
        print x.shape, x.shape, ss.correlate(x, s, mode='same').shape
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


def correlate1d(x, s, axis):
    assert x.ndim == s.ndim
    cgt.utils.warn("SKIPPING correlate1d")
    return x
    return core.Result(Correlate1d(axis, x.ndim), [x,s])


class CopyTask(object):
    def __init__(self, batch_size, seq_length, input_dim):
        self.b = batch_size
        self.t = seq_length
        self.k = input_dim
    def gen_batch(self):
        x_tbk = np.zeros((2*self.t + 2, self.b, self.k),floatX)
        x_tbk[0, :, 0] = 1 # start symbol
        message = nr.rand(self.t, self.b, self.k)
        x_tbk[1:self.t+1] = message  
        x_tbk[self.t+1, :, 1] = 1 # end symbol
        y_tbk = np.zeros((2*self.t + 2, self.b, self.k),floatX)
        y_tbk[self.t+2:] = message # desired output
        return x_tbk, y_tbk
    def loss_timesteps(self):
        return range(self.t+1, 2*self.t+2)
    def total_time(self):
        return 2*self.t+2


def main():
    cgt.set_precision('quad')
    np.random.seed(0)
    x = cgt.vector('x')
    y = cgt.vector('y')
    xval = nr.randn(100)
    yval = nr.randn(3)

    # model parameters
    opt = NTMOpts(
        b = 7, # batch size
        h = 3, # number of heads
        n = 13, # number of memory sites
        m = 5, # dimension at each memory site
        k = 2, # dimension of input
        p = 2, # dimension of output
    )

    # task parameters
    seq_length = 2

    ntm = make_ntm(opt)
    params = ntm.get_parameters()
    th = nn.setup_contiguous_storage(params)
    th[:] = nr.uniform(-0.08, 0.08, th.shape)

    task = CopyTask(opt.b, seq_length, opt.k)
    f_loss, f_loss_and_grad = make_funcs(opt, ntm, task.total_time(), task.loss_timesteps())

    if True:
        x,y = task.gen_batch()
        def f(thnew):
            thold = th.copy()
            th[:] = thnew
            loss = f_loss(x,y)
            th[:] = thold
            return loss
        from cgt.numeric_diff import numeric_grad
        g_num = numeric_grad(f, th,eps=1e-10)
        _, g_anal = f_loss_and_grad(x,y)
        assert np.allclose(g_num, g_anal, atol=1e-4)
        print "Gradient check succeeded!"
        return



    x,y = task.gen_batch()
    print f_loss_and_grad(x, y)


    

if __name__ == "__main__":
    main()
