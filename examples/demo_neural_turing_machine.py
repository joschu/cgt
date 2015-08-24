__doc__ = """

Neural turing machine.

Names of arrays
---------------

Along with range of elements [low, high]

Mprev_bnm:          previous memory state. [-inf, inf]
X_bk:               external inputs. [0,1]
wprev_bHn:          previous weights (read & write). [0, 1]. Normalized along axis 2.
rprev_bhm:          previous vector read from memory. [-1, 1]
k_bHm:              key vectors [-1, 1]
beta_bH:            key strength [0, infinity]
g_bH:               gating for weight update.  [0, 1]
s_bH3:              shift weighting. [0, 1]. Normalized along axis 2. 
gamma:              sharpening [1, 2]
e_bhm:              erase [0, 1]
a_bhm:              add [-1, 1]

Names of subscripts
-------------------

- b: batch size
- h: number of read heads == number of write heads
- H: number of read + write heads == 2*h
- n: number of memory sites
- m: dimension at each memory site
- k: dimension of input
- p: dimension of output

"""

import cgt, numpy as np, numpy.random as nr
from cgt import core, nn
from collections import namedtuple
from cgt.core import infer_shape
from example_utils import fmt_row
from param_collection import ParamCollection


# Subscript indicate dimensions of array, and what each dimension indexes over
NTMOpts = namedtuple("NTMOpts",[
    "b", # batch size
    "h", # number of heads
    "n", # number of memory sites
    "m", # dimension at each memory site
    "k", # dimension of input
    "p", # dimension of output
    "ff_hid_sizes", # hidden layer sizes of feedforward controller
])

def make_ff_controller(opt):

    b, h, m, p, k = opt.b, opt.h, opt.m, opt.p, opt.k

    H = 2*h
    in_size = k + h*m
    out_size = H*m + H + H + H*3 + H + h*m + h*m + p

    # Previous reads
    r_bhm = cgt.tensor3("r", fixed_shape = (b,h,m))
    # External inputs
    X_bk = cgt.matrix("x", fixed_shape = (b,k))
    r_b_hm = r_bhm.reshape([r_bhm.shape[0], r_bhm.shape[1]*r_bhm.shape[2]])
    # Input to controller
    inp_bq = cgt.concatenate([X_bk, r_b_hm], axis=1)

    hid_sizes = opt.ff_hid_sizes
    activation = cgt.tanh

    layer_out_sizes = [in_size] + hid_sizes + [out_size]
    last_out = inp_bq
    # feedforward part. we could simplify a bit by using nn.Affine
    for i in xrange(len(layer_out_sizes)-1):
        indim = layer_out_sizes[i]
        outdim = layer_out_sizes[i+1]        
        W = cgt.shared(.02*nr.randn(indim, outdim), name="W%i"%i, fixed_shape_mask="all")
        bias = cgt.shared(.02*nr.randn(1, outdim), name="b%i"%i, fixed_shape_mask="all")
        last_out = cgt.broadcast("+",last_out.dot(W),bias,"xx,1x")
        # Don't apply nonlinearity at the last layer
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
    beta_bH = nn.softplus(beta_bH)
    g_bH = cgt.sigmoid(g_bH)
    s_bH3 = sum_normalize2(cgt.exp(s_bH3))
    gamma_bH = cgt.sigmoid(gamma_bH)+1
    e_bhm = cgt.sigmoid(e_bhm)
    a_bhm = cgt.tanh(a_bhm)
    # y_bp = y_bp

    assert infer_shape(k_bHm) == (b,H,m)
    assert infer_shape(beta_bH) == (b,H)
    assert infer_shape(g_bH) == (b,H)
    assert infer_shape(s_bH3) == (b,H,3)
    assert infer_shape(gamma_bH) == (b,H)
    assert infer_shape(e_bhm) == (b,h,m)
    assert infer_shape(a_bhm) == (b,h,m)
    assert infer_shape(y_bp) == (b,p)

    return nn.Module([r_bhm, X_bk], [k_bHm, beta_bH, g_bH, s_bH3, gamma_bH, e_bhm, a_bhm, y_bp])

def make_ntm_initial_states(opt):
    n, m, h, b = opt.n, opt.m, opt.h, opt.b
    M_1nm = cgt.shared(.1*nr.randn(1,n,m))
    winit_1Hn = cgt.shared(.1*nr.rand(1,2*h,n))
    winit_1Hn = sum_normalize2(cgt.exp(winit_1Hn))
    rinit_1hm = cgt.shared(np.zeros((1,h,m)))
    return [cgt.repeat(arr, b, axis=0) for arr in (M_1nm, winit_1Hn, rinit_1hm)]

def ntm_address(opt, wprev_bhn, M_bnm, k_bhm, beta_bh, g_bh, s_bh3, gamma_bh):

    # Content addressing

    # Cosine similarity
    # take inner product along memory axis k * M
    numer_bhn = cgt.einsum("bhm,bnm->bhn", k_bhm, M_bnm) 
    # compute denominator |k| * |m|
    denom_bhn = cgt.broadcast("*",
        cgt.norm(k_bhm, axis=2, keepdims=True), # -> shape bh1
        cgt.norm(M_bnm, axis=2, keepdims=True).transpose([0,2,1]), # -> bn1 -> b1n
        "xx1,x1x"
    )
    csim_bhn =  numer_bhn / denom_bhn
    assert infer_shape(csim_bhn) == (opt.b, 2*opt.h, opt.n)
    # scale by beta
    tmp_bhn = cgt.broadcast("*", beta_bh[:,:,None], csim_bhn, "xx1,xxx")
    wc_bhn = sum_normalize2(cgt.exp( tmp_bhn ))
    # Interpolation
    g_bh1 = g_bh[:,:,None]
    wg_bhn = cgt.broadcast("*", wprev_bhn, (1 - g_bh1), "xxx,xx1") \
            + cgt.broadcast("*", wc_bhn, g_bh1, "xxx,xx1")
    # Shift
    wtil_bhn = circ_conv_1d(wg_bhn, s_bh3, axis=2)
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

    if False: # Here's the version that's faithful to the paper
        # weighted erases                  bhn1                bh1m
        # ideally we wouldn't create this big 4-tensor but this operation 
        # requires a more general kind of contraction than is provided by einsum
        we_bhmn = cgt.broadcast("*", w_bhn[:,:,:,None], e_bhm[:,:,None,:], "xxx1,xx1x")
        # take produce of erasing factors
        mult_bmn = (1 - we_bhmn).prod(axis=1)
        M_bnm = M_bnm * mult_bmn # Equation 3 http://arxiv.org/pdf/1410.5401v2.pdf
    else: # This version just does a regular contraction
        erase_bnm = cgt.einsum( "bhn,bhm->bnm", w_bhn, e_bhm)
        M_bnm = M_bnm*(1-erase_bnm)

    # Now do the same thing with adds
    # But now it's just a regular contraction since we are adding rather than taking product
    add_bnm = cgt.einsum( "bhn,bhm->bnm", w_bhn, a_bhm)
    M_bnm = M_bnm + add_bnm

    return M_bnm

def ntm_step(opt, Mprev_bnm, X_bk, wprev_bHn, rprev_bhm, controller):
    n_heads = opt.h
    k_bHm, beta_bH, g_bH, s_bH3, gamma_bH, e_bhm, a_bhm, y_bp = controller([rprev_bhm, X_bk])
    w_bHn = ntm_address(opt, wprev_bHn, Mprev_bnm, k_bHm, beta_bH, g_bH, s_bH3, gamma_bH)
    wr_bhn = w_bHn[:,:n_heads,:]
    ww_bhn = w_bHn[:,n_heads:,:]    
    r_bhm = ntm_read(Mprev_bnm, wr_bhn)
    M_bnm = ntm_write(Mprev_bnm, ww_bhn, e_bhm, a_bhm)
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

def bernoulli_crossentropy(bins, probs):
    "bins = binary values. probs = Pr(b=1)"
    return -( bins*cgt.log(probs) + (1-bins)*cgt.log(1-probs))

def make_funcs(opt, ntm, total_time, loss_timesteps):    
    x_tbk = cgt.tensor3("x", fixed_shape=(total_time, opt.b, opt.k))
    y_tbp = cgt.tensor3("y", fixed_shape=(total_time, opt.b, opt.p))
    loss_timesteps = set(loss_timesteps)

    initial_states = make_ntm_initial_states(opt)
    params = ntm.get_parameters() + get_parameters(initial_states)
    # params = ntm.get_parameters()

    lossCE = 0
    loss01 = 0

    state_arrs = initial_states
    for t in xrange(total_time):
        tmp = ntm([x_tbk[t]] + state_arrs)
        raw_pred = tmp[0]
        state_arrs = tmp[1:4]

        if t in loss_timesteps:
            p_pred = cgt.sigmoid(raw_pred)
            ce = bernoulli_crossentropy(y_tbp[t] , p_pred).sum() # cross-entropy of bernoulli distribution
            lossCE = lossCE + ce
            loss01 = loss01 + cgt.cast(cgt.equal(y_tbp[t], round01(p_pred)),cgt.floatX).sum()


    lossCE = lossCE / (len(loss_timesteps) * opt.p * opt.b) / np.log(2)
    loss01 = loss01 / (len(loss_timesteps) * opt.p * opt.b)
    gradloss = cgt.grad(lossCE, params)

    flatgrad = flatcat(gradloss)

    f_loss = cgt.function([x_tbk, y_tbp], lossCE)
    f_loss_and_grad = cgt.function([x_tbk, y_tbp], [lossCE, loss01, flatgrad])

    return f_loss, f_loss_and_grad, params

def round01(x):
    return cgt.cast(x>.5,cgt.floatX)

def flatcat(xs):
    return cgt.concatenate([x.flatten() for x in xs])

class CopyTask(object):
    def __init__(self, batch_size, seq_length, output_dim):
        self.b = batch_size
        self.t = seq_length
        self.k = output_dim+2
        self.p = output_dim
    def gen_batch(self):
        assert self.k == self.p + 2
        x_tbk = np.zeros((2*self.t + 2, self.b, self.k),cgt.floatX)
        x_tbk[0, :, 0] = 1 # start symbol
        message = (nr.rand(self.t, self.b, self.p) > .5).astype(cgt.floatX)
        # message = (nr.rand(self.t, self.b, self.p)).astype(cgt.floatX)

        x_tbk[1:self.t+1,:,2:] = message
        x_tbk[self.t+1, :, 1] = 1 # end symbol
        y_tbk = np.zeros((2*self.t+2, self.b, self.p),cgt.floatX)
        y_tbk[self.t+2:] = message # desired output

        return x_tbk, y_tbk
    def loss_timesteps(self):
        return range(self.t+1, 2*self.t+2)
    def total_time(self):
        return 2*self.t+2

def circ_conv_1d(wg_bhn, s_bh3, axis=2):
    "VERY inefficient way to implement circular convolution for the special case of filter size 3"
    assert axis == 2
    n = cgt.size(wg_bhn,2)
    wback = cgt.concatenate([wg_bhn[:,:,n-1:n], wg_bhn[:,:,:n-1]], axis=2)
    w = wg_bhn
    wfwd = cgt.concatenate([wg_bhn[:,:,1:n], wg_bhn[:,:,0:1]], axis=2)
    return cgt.broadcast("*", s_bh3[:,:,0:1] , wback, "xx1,xxx")\
     + cgt.broadcast("*", s_bh3[:,:,1:2] , w, "xx1,xxx")\
     + cgt.broadcast("*", s_bh3[:,:,2:3] , wfwd, "xx1,xxx")

def rmsprop_update(grad, state):
    state.sqgrad[:] *= state.decay_rate
    np.square(grad, out=state.scratch) # scratch=g^2
    state.sqgrad[:] += state.scratch
    np.sqrt(state.sqgrad, out=state.scratch) # scratch = scaling
    np.divide(grad, state.scratch, out=state.scratch) # scratch = grad/scaling
    np.multiply(state.scratch, state.step_size, out=state.scratch)
    state.theta[:] -= state.scratch

class Table(dict):
    "dictionary-like object that exposes its keys as attributes"
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self

def make_rmsprop_state(theta, step_size, decay_rate):
    return Table(theta=theta, sqgrad=np.zeros_like(theta)+1e-6, scratch=np.empty_like(theta), 
        step_size=step_size, decay_rate=decay_rate)

def get_parameters(xs):
    # XXX
    out = []
    for node in core.topsorted(xs):
        if node.is_data():
            out.append(node)
    return out

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--grad_check",action="store_true")
    parser.add_argument("--n_batches",type=int,default=1000000)
    parser.add_argument("--profile",action="store_true")
    args = parser.parse_args()
    np.seterr("raise")

    cgt.set_precision("quad" if args.grad_check else "double")
    np.random.seed(0)

    # model parameters
    if args.grad_check:
        opt = NTMOpts(
            b = 1, # batch size
            h = 1, # number of heads
            n = 2, # number of memory sites
            m = 3, # dimension at each memory site
            k = 4, # dimension of input
            p = 2, # dimension of output
            ff_hid_sizes = []
        )
        seq_length = 2

    else:
        opt = NTMOpts(
            b = 64, # batch size
            h = 3, # number of heads
            n = 128, # number of memory sites
            m = 20, # dimension at each memory site
            k = 3, # dimension of input
            p = 1, # dimension of output
            ff_hid_sizes = [128,128]
        )

    # task parameters
        seq_length = 10

    ntm = make_ntm(opt)

    task = CopyTask(opt.b, seq_length, opt.p)
    f_loss, f_loss_and_grad, params = make_funcs(opt, ntm, task.total_time(), task.loss_timesteps())

    pc = ParamCollection(params)
    pc.set_value_flat(nr.uniform(-.1, .1, size=(pc.get_total_size(),)))

    if args.grad_check:
        x,y = task.gen_batch()
        def f(thnew):
            thold = th.copy()
            pc.set_value_flat(thnew)
            loss = f_loss(x,y)
            pc.set_value_flat(thold)
            return loss
        from cgt.numeric_diff import numeric_grad
        g_num = numeric_grad(f, th,eps=1e-8)
        _, _, g_anal = f_loss_and_grad(x,y)
        assert np.allclose(g_num, g_anal, atol=1e-8)
        print "Gradient check succeeded!"
        print "%i/%i elts of grad are nonzero"%( (g_anal != 0).sum(), g_anal.size )
        return

    seq_num = 0
    state = make_rmsprop_state(pc.get_value_flat(), .01, .95)
    print fmt_row(13, ["seq num", "CE (bits)", "accuracy", "|g|_inf"], header=True)
    
    if args.profile: cgt.profiler.start()
    
    for i in xrange(args.n_batches):
        x,y = task.gen_batch()
        seq_num += x.shape[1]
        l,l01,g = f_loss_and_grad(x,y)
        print fmt_row(13, [seq_num, l,l01,np.abs(g).max()])
        rmsprop_update(g, state)        
        pc.set_value_flat(state.theta)
        if not np.isfinite(l): break

    
    if args.profile: cgt.profiler.print_stats()

    

if __name__ == "__main__":
    main()
