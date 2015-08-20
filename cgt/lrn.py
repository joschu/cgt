import ctypes
from cgt.core import *
from collections import namedtuple

LRNInfo = namedtuple("LRNInfo",["N","alpha","beta"])

def make_closure(info):
    return [
        ("N",ctypes.c_int,info.N),
        ("alpha",ctypes.c_double,info.alpha),
        ("beta",ctypes.c_double,info.beta)
    ]

class LRNForward(Op):
    available_impls = ("native_gpu",)    
    def __init__(self, info):
        assert isinstance(info, LRNInfo)
        self.info = info

    def get_native_compile_info(self, input_types, devtype):
        code = r"""
            CGT_EXPORT_C void $function($closure* cldata, cgtArray** reads, cgtTuple* write) {
                cgtArray* X = reads[0];
                int num_img = X->shape()[0],
                    channels = X->shape()[1],
                    height = X->shape()[2],
                    width = X->shape()[3];
                cgtArray* top = (cgtArray*)write->getitem(0);                
                cgtArray* scale = (cgtArray*)write->getitem(1);
                int size = num_img * height * width;
                int nblocks, nthreads;
                cgt_get_bt(size, &nblocks, &nthreads);
                LRNFillScale<%(cdtype)s><<<nblocks, nthreads, 0>>>(
                    size, (%(cdtype)s*)X->data(), num_img, channels, height, width, cldata->N, cldata->alpha / cldata->N, (%(cdtype)s*)scale->data());
                CUDA_CHECK_ERROR("LRNFillScale");

                size = num_img * channels * width * height;
                cgt_get_bt(size, &nblocks, &nthreads);
                LRNComputeOutput<%(cdtype)s><<<nblocks, nthreads, 0>>>(size, (%(cdtype)s*)X->data(), (%(cdtype)s*)scale->data(), -cldata->beta, (%(cdtype)s*)top->data());
                CUDA_CHECK_ERROR("LRNComputeOutput");
            }"""%dict(cdtype=np2c[input_types[0].dtype])
        return NativeCompileInfo(code, lang="cuda", closure_triples = make_closure(self.info),
            includes=["lrn.cuh"], link_flags="-lcudart", gpu_deref_mask=(True,))
    def shp_apply(self, inputs):
        return (inputs[0].shape,inputs[0].shape)
    def typ_apply(self, input_types):
        return TupleType(input_types[0], input_types[0])
    def pullback(self, inputs, output, gout):
        top, scaling = cgt.core.unpack(output)
        gtop, _ = gout
        return [Result(LRNBackward(self.info), [inputs[0], top, scaling, gtop])]

class LRNBackward(Op):
    available_impls = ("native_gpu",)    
    def __init__(self, info):
        self.info = info
    def get_native_compile_info(self, input_types, devtype):
        code="""
            CGT_EXPORT_C void $function($closure* cldata, cgtArray** reads, cgtArray* bottom_diff) {
            cgtArray *X=reads[0], *top=reads[1], *scaling=reads[2], *top_diff=reads[3];
            int num_img = X->shape()[0],
                channels = X->shape()[1],
                height = X->shape()[2],
                width = X->shape()[3];
            int nblocks, nthreads;
            int size = num_img * width * height;
            cgt_get_bt(size, &nblocks, &nthreads);
            LRNComputeDiff<%(cdtype)s><<<nblocks, nthreads, 0>>>(size, (%(cdtype)s*)X->data(), (%(cdtype)s*)top->data(), 
                (%(cdtype)s*)scaling->data(), (%(cdtype)s*)top_diff->data(),  num_img, channels, height, width, cldata->N, 
                -cldata->beta, 2. * cldata->alpha * cldata->beta / cldata->N, (%(cdtype)s*)bottom_diff->data());
            CUDA_CHECK_ERROR("LRNBackward");
            }"""%dict(cdtype=np2c[input_types[0].dtype])
        return NativeCompileInfo(code, lang="cuda", closure_triples = make_closure(self.info),
            includes=["lrn.cuh"], link_flags="-lcudart", gpu_deref_mask=(True,True,True,True))
    def shp_apply(self, inputs):
        return cgt.shape(inputs[0])
    def typ_apply(self, _inputs):
        return TensorType(cgt.floatX, 4)

def lrn(X, windowsize, alpha, beta):
    assert X.ndim == 4
    return Result(LRNForward(LRNInfo(windowsize,alpha,beta)), [X])

if __name__ == "__main__":
    from cgt.tests import gradcheck_model
    cgt.set_precision('double')
    import numpy.random as nr
    nr.seed(9)
    Xval = nr.randn(4,8,16,16)
    X = cgt.shared(Xval, name="X", fixed_shape_mask="all")
    # X = cgt.tensor4(name='X')
    y,_ = lrn(X, windowsize=4, alpha=.1, beta=.5)
    f = cgt.function([],y)
    print f().sum()
    print f().sum()
    print f().sum()
    assert np.isfinite(f().sum())
    # print f(Xval).sum()
    a = nr.rand(*cgt.infer_shape(y))
    loss = (y*a).sum()
    gradcheck_model(loss, [X],eps=1e-5)



    # print q[:-1].sum(), s[:-1].sum()


