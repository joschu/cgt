import ctypes
import cgt
from cgt import core
from collections import namedtuple

LRNInfo = namedtuple("LRNInfo",["localsize","alpha","beta"])

def make_closure(info):
    return [
        ("localsize",ctypes.c_int,info.localsize),
        ("alpha",ctypes.c_double,info.alpha),
        ("beta",ctypes.c_double,info.beta)
    ]

class CrossChannelLRNForward(core.Op):
    available_impls = ("native_gpu",)    
    def __init__(self, info):
        assert isinstance(info, LRNInfo)
        self.info = info

    def get_native_compile_info(self, input_types, devtype):
        assert devtype == "gpu"
        d = dict(cdtype=core.np2c[input_types[0].dtype])
        cuda_code = r"""
            #include "cgt_cuda.h"
            #include "lrn.cuh"
            void launchker_$function(int num_img, int channels, int height, int width, int localsize, double alpha, double beta, %(cdtype)s* Xdata, %(cdtype)s* topdata, %(cdtype)s* scaledata) {
                int size = num_img * height * width;
                int nblocks, nthreads;
                cgt_get_bt(size, nblocks, nthreads);
                LRNFillScale<%(cdtype)s><<<nblocks, nthreads, 0>>>(
                    size, Xdata, num_img, channels, height, width, localsize, alpha / localsize, scaledata);
                CUDA_CHECK_ERROR("LRNFillScale");

                size = num_img * channels * width * height;
                cgt_get_bt(size, nblocks, nthreads);
                LRNComputeOutput<%(cdtype)s><<<nblocks, nthreads, 0>>>(size, Xdata, scaledata, -beta, topdata);
                CUDA_CHECK_ERROR("LRNComputeOutput");
            }"""%d
        code = r"""
            extern void launchker_$function(int num_img, int channels, int height, int width, int localsize, double alpha, double beta, %(cdtype)s* Xdata, %(cdtype)s* topdata, %(cdtype)s* scaledata);
            CGT_EXPORT_C void $function($closure* cldata, cgtArray** reads, cgtTuple* write) {
                cgtArray* X = reads[0];
                int num_img = X->shape()[0],
                    channels = X->shape()[1],
                    height = X->shape()[2],
                    width = X->shape()[3];
                cgtArray* top = (cgtArray*)write->getitem(0);                
                cgtArray* scale = (cgtArray*)write->getitem(1);
                launchker_$function(num_img, channels, height, width, cldata->localsize, cldata->alpha, cldata->beta, (%(cdtype)s*)X->data(), (%(cdtype)s*)top->data(), (%(cdtype)s*)scale->data());

            }"""%d
        return core.NativeCompileInfo(code, closure_triples = make_closure(self.info),
            link_flags="-lcudart", gpu_deref_mask=(True,), 
            extra_srcs=[core.SrcFile("cuda",cuda_code)])
    def shp_apply(self, inputs):
        return (inputs[0].shape,inputs[0].shape)
    def typ_apply(self, input_types):
        return core.TupleType(input_types[0], input_types[0])
    def pullback(self, inputs, output, gout):
        top, scaling = cgt.core.unpack(output)
        gtop, _ = gout
        return [core.Result(CrossChannelLRNBackward(self.info), [inputs[0], top, scaling, gtop])]

class CrossChannelLRNBackward(core.Op):
    available_impls = ("native_gpu",)    
    def __init__(self, info):
        self.info = info
    def get_native_compile_info(self, input_types, devtype):
        assert devtype == "gpu"   
        d = dict(cdtype=core.np2c[input_types[0].dtype])     
        cuda_code=r"""
            #include "cgt_cuda.h"
            #include "lrn.cuh"
            void launchker_$function(int num_img, int channels, int height, int width, int localsize, double alpha, double beta, %(cdtype)s* Xdata, 
                %(cdtype)s* topdata, %(cdtype)s* scalingdata, %(cdtype)s* topdiffdata, %(cdtype)s* bottomdiffdata) {
                int nblocks, nthreads;
                int size = num_img * width * height;
                cgt_get_bt(size, nblocks, nthreads);
                LRNComputeDiff<%(cdtype)s><<<nblocks, nthreads, 0>>>(size, (%(cdtype)s*)Xdata, (%(cdtype)s*)topdata, 
                    (%(cdtype)s*)scalingdata, (%(cdtype)s*)topdiffdata,  num_img, channels, height, width, localsize, 
                    -beta, 2. * alpha * beta / localsize, (%(cdtype)s*)bottomdiffdata);
                CUDA_CHECK_ERROR("CrossChannelLRNBackward");
            }
        """%d
        code = """        
            void launchker_$function(int num_img, int channels, int height, int width, int localsize, double alpha, double beta, %(cdtype)s* Xdata, 
                %(cdtype)s* topdata, %(cdtype)s* scaledata, %(cdtype)s* topdiffdata, %(cdtype)s* bottomdiffdata);
            CGT_EXPORT_C void $function($closure* cldata, cgtArray** reads, cgtArray* bottom_diff) {
            cgtArray *X=reads[0], *top=reads[1], *scaling=reads[2], *top_diff=reads[3];
            int num_img = X->shape()[0],
                channels = X->shape()[1],
                height = X->shape()[2],
                width = X->shape()[3];
            launchker_$function(num_img, channels, height, width, cldata->localsize, cldata->alpha, cldata->beta, (%(cdtype)s*)X->data(), 
                (%(cdtype)s*)top->data(), (%(cdtype)s*)scaling->data(), (%(cdtype)s*)top_diff->data(), (%(cdtype)s*)bottom_diff->data());            
            }"""%d
        return core.NativeCompileInfo(code, closure_triples = make_closure(self.info),
            link_flags="-lcudart", gpu_deref_mask=(True,True,True,True), 
            extra_srcs=[core.SrcFile("cuda",cuda_code)])
    def shp_apply(self, inputs):
        return cgt.shape(inputs[0])
    def typ_apply(self, _inputs):
        return core.TensorType(cgt.floatX, 4)

def cross_channel_lrn(X, localsize, alpha, beta):
    assert X.ndim == 4
    return core.Result(CrossChannelLRNForward(LRNInfo(localsize,alpha,beta)), [X])[0]


    # print q[:-1].sum(), s[:-1].sum()


