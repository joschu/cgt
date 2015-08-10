import ctypes
from cgt.core import *

def cudnn_conv_closure(*ints):
    return (ctypes.c_int*len(ints))(*ints)

def make_closure(ph, pw, sv, sh):
    return [
        ("ph",ctypes.c_int,ph),
        ("pw",ctypes.c_int,pw),
        ("sv",ctypes.c_int,sv),
        ("sh",ctypes.c_int,sh),
    ]

class CudnnConvForward(Op):
    def __init__(self, ph, pw, sv, sh):
        "pad_height, pad_width, stride_vertical, stride_horizontal"
        self.ph = ph
        self.pw = pw
        self.sv = sv
        self.sh = sh

    def get_closure(self, _inputs):
        return make_closure(self.ph, self.pw, self.sv, self.sh)
    def get_cuda_impl(self, inputs):
        return CUDAImpl(code = """
void CGT_FUNCNAME(void* cldata, cgt_array** io) {
    CudaPerformConvForward(io[0], io[1], io[2], io[3], (conv_closure*)cldata, stream, handle)
}""", includes=["cudnn_conv.cuh"], link_flags="-lcudnn")
    def shp_apply(self, inputs):
        X,W,_b = inputs
        h = cgt.ceil_divide(cgt.size(X,2)  - cgt.size(W, 2) + 1, self.sv)
        w = cgt.ceil_divide(cgt.size(X,3)  - cgt.size(W, 3) + 1, self.sh)
        return [cgt.size(X,0), cgt.size(W,0), h, w]
    def typ_apply(self, _inputs):
        return TensorType(cgt.floatX, 4)
    def pullback(self, inputs, output, gout):
        X,W,b = inputs
        # pass in an extra first argument to make output shape computation simpler
        return [Result(CudnnConvBackwardData(self.ph, self.pw, self.sv, self.sh),   [X,   gout, W]), 
                Result(CudnnConvBackwardFilter(self.ph, self.pw, self.sv, self.sh), [W,   gout, X]), 
                Result(CudnnConvBackwardBias(self.ph, self.pw, self.sv, self.sh),   [b,   gout])]

class CudnnConvBackwardData(Op):
    def __init__(self, ph, pw, sv, sh):
        self.ph = ph
        self.pw = pw
        self.sv = sv
        self.sh = sh    
    def get_cuda_impl(self, _inputs, funcname):
        return CUDAImpl(code="""
void CGT_FUNCNAME(void* cldata, cgt_array** io) {
    CudaPerformConvBackwardData(io[1], io[2], io[3], (conv_closure*)cldata, stream, handle);
}""", includes=["cudnn_conv.cuh"], link_flags="-lcudnn"))
    def cuda_includes(self): 
        return ["cudnn_conv.hpp"]
    def shp_apply(self, inputs):
        return cgt.shape(inputs[0])
    def typ_apply(self, _inputs):
        return TensorType(cgt.floatX, 4)

class CudnnConvBackwardFilter(Op):
    def __init__(self, ph, pw, sv, sh):
        self.ph = ph
        self.pw = pw
        self.sv = sv
        self.sh = sh        
    def get_cuda_impl(self, _inputs, funcname):
        return CUDAImpl("""
void CGT_FUNCNAME(void* cldata, cgt_array** io) {
    CudaPerformConvBackwardFilter(io[1], io[2], io[3],  (conv_closure*)cldata, stream, handle);
}"""%dict(funcname=funcname),includes=["cudnn_conv.cuh"], link_flags="-lcudnn")
    def shp_apply(self, inputs):
        return cgt.shape(inputs[0])
    def typ_apply(self, _inputs):
        return TensorType(cgt.floatX, 4)

class CudnnConvBackwardBias(Op):
    def __init__(self, ph, pw, sv, sh):
        self.ph = ph
        self.pw = pw
        self.sv = sv
        self.sh = sh    
    def get_cuda_impl(self, _inputs, funcname):
        return CUDAImpl("""
void CGT_FUNCNAME(void* cldata, cgt_array** io) {
    CudaPerformConvBackwardBias(io[1], io[2], io[3],  (conv_closure*)cldata, stream, handle);
}"""%dict(funcname=funcname),includes=["cudnn_conv.cuh"], link_flags="-lcudnn")
    def shp_apply(self, inputs):
        return cgt.shape(inputs[0])
    def typ_apply(self, _inputs):
        return TensorType(cgt.floatX, 4)
