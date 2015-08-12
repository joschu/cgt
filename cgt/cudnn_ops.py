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
        ("handle",ctypes.c_void_p),
        ("stream",ctypes.c_void_p),
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
        return CUDAImpl(
            code = """
extern "C" void $function(conv_closure* closure, cgt_array** io) {
    performConvForward(closure, io[0], io[1], io[2], io[3])
}""", 
        includes=["cudnn_support.h"], link_flags="-lcudnn", setup="setupConv", teardown="teardownConv")
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
extern "C" void $function(conv_closure* closure, cgt_array** io) {
    performConvBackwardData(closure io[1], io[2], io[3]);
}
extern "C" void CGT_FUNCNAME_setup(conv_closure* closure) {conv_closure_setup(closure);}
extern "C" void CGT_FUNCNAME_teardown(conv_closure* closure) {conv_closure_setup(closure);}
""", includes=["cudnn_support.h"], link_flags="-lcudnn", setup="setupConv", teardown="teardownConv"))
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
extern "C" void $function(conv_closure* closure, cgt_array** io) {
    performConvBackwardFilter(closure io[1], io[2], io[3]);
}"""%dict(funcname=funcname),includes=["cudnn_support.h"], link_flags="-lcudnn", setup="setupConv", teardown="teardownConv")
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
extern "C" void $function(conv_closure* closure, cgt_array** io) {
    performConvBackwardBias(closure io[1], io[2], io[3]);
}"""%dict(funcname=funcname),includes=["cudnn_support.h"], link_flags="-lcudnn", setup="setupConv", teardown="teardownConv")
    def shp_apply(self, inputs):
        return cgt.shape(inputs[0])
    def typ_apply(self, _inputs):
        return TensorType(cgt.floatX, 4)
