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
        ("handle",ctypes.c_void_p,0),
        ("stream",ctypes.c_void_p,0),
    ]

class CudnnConvForward(Op):
    available_impls = ("native_gpu",)    
    def __init__(self, ph, pw, sv, sh):
        "pad_height, pad_width, stride_vertical, stride_horizontal"
        self.ph = ph
        self.pw = pw
        self.sv = sv
        self.sh = sh

    def get_native_compile_info(self, _input_types, devtype):
        code = """
            CGT_EXPORT_C void $function(conv_closure* closure, cgtArray** reads, cgtArray* write) {
                if (!closure->handle) setupConv(closure);
                performConvForward(closure, reads[0], reads[1], reads[2], write);
            }"""
        return NativeCompileInfo(code, closure_triples = make_closure(self.ph, self.pw, self.sv, self.sh),
            includes=["cudnn_support.h"], link_flags="-lcudnn -lcudart", gpu_deref_mask=(True,True,True,True))
    def shp_apply(self, inputs):
        X,W,_b = inputs
        h = cgt.ceil_divide(cgt.size(X,2) + self.ph*2 - cgt.size(W, 2) + 1, self.sv)
        w = cgt.ceil_divide(cgt.size(X,3) + self.pw*2 - cgt.size(W, 3) + 1, self.sh)
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
    available_impls = ("native_gpu",)    
    def __init__(self, ph, pw, sv, sh):
        self.ph = ph
        self.pw = pw
        self.sv = sv
        self.sh = sh    
    def get_native_compile_info(self, input_types, devtype):
        code="""
            CGT_EXPORT_C void $function(conv_closure* closure, cgtArray** reads, cgtArray* write) {
                if (!closure->handle) setupConv(closure);
                performConvBackwardData(closure, reads[1], reads[2], write);
            }"""
        return NativeCompileInfo(code, closure_triples = make_closure(self.ph, self.pw, self.sv, self.sh),
            includes=["cudnn_support.h"], link_flags="-lcudnn -lcudart")
    def shp_apply(self, inputs):
        return cgt.shape(inputs[0])
    def typ_apply(self, _inputs):
        return TensorType(cgt.floatX, 4)

class CudnnConvBackwardFilter(Op):
    available_impls = ("native_gpu",)    
    def __init__(self, ph, pw, sv, sh):
        self.ph = ph
        self.pw = pw
        self.sv = sv
        self.sh = sh        
    def get_native_compile_info(self, input_types, devtype):
        code = """
            CGT_EXPORT_C void $function(conv_closure* closure, cgtArray** reads, cgtArray* write) {
                if (!closure->handle) setupConv(closure);
                performConvBackwardFilter(closure, reads[1], reads[2], write);
            }"""
        return NativeCompileInfo(code, closure_triples = make_closure(self.ph, self.pw, self.sv, self.sh),
            includes=["cudnn_support.h"], link_flags="-lcudnn -lcudart")
    def shp_apply(self, inputs):
        return cgt.shape(inputs[0])
    def typ_apply(self, _inputs):
        return TensorType(cgt.floatX, 4)

class CudnnConvBackwardBias(Op):
    available_impls = ("native_gpu",)
    def __init__(self, ph, pw, sv, sh):
        self.ph = ph
        self.pw = pw
        self.sv = sv
        self.sh = sh    
    def get_native_compile_info(self, input_types, devtype):
        code = """
            CGT_EXPORT_C void $function(conv_closure* closure, cgtArray** reads, cgtArray* write) {
                if (!closure->handle) setupConv(closure);
                performConvBackwardBias(closure, reads[1], write);
            }"""
        return NativeCompileInfo(code, closure_triples = make_closure(self.ph, self.pw, self.sv, self.sh),
            includes=["cudnn_support.h"], link_flags="-lcudnn -lcudart")            
    def shp_apply(self, inputs):
        return cgt.shape(inputs[0])
    def typ_apply(self, _inputs):
        return TensorType(cgt.floatX, 4)


