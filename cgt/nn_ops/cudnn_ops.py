import ctypes
import cgt
from cgt import core
from collections import namedtuple

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

class CudnnConvForward(core.Op):
    available_impls = ("native_gpu",)    
    def __init__(self, ph, pw, sv, sh):
        "pad_height, pad_width, stride_vertical, stride_horizontal"
        self.ph = ph
        self.pw = pw
        self.sv = sv
        self.sh = sh

    def get_native_compile_info(self, _input_types, devtype):
        assert devtype=="gpu"
        code = """
            CGT_EXPORT_C void $setup(conv_closure* closure) {setup_cudnn(closure);}
            CGT_EXPORT_C void $teardown(conv_closure* closure) {teardown_cudnn(closure);}
            CGT_EXPORT_C void $function(conv_closure* closure, cgtArray** reads, cgtArray* write) {
                if (!closure->handle) setup_cudnn(closure);
                performConvForward(closure, reads[0], reads[1], reads[2], write);
            }"""
        return core.NativeCompileInfo(code, closure_triples = make_closure(self.ph, self.pw, self.sv, self.sh),
            includes=["cudnn_support.h"], link_flags="-lcudnn -lcudart")
    def shp_apply(self, inputs):
        X,W,_b = inputs
        h = cgt.ceil_divide(cgt.size(X,2) + self.ph*2 - cgt.size(W, 2) + 1, self.sv)
        w = cgt.ceil_divide(cgt.size(X,3) + self.pw*2 - cgt.size(W, 3) + 1, self.sh)
        return [cgt.size(X,0), cgt.size(W,0), h, w]
    def typ_apply(self, _inputs):
        return core.TensorType(cgt.floatX, 4)
    def pullback(self, inputs, _output, gout):
        X,W,b = inputs
        # pass in an extra first argument to make output shape computation simpler
        return [core.Result(CudnnConvBackwardData(self.ph, self.pw, self.sv, self.sh),   [X,   gout, W]), 
                core.Result(CudnnConvBackwardFilter(self.ph, self.pw, self.sv, self.sh), [W,   gout, X]), 
                core.Result(CudnnConvBackwardBias(self.ph, self.pw, self.sv, self.sh),   [b,   gout])]

class CudnnConvBackwardData(core.Op):
    available_impls = ("native_gpu",)    
    def __init__(self, ph, pw, sv, sh):
        self.ph = ph
        self.pw = pw
        self.sv = sv
        self.sh = sh    
    def get_native_compile_info(self, input_types, devtype):
        assert devtype=="gpu"
        code="""
            CGT_EXPORT_C void $setup(conv_closure* closure) {setup_cudnn(closure);}
            CGT_EXPORT_C void $teardown(conv_closure* closure) {teardown_cudnn(closure);}
            CGT_EXPORT_C void $function(conv_closure* closure, cgtArray** reads, cgtArray* write) {
                if (!closure->handle) setup_cudnn(closure);
                performConvBackwardData(closure, reads[1], reads[2], write);
            }"""
        return core.NativeCompileInfo(code, closure_triples = make_closure(self.ph, self.pw, self.sv, self.sh),
            includes=["cudnn_support.h"], link_flags="-lcudnn -lcudart")
    def shp_apply(self, inputs):
        return cgt.shape(inputs[0])
    def typ_apply(self, _inputs):
        return core.TensorType(cgt.floatX, 4)

class CudnnConvBackwardFilter(core.Op):
    available_impls = ("native_gpu",)    
    def __init__(self, ph, pw, sv, sh):
        self.ph = ph
        self.pw = pw
        self.sv = sv
        self.sh = sh        
    def get_native_compile_info(self, input_types, devtype):
        assert devtype=="gpu"
        code = """
            CGT_EXPORT_C void $setup(conv_closure* closure) {setup_cudnn(closure);}
            CGT_EXPORT_C void $teardown(conv_closure* closure) {teardown_cudnn(closure);}
            CGT_EXPORT_C void $function(conv_closure* closure, cgtArray** reads, cgtArray* write) {
                if (!closure->handle) setup_cudnn(closure);
                performConvBackwardFilter(closure, reads[1], reads[2], write);
            }"""
        return core.NativeCompileInfo(code, closure_triples = make_closure(self.ph, self.pw, self.sv, self.sh),
            includes=["cudnn_support.h"], link_flags="-lcudnn -lcudart")
    def shp_apply(self, inputs):
        return cgt.shape(inputs[0])
    def typ_apply(self, _inputs):
        return core.TensorType(cgt.floatX, 4)

class CudnnConvBackwardBias(core.Op):
    available_impls = ("native_gpu",)
    def __init__(self, ph, pw, sv, sh):
        self.ph = ph
        self.pw = pw
        self.sv = sv
        self.sh = sh    
    def get_native_compile_info(self, input_types, devtype):
        assert devtype == "gpu"
        code = """
            CGT_EXPORT_C void $setup(conv_closure* closure) {setup_cudnn(closure);}
            CGT_EXPORT_C void $teardown(conv_closure* closure) {teardown_cudnn(closure);}
            CGT_EXPORT_C void $function(conv_closure* closure, cgtArray** reads, cgtArray* write) {
                if (!closure->handle) setup_cudnn(closure);
                performConvBackwardBias(closure, reads[1], write);
            }"""
        return core.NativeCompileInfo(code, closure_triples = make_closure(self.ph, self.pw, self.sv, self.sh),
            includes=["cudnn_support.h"], link_flags="-lcudnn -lcudart")            
    def shp_apply(self, inputs):
        return cgt.shape(inputs[0])
    def typ_apply(self, _inputs):
        return core.TensorType(cgt.floatX, 4)

PoolInfo = namedtuple("PoolInfo", ["kernel_h", "kernel_w", "pad_h", "pad_w", "stride_h", "stride_w"])

def poolinfo2closure(info):
    return [
        ("kernel_h", ctypes.c_int, info.kernel_h),
        ("kernel_w", ctypes.c_int, info.kernel_w),
        ("pad_h", ctypes.c_int, info.pad_h),
        ("pad_w", ctypes.c_int, info.pad_w),
        ("stride_h", ctypes.c_int, info.stride_h),
        ("stride_w", ctypes.c_int, info.stride_w),
        ("handle",ctypes.c_void_p,0),
        ("stream",ctypes.c_void_p,0),
    ]    

class CudnnPoolForward(core.Op):
    available_impls = ("native_gpu",)    
    def __init__(self, info):
        self.info = info

    def get_native_compile_info(self, _input_types, devtype):
        assert devtype == "gpu"
        code = """
            CGT_EXPORT_C void $setup(pooling_closure* closure) {setup_cudnn(closure);}
            CGT_EXPORT_C void $teardown(pooling_closure* closure) {teardown_cudnn(closure);}
            CGT_EXPORT_C void $function(pooling_closure* closure, cgtArray** reads, cgtArray* write) {
                if (!closure->handle) setup_cudnn(closure);
                performPoolingForward(closure, reads[0], write);
            }"""
        return core.NativeCompileInfo(code, closure_triples = poolinfo2closure(self.info),
            includes=["cudnn_support.h"], link_flags="-lcudnn -lcudart")
    def shp_apply(self, inputs):
        info = self.info
        batch_size, channels, height, width = cgt.shape(inputs[0])
        pooled_height =  cgt.ceil_divide(height + 2*info.pad_h - info.kernel_h, info.stride_h)
        pooled_width = cgt.ceil_divide(width + 2*info.pad_w - info.kernel_w, info.stride_w)
        outshape = [batch_size ,  channels, pooled_height, pooled_width]
        return outshape
    def typ_apply(self, input_types):
        return input_types[0]
    def pullback(self, inputs, output, gout):
        return [core.Result(CudnnPoolBackward(self.info),   [inputs[0], output, gout])]


class CudnnPoolBackward(core.Op):
    available_impls = ("native_gpu",)    
    def __init__(self, info):
        self.info = info

    def get_native_compile_info(self, _input_types, devtype):
        assert devtype == "gpu"
        code = """
            CGT_EXPORT_C void $setup(pooling_closure* closure) {setup_cudnn(closure);}
            CGT_EXPORT_C void $teardown(pooling_closure* closure) {teardown_cudnn(closure);}
            CGT_EXPORT_C void $function(pooling_closure* closure, cgtArray** reads, cgtArray* write) {
                if (!closure->handle) setup_cudnn(closure);
                performPoolingBackward(closure, reads[0], reads[1], reads[2], write);
            }"""
        return core.NativeCompileInfo(code, closure_triples = poolinfo2closure(self.info),
            includes=["cudnn_support.h"], link_flags="-lcudnn -lcudart")
    def shp_apply(self, inputs):
        return cgt.shape(inputs[0])
    def typ_apply(self, input_types):
        return input_types[0]
    def pullback(self, inputs, output, gout):
        raise NotImplementedError



