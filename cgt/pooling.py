import cgt
from cgt import core
import ctypes
from collections import namedtuple
import numpy as np


# <Copied from im2col.py>
PoolInfo = namedtuple("PoolInfo", ["kernel_h", "kernel_w", "pad_h", "pad_w", "stride_h", "stride_w"])

def max_pool_2d(x, kernelshape, pad = (0,0), stride=(1,1)):
    kernel_h, kernel_w = kernelshape
    pad_h, pad_w = pad
    stride_h, stride_w = stride
    return core.Result(MaxPool(PoolInfo(kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w)), [x])[0]

def info2closure(info):
    return [
        ("kernel_h", ctypes.c_int, info.kernel_h),
        ("kernel_w", ctypes.c_int, info.kernel_w),
        ("pad_h", ctypes.c_int, info.pad_h),
        ("pad_w", ctypes.c_int, info.pad_w),
        ("stride_h", ctypes.c_int, info.stride_h),
        ("stride_w", ctypes.c_int, info.stride_w),
    ]    
# </Copied>

class MaxPool(core.Op):
    available_impls = ("native_cpu",)    
    def __init__(self, info):
        assert info.stride_h>0 and info.stride_w>0        
        self.info = info
    def get_diff(self, _):
        return [True]
    def get_py_impl(self):
        raise core.MethodNotDefined
    def pullback(self, (x,), y, gy):
        pool,mask = core.unpack(y)
        gpool,_gmask = gy
        return [core.Result(MaxPoolPullback(self.info), [x,pool,mask,gpool])]
    def shp_apply(self, inputs):
        # pooled_height_ = static_cast<int>(ceil(static_cast<float>(height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
        # pooled_width_ = static_cast<int>(ceil(static_cast<float>(width_ + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;
        info = self.info
        batch_size, channels, height, width = cgt.shape(inputs[0])
        pooled_height =  cgt.ceil_divide(height + 2*info.pad_h - info.kernel_h, info.stride_h)
        pooled_width = cgt.ceil_divide(width + 2*info.pad_w - info.kernel_w, info.stride_w)
        outshape = [batch_size ,  channels, pooled_height, pooled_width]
        return (outshape, outshape)
    def typ_apply(self, inputs):
        return core.TupleType(core.TensorType(inputs[0].dtype, 4), core.TensorType('i4', 4))
    def get_closure(self, _inputs):
        return info2closure(self.info)
    def get_native_compile_info(self, input_types, devtype):
        code = r"""
CGT_EXPORT_C void $function(conv_closure* cl, cgtArray** reads, cgtTuple* write) {
    max_pool<%(cdtype)s>(cl, reads[0], static_cast<cgtArray*>(write->getitem(0)), static_cast<cgtArray*>(write->getitem(1)));
}"""%dict(cdtype=core.np2c[input_types[0].dtype])
        return core.NativeCompileInfo(self, 1, "c++", code, 
            closure_triples=info2closure(self.info), includes=["pooling.h"])

class MaxPoolPullback(core.Op):
    available_impls = ("native_cpu",)
    def __init__(self, info):
        self.info = info
    def get_py_impl(self):
        raise core.MethodNotDefined
    def shp_apply(self, inputs):
        return cgt.shape(inputs[0])
    def typ_apply(self, inputs):
        return core.TensorType(inputs[0].dtype, 4)
    def get_closure(self, _inputs):
        return info2closure(self.info)
    def get_native_compile_info(self, input_types, devtype):
        code = r"""
CGT_EXPORT_C void $function(conv_closure* cl, cgtArray** reads, cgtArray* write) {
    max_pool_pullback<%(cdtype)s>(reads[0], reads[1], reads[2], reads[3], write);
}"""%dict(cdtype=core.np2c[input_types[0].dtype])
        return core.NativeCompileInfo(self, 4, "c++", code, 
            closure_triples=info2closure(self.info), includes=["pooling.h"])

def test():
    np.random.seed(0)
    cgt.set_precision("quad")
    x = cgt.tensor4("x", fixed_shape=(2,3,5,7))
    y = max_pool_2d(x, (4,4),(0,0),(1,1))
    xval = np.random.randn(2,3,5,7)
    hval = np.random.randn(*core.infer_shape(y))
    h = cgt.constant(hval)

    cost = (y*h).sum()

    fcost = cgt.function([x], cost)
    fgrad = cgt.function([x], cgt.grad(cost, [x])[0])

    from cgt.numeric_diff import numeric_grad
    gnum = numeric_grad(fcost, xval)
    gana = fgrad(xval)

    assert np.allclose(gnum,gana)

if __name__ == "__main__":
    test()