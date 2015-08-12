import cgt
from cgt import core
import ctypes
from collections import namedtuple
import numpy as np

Im2ColInfo = namedtuple("Im2ColInfo", ["kernel_h", "kernel_w", "pad_h", "pad_w", "stride_h", "stride_w"])

def im2col(x, info):
    return core.Result(Im2Col(info), [x])

def info2closure(info):
    return [
        ("kernel_h", ctypes.c_int, info.kernel_h),
        ("kernel_w", ctypes.c_int, info.kernel_w),
        ("pad_h", ctypes.c_int, info.pad_h),
        ("pad_w", ctypes.c_int, info.pad_w),
        ("stride_h", ctypes.c_int, info.stride_h),
        ("stride_w", ctypes.c_int, info.stride_w),
    ]    


class Im2Col(core.Op):
    def __init__(self, info):
        self.info = info
    def get_diff(self, _):
        return [True]
    def get_py_impl(self):
        raise core.MethodNotDefined
    def pullback(self, (x,), _y, gy):
        return core.Result(Col2Im(self.info), [gy] + cgt.shape(x))
    def shp_apply(self, inputs):
        info = self.info
        batch_size, channels, height, width = cgt.shape(inputs[0])
        height_out = (height + 2 * info.pad_h - info.kernel_h) // info.stride_h + 1
        width_out = (width + 2 * info.pad_w - info.kernel_w) // info.stride_w + 1
        return [batch_size ,  height_out,  width_out, channels * info.kernel_w * info.kernel_h]
    def typ_apply(self, inputs):
        return core.TensorType(inputs[0].dtype, 4)
    def get_closure(self, _inputs):
        return info2closure(self.info)
    def get_c_impl(self, inputs):
        code = r"""
extern "C" void $function($closure* cl, cgtArray** reads, cgtArray* write) {
    cgtArray* im = reads[0];
    const size_t* imshape = im->shape();
    int batchsize = imshape[0],
        channels = imshape[1],
        height = imshape[2],
        width = imshape[3];
    for (int i=0; i < batchsize; ++i) {
        im2col_cpu((%(cdtype)s*)im->data() + im->stride(0)*i, channels, height, width,
            cl->kernel_h, cl->kernel_w, cl->pad_h, cl->pad_w, cl->stride_h, 
            cl->stride_w, (%(cdtype)s*)write->data() + write->stride(0)*i);
    }
}"""%dict(cdtype=core.np2c[inputs[0].dtype])
        return core.CImpl(code=code, includes=["im2col.h"])

class Col2Im(core.Op):
    def __init__(self, info):
        self.info = info
    def get_diff(self, _):
        return [True]
    def get_py_impl(self):
        raise core.MethodNotDefined
    def shp_apply(self, inputs):
        return inputs[1:]
    def typ_apply(self, inputs):
        return core.TensorType(inputs[0].dtype, 4)
    def get_closure(self, _inputs):
        return info2closure(self.info)
    def get_c_impl(self, inputs):
        code = r"""
extern "C" void $function($closure* cl, cgtArray** reads, cgtArray* write) {
    cgtArray* col = reads[0];
    size_t batchsize = reads[1]->at<size_t>(0),
           channels  = reads[2]->at<size_t>(0),
           height    = reads[3]->at<size_t>(0),
           width     = reads[4]->at<size_t>(0);
    for (int i=0; i < batchsize; ++i) {
        col2im_cpu((%(cdtype)s*)col->data() + col->stride(0)*i, channels, height, width,
            cl->kernel_h, cl->kernel_w, cl->pad_h, cl->pad_w, cl->stride_h, 
            cl->stride_w, (%(cdtype)s*)write->data() + write->stride(0)*i);
    }
}"""%dict(cdtype=core.np2c[inputs[0].dtype])
        return core.CImpl(code=code, includes=["im2col.h"])

def test():
    np.random.seed(0)
    cgt.set_precision("quad")
    x = cgt.tensor4("x", fixed_shape=(2,3,5,7))
    info = Im2ColInfo(4,4,0,0,1,1)
    y = cgt.core.Result(Im2Col(info), [x])
    xval = np.arange(2*3*5*7).reshape(2,3,5,7).astype(cgt.floatX)
    h = cgt.constant(np.random.randn(*core.infer_shape(y)))
    cost = (y*h).sum()
    fcost = cgt.function([x],cost)
    fgrad = cgt.function([x], cgt.grad(cost, [x])[0])

    from cgt.numeric_diff import numeric_grad
    gnum = numeric_grad(fcost, xval)
    gana = fgrad(xval)
    assert np.allclose(gnum, gana)

    fy = cgt.function([x],y)
    yval = fy(xval)
    assert np.allclose(yval[0,0,0] , xval[0,:,0:4,0:4].flatten())
