import cgt
from cgt import core
import ctypes
from collections import namedtuple

Im2ColInfo = namedtuple("Im2ColInfo", ["kernel_h", "kernel_w", "pad_h", "pad_w", "stride_h", "stride_w"])

def im2col(x, kernelshape, pad, stride):
    assert stride[0] > 0 and stride[1] > 0
    assert kernelshape[0] > 0 and kernelshape[1] > 0
    kernelshape, pad, stride = map(tuple, (kernelshape, pad, stride))
    return core.Result(Im2Col(Im2ColInfo(*(kernelshape+pad+stride))), [x])

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
    available_impls = ("native_cpu",)        
    def __init__(self, info):
        assert info.stride_h>0 and info.stride_w>0
        self.info = info
    def get_diff(self, _):
        return [True]
    def get_py_impl(self):
        raise core.MethodNotDefined
    def pullback(self, (x,), _y, gy):
        return [core.Result(Col2Im(self.info), [gy] + cgt.shape(x))]
    def shp_apply(self, inputs):
        info = self.info
        batch_size, channels, height, width = cgt.shape(inputs[0])
        height_out = (height + 2 * info.pad_h - info.kernel_h) // info.stride_h + 1
        width_out = (width + 2 * info.pad_w - info.kernel_w) // info.stride_w + 1
        return [batch_size ,  height_out,  width_out, channels * info.kernel_w * info.kernel_h]
    def typ_apply(self, inputs):
        assert inputs[0].ndim == 4
        return core.TensorType(inputs[0].dtype, 4)
    def get_native_compile_info(self, input_types, devtype):
        assert devtype == "cpu"
        d = dict(cdtype=core.np2c[input_types[0].dtype])
        d.update(self.info.__dict__)
        code = r"""
            CGT_EXPORT_C void $function($closure* cl, cgtArray** reads, cgtArray* write) {
                cgtArray* im = reads[0];
                const size_t* imshape = im->shape();
                int batchsize = imshape[0],
                    channels = imshape[1],
                    height = imshape[2],
                    width = imshape[3];
                for (int i=0; i < batchsize; ++i) {
                    im2col_cpu<%(cdtype)s, %(kernel_h)s,%(kernel_w)s,%(pad_h)s,%(pad_w)s,%(stride_h)s,%(stride_w)s>
                    ((%(cdtype)s*)im->data() + im->stride(0)*i, channels, height, width, (%(cdtype)s*)write->data() + write->stride(0)*i);
                }
            }"""%d
        return core.NativeCompileInfo(code, includes=["im2col.h"], closure_triples=info2closure(self.info))

class Col2Im(core.Op):
    available_impls = ("native_cpu",)            
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
    def get_native_compile_info(self, input_types, devtype):
        d = dict(cdtype=core.np2c[input_types[0].dtype])
        d.update(self.info.__dict__)
        code = r"""
            CGT_EXPORT_C void $function($closure* cl, cgtArray** reads, cgtArray* write) {
                cgtArray* col = reads[0];
                size_t batchsize = reads[1]->at<size_t>(0),
                       channels  = reads[2]->at<size_t>(0),
                       height    = reads[3]->at<size_t>(0),
                       width     = reads[4]->at<size_t>(0);
                for (int i=0; i < batchsize; ++i) {
                    col2im_cpu<%(cdtype)s, %(kernel_h)s,%(kernel_w)s,%(pad_h)s,%(pad_w)s,%(stride_h)s,%(stride_w)s>
                    ((%(cdtype)s*)col->data() + col->stride(0)*i, channels, height, width,(%(cdtype)s*)write->data() + write->stride(0)*i);
                }
            }"""%d
        return core.NativeCompileInfo(code, includes=["im2col.h"], closure_triples=info2closure(self.info))

