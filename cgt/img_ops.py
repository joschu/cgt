import cgt
from cgt.core import Op, Result, Tensor, size, shape, ceil_divide
import ctypes

# Maybe we shouldn't have special CuDNN ops, we should just have the same
# convolution interface with various implementations
# see nice blog post http://benanne.github.io/2014/12/09/theano-metaopt.html

def cudnn_conv_closure(*ints):
    return (ctypes.c_int*len(ints))(*ints)

class CudnnConvForward(Op):
    def __init__(self, ph, pw, sv, sh):
        "pad_height, pad_width, stride_vertical, stride_horizontal"
        self.ph = ph
        self.pw = pw
        self.sv = sv
        self.sh = sh

    def cuda_code(self, _inputs, funcname):
        return """
void %(funcname)s(void* cldata, cgt_array** io) {
    CudaPerformConvForward(io[0], io[1], io[2], io[3], (conv_closure*)cldata, stream, handle)
}
"""%dict(funcname=funcname)
    def cuda_includes(self): 
        return ["cudnn_conv.cuh"]
    def impl_data(self):        
        return (self.__class__.__name__,), cudnn_conv_closure(self.ph, self.pw, self.sv, self.sh)
    def shp_apply(self, inputs):
        X,W,_b = inputs
        h = ceil_divide(size(X,2)  - size(W, 2) + 1, self.sv)
        w = ceil_divide(size(X,3)  - size(W, 3) + 1, self.sh)
        return [size(X,0), size(W,0), h, w]
    def typ_apply(self, _inputs):
        return Tensor(cgt.floatX, 4)
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
    def cuda_code(self, _inputs, funcname):
        return """
void %(funcname)s(void* cldata, cgt_array** io) {
    CudaPerformConvBackwardData(io[1], io[2], io[3], (conv_closure*)cldata, stream, handle);
}
"""%dict(funcname=funcname)
    def cuda_includes(self): 
        return ["cudnn_conv.hpp"]
    def shp_apply(self, inputs):
        return shape(inputs[0])
    def typ_apply(self, _inputs):
        return Tensor(cgt.floatX, 4)

class CudnnConvBackwardFilter(Op):
    def __init__(self, ph, pw, sv, sh):
        self.ph = ph
        self.pw = pw
        self.sv = sv
        self.sh = sh        
    def cuda_code(self, _inputs, funcname):
        return """
void %(funcname)s(void* cldata, cgt_array** io) {
    CudaPerformConvBackwardFilter(io[1], io[2], io[3],  (conv_closure*)cldata, stream, handle);
}
"""%dict(funcname=funcname)
    def cuda_includes(self): 
        return ["cudnn_conv.hpp"]
    def impl_data(self):        
        return (self.__class__.__name__,), cudnn_conv_closure(self.ph, self.pw, self.sv, self.sh)
    def shp_apply(self, inputs):
        return shape(inputs[0])
    def typ_apply(self, _inputs):
        return Tensor(cgt.floatX, 4)

class CudnnConvBackwardBias(Op):
    def __init__(self, ph, pw, sv, sh):
        self.ph = ph
        self.pw = pw
        self.sv = sv
        self.sh = sh    
    def cuda_code(self, _inputs, funcname):
        return """
void %(funcname)s(void* cldata, cgt_array** io) {
    CudaPerformConvBackwardBias(io[1], io[2], io[3],  (conv_closure*)cldata, stream, handle);
}
"""%dict(funcname=funcname)
    def cuda_includes(self): 
        return ["cudnn_conv.hpp"]
    def shp_apply(self, inputs):
        return shape(inputs[0])
    def typ_apply(self, _inputs):
        return Tensor(cgt.floatX, 4)

# def pool(x_ncuv, rows_in, cols_in, poolshp, pool_type='max'):
#     if rows_in % poolshp[0] != 0 or cols_in % poolshp[0] != 0:
#         row_residue = rows_in%poolshp[0]
#         col_residue = cols_in%poolshp[1]
#         warn("image shape not divisible by pool size. cropping %i/%i on top, %i/%i on left"%(row_residue,rows_in,col_residue,cols_in))
#         x_ncuv = x_ncuv[:,:,:rows_in - row_residue, :cols_in - col_residue]
#     x_ncpaqb = x_ncuv.reshape( (x_ncuv.shape[0], x_ncuv.shape[1], rows_in // poolshp[0], poolshp[0], cols_in // poolshp[1], poolshp[1]) )
#     x_ncpqab = x_ncpaqb.transpose([0,1,2,4,3,5])
#     x_ncpq_ab = cgt.reshape(x_ncpqab, shape(x_ncpqab)[:4] + [size(x_ncpqab,4)*size(x_ncpqab,5)])
#     if pool_type == 'max':
#         x_ncpq = x_ncpq_ab.max(axis=4)
#     elif pool_type == 'mean':
#         x_ncpq = x_ncpq_ab.mean(axis=4)
#     elif pool_type == '2norm':
#         x_ncpq = cgt.sqrt(cgt.square(x_ncpq_ab).sum(axis=4)) #pylint: disable=E1111
#     elif pool_type == 'softmax':
#         x_ncpq = cgt.log(cgt.exp(x_ncpq_ab).sum(axis=4)) #pylint: disable=E1111
#     assert x_ncpq.ndim==4
#     return x_ncpq


class Pool(Op):
    def __init__(self, kind, stride, kernel, pad):
        self.kind = kind
        self.stride = stride
        self.kernel = kernel
        self.pad = pad
    def get_diff(self, _):
        return [True]
    def get_name(self):
        return "%spool"%self.kind
    def get_numeric_py(self):
        raise cgt.exceptions.Todo
    def pullback(self, inputs, output, goutput):
        raise cgt.exceptions.Todo
    def shp_apply(self, inputs):        
        x = inputs[0]
        assert x.ndim == 4
        return [size(x,0), size(x,1), (size(x,2)-self.pad[0]-self.kernel[0]+1)//self.stride[0],  
                                      (size(x,3)-self.pad[1]-self.kernel[1]+1)//self.stride[0]] 
                                      # XXX round up or down?
    def typ_apply(self, inputs):
        return inputs[0].get_type()

def pool(kind, x, stride, kernel, pad):
    return Result(Pool(kind,stride,kernel,pad), [x])

def lrn(x, alpha, beta, local_size):
    s = Result(CudaLRNScaling(alpha, local_size), [x])
    return s/cgt.power(s, -beta)

# XXX needs params
class CudaLRNScaling(Op):
    def __init__(self, alpha, local_size):
        self.alpha = alpha
        self.local_size = local_size
    def cuda_code(self, _inputs, funcname):
        return """
void %(funcname)s(void* cldata, cgt_array** io) {
  int block, thread, size;
  size = num_img * height * width;
  FindConfiguration(size, block, thread);
  cgt_array* bottom=io[0], *scale=io[1];
  ing num_img = bottom->shape[0], channel = bottom->shape[1],
    height = bottom->shape[2], width=bottom->shape[2];
  LRNFillScale<<<block, thread, 0, stream>>>(
      size, bottom->data, num_img, channel, height, width, cl->local_size,
      cl->alpha / cl->local_size, scale->data);
}"""%dict(funcname=funcname)
    def cuda_headers(self):
        return ["cgt_cuda.h","lrn.cuh"]
    def shp_apply(self, inputs):
        return shape(inputs[0])
    def typ_apply(self, _inputs):
        return Tensor(cgt.floatX, 4)
