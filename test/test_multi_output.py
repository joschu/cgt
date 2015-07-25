import cgt, numpy as np
class SinCos(cgt.Op):
#     def c_code(self, inputs):
#         return """
# void CGT_FUNCNAME(void* cldata, cgt_array** io) {
#     float* x = io[0]->data;
#     float* y = io[1]->data;
#     float* z = io[2]->data;
#     y[0] = sinf(x[0]);
#     z[0] = cosf(x[0]);
# }
#         """
    def typ_apply(self, inputs):
        return cgt.Tuple(cgt.Tensor(cgt.floatX, 0), cgt.Tensor(cgt.floatX, 0))
    def get_numeric_py(self):
        def fn(x):
            return (np.sin(x),np.cos(x))
        return fn
    def shp_apply(self, inputs):
        return (cgt.shape(inputs[0]), cgt.shape(inputs[0]))
    c_extra_link_flags = "-lm"
    c_extra_includes = ["math.h"]


x = cgt.scalar('x')
y,z = cgt.unpack(cgt.Result(SinCos(), [x]))
xnum = 1.0
yznum = cgt.numeric_eval([y,z], {x:xnum})
print yznum, (np.sin(1),np.cos(1))
f = cgt.make_function([x],[y,z])
print f(xnum)