import cgt, numpy as np
import unittest


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
    call_type = "valret"
    def typ_apply(self, inputs):
        return cgt.Tuple(cgt.Tensor(cgt.floatX, 0), cgt.Tensor(cgt.floatX, 0))
    def shp_apply(self, inputs):
        return (cgt.shape(inputs[0]), cgt.shape(inputs[0]))
    def get_py_impl(self):
        def f(reads):
            x = reads[0]
            return (np.sin(x), np.cos(x))
        return cgt.PyImpl(valret_func=f)
    c_extra_link_flags = "-lm"
    c_extra_includes = ["math.h"]

class SinCos2(cgt.Op):
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
    call_type = "inplace"
    def typ_apply(self, inputs):
        return cgt.Tuple(cgt.Tensor(cgt.floatX, 0), cgt.Tensor(cgt.floatX, 0))
    def shp_apply(self, inputs):
        return (cgt.shape(inputs[0]), cgt.shape(inputs[0]))
    def get_py_impl(self):
        def f(reads, write):
            x = reads[0]
            write[0][...] = np.sin(x)
            write[1][...] = np.cos(x)
        return cgt.PyImpl(inplace_func=f)
    c_extra_link_flags = "-lm"
    c_extra_includes = ["math.h"]

class MultiOutputTestCase(unittest.TestCase):
    def runTest(self):
        x = cgt.scalar('x')
        for cls in (SinCos, SinCos2):
            y,z = cgt.unpack(cgt.Result(cls(), [x]))
            xnum = 1.0
            yznum = cgt.numeric_eval([y,z], {x:xnum})
            np.testing.assert_allclose(yznum, (np.sin(1),np.cos(1)))
            f = cgt.make_function([x],[y,z])
            np.testing.assert_allclose(f(xnum), yznum)


if __name__ == "__main__":
    MultiOutputTestCase().runTest()
