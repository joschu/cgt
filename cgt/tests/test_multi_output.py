import cgt, numpy as np
from cgt import core

class SinCos(core.Op):
    call_type = "byval"
    available_impls = ("python",)
    def typ_apply(self, inputs):
        assert inputs[0].dtype == 'f4'
        d = inputs[0].ndim
        return core.TupleType(core.TensorType(cgt.floatX, d), core.TensorType(cgt.floatX, d))
    def shp_apply(self, inputs):
        return (cgt.shape(inputs[0]), cgt.shape(inputs[0]))
    def get_py_func(self, inputs):
        def f(reads):
            x = reads[0]
            return (np.sin(x), np.cos(x))
        return f
    # c_extra_link_flags = "-lm"
    # c_extra_includes = ["math.h"]
#     def get_c_impl(self, inputs):
#         code = """
# void CGT_FUNCNAME(void* cldata, cgt_array** io) {
#     float* x = io[0]->data();
#     float* y = io[1]->data();
#     float* z = io[2]->data();
#     y[0] = sinf(x[0]);
#     z[0] = cosf(x[0]);
# }"""
#         return CImpl(code, includes=["math.h"], link_flags="-lm")

class SinCos2(core.Op):
    call_type = "byref"
    available_impls = ("python","native_cpu")    
    def typ_apply(self, inputs):
        ndim = inputs[0].ndim
        return core.TupleType(core.TensorType(cgt.floatX, ndim), core.TensorType(cgt.floatX, ndim))
    def shp_apply(self, inputs):
        return (cgt.shape(inputs[0]), cgt.shape(inputs[0]))
    def get_py_func(self, inputs):
        def f(reads, write):
            x = reads[0]
            write[0][...] = np.sin(x)
            write[1][...] = np.cos(x)
        return f
    def get_native_compile_info(self, input_types,devtype):
        assert devtype=="cpu"
        code = """
CGT_EXPORT_C void $function(void* cldata, cgtArray** reads, cgtTuple* write) {
    float* x = static_cast<float*>(reads[0]->data());
    float* y = static_cast<float*>(static_cast<cgtArray*>(write->getitem(0))->data());
    float* z = static_cast<float*>(static_cast<cgtArray*>(write->getitem(1))->data());
    for (int i=0; i < reads[0]->size(); ++i) {
        y[i] = sinf(x[i]);
        z[i] = cosf(x[i]);    
    }
}"""
        return core.NativeCompileInfo(self, 1, "c++", code, includes=["math.h"], link_flags="-lm")

def test_multi_output():
    cgt.reset_config()
    cgt.set_precision("single")    
    for x in (cgt.scalar('x'), cgt.vector('x'), cgt.matrix('x')):
        for cls in (SinCos, SinCos2):
            y,z = core.unpack(core.Result(cls(), [x]))
            xnum = np.ones((3,)*x.ndim, cgt.floatX)
            correct = (np.sin(xnum),np.cos(xnum))
            yznum = cgt.numeric_eval([y,z], {x:xnum})
            np.testing.assert_allclose(yznum, correct)
            f = cgt.function([x],[y,z])
            np.testing.assert_allclose(f(xnum), correct)


