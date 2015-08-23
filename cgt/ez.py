import cgt
from cgt import core

class EasyCustomOp(core.Op):
    return_type = "byval"
    def __init__(self, input_types, output_type, forward_impl, pullback_impl=None,shapefun=None):
        assert all(isinstance(typ, core.TensorType) for typ in input_types)
        assert isinstance(output_type, core.Type)
        self.input_types = input_types
        self.output_type = output_type
        self.forward_impl = forward_impl
        self.pullback_impl = pullback_impl
        self.shapefun = shapefun
    def shp_apply(self, parents):
        if self.shapefun:
            return self.shapefun(parents)
        else:
            return cgt.shape(self)
    def typ_apply(self, _parents):
        return self.output_type
    def get_hash(self):
        return str(id(self))
    def pullback(self, inputs, output, goutput):
        if self.pullback_impl is None:
            raise core.MethodNotDefined
        pb_input_types = self.input_types + [self.output_type]*2
        pb_output_type = core.TupleType(*self.input_types)
        pbop = EasyCustomOp(pb_input_types, pb_output_type, 
            forward_impl=self.pullback_impl, pullback_impl=None,
            shapefun = lambda *args : tuple(cgt.shape(x) for x in inputs)  )
        return cgt.core.unpack(core.Result(pbop, inputs + [output, goutput]))
    def py_apply_valret(self, reads):
        return self.forward_impl(*reads)