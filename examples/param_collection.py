import cgt, numpy as np


class ParamCollection(object):

    """
    A utility class containing a collection of parameters
    which makes it convenient to write optimization code that uses flat vectors
    """

    def __init__(self,params): #pylint: disable=W0622
        assert all(param.is_data() and param.dtype == cgt.floatX for param in params)
        self._params = params

    @property
    def params(self):
        return self._params

    def get_values(self):
        return [param.op.get_value() for param in self._params]

    def get_shapes(self):
        return [param.op.get_shape() for param in self._params]

    def get_total_size(self):
        return sum(np.prod(shape) for shape in self.get_shapes())

    def num_vars(self):
        return len(self._params)

    def set_values(self, parvals):
        assert len(parvals) == len(self._params)
        for (param, newval) in zip(self._params, parvals):
            param.op.set_value(newval)
            param.op.get_shape() == newval.shape

    def set_value_flat(self, theta):
        theta = theta.astype(cgt.floatX)
        arrs = []
        n = 0        
        for shape in self.get_shapes():
            size = np.prod(shape)
            arrs.append(theta[n:n+size].reshape(shape))
            n += size
        assert theta.size == n
        self.set_values(arrs)
    
    def get_value_flat(self):
        theta = np.empty(self.get_total_size(),dtype=cgt.floatX)
        n = 0
        for param in self._params:
            s = param.op.get_size()
            theta[n:n+s] = param.op.get_value().flat
            n += s
        assert theta.size == n
        return theta