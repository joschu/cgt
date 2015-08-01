import cgt
from . import core

class Distribution(object):
    @staticmethod
    def lik(x, p):
        raise NotImplementedError
    @staticmethod
    def loglik(x, p):
        raise NotImplementedError
    @staticmethod
    def crossent(p, q):
        raise NotImplementedError
    @staticmethod
    def kl(p, q):
        raise NotImplementedError
    @staticmethod
    def sample(p):
        raise NotImplementedError

class Bernoulli(Distribution):
    @staticmethod
    def sample(p, shape=None):
        p = core.as_node(p)
        shape = shape or cgt.shape(p)
        return cgt.rand(*shape) <= p

class Categorical(Distribution):
    @staticmethod
    def crossent(p, q):
        assert p.ndim==2 and q.ndim==2
        return -(p*cgt.log(q)).sum(axis=1)
    @staticmethod
    def loglik(labels, p):
        return cgt.log(p[cgt.arange(cgt.size(labels,0)),labels])

class DiagonalGaussian(Distribution):
    pass

class ProductDistribution(Distribution):    
    r"""
    Factored distribution obtained by taking the product of several component distributions
    E.g. suppose we have p0(x), p1(y), p2(z),
    then p3 := ProductDistribution(p1,p2,p3) is a distribution satisfying
    p3(x,y,z) = p0(x)p1(y)p2(z)
    """
    pass
