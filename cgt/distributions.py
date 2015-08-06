import cgt
from . import core

class Distribution(object):
    def lik(self, x, p):
        raise NotImplementedError
    def loglik(self, x, p):
        raise NotImplementedError
    def crossent(self, p, q):
        raise NotImplementedError
    def kl(self, p, q):
        raise NotImplementedError
    def sample(self, p):
        raise NotImplementedError

class _Bernoulli(Distribution):
    def sample(self, p, shape=None):
        p = core.as_node(p)
        shape = shape or cgt.shape(p)
        return cgt.rand(*shape) <= p

bernoulli = _Bernoulli()

class _Categorical(Distribution):
    def crossent(self, p, q):
        assert p.ndim==2 and q.ndim==2
        return -(p*cgt.log(q)).sum(axis=1)
    def loglik(self, labels, p):
        return cgt.log(p[cgt.arange(cgt.size(labels,0)),labels])
categorical = _Categorical()

class _DiagonalGaussian(Distribution):
    pass

class Product(Distribution):    
    r"""
    Factored distribution obtained by taking the product of several component distributions
    E.g. suppose we have p0(x), p1(y), p2(z),
    then p3 := ProductDistribution(p1,p2,p3) is a distribution satisfying
    p3(x,y,z) = p0(x)p1(y)p2(z)
    """
    pass
