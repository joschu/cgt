import theano, theano.tensor as TT
from cgt.utils import Message
import time
import numpy as np

def normc(x):
    assert x.ndim == 2
    return x/norms(x,0)[None,:]
def randnf(*shp):
    return np.random.randn(*shp).astype(theano.config.floatX)
def norms(x,ax):
    return np.sqrt(np.square(x).sum(axis=ax))

class GRUCell(object):
    """
    Gated Recurrent Unit. E.g., see
    Chung, Junyoung, et al. "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling." arXiv preprint arXiv:1412.3555 (2014).
    """    
    def __init__(self,input_sizes,mem_size,name_prefix=""):

        Wiz_vals = [normc(randnf(input_size,mem_size)) for input_size in input_sizes]
        self.Wizs = [theano.shared(Wiz_val,name=name_prefix+"Wiz") for Wiz_val in Wiz_vals]
        Wmz_val = normc(randnf(mem_size,mem_size))
        self.Wmz = theano.shared(Wmz_val,name=name_prefix+"Wmz")
        bz = np.zeros((1,mem_size),theano.config.floatX)
        self.bz = theano.shared(bz,name=name_prefix+"bz")
        self.bz.type.broadcastable = (True,False)        

        Wir_vals = [normc(randnf(input_size,mem_size)) for input_size in input_sizes]
        self.Wirs = [theano.shared(Wir_val,name=name_prefix+"Wir") for Wir_val in Wir_vals]
        Wmr_val = normc(randnf(mem_size,mem_size))
        self.Wmr = theano.shared(Wmr_val,name=name_prefix+"Wmr")
        br = np.zeros((1,mem_size),theano.config.floatX)
        self.br = theano.shared(br,name=name_prefix+"br")
        self.br.type.broadcastable = (True,False)

        Wim_vals = [normc(randnf(input_size,mem_size)) for input_size in input_sizes]
        self.Wims = [theano.shared(Wim_val,name=name_prefix+"Wim") for Wim_val in Wim_vals]
        Wmm_val = normc(np.eye(mem_size,dtype=theano.config.floatX))
        self.Wmm = theano.shared(Wmm_val,name=name_prefix+"Wmm")
        bm = np.zeros((1,mem_size),theano.config.floatX)
        self.bm = theano.shared(bm,name=name_prefix+"bm")
        self.bm.type.broadcastable = (True,False)

    def __call__(self,M,*inputs):
        assert len(inputs) == len(self.Wizs)
        summands = [Xi.dot(Wiz) for (Xi,Wiz) in zip(inputs,self.Wizs)] + [M.dot(self.Wmz),self.bz]
        z = TT.nnet.sigmoid(TT.add(*summands))

        summands = [Xi.dot(Wir) for (Xi,Wir) in zip(inputs,self.Wirs)] + [M.dot(self.Wmr),self.br]
        r = TT.nnet.sigmoid(TT.add(*summands))

        summands = [Xi.dot(Wim) for (Xi,Wim) in zip(inputs,self.Wims)] + [(r*M).dot(self.Wmm),self.bm]
        Mtarg = TT.tanh(TT.add(*summands)) #pylint: disable=E1111

        Mnew = (1-z)*M + z*Mtarg
        return Mnew

    def params(self):
        out = []
        out.extend(self.Wizs)
        out.append(self.Wmz)
        out.append(self.bz)        
        out.extend(self.Wirs)
        out.append(self.Wmr)
        out.append(self.br)        
        out.extend(self.Wims)
        out.append(self.Wmm)
        out.append(self.bm)        
        return out

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon",type=int)
    args = parser.parse_args()
    horizon =args.horizon
    assert horizon is not None
    size=128
    batchsize=64
    cell = GRUCell([size],size)
    X = TT.tensor3()
    init = TT.zeros((batchsize, size),theano.config.floatX)

    prev_h = init
    for i in range(horizon):
        prev_h = cell(X[i], prev_h)

    with Message("compiling"):
        f = theano.function([X],theano.grad(prev_h.sum(), cell.params()))
    with Message("running"):
        x = np.zeros((horizon,batchsize,size),theano.config.floatX)
        for i in range(100): 
            f(x)