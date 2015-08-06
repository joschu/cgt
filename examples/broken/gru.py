import cgt
import numpy as np

def normc(x):
    assert x.ndim == 2
    return x/norms(x,0)[None,:]
def randnf(*shp):
    return np.random.randn(*shp).astype(cgt.floatX)
def norms(x,ax):
    return np.sqrt(np.square(x).sum(axis=ax))


class GRUCell(object):
    """
    Gated Recurrent Unit. E.g., see
    Chung, Junyoung, et al. "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling." arXiv preprint arXiv:1412.3555 (2014).
    """    
    def __init__(self,input_sizes,mem_size,name_prefix=""):

        Wiz_vals = [normc(randnf(input_size,mem_size)) for input_size in input_sizes]
        self.Wizs = [cgt.shared(Wiz_val,name=name_prefix+"Wiz") for Wiz_val in Wiz_vals]
        Wmz_val = normc(randnf(mem_size,mem_size))
        self.Wmz = cgt.shared(Wmz_val,name=name_prefix+"Wmz")
        bz = np.zeros((1,mem_size),cgt.floatX)
        self.bz = cgt.shared(bz,name=name_prefix+"bz")

        Wir_vals = [normc(randnf(input_size,mem_size)) for input_size in input_sizes]
        self.Wirs = [cgt.shared(Wir_val,name=name_prefix+"Wir") for Wir_val in Wir_vals]
        Wmr_val = normc(randnf(mem_size,mem_size))
        self.Wmr = cgt.shared(Wmr_val,name=name_prefix+"Wmr")
        br = np.zeros((1,mem_size),cgt.floatX)
        self.br = cgt.shared(br,name=name_prefix+"br")

        Wim_vals = [normc(randnf(input_size,mem_size)) for input_size in input_sizes]
        self.Wims = [cgt.shared(Wim_val,name=name_prefix+"Wim") for Wim_val in Wim_vals]
        Wmm_val = normc(np.eye(mem_size,dtype=cgt.floatX))
        self.Wmm = cgt.shared(Wmm_val,name=name_prefix+"Wmm")
        bm = np.zeros((1,mem_size),cgt.floatX)
        self.bm = cgt.shared(bm,name=name_prefix+"bm")

    def __call__(self,M,*inputs):
        assert len(inputs) == len(self.Wizs)
        summands = [Xi.dot(Wiz) for (Xi,Wiz) in zip(inputs,self.Wizs)] + [M.dot(self.Wmz),self.bz]
        z = cgt.sigmoid(cgt.add_multi(summands))

        summands = [Xi.dot(Wir) for (Xi,Wir) in zip(inputs,self.Wirs)] + [M.dot(self.Wmr),self.br]
        r = cgt.sigmoid(cgt.add_multi(summands))

        summands = [Xi.dot(Wim) for (Xi,Wim) in zip(inputs,self.Wims)] + [(r*M).dot(self.Wmm),self.bm]
        Mtarg = cgt.tanh(cgt.add_multi(summands)) #pylint: disable=E1111

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

