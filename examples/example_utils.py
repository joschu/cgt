import os, os.path as osp, shutil, numpy as np, urllib

def train_test_val_slices(n, trainfrac, testfrac, valfrac):
    assert trainfrac+testfrac+valfrac==1.0
    ntrain = int(np.round(n*trainfrac))
    ntest = int(np.round(n*testfrac))
    nval = n - ntrain - ntest
    return slice(0,ntrain), slice(ntrain,ntrain+ntest), slice(ntrain+ntest,ntrain+ntest+nval)

# helper methods to print nice table
def fmt_item(x, l):
    if isinstance(x, np.ndarray):
        assert x.ndim==0
        x = x.item()
    if isinstance(x, (str,int)): rep = str(x)
    elif isinstance(x, float): rep = "%g"%x
    return " "*(l - len(rep)) + rep

def fmt_row(width, row):
    return " | ".join(fmt_item(x, width) for x in row)

def fetch_dataset(url):    
    fname = osp.basename(url)
    extension =  osp.splitext(fname)[-1]
    assert extension in [".npz"]
    from cgt.core import get_cgt_src_root
    datadir = osp.join(get_cgt_src_root(),"downloads")
    datapath = osp.join(datadir, fname)
    if not osp.exists(datapath):
        print "downloading %s to %s"%(url, datapath)
        if not osp.exists(datadir): os.makedirs(datadir)
        urllib.urlretrieve(url, datapath)
    if extension == ".npz":
        return np.load(datapath)
    else:
        raise NotImplementedError
