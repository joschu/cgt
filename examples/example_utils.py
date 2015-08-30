import cPickle as pickle
import os, os.path as osp, shutil, numpy as np, urllib

def train_val_test_slices(n, trainfrac, valfrac, testfrac):
    assert trainfrac+valfrac+testfrac==1.0
    ntrain = int(np.round(n*trainfrac))
    nval = int(np.round(n*valfrac))
    ntest = n - ntrain - nval
    return slice(0,ntrain), slice(ntrain,ntrain+nval), slice(ntrain+nval,ntrain+nval+ntest)

# helper methods to print nice table
def fmt_item(x, l):
    if isinstance(x, np.ndarray):
        assert x.ndim==0
        x = x.item()
    if isinstance(x, float): rep = "%g"%x
    else: rep = str(x)
    return " "*(l - len(rep)) + rep

def fmt_row(width, row, header=False):
    out = " | ".join(fmt_item(x, width) for x in row)
    if header: out = out + "\n" + "-"*len(out)
    return out

def download(url):
    "download and return path to file"
    fname = osp.basename(url)
    from cgt.core import get_cgt_src_root
    datadir = osp.join(get_cgt_src_root(),"downloads")
    datapath = osp.join(datadir, fname)
    if not osp.exists(datapath):
        print "downloading %s to %s"%(url, datapath)
        if not osp.exists(datadir): os.makedirs(datadir)
        urllib.urlretrieve(url, datapath)
    return datapath


def fetch_dataset(url):
    datapath = download(url)
    fname = osp.basename(url)
    extension =  osp.splitext(fname)[-1]
    assert extension in [".npz", ".pkl"]
    if extension == ".npz":
        return np.load(datapath)
    elif extension == ".pkl":
        with open(datapath, 'rb') as fin:
            return pickle.load(fin)
    else:
        raise NotImplementedError
