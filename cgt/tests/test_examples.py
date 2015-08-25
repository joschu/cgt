import subprocess, os.path as osp
from nose.plugins.skip import SkipTest

thisdir = osp.dirname(__file__)

def run_example(cmd):
    subprocess.check_call(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
def skip_example(cmd):
    raise SkipTest(cmd)

def test_examples():
    if osp.exists("%s/../../downloads/mnist.npz"%thisdir):
        yield run_example, "CGT_FLAGS=backend=python python %s/../../examples/demo_mnist.py --unittest"%thisdir
        yield run_example, "CGT_FLAGS=backend=native python %s/../../examples/demo_mnist.py --unittest"%thisdir
        yield run_example, "CGT_FLAGS=backend=native python %s/../../examples/demo_cifar.py --unittest"%thisdir
        yield run_example, "python %s/../../examples/cgt_theano_feedforward_comparison.py --unittest"%thisdir
    yield run_example, "cd %s/../../examples/ && CGT_FLAGS=backend=native python demo_char_rnn.py --unittest"%thisdir
    yield run_example, "CGT_FLAGS=backend=native python %s/../../examples/demo_neural_turing_machine.py --unittest"%thisdir
    runipycmd = "runipy %s/../../examples/tutorial.ipynb"%thisdir
    try:
        import graphviz
        yield run_example, runipycmd
    except ImportError:
        yield skip_example, runipycmd