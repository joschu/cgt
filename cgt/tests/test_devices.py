import numpy as np
import cgt
from cgt.tests import across_configs
from nose.plugins.skip import SkipTest

@across_configs(backends=("native",))
def test_devices():
    N = 10
    K = 3

    compile_info = cgt.compilation.get_compile_info()
    cuda_enabled = compile_info["CGT_ENABLE_CUDA"]
    if not cuda_enabled:
        raise SkipTest("cuda disabled")

    Xval = np.random.randn(N,K).astype(cgt.floatX)
    wval = np.random.randn(K).astype(cgt.floatX)
    bval = np.asarray(np.random.randn()).astype(cgt.floatX)
    yval = np.random.randn(N).astype(cgt.floatX)

    with cgt.scoped_update_config(default_device=cgt.Device(devtype="gpu")):

        X_nk = cgt.shared(Xval, "X", device=cgt.Device(devtype='gpu'))
        y_n = cgt.shared(yval, "y")
        w_k = cgt.shared(wval, "w")
        b = cgt.shared(bval, name="b")

        print "bval",bval

        ypred = cgt.dot(cgt.square(X_nk), w_k) + b

        err = cgt.sum(cgt.sin(ypred - y_n))
        g = cgt.grad(err, [w_k, b])
        outputs = [err]+g
        f = cgt.function([], [err]+g)
        results = f()
        print results
        assert np.allclose(results[0] , np.sin(np.square(Xval).dot(wval)+bval-yval).sum())


if __name__ == "__main__":
    import nose
    nose.runmodule()
