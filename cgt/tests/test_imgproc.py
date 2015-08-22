import numpy as np, numpy.random as nr
from cgt.numeric_diff import numeric_grad_multi
import cgt
from cgt.nn import max_pool_2d, im2col, cross_channel_lrn
from cgt.compilation import get_compile_info
from cgt import utils


def test_cudnn():
    with cgt.scoped_update_config(precision="double",backend="native"):
        if not get_compile_info()["CGT_ENABLE_CUDNN"]:
            utils.warn("CUDNN not enabled. Skipping this test")
            return

        Xval = nr.randn(2,3,19,18)
        Wval = nr.randn(5,3,3,3)
        bval = nr.randn(1,5,1,1)

        X = cgt.tensor4("X", fixed_shape=Xval.shape)
        W = cgt.tensor4("W", fixed_shape=Wval.shape)
        b = cgt.tensor4("b", fixed_shape=bval.shape)


        Y = cgt.core.Result(cudnn_ops.CudnnConvForward(1,1,1,1),[X, W, b])

        Y2 = nr.randn(*cgt.core.infer_shape(Y))

        fY = cgt.function([X,W,b],Y)
        Yval = fY(Xval,Wval,bval)
        cost = (Y*Y2).sum()
        fcost = cgt.function([X,W,b],cost)
        fgrad = cgt.function([X,W,b],cgt.grad(cost, [X,W,b]))
        angrads = fgrad(Xval,Wval,bval)
        nugrads = numeric_grad_multi(fcost, [Xval, Wval, bval],eps=1e-3)
        for (nugrad,angrad) in zip(nugrads,angrads):
            assert np.allclose(nugrad, angrad)

def test_cpu_pool():
    with cgt.scoped_update_config(precision="quad",backend="native"):
        print cgt.get_precision()
        ci = get_compile_info()

        np.random.seed(0)
        x = cgt.tensor4("x", fixed_shape=(2,3,5,7))
        y = max_pool_2d(x, (4,4),(0,0),(1,1))
        xval = np.random.randn(2,3,5,7)
        hval = np.random.randn(*cgt.infer_shape(y))
        h = cgt.constant(hval)

        cost = (y*h).sum()

        fcost = cgt.function([x], cost)
        fgrad = cgt.function([x], cgt.grad(cost, [x])[0])

        from cgt.numeric_diff import numeric_grad
        gnum = numeric_grad(fcost, xval)
        gana = fgrad(xval)

        assert np.allclose(gnum,gana)

def test_im2col():

    with cgt.scoped_update_config(precision="quad",backend="native"):


        for settings in [ ((4,4),(0,0),(1,1)), ((3,3),(1,1),(2,2)), ((3,3),(1,1),(3,3)) ]:
            xval = np.arange(2*1*28*28).reshape(2,1,28,28).astype(cgt.floatX)
            x = cgt.tensor4("x", fixed_shape=xval.shape)
            y = im2col(x, *settings)
            h = cgt.constant(np.random.randn(*cgt.infer_shape(y)))
            cost = (y*h).sum()

            fcost = cgt.function([x],cost)
            fgrad = cgt.function([x], cgt.grad(cost, [x])[0])

            from cgt.numeric_diff import numeric_grad
            gnum = numeric_grad(fcost, xval,eps=1e-5)
            gana = fgrad(xval)
            assert np.allclose(gnum, gana)
            # fy = cgt.function([x],y)
            # yval = fy(xval)
            # assert np.allclose(yval[0,0,0] , xval[0,:,0:4,0:4].flatten())

def test_lrn():
    if not get_compile_info()["CGT_ENABLE_CUDA"]:
        utils.warn("CUDA not enabled. Skipping this test")
        return

    with cgt.scoped_update_config(precision="double",backend="native"):
        from cgt.tests import gradcheck_model
        cgt.set_precision('double')
        nr.seed(0)
        Xval = nr.randn(4,8,16,16)
        X = cgt.shared(Xval, name="X", fixed_shape_mask="all")
        # X = cgt.tensor4(name='X')
        y = cross_channel_lrn.cross_channel_lrn(X, localsize=4, alpha=.1, beta=.5)
        f = cgt.function([],y)
        print f().sum()
        print f().sum()
        print f().sum()
        assert np.isfinite(f().sum())
        # print f(Xval).sum()
        a = nr.rand(*cgt.infer_shape(y))
        loss = (y*a).sum()
        gradcheck_model(loss, [X],eps=1e-5)


if __name__=="__main__":
    test_im2col()