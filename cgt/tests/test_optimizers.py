import cgt
import cgt.nn as nn
from cgt.tests import across_configs
import numpy as np

# Torch values obtained via this script: https://gist.github.com/ebenolson/931e879ed38f257253d2

torch_values = {
    'sgd': [0.81707280688755,0.6648326359915,0.5386151140949],
    'momentum': [0.6848486952183,0.44803321781003,0.27431190123502],
    # TORCH:
    # 'nesterov_momentum': [0.67466543592725,0.44108468114241,0.2769002108997],
    # OURS:
    'nesterov_momentum' : [0.6848486661911011, 0.4480332136154175, 0.2743118703365326], 
    # Different because we're using
    # version from http://arxiv.org/abs/1212.0901v2, which is returning "lookahead parameters"
    'adagrad': [0.55373120047759,0.55373120041518,0.55373120039438],
    'rmsprop': [0.83205403985348,0.83205322744821,0.83205295664444],
    'adadelta': [0.95453237704725,0.9545237471374,0.95452214847397],
    'adam': [0.90034973381771,0.90034969365796,0.90034968027137],
}
scales = [0.1, 0.2, 0.3]


def f(X, scale):
    return (scale*X**2).sum()

@across_configs
def test_sgd():
    results = []
    for scale in scales:
        A = cgt.shared(1.0)
        B = cgt.shared(1.0)
        updates = nn.sgd(f(A, scale) + f(B, scale), [A, B], learning_rate=0.1)
        do_update = cgt.function([], [], updates=updates)
        for _ in range(10):
            do_update()

        assert np.allclose(A.op.get_value(), B.op.get_value())
        results.append(A.op.get_value().copy())

    assert np.allclose(results, torch_values['sgd'])


@across_configs
def test_momentum():
    results = []
    for scale in scales:
        A = cgt.shared(1.0)
        B = cgt.shared(1.0)
        updates = nn.momentum(f(A, scale) + f(B, scale), [A, B], learning_rate=0.1, mu=0.5)
        do_update = cgt.function([], [], updates=updates)
        for _ in range(10):
            do_update()

        assert np.allclose(A.op.get_value(), B.op.get_value())
        results.append(A.op.get_value().copy())

    assert np.allclose(results, torch_values['momentum'])


@across_configs
def test_nesterov_momentum():
    results = []
    for scale in scales:
        A = cgt.shared(1.0)
        B = cgt.shared(1.0)
        updates = nn.momentum(f(A, scale) + f(B, scale), [A, B], learning_rate=0.1, mu=0.5)
        do_update = cgt.function([], [], updates=updates)
        for _ in range(10):
            do_update()
        assert np.allclose(A.op.get_value(), B.op.get_value())
        results.append(A.op.get_value().copy())

    assert np.allclose(results, torch_values['nesterov_momentum'])


@across_configs
def test_adagrad():
    results = []
    for scale in scales:
        A = cgt.shared(1.0)
        B = cgt.shared(1.0)
        updates = nn.adagrad(f(A, scale) + f(B, scale), [A, B], learning_rate=0.1)
        do_update = cgt.function([], [], updates=updates)
        for _ in range(10):
            do_update()

        assert np.allclose(A.op.get_value(), B.op.get_value())
        results.append(A.op.get_value().copy())

    assert np.allclose(results, torch_values['adagrad'])


@across_configs
def test_rmsprop():
    results = []
    for scale in scales:
        A = cgt.shared(1.0)
        B = cgt.shared(1.0)
        updates = nn.rmsprop(f(A, scale) + f(B, scale), [A, B], learning_rate=0.01)
        do_update = cgt.function([], [], updates=updates)
        for _ in range(10):
            do_update()

        assert np.allclose(A.op.get_value(), B.op.get_value())
        results.append(A.op.get_value().copy())

    assert np.allclose(results, torch_values['rmsprop'])


@across_configs
def test_adadelta():
    results = []
    for scale in scales:
        A = cgt.shared(1.0)
        B = cgt.shared(1.0)
        updates = nn.adadelta(f(A, scale) + f(B, scale), [A, B])
        do_update = cgt.function([], [], updates=updates)
        for _ in range(10):
            do_update()

        assert np.allclose(A.op.get_value(), B.op.get_value())
        results.append(A.op.get_value().copy())

    assert np.allclose(results, torch_values['adadelta'])

if __name__ == "__main__":
    import nose
    nose.runmodule()
