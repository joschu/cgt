import cgt
import cgt.nn as nn
import numpy as np

# Torch values obtained via this script: https://gist.github.com/ebenolson/931e879ed38f257253d2

torch_values = {'sgd': [0.81707280688755, 0.6648326359915, 0.5386151140949],
                'momentum': [0.817072808743, 0.664832651615, 0.538615107536],
                'nesterov_momentum': [0.817072808743, 0.664832651615, 0.538615107536],
                'adagrad': [0.725280165672, 0.621351599693, 0.546548962593],
                'rmsprop': [0.940282583237, 0.920392453671, 0.903152763844],
                'adadelta': [0.991312503815, 0.988132655621, 0.985075354576]}

scales = [0.1, 0.2, 0.3]


def f(X, scale):
    return (scale*X**2).sum()


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
        results.append(A.op.get_value())

    assert np.allclose(results, torch_values['sgd'])


def test_momentum():
    results = []
    for scale in scales:
        A = cgt.shared(1.0)
        B = cgt.shared(1.0)
        updates = nn.momentum(f(A, scale) + f(B, scale), [A, B], learning_rate=0.1, momentum=0.5)
        do_update = cgt.function([], [], updates=updates)
        for _ in range(10):
            do_update()

        assert np.allclose(A.op.get_value(), B.op.get_value())
        results.append(A.op.get_value())

    assert np.allclose(results, torch_values['momentum'])


def test_nesterov_momenutm():
    results = []
    for scale in scales:
        A = cgt.shared(1.0)
        B = cgt.shared(1.0)
        updates = nn.momentum(f(A, scale) + f(B, scale), [A, B], learning_rate=0.1, momentum=0.5)
        do_update = cgt.function([], [], updates=updates)
        for _ in range(10):
            do_update()

        assert np.allclose(A.op.get_value(), B.op.get_value())
        results.append(A.op.get_value())

    assert np.allclose(results, torch_values['nesterov_momentum'])


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
        results.append(A.op.get_value())

    assert np.allclose(results, torch_values['adagrad'])


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
        results.append(A.op.get_value())

    assert np.allclose(results, torch_values['rmsprop'])


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
        results.append(A.op.get_value())

    assert np.allclose(results, torch_values['adadelta'])
