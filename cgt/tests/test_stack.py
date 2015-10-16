import numpy as np
import cgt
from cgt.tests import across_configs

@across_configs
def test_stack():
    x = cgt.scalar()
    y = cgt.scalar()
    z = cgt.scalar()
    s0 = cgt.stack([x, y, z], axis=0)
    assert cgt.numeric_eval(s0, {x: 1, y: 2, z: 3}).shape == (3,)

    x = cgt.vector()
    y = cgt.vector()
    z = cgt.vector()
    v0 = cgt.stack([x, y, z], axis=0)
    assert cgt.numeric_eval(v0, {x: np.zeros(2), y: np.zeros(2), z: np.zeros(2)}).shape == (3, 2)
    v1 = cgt.stack([x, y, z], axis=1)
    assert cgt.numeric_eval(v1, {x: np.zeros(2), y: np.ones(2), z: np.zeros(2)}).shape == (2, 3)

    x = cgt.matrix()
    y = cgt.matrix()
    z = cgt.matrix()
    m0 = cgt.stack([x, y, z], axis=0)
    assert cgt.numeric_eval(m0, {x: np.zeros((2, 4)), y: np.zeros((2, 4)), z: np.zeros((2, 4))}).shape == (3, 2, 4)
    m1 = cgt.stack([x, y, z], axis=1)
    assert cgt.numeric_eval(m1, {x: np.zeros((2, 4)), y: np.zeros((2, 4)), z: np.zeros((2, 4))}).shape == (2, 3, 4)
    m2 = cgt.stack([x, y, z], axis=2)
    assert cgt.numeric_eval(m2, {x: np.zeros((2, 4)), y: np.zeros((2, 4)), z: np.zeros((2, 4))}).shape == (2, 4, 3)

if __name__ == "__main__":
    import nose
    nose.runmodule()
