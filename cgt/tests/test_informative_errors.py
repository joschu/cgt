import cgt, numpy as np
from StringIO import StringIO
import sys

class SetStderr(object):
    def __init__(self):
        self.origstderr = sys.stderr
    def __enter__(self):
        self.s = StringIO()
        sys.stderr = self.s
        return self.s
    def __exit__(self, *args):
        self.stderr = self.origstderr

def test_shape_err():
    with SetStderr() as s:
        with cgt.scoped_update_config(debug=True):
            x = cgt.vector()
            y = cgt.vector()
            f = cgt.function([x,y],x+y)
            f(np.zeros(3),np.zeros(4))
    assert "f = cgt.function([x,y],x+y)" in s.getvalue()

if __name__ == "__main__":
    test_shape_err()