import cgt, numpy as np
from nose.tools import raises
from StringIO import StringIO
import sys

class CaptureStderr(object):
    def __init__(self):
        self.origstderr = sys.stderr
    def __enter__(self):
        self.s = StringIO()
        sys.stderr = self.s
        return self.s
    def __exit__(self, *args):
        self.stderr = self.origstderr

@raises(RuntimeError)
def test_shape_err():
    with CaptureStderr():
        with cgt.scoped_update_config(debug=True, backend="python"):
            x = cgt.vector()
            y = cgt.vector()
            f = cgt.function([x,y],x+y)
            f(np.zeros(3),np.zeros(4))

if __name__ == "__main__":
    import nose
    nose.runmodule()
