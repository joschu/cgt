import cgt
from functools import wraps

def reset_config(fn):
    @wraps(fn)
    def newfn(*args,**kw):
        fn(*args, **kw)
        cgt.load_config(True)
    return newfn