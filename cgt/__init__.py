from .api import *
from .display import print_tree, print_expr, print_text, as_dot
from .compilation import function, numeric_eval, profiler
from .core import (grad, get_config, update_config, simplify, reset_config, 
    Device, scoped_update_config, infer_shape, count_nodes)
from .ez import EasyCustomOp
try: 
    import cycgt
except ImportError:
    import warnings
    msg = """
    Could not import the compiled extension module cycgt
    Only pure python mode is available. If you have compiled "
    this extension (via 'make'), you may need to add build/lib 
    to your PYTHONPATH. Try 'import cycgt' to debug the problem further."""
    warnings.warn(msg, UserWarning)
    del warnings


floatX = "f4"
complexX = "c8"

# Get rid of names we don't want to export
del np
del cgt
del operator
del sys

get_config()