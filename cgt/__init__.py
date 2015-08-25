from .api import *
from .display import print_tree, print_expr, print_text, as_dot
from .compilation import function, numeric_eval, profiler
from .core import grad, get_config, update_config, simplify, reset_config, Device, scoped_update_config, infer_shape, count_nodes
from .ez import EasyCustomOp
try: 
    import cycgt
except ImportError as e: 
    utils.warn("cgt/__init__.py: 'import cycgt' failed. Exception: %s"%e)


floatX = "f4"
complexX = "c8"

# Get rid of names we don't want to export
del np
del cgt
del operator
del sys

get_config()