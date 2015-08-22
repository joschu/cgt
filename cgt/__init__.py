from .api import *
from .display import print_tree, print_expr, print_text, as_dot
from .compilation import function, numeric_eval, profiler
from .core import grad, get_config, update_config, simplify, reset_config, Device, scoped_update_config, infer_shape
from .ez import EasyCustomOp
try: import cycgt
except ImportError: pass 

floatX = "f4"
complexX = "c8"

# Get rid of names we don't want to export
del np
del cgt
del operator
del sys