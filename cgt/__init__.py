from .api import *
from .display import print_tree, print_expr, print_text, as_dot
from .execution import function, numeric_eval
from .core import grad, load_config, update_config, simplify, reset_config
from .ez import EasyCustomOp

floatX = "f4"
complexX = "c8"

# Get rid of names we don't want to export
del np
del cgt
del operator
del sys