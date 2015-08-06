
def abs(x):
    return core.Result(core.ElwiseUnary("abs"), [x])
    
def ceil(x):
    return core.Result(core.ElwiseUnary("ceil"), [x])
    
def conj(x):
    return core.Result(core.ElwiseUnary("conj"), [x])
    
def cos(x):
    return core.Result(core.ElwiseUnary("cos"), [x])
    
def exp(x):
    return core.Result(core.ElwiseUnary("exp"), [x])
    
def iceil(x):
    return core.Result(core.ElwiseUnary("iceil"), [x])
    
def ifloor(x):
    return core.Result(core.ElwiseUnary("ifloor"), [x])
    
def log(x):
    return core.Result(core.ElwiseUnary("log"), [x])
    
def negative(x):
    return core.Result(core.ElwiseUnary("neg"), [x])
    
def sigmoid(x):
    return core.Result(core.ElwiseUnary("sigmoid"), [x])
    
def sign(x):
    return core.Result(core.ElwiseUnary("sign"), [x])
    
def sin(x):
    return core.Result(core.ElwiseUnary("sin"), [x])
    
def sqrt(x):
    return core.Result(core.ElwiseUnary("sqrt"), [x])
    
def square(x):
    return core.Result(core.ElwiseUnary("square"), [x])
    
def tanh(x):
    return core.Result(core.ElwiseUnary("tanh"), [x])
    
def add(x, y):
    return core.elwise_binary("+", x,y)
    
def divide(x, y):
    return core.elwise_binary("/", x,y)
    
def equal(x, y):
    return core.elwise_binary("==", x,y)
    
def greater(x, y):
    return core.elwise_binary(">", x,y)
    
def greater_equal(x, y):
    return core.elwise_binary(">=", x,y)
    
def less(x, y):
    return core.elwise_binary("<", x,y)
    
def less_equal(x, y):
    return core.elwise_binary("<=", x,y)
    
def multiply(x, y):
    return core.elwise_binary("*", x,y)
    
def not_equal(x, y):
    return core.elwise_binary("!=", x,y)
    
def power(x, y):
    return core.elwise_binary("**", x,y)
    
def subtract(x, y):
    return core.elwise_binary("-", x,y)
    
from . import core
