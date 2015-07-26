from core import ElwiseUnary, ElwiseBinary, Result, elwise_binary

def abs(x):
    return Result(ElwiseUnary("abs"), [x])
    
def ceil(x):
    return Result(ElwiseUnary("ceil"), [x])
    
def conj(x):
    return Result(ElwiseUnary("conj"), [x])
    
def cos(x):
    return Result(ElwiseUnary("cos"), [x])
    
def exp(x):
    return Result(ElwiseUnary("exp"), [x])
    
def iceil(x):
    return Result(ElwiseUnary("iceil"), [x])
    
def ifloor(x):
    return Result(ElwiseUnary("ifloor"), [x])
    
def log(x):
    return Result(ElwiseUnary("log"), [x])
    
def negative(x):
    return Result(ElwiseUnary("neg"), [x])
    
def sigmoid(x):
    return Result(ElwiseUnary("sigmoid"), [x])
    
def sign(x):
    return Result(ElwiseUnary("sign"), [x])
    
def sin(x):
    return Result(ElwiseUnary("sin"), [x])
    
def sqrt(x):
    return Result(ElwiseUnary("sqrt"), [x])
    
def square(x):
    return Result(ElwiseUnary("square"), [x])
    
def tanh(x):
    return Result(ElwiseUnary("tanh"), [x])
    
def equal(x, y):
    return elwise_binary("==", x,y)
    
def add(x, y):
    return elwise_binary("+", x,y)
    
def multiply(x, y):
    return elwise_binary("*", x,y)
    
def subtract(x, y):
    return elwise_binary("-", x,y)
    
def divide(x, y):
    return elwise_binary("/", x,y)
    
def power(x, y):
    return elwise_binary("**", x,y)
    
def less(x, y):
    return elwise_binary("<", x,y)
    