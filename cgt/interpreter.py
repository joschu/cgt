__doc__ = """
Python implementation of interpreter for execution graph

The parallel interpreter is mostly for prototyping/experimental purposes
"""

from . import core
import time, numpy as np

################################################################
### Interpreters 
################################################################
 
class Interpreter(object):
    def __call__(self, args):
        raise NotImplementedError
    def get(self, mem):
        raise NotImplementedError
    def set(self, mem, val):
        raise NotImplementedError
    def getarg(self, i):
        raise NotImplementedError


class SequentialInterpreter(Interpreter):
    """
    Runs an execution graph
    """
    def __init__(self, eg, output_locs, input_types, copy_outputs=True):
        self.eg = eg
        self.input_types = input_types
        self.output_locs = output_locs
        self.storage = [None for _ in xrange(self.eg.n_locs)]
        self.args = None
        self.copy_outputs = copy_outputs
    def __call__(self, *args):
        assert len(args) == len(self.input_types), "Wrong number of inputs provided"
        self.args = tuple(core.as_valid_array(arg, intype) for (arg, intype) in zip(args, self.input_types))
        for instr in self.eg.instrs:
            if profiler.on: tstart = time.time()
            instr.fire(self)
            if profiler.on: profiler.update(instr, time.time()-tstart)
        outputs = [self.get(loc) for loc in self.output_locs]
        if self.copy_outputs: outputs = map(_copy, outputs)
        return outputs
        # need to copy because otherwise we might mess up the data when we call func again
        # todo: add option that prevents this behavior
    def get(self, mem):
        return self.storage[mem.index]
    def set(self, mem, val):
        self.storage[mem.index] = val
    def getarg(self, i):
        return self.args[i]


################################################################
### Profiler
################################################################
 
class _Profiler(object):
    def __init__(self):
        self.instr2stats = {}
        self.on = False
        self.t_total = 0.0
    def start(self): self.on = True
    def stop(self): self.on = False
    def update(self, instr, elapsed):
        (prevcount, prevtime) = self.instr2stats.get(instr, (0,0.0))
        self.instr2stats[instr] = (prevcount+1, prevtime+elapsed)
        self.t_total += elapsed
    def print_stats(self):
        op2stats = {}
        from cgt.execution import ReturnByRef,ReturnByVal,Alloc
        # Collapse by Op, rather than instruction
        for (instr,(count,t)) in self.instr2stats.iteritems():
            if isinstance(instr, (ReturnByRef, ReturnByVal)):
                opkey = str(instr.op)
            elif isinstance(instr, Alloc):
                opkey = "Alloc{dtype=%s,ndim=%i}"%(instr.dtype, len(instr.read_locs))
            else:
                opkey = instr.__class__.__name__

            (prevcount, prevtime) = op2stats.get(opkey, (0, 0.0))
            op2stats[opkey] = (prevcount+count, prevtime+t)

        print "Total time elapsed: %.3g seconds"%self.t_total
        # _print_heading("By instruction")
        # _print_stats(self.instr2stats, self.t_total)
        _print_heading("By Op")
        _print_stats(op2stats, self.t_total)
    def clear_stats(self):
        self.instr2stats = {}
        self.t_total = 0.0

profiler = _Profiler()

def _print_heading(heading):
    heading = "  " + heading + "  "
    width = 60
    assert len(heading) < width-10
    print
    print "*"*width
    padleft = (width-len(heading))//2
    padright = width-len(heading)-padleft
    print "*"*padleft + heading + "*"*padright
    print "*"*width

def _print_stats(key2stats, t_total):
    rows = []
    for (key, (count,t)) in key2stats.iteritems():
        rows.append([str(key), count, t, t/t_total])
    rows = sorted(rows, key=lambda row: row[2], reverse=True)
    cumsum = 0
    for row in rows:
        cumsum += row[3]
        row.append(cumsum)
    from thirdparty.tabulate import tabulate
    print tabulate(rows, headers=["Instruction","Count","Time","Frac","Frac cumsum"])

def _copy(x):
    if isinstance(x, np.ndarray): return x.copy()
    elif isinstance(x, tuple): return tuple(el.copy() for el in x)
    elif np.isscalar(x): return x # xxx is this case ok?
    else: raise NotImplementedError


def typecheck_args(numargs, types):
    assert len(numargs)==len(types), "wrong number of arguments. got %i, expected %i"%(len(numargs),len(types))
    for (numarg,typ) in zip(numargs,types):
        if isinstance(typ, core.TensorType):
            assert numarg.dtype==typ.dtype and numarg.ndim==typ.ndim
    

