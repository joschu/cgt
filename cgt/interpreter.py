__doc__ = """
Python implementation of interpreter for execution graph

The parallel interpreter is mostly for prototyping/experimental purposes
"""

import multiprocessing
from multiprocessing.pool import ThreadPool
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

# class ParallelInterpreter(Interpreter):
#     """
#     Runs an execution graph in parallel using Python threads
#     """
#     def __init__(self, eg, output_locs, input_types, copy_outputs=True):
#         self.eg = eg
#         self.input_types = input_types
#         self.output_locs = output_locs
#         self.storage = [None for _ in xrange(self.eg.n_locs)]
#         self.args = None
#         self.copy_outputs = copy_outputs
#         # XXX may want to specify max threads different way
#         self.pool = ThreadPool(multiprocessing.cpu_count())
#     def __call__(self, *args):
#         self.args = tuple(core.as_valid_arg(arg) for arg in args)
#         typecheck_args(self.args, self.input_types)
#         self.instrs_left = set(xrange(len(self.eg.instrs)))
#         # XXX removed profiler
#         self.setup_instr_locs()
#         while self.instrs_left:
#             self.pool.map(lambda instr: instr.fire(self),
#                     [self.eg.instrs[k] for k in self.ready_instr_inds])
#             for instr_ind in self.ready_instr_inds:
#                 instr = self.eg.instrs[instr_ind]
#                 self.write_queue[instr.write_loc.index].pop(0)
#                 self.instrs_left.remove(instr_ind)
#             self.update_instr_locs()
#         outputs = [self.get(loc) for loc in self.output_locs]
#         if self.copy_outputs: outputs = map(_copy, outputs)
#         return outputs
#         # need to copy because otherwise we might mess up the data when we call func again
#         # todo: add option that prevents this behavior
#     def get(self, mem):
#         return self.storage[mem.index]
#     def set(self, mem, val):
#         self.storage[mem.index] = val
#     def getarg(self, i):
#         return self.args[i]
#     def apply_inplace(self, node, reads, write):
#         self.oplib.py_apply_inplace(node, reads, write)
#     def apply_valret(self, node, reads):
#         return self.oplib.py_apply_valret(node, reads)
#     def setup_instr_locs(self):
#         # instructions which write to location, fifo
#         self.write_queue = [list() for _ in xrange(self.eg.n_locs)]
#         for k in xrange(len(self.eg.instrs)):
#             instr = self.eg.instrs[k]
#             self.write_queue[instr.write_loc.index].append(k)
#         self.update_instr_locs()
#     def update_instr_locs(self):
#         self.ready_instr_inds = list()
#         for k in self.instrs_left:
#             if self.ready_to_fire(k):
#                 self.ready_instr_inds.append(k)
#     def ready_to_fire(self, instr_ind):
#         instr = self.eg.instrs[instr_ind]
#         read_rdy = True
#         for read_loc in instr.read_locs:
#             for k in self.write_queue[read_loc.index]:
#                 if k < instr_ind:
#                     read_rdy = False
#         write_rdy = self.write_queue[instr.write_loc.index][0] == instr_ind
#         return read_rdy and write_rdy

################################################################
### 
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
        # Collapse by Op, rather than instruction
        for (instr,(count,t)) in self.instr2stats.iteritems():
            opkey = str(instr) # XXX
            (prevcount, prevtime) = op2stats.get(opkey, (0, 0.0))
            op2stats[opkey] = (prevcount+count, prevtime+t)

        print "Total time elapsed: %.3g seconds"%self.t_total
        _print_heading("By instruction")
        _print_stats(self.instr2stats, self.t_total)
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
    

