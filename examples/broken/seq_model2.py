import gru,cgt, numpy as np
import sys
from time import time

horizon = 3000
batch_size = 6
dim_x = 16
mem_size = 10

X_tnk = cgt.tensor3("X")

cell = gru.GRUCell([dim_x], mem_size)

# Minit_nk = cgt.zeros((X_tnk.shape[0], X_tnk.shape[1]),cgt.floatX)
# M = Minit_nk

Min = cgt.matrix("M_in")
Xin = cgt.matrix("X_in")
Mout = cell(Min, Xin)


cellop = cgt.CallableComposition([Min, Xin], [Mout])


M=Min
for t in xrange(horizon):
    M, = cellop(M, X_tnk[t])

# cgt.print_tree(M)
print "simplifying..."
M_simp = cgt.simplify(M)
print "done"
# cgt.print_tree(M_simp)
print "before:",cgt.count_nodes(M)
print "after:",cgt.count_nodes(M_simp)

# M = cgt.simplify(M)
