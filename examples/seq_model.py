import gru,cgt, numpy as np
import sys
from time import time

elapsed = []
horizons = 2**np.arange(14)

for horizon in horizons:
    print "HORIZON",horizon
    tstart = time()

    batch_size = 6
    dim_x = 16
    mem_size = 10

    X_tnk = cgt.tensor3("X")

    cell = gru.GRUCell([dim_x], mem_size)

    Minit_nk = cgt.zeros((X_tnk.shape[0], X_tnk.shape[1]),cgt.floatX)
    M = Minit_nk

    for t in xrange(horizon):
        M = cell(M, X_tnk[t])

    # cgt.print_tree(M)
    print "simplifying..."
    M_simp = cgt.simplify(M)
    print "done"
    # cgt.print_tree(M_simp)
    print "fn before:",cgt.count_nodes(M)
    print "fn after:",cgt.count_nodes(M_simp)

    gs = cgt.grad(cgt.sum(M), cell.params())
    print "grad before", cgt.count_nodes(gs)
    g_simp = cgt.simplify(gs)
    print "grad after",cgt.count_nodes(g_simp)

    # M = cgt.simplify(M)
    elapsed.append(time()-tstart)

import matplotlib.pyplot as plt
plt.plot(horizons,elapsed)
plt.show()

