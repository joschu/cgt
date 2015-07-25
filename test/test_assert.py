import cgt



def myfunc(x):
    print "x",x


x = cgt.scalar()
with cgt.debug_context() as dbg:
    cgt.assert_(cgt.equal(x, 1),"yoyoyo")
    cgt.dbg_call(myfunc, x)
    print "dbg",dbg.nodes
    # cgt.assert_(cgt.equal(x, 2))

f = cgt.make_function([x],[x],dbg=dbg)
f(1)
try:
    f(2)
    raise RuntimeError("expecting AssertionError!")
except AssertionError:
    pass
