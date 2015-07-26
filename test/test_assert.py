import cgt
import unittest

def myfunc(x):
    print "x",x

class AssertTestCase(unittest.TestCase):
    def test_assertion(self):
        x = cgt.scalar()
        with cgt.debug_context() as dbg:
            cgt.assert_(cgt.equal(x, 1),"yoyoyo")
            cgt.dbg_call(myfunc, x)
            print "dbg",dbg.nodes
            # cgt.assert_(cgt.equal(x, 2))

        f = cgt.make_function([x],[x],dbg=dbg)
        f(1)
        with self.assertRaises(AssertionError):
            f(2)


if __name__ == "__main__":
    unittest.main()
