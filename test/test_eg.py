import cgt
import unittest
import numpy as np 
class EgTestCase(unittest.TestCase):
    def runTest(self):
        x = cgt.vector()
        y = cgt.square(x)
        eg = cgt.execution.make_execution_graph([x],[y+y])
        print eg.to_json()
        import cycgt2
        interp = cycgt2.cInterpreter(eg)
        print interp([np.array([3,4,5,6],'f4')])

if __name__ == "__main__":
    EgTestCase().runTest()