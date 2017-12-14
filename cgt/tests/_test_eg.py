import cgt
import unittest
import numpy as np 
import pprint
class EgTestCase(unittest.TestCase):
    def runTest(self):
        cgt.set_precision('double')
        x = cgt.vector()
        y = cgt.square(x)
        eg = cgt.execution.compilation_pipeline([x],[y+y],[])
        pprint.pprint(eg.to_json())
        import cycgt
        interp = cycgt.cInterpreter(eg)
        print(interp(np.array([3,4,5,6],'f8')))

if __name__ == "__main__":
    EgTestCase().runTest()