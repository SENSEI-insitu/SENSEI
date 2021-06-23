"""Test support of SVTK singleton objects

Created on Aug 30, 2017 by David Gobbi

"""

import sys
import svtk
from svtk.test import Testing

class TestSingleton(Testing.svtkTest):
    def testOutputWindow(self):
        a = svtk.svtkOutputWindow()
        b = svtk.svtkOutputWindow()
        self.assertNotEqual(a, b)

        c = svtk.svtkOutputWindow.GetInstance()
        d = svtk.svtkOutputWindow.GetInstance()
        self.assertIs(c, d)

    def testObject(self):
        a = svtk.svtkObject()
        b = svtk.svtkObject()
        self.assertNotEqual(a, b)

if __name__ == "__main__":
    Testing.main([(TestSingleton, 'test')])
