"""Test if SVTK-Python objects support weak referencing.

Run this test like so:
 $ svtkpython TestWeakref.py
or
 $ python TestWeakref.py

Created on July, 8 2005
Prabhu Ramachandran <prabhu_r at users dot sf dot net>

"""

import sys
import svtk
from svtk.test import Testing
try:
    import weakref
except ImportError:
    print("No weakref in this version of Python.  Time to upgrade?")
    print("Python version:", sys.version)
    from svtk.test import Testing
    Testing.skip()


class TestWeakref(Testing.svtkTest):
    def testWeakref(self):
        o = svtk.svtkObject()
        ref = weakref.ref(o)
        self.assertEqual(ref().GetClassName(), 'svtkObject')
        del o
        self.assertEqual(ref(), None)

    def testProxy(self):
        o = svtk.svtkObject()
        proxy = weakref.proxy(o)
        self.assertEqual(proxy.GetClassName(), 'svtkObject')
        del o
        self.assertRaises(ReferenceError, getattr,
                          proxy, 'GetClassName')


if __name__ == "__main__":
    Testing.main([(TestWeakref, 'test')])
