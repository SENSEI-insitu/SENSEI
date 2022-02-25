#!/usr/bin/env python
"""Test template support in SVTK-Python

SVTK-python decides which template specializations
to wrap according to which ones are used in typedefs
and which ones appear as superclasses of other classes.
In addition, the wrappers are hard-coded to wrap the
svtkDenseArray and svtkSparseArray classes over a broad
range of types.

Created on May 29, 2011 by David Gobbi
"""

import sys
try:
    # for Python2
    import exceptions
except ImportError:
    # In Python 3 standard exceptions were moved to builtin module.
    pass
import svtk
from svtk.test import Testing

arrayTypes = ['char', 'int8', 'uint8', 'int16', 'uint16',
              'int32', 'uint32', int, 'uint', 'int64', 'uint64',
              'float32', float, str, 'unicode', svtk.svtkVariant]

arrayCodes = ['c', 'b', 'B', 'h', 'H',
              'i', 'I', 'l', 'L', 'q', 'Q',
              'f', 'd']

# create a unicode string for python 2 and python 3
if sys.hexversion >= 0x03000000:
    francois = 'Fran\xe7ois'
else:
    francois = unicode('Fran\xe7ois', 'latin1')

class TestTemplates(Testing.svtkTest):

    def testDenseArray(self):
        """Test svtkDenseArray template"""
        for t in (arrayTypes + arrayCodes):
            a = svtk.svtkDenseArray[t]()
            a.Resize(1)
            i = svtk.svtkArrayCoordinates(0)
            if t in ['bool', '?']:
                value = 1
                a.SetValue(i, value)
                result = a.GetValue(i)
                self.assertEqual(value, result)
            elif t in ['float32', 'float64', 'float', 'f', 'd']:
                value = 3.125
                a.SetValue(i, value)
                result = a.GetValue(i)
                self.assertEqual(value, result)
            elif t in ['char', 'c']:
                value = 'c'
                a.SetValue(i, value)
                result = a.GetValue(i)
                self.assertEqual(value, result)
            elif t in [str, 'str']:
                value = "hello"
                a.SetValue(i, value)
                result = a.GetValue(i)
                self.assertEqual(value, result)
            elif t in ['unicode']:
                value = francois
                a.SetValue(i, value)
                result = a.GetValue(i)
                self.assertEqual(value, result)
            elif t in ['svtkVariant', svtk.svtkVariant]:
                value = svtk.svtkVariant("world")
                a.SetValue(i, value)
                result = a.GetValue(i)
                self.assertEqual(value, result)
            else:
                value = 12
                a.SetValue(i, value)
                result = a.GetValue(i)
                self.assertEqual(value, result)

    def testSparseArray(self):
        """Test svtkSparseArray template"""
        for t in (arrayTypes + arrayCodes):
            a = svtk.svtkSparseArray[t]()
            a.Resize(1)
            i = svtk.svtkArrayCoordinates(0)
            if t in ['bool', '?']:
                value = 0
                a.SetValue(i, value)
                result = a.GetValue(i)
                self.assertEqual(value, result)
            elif t in ['float32', 'float64', 'float', 'f', 'd']:
                value = 3.125
                a.SetValue(i, value)
                result = a.GetValue(i)
                self.assertEqual(value, result)
            elif t in ['char', 'c']:
                value = 'c'
                a.SetValue(i, value)
                result = a.GetValue(i)
                self.assertEqual(value, result)
            elif t in [str, 'str']:
                value = "hello"
                a.SetValue(i, value)
                result = a.GetValue(i)
                self.assertEqual(value, result)
            elif t in ['unicode']:
                value = francois
                a.SetValue(i, value)
                result = a.GetValue(i)
                self.assertEqual(value, result)
            elif t in ['svtkVariant', svtk.svtkVariant]:
                value = svtk.svtkVariant("world")
                a.SetValue(i, value)
                result = a.GetValue(i)
                self.assertEqual(value, result)
            else:
                value = 12
                a.SetValue(i, value)
                result = a.GetValue(i)
                self.assertEqual(value, result)

    def testArray(self):
        """Test array CreateArray"""
        o = svtk.svtkArray.CreateArray(svtk.svtkArray.DENSE, svtk.SVTK_DOUBLE)
        self.assertEqual(o.__class__, svtk.svtkDenseArray[float])

    def testVector(self):
        """Test vector templates"""
        # make sure Rect inherits operators
        r = svtk.svtkRectf(0, 0, 2, 2)
        self.assertEqual(r[2], 2.0)
        c = svtk.svtkColor4ub(0, 0, 0)
        self.assertEqual(list(c), [0, 0, 0, 255])
        e = svtk.svtkVector['float32', 3]([0.0, 1.0, 2.0])
        self.assertEqual(list(e), [0.0, 1.0, 2.0])
        i = svtk.svtkVector3['i'](0)
        self.assertEqual(list(i), [0, 0, 0])

if __name__ == "__main__":
    Testing.main([(TestTemplates, 'test')])
