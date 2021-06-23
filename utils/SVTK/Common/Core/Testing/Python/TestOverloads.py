"""Test overloaded method resolution in SVTK-Python

The wrappers should call overloaded C++ methods using similar
overload resolution rules as C++.  Python itself does not have
method overloading.

Created on Feb 15, 2015 by David Gobbi

"""

import sys
import svtk
from svtk.test import Testing

class TestOverloads(Testing.svtkTest):

    def testMethods(self):
        """Test overloaded methods"""
        # single-argument method svtkTransform::SetMatrix()
        t = svtk.svtkTransform()
        m = svtk.svtkMatrix4x4()
        m.SetElement(0, 0, 2)
        t.SetMatrix(m)
        self.assertEqual(t.GetMatrix().GetElement(0, 0), 2)
        t.SetMatrix([0,1,0,0, 1,0,0,0, 0,0,-1,0, 0,0,0,1])
        self.assertEqual(t.GetMatrix().GetElement(0, 0), 0)
        # mixed number of arguments
        fd = svtk.svtkFieldData()
        fa = svtk.svtkFloatArray()
        fa.SetName("Real")
        ia = svtk.svtkIntArray()
        ia.SetName("Integer")
        fd.AddArray(fa)
        fd.AddArray(ia)
        a = fd.GetArray("Real")
        self.assertEqual(id(a), id(fa))
        i = svtk.reference(0)
        a = fd.GetArray("Integer", i)
        self.assertEqual(id(a), id(ia))
        self.assertEqual(i, 1)

    def testConstructors(self):
        """Test overloaded constructors"""
        # resolve by number of arguments
        v = svtk.svtkVector3d(3, 4, 5)
        self.assertEqual((v[0], v[1], v[2]), (3, 4, 5))
        v = svtk.svtkVector3d(6)
        self.assertEqual((v[0], v[1], v[2]), (6, 6, 6))
        # resolve by argument type
        v = svtk.svtkVariant(3.0)
        self.assertEqual(v.GetType(), svtk.SVTK_DOUBLE)
        v = svtk.svtkVariant(1)
        self.assertEqual(v.GetType(), svtk.SVTK_INT)
        v = svtk.svtkVariant("hello")
        self.assertEqual(v.GetType(), svtk.SVTK_STRING)
        v = svtk.svtkVariant(svtk.svtkObject())
        self.assertEqual(v.GetType(), svtk.SVTK_OBJECT)

    def testArgumentConversion(self):
        """Test argument conversion via implicit constructors"""
        # automatic conversion to svtkVariant
        a = svtk.svtkVariantArray()
        a.InsertNextValue(2.5)
        a.InsertNextValue(svtk.svtkObject())
        self.assertEqual(a.GetValue(0), svtk.svtkVariant(2.5))
        self.assertEqual(a.GetValue(1).GetType(), svtk.SVTK_OBJECT)
        # same, but this one is via "const svtkVariant&" argument
        a = svtk.svtkDenseArray[float]()
        a.Resize(1)
        a.SetVariantValue(0, 2.5)
        self.assertEqual(a.GetVariantValue(0).ToDouble(), 2.5)


if __name__ == "__main__":
    Testing.main([(TestOverloads, 'test')])
