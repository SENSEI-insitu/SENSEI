"""Test the svtk.reference() type and test pass-by-reference.

Created on Sept 19, 2010 by David Gobbi
"""

import sys
import svtk
from svtk.test import Testing

class TestPassByReference(Testing.svtkTest):
    def testFloatReference(self):
        m = svtk.reference(3.0)
        n = svtk.reference(4.0)
        m *= 2
        self.assertEqual(m, 6.0)
        self.assertEqual(str(m), str(m.get()))
        o = n + m
        self.assertEqual(o, 10.0)

    def testIntReference(self):
        m = svtk.reference(3)
        n = svtk.reference(4)
        m |= n
        self.assertEqual(m, 7.0)
        self.assertEqual(str(m), str(m.get()))

    def testStringReference(self):
        m = svtk.reference("%s %s!")
        m %= ("hello", "world")
        self.assertEqual(m, "hello world!")

    def testTupleReference(self):
        m = svtk.reference((0,))
        self.assertEqual(m, (0,))
        self.assertEqual(len(m), 1)
        self.assertEqual(m[0], 0)

    def testPassByReferenceerence(self):
        t = svtk.reference(0.0)
        p0 = (0.5, 0.0, 0.0)
        n = (1.0, 0.0, 0.0)
        p1 = (0.0, 0.0, 0.0)
        p2 = (1.0, 1.0, 1.0)
        x = [0.0, 0.0, 0.0]
        svtk.svtkPlane.IntersectWithLine(p1, p2, n, p0, t, x)
        self.assertEqual(round(t,6), 0.5)
        self.assertEqual(round(x[0],6), 0.5)
        self.assertEqual(round(x[1],6), 0.5)
        self.assertEqual(round(x[2],6), 0.5)
        svtk.svtkPlane().IntersectWithLine(p1, p2, n, p0, t, x)
        self.assertEqual(round(t,6), 0.5)
        self.assertEqual(round(x[0],6), 0.5)
        self.assertEqual(round(x[1],6), 0.5)
        self.assertEqual(round(x[2],6), 0.5)
        t.set(0)
        p = svtk.svtkPlane()
        p.SetOrigin(0.5, 0.0, 0.0)
        p.SetNormal(1.0, 0.0, 0.0)
        p.IntersectWithLine(p1, p2, t, x)
        self.assertEqual(round(t,6), 0.5)
        self.assertEqual(round(x[0],6), 0.5)
        self.assertEqual(round(x[1],6), 0.5)
        self.assertEqual(round(x[2],6), 0.5)
        svtk.svtkPlane.IntersectWithLine(p, p1, p2, t, x)
        self.assertEqual(round(t,6), 0.5)
        self.assertEqual(round(x[0],6), 0.5)
        self.assertEqual(round(x[1],6), 0.5)
        self.assertEqual(round(x[2],6), 0.5)

    def testPassTupleByReference(self):
        n = svtk.reference(0)
        t = svtk.reference((0,))
        ca = svtk.svtkCellArray()
        ca.InsertNextCell(3, (1, 3, 0))
        ca.GetCell(0, n, t)
        self.assertEqual(n, 3)
        self.assertEqual(tuple(t), (1, 3, 0))

if __name__ == "__main__":
    Testing.main([(TestPassByReference, 'test')])
