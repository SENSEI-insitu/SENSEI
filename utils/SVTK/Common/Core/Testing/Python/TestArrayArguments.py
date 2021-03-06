"""Test passing arguments of various types

Created on Sept 19, 2010 by David Gobbi
"""

import sys
import svtk
from svtk.test import Testing

arrays = [
  svtk.svtkFloatArray,
  svtk.svtkDoubleArray,
  svtk.svtkSignedCharArray,
  svtk.svtkShortArray,
  svtk.svtkIntArray,
  svtk.svtkLongArray,
  svtk.svtkTypeInt64Array,
  svtk.svtkIdTypeArray,
]

unsignedArrays = [
  svtk.svtkUnsignedCharArray,
  svtk.svtkUnsignedShortArray,
  svtk.svtkUnsignedIntArray,
  svtk.svtkUnsignedLongArray,
  svtk.svtkTypeUInt64Array,
]

class TestArrayArguments(Testing.svtkTest):
    def testArrayArguments(self):
        for arrayClass in arrays:
            a = arrayClass()
            a.SetNumberOfComponents(3)
            a.SetNumberOfTuples(1)
            ti = [0, a.GetDataTypeValueMin(), a.GetDataTypeValueMax()]
            to = [0, 0, 0]
            a.SetTypedTuple(0, ti)
            a.GetTypedTuple(0, to);
            self.assertEqual(ti, to)
            d1 = a.GetTuple(0)
            d2 = [float(x) for x in ti]
            r1 = [round(x) for x in d1]
            r2 = [round(x) for x in d2]
            self.assertEqual(r1, r2)

    def testUnsignedArrayArguments(self):
        for arrayClass in unsignedArrays:
            a = arrayClass()
            a.SetNumberOfComponents(3)
            a.SetNumberOfTuples(1)
            ti = [0, a.GetDataTypeValueMin(), a.GetDataTypeValueMax()]
            to = [0, 0, 0]
            a.SetTypedTuple(0, ti)
            a.GetTypedTuple(0, to);
            self.assertEqual(ti, to)
            d1 = a.GetTuple(0)
            d2 = [float(x) for x in ti]
            r1 = [round(x) for x in d1]
            r2 = [round(x) for x in d2]
            self.assertEqual(r1, r2)

    def testCharArrayArguments(self):
        a = svtk.svtkCharArray()
        a.SetNumberOfComponents(3)
        a.SetNumberOfTuples(1)
        ti = "opn"
        a.SetTypedTuple(0, ti)
        # python strings are immutable, so this should NOT work
        #to = "***"
        #a.GetTypedTuple(0, to);
        d1 = list(a.GetTuple(0))
        d2 = [ord(x) for x in ti]
        self.assertEqual(d1, d2)

    def testBitArrayArguments(self):
        a = svtk.svtkBitArray()
        a.SetNumberOfComponents(2)
        a.SetNumberOfTuples(1)
        ti = [0,1]
        to = [0,0]
        a.SetTuple(0, ti)
        a.GetTuple(0, to);
        self.assertEqual(ti, [int(x) for x in to])

    def testNDimArrayArguments(self):
        a = [[0,0,0],[0,0,0],[0,0,0]]
        svtk.svtkMath.Identity3x3(a)
        x = [0.5, 0.2, 0.1]
        y = [0.0, 0.0, 0.0]
        svtk.svtkMath.Multiply3x3(a, x, y)
        self.assertEqual(x, y)

    def testInformationVectorKeys(self):
        a = svtk.svtkImageGridSource()
        spacing = (3.0, 2.0, 1.0)
        a.SetDataSpacing(spacing)
        a.UpdateInformation()
        info = a.GetOutputInformation(0)
        t = info.Get(svtk.svtkDataObject.SPACING())
        self.assertEqual(t, spacing)

    def testArrayIterator(self):
        # try a string array
        a = svtk.svtkStringArray()
        a.InsertNextValue("hello")
        i = a.NewIterator()
        self.assertEqual(a.GetValue(0), i.GetValue(0))
        # try the various data array subclasses
        for arrayClass in arrays:
            a = arrayClass()
            a.SetNumberOfComponents(2)
            a.SetNumberOfTuples(1)
            tupleIn = (a.GetDataTypeValueMin(), a.GetDataTypeValueMax())
            a.SetTypedTuple(0, tupleIn)
            i = a.NewIterator()
            # make sure iterator's GetTuple method is wrapped
            tupleOut = i.GetTuple(0)
            self.assertEqual(tupleIn, tupleOut)
            # make sure the GetValue method returns expected result
            self.assertEqual(tupleIn[0], i.GetValue(0))
            self.assertEqual(tupleIn[1], i.GetValue(1))

if __name__ == "__main__":
    Testing.main([(TestArrayArguments, 'test')])
