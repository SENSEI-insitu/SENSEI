"""Test svtkVariant support in SVTK-Python

The following svtkVariant features have to be tested:
- Construction from various types
- Automatic arg conversion for methods with svtkVariant args
- Access of various types
- Operators < <= == > >=
- Use of svtkVariant as a dict key

The following features are not supported
- The ToNumeric method

Created on May 12, 2010 by David Gobbi

"""

import sys
import svtk
from svtk.test import Testing

if sys.hexversion >= 0x03000000:
    cedilla = 'Fran\xe7ois'
else:
    cedilla = unicode('Fran\xe7ois', 'latin1')


class TestVariant(Testing.svtkTest):
    def testDefaultConstructor(self):
        """Default constructor"""
        v = svtk.svtkVariant()
        self.assertEqual(v.IsValid(), False)
        self.assertEqual(v.GetType(), 0)

    def testCopyConstructor(self):
        """Construct from another svtkVariant"""
        u = svtk.svtkVariant('test')
        v = svtk.svtkVariant(u)
        self.assertEqual(v.GetType(), svtk.SVTK_STRING)
        self.assertEqual(v.ToString(), u.ToString())

    def testIntConstructor(self):
        """Construct from int"""
        v = svtk.svtkVariant(10)
        self.assertEqual(v.GetType(), svtk.SVTK_INT)
        self.assertEqual(v.ToInt(), 10)

    def testFloatConstructor(self):
        """Construct from float"""
        v = svtk.svtkVariant(10.0)
        self.assertEqual(v.GetType(), svtk.SVTK_DOUBLE)
        self.assertEqual(v.ToDouble(), 10.0)

    def testStringConstructor(self):
        """Construct from string"""
        v = svtk.svtkVariant('hello')
        self.assertEqual(v.GetType(), svtk.SVTK_STRING)
        self.assertEqual(v.ToString(), 'hello')

    def testBytesConstructor(self):
        """Construct from bytes"""
        v = svtk.svtkVariant(b'hello')
        self.assertEqual(v.GetType(), svtk.SVTK_STRING)
        self.assertEqual(v.ToString(), 'hello')

    def testUnicodeConstructor(self):
        """Construct from unicode"""
        v = svtk.svtkVariant(cedilla)
        self.assertEqual(v.GetType(), svtk.SVTK_UNICODE_STRING)
        self.assertEqual(v.ToUnicodeString(), cedilla)

    def testObjectConstructor(self):
        """Construct from SVTK object"""
        o = svtk.svtkIntArray()
        v = svtk.svtkVariant(o)
        self.assertEqual(v.GetType(), svtk.SVTK_OBJECT)
        self.assertEqual(v.GetTypeAsString(), o.GetClassName())
        self.assertEqual(v.ToSVTKObject(), o)

    def testTwoArgConstructor(self):
        """Construct with a specific type"""
        # construct with conversion to int
        v = svtk.svtkVariant('10')
        u = svtk.svtkVariant(v, svtk.SVTK_INT)
        self.assertEqual(u.GetType(), svtk.SVTK_INT)
        self.assertEqual(v.ToInt(), u.ToInt())

        # construct with conversion to double
        v = svtk.svtkVariant(10)
        u = svtk.svtkVariant(v, svtk.SVTK_DOUBLE)
        self.assertEqual(u.GetType(), svtk.SVTK_DOUBLE)
        self.assertEqual(u.ToDouble(), 10.0)

        # failed conversion to svtkObject
        v = svtk.svtkVariant(10)
        u = svtk.svtkVariant(v, svtk.SVTK_OBJECT)
        self.assertEqual(u.IsValid(), False)

    def testAutomaticArgConversion(self):
        """Automatic construction of variants to resolve args"""
        # use with one of svtkVariant's own constructors
        v = svtk.svtkVariant('10', svtk.SVTK_INT)
        self.assertEqual(v.ToInt(), 10)
        self.assertEqual(v.GetType(), svtk.SVTK_INT)

        # use with svtkVariantArray
        a = svtk.svtkVariantArray()
        i = a.InsertNextValue(10)
        v = a.GetValue(i)
        self.assertEqual(v.GetType(), svtk.SVTK_INT)
        self.assertEqual(v.ToInt(), 10)
        i = a.InsertNextValue(10.0)
        v = a.GetValue(i)
        self.assertEqual(v.GetType(), svtk.SVTK_DOUBLE)
        self.assertEqual(v.ToDouble(), 10.0)
        i = a.InsertNextValue('10')
        v = a.GetValue(i)
        self.assertEqual(v.GetType(), svtk.SVTK_STRING)
        self.assertEqual(v.ToString(), '10')

    def testCompare(self):
        """Use comparison operators to sort a list of svtkVariants"""
        original = [1, 2.5, svtk.svtkVariant(), "0", cedilla]
        ordered = [svtk.svtkVariant(), "0", 1, 2.5, cedilla]
        l = [svtk.svtkVariant(x) for x in original]
        s = [svtk.svtkVariant(x) for x in ordered]
        l.sort()
        self.assertEqual(l, s)

    def testComparisonMethods(self):
        v1 = svtk.svtkVariant(10)
        v2 = svtk.svtkVariant("10")
        # compare without regards to type
        self.assertEqual(svtk.svtkVariantEqual(v1, v2), True)
        self.assertEqual(svtk.svtkVariantLessThan(v1, v2), False)
        # compare with different types being non-equivalent
        self.assertEqual(svtk.svtkVariantStrictEquality(v1, v2), False)
        if sys.hexversion >= 0x03000000:
            self.assertEqual(svtk.svtkVariantStrictWeakOrder(v1, v2), True)
        else:
            # for Python 2, it worked like the cmp() function
            self.assertEqual(svtk.svtkVariantStrictWeakOrder(v1, v2), -1)

    def testStrictWeakOrder(self):
        """Use svtkVariantStrictWeakOrder to sort a list of svtkVariants"""
        original = [1, 2.5, svtk.svtkVariant(), "0", cedilla]
        ordered = [svtk.svtkVariant(), 1, 2.5, "0", cedilla]
        l = [svtk.svtkVariant(x) for x in original]
        s = [svtk.svtkVariant(x) for x in ordered]
        l.sort(key=svtk.svtkVariantStrictWeakOrderKey)
        self.assertEqual(l, s)

    def testVariantExtract(self):
        """Use svtkVariantExtract"""
        l = [1, '2', cedilla, 4.0, svtk.svtkObject()]
        s = [svtk.svtkVariant(x) for x in l]
        m = [svtk.svtkVariantExtract(x) for x in s]
        self.assertEqual(l, m)

    def testHash(self):
        """Use a variant as a dict key"""
        d = {}
        # doubles, ints, strings, all hash as strings
        d[svtk.svtkVariant(1.0)] = 'double'
        d[svtk.svtkVariant(1)] = 'int'
        self.assertEqual(d[svtk.svtkVariant('1')], 'int')

        # strings and unicode have the same hash
        d[svtk.svtkVariant('s').ToString()] = 'string'
        d[svtk.svtkVariant('s').ToUnicodeString()] = 'unicode'
        self.assertEqual(d[svtk.svtkVariant('s').ToString()], 'unicode')

        # every svtkObject is hashed by memory address
        o1 = svtk.svtkIntArray()
        o2 = svtk.svtkIntArray()
        d[svtk.svtkVariant(o1)] = 'svtkIntArray1'
        d[svtk.svtkVariant(o2)] = 'svtkIntArray2'
        self.assertEqual(d[svtk.svtkVariant(o1)], 'svtkIntArray1')
        self.assertEqual(d[svtk.svtkVariant(o2)], 'svtkIntArray2')

        # invalid variants all hash the same
        d[svtk.svtkVariant()] = 'invalid'
        self.assertEqual(d[svtk.svtkVariant()], 'invalid')

    def testPassByValueReturnByReference(self):
        """Pass svtkVariant by value, return by reference"""
        a = svtk.svtkVariantArray()
        a.SetNumberOfValues(1)
        v = svtk.svtkVariant(1)
        a.SetValue(0, v)
        u = a.GetValue(0)
        self.assertEqual(u.ToInt(), v.ToInt())
        self.assertEqual(u.GetType(), v.GetType())
        self.assertEqual(u.IsValid(), v.IsValid())

    def testPassByReferenceReturnByValue(self):
        """Pass svtkVariant by reference, return by value."""
        a = svtk.svtkArray.CreateArray(1, svtk.SVTK_INT)
        a.Resize(1,1)
        v = svtk.svtkVariant(1)
        a.SetVariantValue(0, 0, v)
        u = a.GetVariantValue(0, 0)
        self.assertEqual(u.ToInt(), v.ToInt())
        self.assertEqual(u.GetType(), v.GetType())
        self.assertEqual(u.IsValid(), v.IsValid())

if __name__ == "__main__":
    Testing.main([(TestVariant, 'test')])
