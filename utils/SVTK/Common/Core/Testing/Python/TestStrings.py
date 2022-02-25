"""Test string and unicode support in SVTK-Python

The following string features have to be tested for string and unicode
- Pass a string arg by value
- Pass a string arg by reference
- Return a string arg by value
- Return a string arg by reference

The following features are not supported
- Pointers to strings, arrays of strings
- Passing a string arg by reference and returning a value in it

Created on May 12, 2010 by David Gobbi
"""

import sys
import svtk
from svtk.test import Testing

if sys.hexversion >= 0x03000000:
    cedilla = 'Fran\xe7ois'
    nocedilla = 'Francois'
    eightbit = 'Francois'.encode('ascii')
else:
    cedilla = unicode('Fran\xe7ois', 'latin1')
    nocedilla = unicode('Francois')
    eightbit = 'Francois'

class TestString(Testing.svtkTest):
    def testPassByValue(self):
        """Pass string by value... hard to find examples of this,
        because "const char *" methods shadow "svtkStdString" methods.
        """
        self.assertEqual('y', 'y')

    def testReturnByValue(self):
        """Return a string by value."""
        a = svtk.svtkArray.CreateArray(1, svtk.SVTK_INT)
        a.Resize(1,1)
        a.SetDimensionLabel(0, 'x')
        s = a.GetDimensionLabel(0)
        self.assertEqual(s, 'x')

    def testPassByReference(self):
        """Pass a string by reference."""
        a = svtk.svtkArray.CreateArray(0, svtk.SVTK_STRING)
        a.SetName("myarray")
        s = a.GetName()
        self.assertEqual(s, "myarray")

    def testReturnByReference(self):
        """Return a string by reference."""
        a = svtk.svtkStringArray()
        s = "hello"
        a.InsertNextValue(s)
        t = a.GetValue(0)
        self.assertEqual(t, s)

    def testPassAndReturnUnicodeByReference(self):
        """Pass a unicode string by const reference"""
        a = svtk.svtkUnicodeStringArray()
        a.InsertNextValue(cedilla)
        u = a.GetValue(0)
        self.assertEqual(u, cedilla)

    def testPassBytesAsUnicode(self):
        """Pass 8-bit string when unicode is expected.  Should fail."""
        a = svtk.svtkUnicodeStringArray()
        self.assertRaises(TypeError,
                          a.InsertNextValue, eightbit)

    def testPassUnicodeAsString(self):
        """Pass unicode where string is expected.  Should succeed."""
        a = svtk.svtkStringArray()
        a.InsertNextValue(nocedilla)
        s = a.GetValue(0)
        self.assertEqual(s, 'Francois')

    def testPassBytesAsString(self):
        """Pass 8-bit string where string is expected.  Should succeed."""
        a = svtk.svtkStringArray()
        a.InsertNextValue(eightbit)
        s = a.GetValue(0)
        self.assertEqual(s, 'Francois')

    def testPassEncodedString(self):
        """Pass encoded 8-bit strings."""
        a = svtk.svtkStringArray()
        # latin1 encoded string will be returned as "bytes", which is
        # just a normal str object in Python 2
        encoded = cedilla.encode('latin1')
        a.InsertNextValue(encoded)
        result = a.GetValue(0)
        self.assertEqual(type(result), bytes)
        self.assertEqual(result, encoded)
        # utf-8 encoded string will be returned as "str", which is
        # actually unicode in Python 3
        a = svtk.svtkStringArray()
        encoded = cedilla.encode('utf-8')
        a.InsertNextValue(encoded)
        result = a.GetValue(0)
        self.assertEqual(type(result), str)
        if sys.hexversion >= 0x03000000:
            self.assertEqual(result.encode('utf-8'), encoded)
        else:
            self.assertEqual(result, encoded)

if __name__ == "__main__":
    Testing.main([(TestString, 'test')])
