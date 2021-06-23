"""Test operator support in SVTK-Python

The following operators are supported:
- The << operator becomes python str() and print()
- The < <= == != > >= operators become richcompare
- The [int] operator become the sequence protocol

The following operators are not yet supported:
- The () operator
- The [] operator for the mapping protocol
- Arithmetic operators + - * / %

Created on May 7, 2011 by David Gobbi

"""

import sys
import svtk
from svtk.test import Testing

class TestOperators(Testing.svtkTest):

    def testPrint(self):
        """Use str slot"""
        c1 = svtk.svtkArrayRange(3,4)
        s1 = str(c1)
        s2 = '[3, 4)'
        self.assertEqual(s1, s2)

    def testCompare(self):
        """Use comparison operators"""
        c1 = svtk.svtkArrayRange(3,4)
        c2 = svtk.svtkArrayRange(3,4)
        # will fail if the "==" operator is not wrapped
        self.assertEqual(c1, c2)

    def testSequence(self):
        """Use sequence operators"""
        c1 = svtk.svtkArrayCoordinates()
        c1.SetDimensions(3)
        n = len(c1) # sq_length slot
        self.assertEqual(n, 3)
        c1[1] = 5  # sq_ass_item slot
        n = c1[1]  # sq_item slot
        self.assertEqual(n, 5)
        r = svtk.svtkArrayRange(3,4)
        e = svtk.svtkArrayExtents()
        e.SetDimensions(2)
        e[0] = r
        s = e[0]
        self.assertEqual(s, r)

if __name__ == "__main__":
    Testing.main([(TestOperators, 'test')])
