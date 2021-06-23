"""Test ghost object support in SVTK-Python

When PySVTKObject is destroyed, the svtkObjectBase that it
contained often continues to exist because references to
it still exist within SVTK.  When that svtkObjectBase is
returned to python, a new PySVTKObject is created.

If the PySVTKObject has a custom class or a custom dict,
then we make a "ghost" of the PySVTKObject when it is
destroyed, so that if its svtkObjectBase returns to python,
the PySVTKObject can be restored with the proper class and
dict.  Each ghost has a weak pointer to its svtkObjectBase
so that it can be erased if the svtkObjectBase is destroyed.

To be tested:
 - make sure custom dicts are restored
 - make sure custom classes are restored

Created on Aug 19, 2010 by David Gobbi

"""

import sys
import svtk
from svtk.test import Testing

class svtkCustomObject(svtk.svtkObject):
    pass

class TestGhost(Testing.svtkTest):
    def testGhostForDict(self):
        """Ghost an object to save the dict"""
        o = svtk.svtkObject()
        o.customattr = 'hello'
        a = svtk.svtkVariantArray()
        a.InsertNextValue(o)
        i = id(o)
        del o
        o = svtk.svtkObject()
        o = a.GetValue(0).ToSVTKObject()
        # make sure the id has changed, but dict the same
        self.assertEqual(o.customattr, 'hello')
        self.assertNotEqual(i, id(o))

    def testGhostForClass(self):
        """Ghost an object to save the class"""
        o = svtkCustomObject()
        a = svtk.svtkVariantArray()
        a.InsertNextValue(o)
        i = id(o)
        del o
        o = svtk.svtkObject()
        o = a.GetValue(0).ToSVTKObject()
        # make sure the id has changed, but class the same
        self.assertEqual(o.__class__, svtkCustomObject)
        self.assertNotEqual(i, id(o))

if __name__ == "__main__":
    Testing.main([(TestGhost, 'test')])
