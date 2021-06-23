"""Test subclassing support in SVTK-Python

SVTK classes can be subclassed in Python.  There are
some caveats, such as:
 - protected items are inaccessible to the python class
 - virtual method calls from C++ are not propagated to python

To be tested:
 - make sure that subclassing works
 - make sure that unbound superclass methods can be called

Created on Sept 26, 2010 by David Gobbi

"""

import sys
import svtk
from svtk.test import Testing

class svtkCustomObject(svtk.svtkObject):
    def __init__(self, extra=None):
        """Initialize all attributes."""
        if extra is None:
            extra = svtk.svtkObject()
        self._ExtraObject = extra

    def GetClassName(self):
        """Get the class name."""
        return self.__class__.__name__

    def GetExtraObject(self):
        """Getter method."""
        return self._ExtraObject

    def SetExtraObject(self, o):
        """Setter method."""
        # make sure it is "None" or a svtkobject instance
        if o == None or isinstance(o, svtk.svtkObjectBase):
            self._ExtraObject = o
            self.Modified()
        else:
            raise TypeError("requires None or a svtkobject")

    def GetMTime(self):
        """Override a method (only works when called from Python)"""
        t = svtk.svtkObject.GetMTime(self)
        if self._ExtraObject:
            t = max(t, self._ExtraObject.GetMTime())
        return t


class TestSubclass(Testing.svtkTest):
    def testSubclassInstantiate(self):
        """Instantiate a python svtkObject subclass"""
        o = svtkCustomObject()
        self.assertEqual(o.GetClassName(), "svtkCustomObject")

    def testConstructorArgs(self):
        """Test the use of constructor arguments."""
        extra = svtk.svtkObject()
        o = svtkCustomObject(extra)
        self.assertEqual(o.GetClassName(), "svtkCustomObject")
        self.assertEqual(id(o.GetExtraObject()), id(extra))

    def testCallUnboundMethods(self):
        """Test calling an unbound method in an overridden method"""
        o = svtkCustomObject()
        a = svtk.svtkIntArray()
        o.SetExtraObject(a)
        a.Modified()
        # GetMTime should return a's mtime
        self.assertEqual(o.GetMTime(), a.GetMTime())
        # calling the svtkObject mtime should give a lower MTime
        self.assertNotEqual(o.GetMTime(), svtk.svtkObject.GetMTime(o))
        # another couple quick unbound method check
        svtk.svtkDataArray.InsertNextTuple1(a, 2)
        self.assertEqual(a.GetTuple1(0), 2)

    def testPythonRTTI(self):
        """Test the python isinstance and issubclass methods """
        o = svtkCustomObject()
        d = svtk.svtkIntArray()
        self.assertEqual(True, isinstance(o, svtk.svtkObjectBase))
        self.assertEqual(True, isinstance(d, svtk.svtkObjectBase))
        self.assertEqual(True, isinstance(o, svtkCustomObject))
        self.assertEqual(False, isinstance(d, svtkCustomObject))
        self.assertEqual(False, isinstance(o, svtk.svtkDataArray))
        self.assertEqual(True, issubclass(svtkCustomObject, svtk.svtkObject))
        self.assertEqual(False, issubclass(svtk.svtkObject, svtkCustomObject))
        self.assertEqual(False, issubclass(svtkCustomObject, svtk.svtkDataArray))

    def testSubclassGhost(self):
        """Make sure ghosting of the class works"""
        o = svtkCustomObject()
        c = svtk.svtkCollection()
        c.AddItem(o)
        i = id(o)
        del o
        o = svtk.svtkObject()
        o = c.GetItemAsObject(0)
        # make sure the id has changed, but class the same
        self.assertEqual(o.__class__, svtkCustomObject)
        self.assertNotEqual(i, id(o))

if __name__ == "__main__":
    Testing.main([(TestSubclass, 'test')])
