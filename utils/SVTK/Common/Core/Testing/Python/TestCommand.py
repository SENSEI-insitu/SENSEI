"""Test the wrapping of svtkCommand

The following svtkCommmand functionality must be tested
- Event enum constants
- Event names

Created on Mar 22, 2012 by David Gobbi
Updated on Feb 12, 2014 by Jean-Christophe Fillion-Robin
"""

import sys
import gc
import svtk
from svtk.test import Testing

class callback:
    def __init__(self):
        self.reset()

    def __call__(self, o, e, d = None):
        self.caller = o
        self.event = e
        self.calldata = d

    def reset(self):
        self.caller = None
        self.event = None
        self.calldata = None


class TestCommand(Testing.svtkTest):
    def testEnumConstants(self):
        """Make sure the event id constants are wrapped
        """
        self.assertEqual(svtk.svtkCommand.ErrorEvent, 39)

    def testCommandWithArgs(self):
        """Test binding a command that has arguments.
        """
        cb = callback()
        o = svtk.svtkObject()
        o.AddObserver(svtk.svtkCommand.ModifiedEvent, cb)
        o.Modified()

        self.assertEqual(cb.caller, o)
        self.assertEqual(cb.event, "ModifiedEvent")

    def testUseEventNameString(self):
        """Test binding with a string event name.
        """
        cb = callback()
        o = svtk.svtkObject()
        o.AddObserver("ModifiedEvent", cb)
        o.Modified()

        self.assertEqual(cb.caller, o)
        self.assertEqual(cb.event, "ModifiedEvent")

    def testPriorityArg(self):
        """Test the optional priority argument
        """
        cb = callback()
        o = svtk.svtkObject()
        o.AddObserver(svtk.svtkCommand.ModifiedEvent, cb, 0.5)
        o.Modified()

        self.assertEqual(cb.caller, o)

    def testRemoveCommand(self):
        """Test the removal of an observer.
        """
        cb = callback()
        o = svtk.svtkObject()
        o.AddObserver(svtk.svtkCommand.ModifiedEvent, cb)
        o.Modified()
        self.assertEqual(cb.caller, o)
        o.RemoveObservers(svtk.svtkCommand.ModifiedEvent)
        cb.caller = None
        cb.event = None
        o.Modified()
        self.assertEqual(cb.caller, None)

    def testGetCommand(self):
        """Test getting the svtkCommand object
        """
        cb = callback()
        o = svtk.svtkObject()
        n = o.AddObserver(svtk.svtkCommand.ModifiedEvent, cb)
        o.Modified()
        self.assertEqual(cb.caller, o)
        c = o.GetCommand(n)
        self.assertEqual((c.IsA("svtkCommand") != 0), True)
        # in the future, o.RemoveObserver(c) should also be made to work
        o.RemoveObserver(n)
        cb.caller = None
        cb.event = None
        o.Modified()
        self.assertEqual(cb.caller, None)

    def testCommandCircularRef(self):
        """Test correct reference loop reporting for commands
        """
        cb = callback()
        o = svtk.svtkObject()
        o.AddObserver(svtk.svtkCommand.ModifiedEvent, cb)
        cb.circular_ref = o
        # referent to check if "o" is deleted
        referent = svtk.svtkObject()
        o.referent = referent
        # make sure gc removes referrer "o" from referent
        s1 = repr(gc.get_referrers(referent))
        del o
        del cb
        gc.collect()
        s2 = repr(gc.get_referrers(referent))

        self.assertNotEqual(s1, s2)
        self.assertNotEqual(s1.count("svtkObject"),0)
        self.assertEqual(s2.count("svtkObject"),0)

    def testAddRemoveObservers(self):
        """Test adding and removing observers
        """
        cb = callback()
        cb2 = callback()
        o = svtk.svtkObject()
        n = o.AddObserver(svtk.svtkCommand.ModifiedEvent, cb)
        n2 = o.AddObserver(svtk.svtkCommand.ModifiedEvent, cb2)
        o.Modified()
        self.assertEqual(cb.caller, o)
        self.assertEqual(cb2.caller, o)
        o.RemoveObserver(n)
        cb.reset()
        cb2.reset()
        o.Modified()
        self.assertEqual(cb.caller, None)
        self.assertEqual(cb2.caller, o)
        o.RemoveObserver(n2)
        cb.reset()
        cb2.reset()
        o.Modified()
        self.assertEqual(cb.caller, None)
        self.assertEqual(cb2.caller, None)

    def testUseCallDataType(self):
        """Test adding an observer associated with a callback expecting a CallData
        """
        cb = callback()
        cb.CallDataType = svtk.SVTK_STRING
        lt = svtk.svtkLookupTable()
        lt.AddObserver(svtk.svtkCommand.ErrorEvent, cb)
        lt.SetTableRange(2, 1)
        self.assertEqual(cb.caller, lt)
        self.assertEqual(cb.event, "ErrorEvent")
        self.assertTrue(cb.calldata.startswith("ERROR: In"))

    def testUseCallDataTypeWithDecoratorAsString0(self):
        """Test adding an observer associated with a callback expecting a CallData.
        This test ensures backward compatibility checking the CallDataType can
        be set to the string 'string0'.
        """
        self.onErrorCalldata = ''

        @svtk.calldata_type('string0')
        def onError(caller, event, calldata):
            self.onErrorCalldata = calldata

        lt = svtk.svtkLookupTable()
        lt.AddObserver(svtk.svtkCommand.ErrorEvent, onError)
        lt.SetTableRange(2, 1)
        self.assertTrue(self.onErrorCalldata.startswith("ERROR: In"))

    def testUseCallDataTypeWithDecorator(self):
        """Test adding an observer associated with a callback expecting a CallData
        """
        self.onErrorCalldata = ''

        @svtk.calldata_type(svtk.SVTK_STRING)
        def onError(caller, event, calldata):
            self.onErrorCalldata = calldata

        lt = svtk.svtkLookupTable()
        lt.AddObserver(svtk.svtkCommand.ErrorEvent, onError)
        lt.SetTableRange(2, 1)
        self.assertTrue(self.onErrorCalldata.startswith("ERROR: In"))

if __name__ == "__main__":
    Testing.main([(TestCommand, 'test')])
