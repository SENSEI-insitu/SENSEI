import unittest
import svtk

from svtk.test import Testing

testInt = 12
testString = "test string"
testFloat = 5.4


class SVTKPythonObjectCalldataInvokeEventTest(Testing.svtkTest):

    @svtk.calldata_type(svtk.SVTK_INT)
    def callbackInt(self, caller, event, calldata):
        self.calldata = calldata

    @svtk.calldata_type(svtk.SVTK_STRING)
    def callbackString(self, caller, event, calldata):
        self.calldata = calldata

    @svtk.calldata_type(svtk.SVTK_DOUBLE)
    def callbackFloat(self, caller, event, calldata):
        self.calldata = calldata

    @svtk.calldata_type(svtk.SVTK_OBJECT)
    def callbackObj(self, caller, event, calldata):
        self.calldata = calldata

    def setUp(self):
        self.svtkObj = svtk.svtkObject()

        self.svtkObjForCallData = svtk.svtkObject()

    def test_int(self):
        self.svtkObj.AddObserver(svtk.svtkCommand.AnyEvent, self.callbackInt)
        self.svtkObj.InvokeEvent(svtk.svtkCommand.ModifiedEvent, testInt)
        self.assertEqual(self.calldata, testInt)

    def test_string(self):
        self.svtkObj.AddObserver(svtk.svtkCommand.AnyEvent, self.callbackString)
        self.svtkObj.InvokeEvent(svtk.svtkCommand.ModifiedEvent, testString)
        self.assertEqual(self.calldata, testString)

    def test_float(self):
        self.svtkObj.AddObserver(svtk.svtkCommand.AnyEvent, self.callbackFloat)
        self.svtkObj.InvokeEvent(svtk.svtkCommand.ModifiedEvent, testFloat)
        self.assertAlmostEqual(self.calldata, testFloat)

    def test_obj(self):
        self.svtkObj.AddObserver(svtk.svtkCommand.AnyEvent, self.callbackObj)
        self.svtkObj.InvokeEvent(svtk.svtkCommand.ModifiedEvent, self.svtkObjForCallData)
        self.assertEqual(self.calldata, self.svtkObjForCallData)



if __name__ == '__main__':
    Testing.main([(SVTKPythonObjectCalldataInvokeEventTest, 'test')])
