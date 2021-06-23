"""Test enum support in SVTK-Python

Created on Nov 13, 2014 by David Gobbi
"""

import sys
import svtk
from svtk.test import Testing

class TestEnum(Testing.svtkTest):
    def testGlobalNamespaceEnum(self):
        """Check that an enum in the global namespace was wrapped.
        """
        # defined in svtkGenericEnSightReader.h
        if hasattr(svtk, 'svtkGenericEnsightReader'):
            self.assertEqual(svtk.SINGLE_PROCESS_MODE, 0)
            self.assertEqual(svtk.SPARSE_MODE, 1)
            self.assertEqual(type(svtk.SINGLE_PROCESS_MODE),
                             svtk.EnsightReaderCellIdMode)
            self.assertEqual(type(svtk.SPARSE_MODE),
                             svtk.EnsightReaderCellIdMode)

    def testClassNamespaceEnum(self):
        """Check that an enum in a class namespace was wrapped.
        """
        # defined in svtkColorSeries.h
        self.assertEqual(svtk.svtkColorSeries.SPECTRUM, 0)
        self.assertEqual(type(svtk.svtkColorSeries.SPECTRUM),
                         svtk.svtkColorSeries.ColorSchemes)
        # defined in svtkErrorCode.h
        self.assertEqual(svtk.svtkErrorCode.FirstSVTKErrorCode, 20000)
        self.assertEqual(type(svtk.svtkErrorCode.FirstSVTKErrorCode),
                         svtk.svtkErrorCode.ErrorIds)

    def testAnonymousEnum(self):
        """Check that anonymous enums are wrapped.
        """
        # defined in svtkAbstractArray.h
        self.assertEqual(svtk.svtkAbstractArray.AbstractArray, 0)

    def testEnumClass(self):
        """Check that "enum class" members are wrapped.
        """
        # defined in svtkEventData.h
        val = svtk.svtkEventDataAction.Unknown
        self.assertEqual(val, -1)
        obj = svtk.svtkEventDataForDevice()
        self.assertEqual(obj.GetAction(), val)
        self.assertEqual(type(obj.GetAction()), svtk.svtkEventDataAction)

if __name__ == "__main__":
    Testing.main([(TestEnum, 'test')])
