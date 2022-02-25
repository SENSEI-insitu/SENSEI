"""Test if the NumPy array support for SVTK data arrays works correctly.
The test requires that numpy (http://numpy.scipy.org) work.

Run this test like so:
 $ svtkpython TestNumpySupport.py
or
 $ python TestNumpySupport.py

"""

import sys
import svtk
from svtk.test import Testing
try:
    import numpy
except ImportError:
    print("Numpy (http://numpy.scipy.org) not found.")
    print("This test requires numpy!")
    from svtk.test import Testing
    Testing.skip()

from svtk.util.numpy_support import numpy_to_svtk, svtk_to_numpy
import svtk.numpy_interface.dataset_adapter as dsa


class TestNumpySupport(Testing.svtkTest):
    def _check_arrays(self, arr, svtk_arr):
        """Private method to check array.
        """
        self.assertEqual(svtk_arr.GetNumberOfTuples(), len(arr))
        if len(arr.shape) == 2:
            dim1 = arr.shape[1]
            self.assertEqual(svtk_arr.GetNumberOfComponents(), dim1)
            for i in range(len(arr)):
                if dim1 in [1,2,3,4,9]:
                    res = getattr(svtk_arr, 'GetTuple%s'%dim1)(i)
                    self.assertEqual(numpy.sum(res - arr[i]), 0)
                else:
                    res = [svtk_arr.GetComponent(i, j) for j in range(dim1)]
                    self.assertEqual(numpy.sum(res - arr[i]), 0)
        else:
            for i in range(len(arr)):
                self.assertEqual(svtk_arr.GetTuple1(i), arr[i])

    def testNumpy2SVTK(self):
        """Test Numeric array to SVTK array conversion and vice-versa."""
        # Put all the test arrays here.
        t_z = []

        # Test the different types of arrays.
        t_z.append(numpy.array([-128, 0, 127], numpy.int8))

        # FIXME: character arrays are a problem since there is no
        # unique mapping to a SVTK data type and back.
        #t_z.append(numpy.array([-128, 0, 127], numpy.character))
        t_z.append(numpy.array([-32768, 0, 32767], numpy.int16))
        t_z.append(numpy.array([-2147483648, 0, 2147483647], numpy.int32))
        t_z.append(numpy.array([0, 255], numpy.uint8))
        t_z.append(numpy.array([0, 65535], numpy.uint16))
        t_z.append(numpy.array([0, 4294967295], numpy.uint32))
        t_z.append(numpy.array([-1.0e38, 0, 1.0e38], 'f'))
        t_z.append(numpy.array([-1.0e299, 0, 1.0e299], 'd'))

        # Check multi-component arrays.
        t_z.append(numpy.array([[1], [2], [300]], 'd'))
        t_z.append(numpy.array([[1, 20], [300, 4000]], 'd'))
        t_z.append(numpy.array([[1, 2, 3], [4, 5, 6]], 'f'))
        t_z.append(numpy.array([[1, 2, 3],[4, 5, 6]], 'd'))
        t_z.append(numpy.array([[1, 2, 3, 400],[4, 5, 6, 700]],
                                 'd'))
        t_z.append(numpy.array([range(9),range(10,19)], 'f'))

        # Test if arrays with number of components not in [1,2,3,4,9] work.
        t_z.append(numpy.array([[1, 2, 3, 400, 5000],
                                  [4, 5, 6, 700, 8000]], 'd'))
        t_z.append(numpy.array([range(10), range(10,20)], 'd'))

        for z in t_z:
            svtk_arr = numpy_to_svtk(z)
            # Test for memory leaks.
            self.assertEqual(svtk_arr.GetReferenceCount(), 1)
            self._check_arrays(z, svtk_arr)
            z1 = svtk_to_numpy(svtk_arr)
            if len(z.shape) == 1:
                self.assertEqual(len(z1.shape), 1)
            self.assertEqual(sum(numpy.ravel(z) - numpy.ravel(z1)), 0)

    def testNumpyView(self):
        "Test if the numpy and SVTK array share the same data."
        # ----------------------------------------
        # Test if the array is copied or not.
        a = numpy.array([[1, 2, 3],[4, 5, 6]], 'd')
        svtk_arr = numpy_to_svtk(a)
        # Change the numpy array and see if the changes are
        # reflected in the SVTK array.
        a[0] = [10.0, 20.0, 30.0]
        self.assertEqual(svtk_arr.GetTuple3(0), (10., 20., 30.))

    def testNumpyConversion(self):
        "Test that converting data copies data properly."
        # ----------------------------------------
        # Test if the array is copied or not.
        a = numpy.array([[1, 2, 3],[4, 5, 6]], 'd')
        svtk_arr = numpy_to_svtk(a, 0, svtk.SVTK_INT)
        # Change the numpy array and see if the changes are
        # reflected in the SVTK array.
        a[0] = [10.0, 20.0, 30.0]
        self.assertEqual(svtk_arr.GetTuple3(0), (1., 2., 3.))

    def testExceptions(self):
        "Test if the right assertion errors are raised."
        # Test if bit arrays raise an exception.
        svtk_arr = svtk.svtkBitArray()
        svtk_arr.InsertNextValue(0)
        svtk_arr.InsertNextValue(1)
        self.assertRaises(AssertionError, svtk_to_numpy, svtk_arr)

    def testNonContiguousArray(self):
        "Test if the non contiguous array are supported"
        a = numpy.array(range(1, 19), 'd')
        a.shape = (3, 6)
        x = a[::2, ::2]
        svtk_array = numpy_to_svtk(x)
        self.assertEqual(svtk_array.GetTuple3(0), (1., 3., 5.))
        self.assertEqual(svtk_array.GetTuple3(1), (13., 15., 17.))

    def testNumpyReferenceWhenDelete(self):
        "Test if the svtk array keeps the numpy reference in memory"
        np_array = numpy.array([[1., 3., 5.], [13., 15., 17.]], 'd')
        svtk_array = numpy_to_svtk(np_array)
        del np_array
        np_array = numpy.array([])

        import gc
        gc.collect()

        self.assertEqual(svtk_array.GetTuple3(0), (1., 3., 5.))
        self.assertEqual(svtk_array.GetTuple3(1), (13., 15., 17.))

    def testNumpyReduce(self):
        "Test that reducing methods return scalars."
        svtk_array = svtk.svtkLongArray()
        for i in range(0, 10):
            svtk_array.InsertNextValue(i)

        numpy_svtk_array = dsa.svtkDataArrayToSVTKArray(svtk_array)
        s = numpy_svtk_array.sum()
        self.assertEqual(s, 45)
        self.assertTrue(isinstance(s, numpy.signedinteger))

        m = numpy_svtk_array.mean()
        self.assertEqual(m, 4.5)
        self.assertTrue(isinstance(m, numpy.floating))

if __name__ == "__main__":
    Testing.main([(TestNumpySupport, 'test')])

