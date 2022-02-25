import svtk
try:
    import numpy
    from svtk.numpy_interface import dataset_adapter as dsa
except ImportError:
    import sys
    sys.exit(0)

# Let's test svtkDataSetAttributes::SetupForCopy() and
# svtkDataSetAttributes::CopyStructuredData() - the code
# path where we copy from a smaller structured data to
# a bigger one that has already some values assigned.
w = svtk.svtkRTAnalyticSource()
w.Update()

w = w.GetOutput()

w2 = svtk.svtkRTAnalyticSource()
ext = (0,5,0,5,0,5)
w2.SetWholeExtent(*ext)
w2.Update()

w2 = dsa.WrapDataObject(w2.GetOutput())
a = w2.PointData['RTData']
a[:] = 2

pd = w.GetPointData()
pd2 = w2.GetPointData()
pd.SetupForCopy(pd2.SVTKObject)

pd.CopyStructuredData(pd2.SVTKObject, w2.GetExtent(), w.GetExtent())

w = dsa.WrapDataObject(w)
assert(numpy.all(w.PointData['RTData'].reshape(w.GetDimensions(), order='F')[10:15, 10:15, 10:15] == 2))
