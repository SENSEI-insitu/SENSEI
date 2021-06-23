from svtkmodules import svtkCommonCore as cc
from svtkmodules import svtkCommonDataModel as dm
from svtkmodules import svtkCommonExecutionModel as em
from svtkmodules import svtkImagingCore as ic
from svtk.util.svtkAlgorithm import SVTKPythonAlgorithmBase

from svtk.test import Testing

class SimpleFilter(SVTKPythonAlgorithmBase):
    def __init__(self):
        SVTKPythonAlgorithmBase.__init__(self)
        self.InputType = "svtkDataSet"
        self.OutputType = "svtkDataObject"
        self.Counter = 0

    def FillInputPortInformation(self, port, info):
        """Sets the required input type to InputType."""
        info.Set(em.svtkAlgorithm.INPUT_REQUIRED_DATA_TYPE(), self.InputType)
        return 1

    def RequestDataObject(self, request, inInfo, outInfo):
        inp = dm.svtkDataObject.GetData(inInfo[0])
        opt = dm.svtkDataObject.GetData(outInfo)

        if opt and opt.IsA(inp.GetClassName()):
            return 1

        opt = inp.NewInstance()
        outInfo.GetInformationObject(0).Set(dm.svtkDataObject.DATA_OBJECT(), opt)
        return 1

    def RequestData(self, request, inInfo, outInfo):
        inp = dm.svtkDataObject.GetData(inInfo[0])
        print("SimpleFilter iter %d: Input: %s"%(self.Counter, inp.GetClassName()))
        opt = dm.svtkDataObject.GetData(outInfo)
        a = cc.svtkTypeUInt8Array()
        a.SetName("counter")
        a.SetNumberOfTuples(1)
        a.SetValue(0, self.Counter)
        a.GetInformation().Set(dm.svtkDataObject.DATA_TYPE_NAME(), inp.GetClassName());
        opt.GetFieldData().AddArray(a)
        self.Counter += 1
        return 1

class PartitionAwareFilter(SVTKPythonAlgorithmBase):
    def __init__(self):
        SVTKPythonAlgorithmBase.__init__(self)
        self.InputType = "svtkDataSet"
        self.OutputType = "svtkDataObject"
        self.Counter = 0

    def FillInputPortInformation(self, port, info):
        """Sets the required input type to InputType."""
        info.Set(em.svtkAlgorithm.INPUT_REQUIRED_DATA_TYPE(), self.InputType)
        info.Append(em.svtkAlgorithm.INPUT_REQUIRED_DATA_TYPE(), "svtkPartitionedDataSet")
        return 1

    def RequestDataObject(self, request, inInfo, outInfo):
        inp = dm.svtkDataObject.GetData(inInfo[0])
        opt = dm.svtkDataObject.GetData(outInfo)

        if opt and opt.IsA(inp.GetClassName()):
            return 1

        opt = inp.NewInstance()
        outInfo.GetInformationObject(0).Set(dm.svtkDataObject.DATA_OBJECT(), opt)
        return 1

    def RequestData(self, request, inInfo, outInfo):
        inp = dm.svtkDataObject.GetData(inInfo[0])
        print("PartitionAwareFilter iter %d: Input: %s"%(self.Counter, inp.GetClassName()))
        opt = dm.svtkDataObject.GetData(outInfo)
        a = cc.svtkTypeUInt8Array()
        a.SetName("counter")
        a.SetNumberOfTuples(1)
        a.SetValue(0, self.Counter)
        a.GetInformation().Set(dm.svtkDataObject.DATA_TYPE_NAME(), inp.GetClassName());
        opt.GetFieldData().AddArray(a)
        self.Counter += 1
        return 1

class PartitionCollectionAwareFilter(SVTKPythonAlgorithmBase):
    def __init__(self):
        SVTKPythonAlgorithmBase.__init__(self)
        self.InputType = "svtkDataSet"
        self.OutputType = "svtkDataObject"
        self.Counter = 0

    def FillInputPortInformation(self, port, info):
        """Sets the required input type to InputType."""
        info.Set(em.svtkAlgorithm.INPUT_REQUIRED_DATA_TYPE(), self.InputType)
        info.Append(em.svtkAlgorithm.INPUT_REQUIRED_DATA_TYPE(), "svtkPartitionedDataSetCollection")
        return 1

    def RequestDataObject(self, request, inInfo, outInfo):
        inp = dm.svtkDataObject.GetData(inInfo[0])
        opt = dm.svtkDataObject.GetData(outInfo)

        if opt and opt.IsA(inp.GetClassName()):
            return 1

        opt = inp.NewInstance()
        outInfo.GetInformationObject(0).Set(dm.svtkDataObject.DATA_OBJECT(), opt)
        return 1

    def RequestData(self, request, inInfo, outInfo):
        inp = dm.svtkDataObject.GetData(inInfo[0])
        print("PartitionCollectionAwareFilter iter %d: Input: %s"%(self.Counter, inp.GetClassName()))
        opt = dm.svtkDataObject.GetData(outInfo)
        a = cc.svtkTypeUInt8Array()
        a.SetName("counter")
        a.SetNumberOfTuples(1)
        a.SetValue(0, self.Counter)
        a.GetInformation().Set(dm.svtkDataObject.DATA_TYPE_NAME(), inp.GetClassName());
        opt.GetFieldData().AddArray(a)
        self.Counter += 1
        return 1

class CompositeAwareFilter(SVTKPythonAlgorithmBase):
    def __init__(self):
        SVTKPythonAlgorithmBase.__init__(self)
        self.InputType = "svtkDataSet"
        self.OutputType = "svtkDataObject"
        self.Counter = 0

    def FillInputPortInformation(self, port, info):
        """Sets the required input type to InputType."""
        info.Set(em.svtkAlgorithm.INPUT_REQUIRED_DATA_TYPE(), self.InputType)
        info.Append(em.svtkAlgorithm.INPUT_REQUIRED_DATA_TYPE(), "svtkCompositeDataSet")
        return 1

    def RequestDataObject(self, request, inInfo, outInfo):
        inp = dm.svtkDataObject.GetData(inInfo[0])
        opt = dm.svtkDataObject.GetData(outInfo)

        if opt and opt.IsA(inp.GetClassName()):
            return 1

        opt = inp.NewInstance()
        outInfo.GetInformationObject(0).Set(dm.svtkDataObject.DATA_OBJECT(), opt)
        return 1

    def RequestData(self, request, inInfo, outInfo):
        inp = dm.svtkDataObject.GetData(inInfo[0])
        print("CompositeAwareFilter iter %d: Input: %s"%(self.Counter, inp.GetClassName()))
        opt = dm.svtkDataObject.GetData(outInfo)
        a = cc.svtkTypeUInt8Array()
        a.SetName("counter")
        a.SetNumberOfTuples(1)
        a.SetValue(0, self.Counter)
        a.GetInformation().Set(dm.svtkDataObject.DATA_TYPE_NAME(), inp.GetClassName());
        opt.GetFieldData().AddArray(a)
        self.Counter += 1
        return 1

class TestPartitionedData(Testing.svtkTest):

    def test(self):

        p = dm.svtkPartitionedDataSet()

        s = ic.svtkRTAnalyticSource()
        s.SetWholeExtent(0, 10, 0, 10, 0, 5)
        s.Update()

        p1 = dm.svtkImageData()
        p1.ShallowCopy(s.GetOutput())

        s.SetWholeExtent(0, 10, 0, 10, 5, 10)
        s.Update()

        p2 = dm.svtkImageData()
        p2.ShallowCopy(s.GetOutput())

        p.SetPartition(0, p1)
        p.SetPartition(1, p2)

        p2 = dm.svtkPartitionedDataSet()
        p2.ShallowCopy(p)

        c = dm.svtkPartitionedDataSetCollection()
        c.SetPartitionedDataSet(0, p)
        c.SetPartitionedDataSet(1, p2)

        # SimpleFilter:
        sf = SimpleFilter()
        sf.SetInputDataObject(c)
        sf.Update()
        self.assertEqual(sf.GetOutputDataObject(0).GetNumberOfPartitionedDataSets(), 2)
        for i in (0, 1):
            pdsc = sf.GetOutputDataObject(0)
            self.assertEqual(pdsc.GetClassName(), "svtkPartitionedDataSetCollection")
            pds = pdsc.GetPartitionedDataSet(i)
            self.assertEqual(pds.GetClassName(), "svtkPartitionedDataSet")
            self.assertEqual(pds.GetNumberOfPartitions(), 2)
            for j in (0, 1):
                part = pds.GetPartition(j)
                countArray = part.GetFieldData().GetArray("counter")
                info = countArray.GetInformation()
                self.assertEqual(countArray.GetValue(0), i * 2 + j);
                self.assertEqual(info.Get(dm.svtkDataObject.DATA_TYPE_NAME()), "svtkImageData")

        # PartitionAwareFilter
        pf = PartitionAwareFilter()
        pf.SetInputDataObject(c)
        pf.Update()
        self.assertEqual(pf.GetOutputDataObject(0).GetNumberOfPartitionedDataSets(), 2)
        for i in (0, 1):
            pdsc = pf.GetOutputDataObject(0)
            self.assertEqual(pdsc.GetClassName(), "svtkPartitionedDataSetCollection")
            pds = pdsc.GetPartitionedDataSet(i)
            self.assertEqual(pds.GetClassName(), "svtkPartitionedDataSet")
            self.assertEqual(pds.GetNumberOfPartitions(), 0)
            countArray = pds.GetFieldData().GetArray("counter")
            info = countArray.GetInformation()
            self.assertEqual(countArray.GetValue(0), i);
            self.assertEqual(info.Get(dm.svtkDataObject.DATA_TYPE_NAME()), "svtkPartitionedDataSet")

        # PartitionCollectionAwareFilter
        pcf = PartitionCollectionAwareFilter()
        pcf.SetInputDataObject(c)
        pcf.Update()
        self.assertEqual(pcf.GetOutputDataObject(0).GetNumberOfPartitionedDataSets(), 0)
        pdsc = pcf.GetOutputDataObject(0)
        self.assertEqual(pdsc.GetClassName(), "svtkPartitionedDataSetCollection")
        countArray = pdsc.GetFieldData().GetArray("counter")
        info = countArray.GetInformation()
        self.assertEqual(countArray.GetValue(0), 0);
        self.assertEqual(info.Get(dm.svtkDataObject.DATA_TYPE_NAME()), "svtkPartitionedDataSetCollection")

        # CompositeAwareFilter
        cf = CompositeAwareFilter()
        cf.SetInputDataObject(c)
        cf.Update()
        self.assertEqual(pcf.GetOutputDataObject(0).GetNumberOfPartitionedDataSets(), 0)
        pdsc = pcf.GetOutputDataObject(0)
        self.assertEqual(pdsc.GetClassName(), "svtkPartitionedDataSetCollection")
        countArray = pdsc.GetFieldData().GetArray("counter")
        info = countArray.GetInformation()
        self.assertEqual(countArray.GetValue(0), 0);
        self.assertEqual(info.Get(dm.svtkDataObject.DATA_TYPE_NAME()), "svtkPartitionedDataSetCollection")

if __name__ == "__main__":
    Testing.main([(TestPartitionedData, 'test')])
