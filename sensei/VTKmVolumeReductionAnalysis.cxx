#include "VTKmVolumeReductionAnalysis.h"

#include "CinemaHelper.h"
#include "DataAdaptor.h"
#include <Timer.h>

#include <vtkCellData.h>
#include <vtkFloatArray.h>
#include <vtkImageData.h>
#ifdef ENABLE_VTK_MPI
#  include <vtkMPICommunicator.h>
#  include <vtkMPIController.h>
#endif
#include <vtkMultiBlockDataSet.h>
#include <vtkNew.h>
#include <vtkObjectFactory.h>
#include <vtkPointData.h>
#include <vtkSmartPointer.h>

#include <algorithm>
#include <vector>

// --- vtkm ---
#include <vtkm/Math.h>
#include <vtkm/cont/ArrayHandle.h>
#include <vtkm/cont/ArrayHandleCounting.h>
#include <vtkm/cont/DataSetBuilderUniform.h>

#include <vtkm/filter/FilterDataSet.h>
#include <vtkm/worklet/DispatcherPointNeighborhood.h>
#include <vtkm/worklet/ScatterPermutation.h>
#include <vtkm/worklet/WorkletPointNeighborhood.h>

#include <vtkm/cont/TryExecute.h>
#include <vtkm/cont/cuda/DeviceAdapterCuda.h>
#include <vtkm/cont/serial/DeviceAdapterSerial.h>
#include <vtkm/cont/tbb/DeviceAdapterTBB.h>

namespace sensei
{
//-----------------------------------------------------------------------------
// VTK-M
//-----------------------------------------------------------------------------

//This is the list of devices to compile in support for. The order of the
//devices determines the runtime preference.
struct DevicesToTry : vtkm::ListTagBase<vtkm::cont::DeviceAdapterTagCuda,
                                        vtkm::cont::DeviceAdapterTagTBB,
                                        vtkm::cont::DeviceAdapterTagSerial>
{
};

struct ImageReductionPolicy : public vtkm::filter::PolicyBase<ImageReductionPolicy>
{
  using DeviceAdapterList = DevicesToTry;
};

struct VoxelMean : public vtkm::worklet::WorkletPointNeighborhood3x3x3
{
  using CountingHandle = vtkm::cont::ArrayHandle<vtkm::Id>;

  VoxelMean(CountingHandle handle): Scatter(handle)
  {
  }

  typedef void ControlSignature(CellSetIn,
                                FieldInNeighborhood<> in,
                                FieldOut<> out);

  typedef void ExecutionSignature(_2, _3);

  using ScatterType = vtkm::worklet::ScatterPermutation<typename CountingHandle::StorageTag>;

  template <typename NeighIn>
  VTKM_EXEC void operator()(const NeighIn& in,
                            vtkm::Float32& out) const
  {
    out = in.Get(0, 0, 0) + in.Get(1, 0, 0)
        + in.Get(0, 1, 0) + in.Get(1, 1, 0)
        + in.Get(0, 0, 1) + in.Get(1, 0, 1)
        + in.Get(0, 1, 1) + in.Get(1, 1, 1);

    out *= 0.125;
  }

  ScatterType GetScatter() const { return this->Scatter; }

private:
  ScatterType Scatter;
};


class ImageReduction : public vtkm::filter::FilterDataSet<ImageReduction>
{
  vtkm::Vec<vtkm::Int32, 3> InputDimensions;
  vtkm::cont::Field::Association FieldAssoc;
  std::string FieldName;

public:

  ImageReduction()
    : InputDimensions(1, 1, 1)
    , FieldAssoc(vtkm::cont::Field::Association::POINTS)
  { }

  void SetInputDimensions(vtkm::Int32 i, vtkm::Int32 j, vtkm::Int32 k)
  {
    this->InputDimensions[0] = i;
    this->InputDimensions[1] = j;
    this->InputDimensions[2] = k;
  }

  void SetField(vtkm::cont::Field::Association assoc, const std::string& name)
  {
    this->FieldName = name;
    this->FieldAssoc = assoc;
  }

  template <typename Policy, typename Device>
  VTKM_CONT vtkm::cont::DataSet DoExecute(const vtkm::cont::DataSet& input,
                                                  vtkm::filter::PolicyBase<Policy> policy,
                                                  Device)

  {
    using DispatcherType = vtkm::worklet::DispatcherPointNeighborhood<VoxelMean, Device>;

    vtkm::cont::ArrayHandle<vtkm::Float32> out;
    vtkm::cont::ArrayHandle<vtkm::Float32> in;

    //get the coordinate system we are using for the 2D area
    const vtkm::cont::DynamicCellSet& cells = input.GetCellSet(this->GetActiveCellSetIndex());

    //get the previous state of the game
    input.GetField(this->FieldName, this->FieldAssoc).GetData().CopyTo(in);

    // Create scatter array
    std::vector<vtkm::Id> ids;
    int kSize = this->InputDimensions[2] - 1;
    int jSize = this->InputDimensions[1] - 1;
    int iSize = this->InputDimensions[0] - 1;
    for (int k = 0; k < kSize; k += 2)
    {
      for (int j = 0; j < jSize; j += 2)
      {
        for (int i = 0; i < iSize; i += 2)
        {
          ids.push_back(i + (j * this->InputDimensions[0]) + (k * this->InputDimensions[0] * this->InputDimensions[1]));
        }
      }
    }

    auto counting = vtkm::cont::make_ArrayHandle(ids);

    //Update the game state
    VoxelMean worklet(counting);
    DispatcherType(worklet, worklet.GetScatter()).Invoke(vtkm::filter::ApplyPolicy(cells, policy), in, out);

    //save the results
    vtkm::cont::DataSet output;
    output.AddCellSet(input.GetCellSet(this->GetActiveCellSetIndex()));
    output.AddCoordinateSystem(input.GetCoordinateSystem(this->GetActiveCoordinateSystemIndex()));

    vtkm::cont::Field outputField(this->FieldName, this->FieldAssoc, out);
    output.AddField(outputField);

    // vtkm::cont::printSummary_ArrayHandle(out, std::cout);

    return output;
  }

  template <typename T, typename StorageType, typename DerivedPolicy, typename DeviceAdapter>
  VTKM_CONT
  bool DoMapField(
    vtkm::cont::DataSet& result,
    const vtkm::cont::ArrayHandle<T, StorageType>& input,
    const vtkm::filter::FieldMetadata& fieldMeta,
    const vtkm::filter::PolicyBase<DerivedPolicy>&,
    DeviceAdapter)
  {
    result.AddField(fieldMeta.AsField(input));
    return true;
  }

};


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
senseiNewMacro(VTKmVolumeReductionAnalysis);

//-----------------------------------------------------------------------------
VTKmVolumeReductionAnalysis::VTKmVolumeReductionAnalysis() : Communicator(MPI_COMM_WORLD), Helper(NULL)
{
}

//-----------------------------------------------------------------------------
VTKmVolumeReductionAnalysis::~VTKmVolumeReductionAnalysis()
{
    delete this->Helper;
}

//-----------------------------------------------------------------------------
void VTKmVolumeReductionAnalysis::Initialize(
  const std::string& meshName,
  const std::string& fieldName,
  const std::string& fieldAssoc,
  const std::string& workingDirectory,
  int reductionFactor,
  MPI_Comm comm
  )
{
  this->MeshName = meshName;
  this->FieldName = fieldName;
  this->FieldAssoc = fieldAssoc == "cell" ?
    vtkm::cont::Field::Association::CELL_SET :
    vtkm::cont::Field::Association::POINTS;
  this->Communicator = comm;
  this->Reduction = reductionFactor;

#ifdef ENABLE_VTK_MPI
  vtkNew<vtkMPIController> con;
  con->Initialize(0, 0, 1); // initialized externally
  vtkMultiProcessController::SetGlobalController(con.GetPointer());
  con->Register(NULL); // Keep ref
#endif

  this->Helper = new CinemaHelper();
  this->Helper->SetWorkingDirectory(workingDirectory);
  this->Helper->SetExportType("vtk-volume");
}

//-----------------------------------------------------------------------------
bool VTKmVolumeReductionAnalysis::Execute(DataAdaptor* data)
{
  Timer::MarkEvent mark("VTKmVolumeReductionAnalysis::execute");
  this->Helper->AddTimeEntry();

  vtkDataObject* mesh = nullptr;
  if (data->GetMesh(this->MeshName, /*structure_only*/true, mesh) || !mesh)
  {
    return false;
  }
  data->AddArray(
    mesh, this->MeshName,
    this->FieldAssoc == vtkm::cont::Field::Association::POINTS ?
    vtkDataObject::POINT : vtkDataObject::CELL,
    this->FieldName);

  vtkMultiBlockDataSet* blocks = vtkMultiBlockDataSet::SafeDownCast(mesh);
  vtkImageData* originalImageData = nullptr;
  for(unsigned int i = 0; i < blocks->GetNumberOfBlocks(); i++)
    {
    vtkImageData* block = vtkImageData::SafeDownCast(blocks->GetBlock(i));
    originalImageData = block ? block : originalImageData;
    }
  if (originalImageData == nullptr) {
    return true;
  }

  vtkNew<vtkImageData> outputDataSet;
  if (this->Reduction > 0)
    {
    Timer::MarkStartEvent("VTKm reduction");

    int dimensions[3];
    originalImageData->GetDimensions(dimensions);
    dimensions[0]--; // cell data to point data
    dimensions[1]--; // cell data to point data
    dimensions[2]--; // cell data to point data

    vtkNew<vtkFloatArray> dataArray;
    dataArray->ShallowCopy(vtkFloatArray::SafeDownCast(originalImageData->GetCellData()->GetScalars()));
    vtkm::cont::ArrayHandle<vtkm::Float32> vtkmArray;
    for (int step = 0 ; step < this->Reduction; step++)
      {
      // Data preparation
      float* inFloat = dataArray->GetPointer(0);
      int size = dimensions[0] * dimensions[1] * dimensions[2];

      // --- vtk-m filtering ---
      // - create vtkm dataset
      vtkm::cont::ArrayHandle<vtkm::Float32> handle = vtkm::cont::make_ArrayHandle(inFloat, size);
      vtkm::cont::DataSetBuilderUniform builder;
      vtkm::cont::DataSet dataset = builder.Create(vtkm::Id3(dimensions[0], dimensions[1], dimensions[2]));
      vtkm::cont::Field scalarField(this->FieldName, this->FieldAssoc, handle);
      dataset.AddField(scalarField);

      // - create and execute filter
      ImageReduction filter;
      filter.SetInputDimensions(dimensions[0], dimensions[1], dimensions[2]);
      filter.SetField(this->FieldAssoc, this->FieldName);
      auto rdata = filter.Execute(dataset, ImageReductionPolicy());

      // - recover data from vtkm
      rdata.GetField(this->FieldName, this->FieldAssoc).GetData().CopyTo(vtkmArray);
      float* t = vtkmArray.GetStorage().GetArray();

      // - rebuild regular vtkImageData
      int newDims[3] = { int(dimensions[0] / 2), int(dimensions[1] / 2), int(dimensions[2] / 2) };
      int reducedSize = newDims[0] * newDims[1] * newDims[2];

      // Update dimensions and array
      dimensions[0] = newDims[0];
      dimensions[1] = newDims[1];
      dimensions[2] = newDims[2];
      dataArray->SetArray(t, reducedSize, 1);
      }
    outputDataSet->SetDimensions(dimensions[0], dimensions[1], dimensions[2]);
    outputDataSet->GetPointData()->SetScalars(dataArray);
    Timer::MarkEndEvent("VTKm reduction");
    }
  else
    {
    int dim[3];
    originalImageData->GetDimensions(dim);
    dim[0]--; // cell data to point data
    dim[1]--; // cell data to point data
    dim[2]--; // cell data to point data
    outputDataSet->SetDimensions(dim[0], dim[1], dim[2]);
    outputDataSet->GetPointData()->SetScalars(originalImageData->GetCellData()->GetScalars());
    }

  // -----------------------

  Timer::MarkStartEvent("Cinema Volume export");
  this->Helper->WriteVolume(outputDataSet.GetPointer());
  this->Helper->WriteMetadata();
  Timer::MarkEndEvent("Cinema Volume export");

  return true;
}

}
