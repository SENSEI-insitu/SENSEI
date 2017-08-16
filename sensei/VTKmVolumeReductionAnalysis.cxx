#include "VTKmVolumeReductionAnalysis.h"
#include "DataAdaptor.h"
#include "CinemaHelper.h"
#include <Timer.h>

#include <vtkCellData.h>
#include <vtkFloatArray.h>
#include <vtkImageData.h>
#include <vtkMPI.h>
#include <vtkMPICommunicator.h>
#include <vtkMPIController.h>
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

public:

  ImageReduction() : InputDimensions(1, 1, 1)
  { }

  void SetInputDimensions(vtkm::Int32 i, vtkm::Int32 j, vtkm::Int32 k)
  {
    this->InputDimensions[0] = i;
    this->InputDimensions[1] = j;
    this->InputDimensions[2] = k;
  }

  template <typename Policy, typename Device>
  VTKM_CONT vtkm::filter::ResultDataSet DoExecute(const vtkm::cont::DataSet& input,
                                                  vtkm::filter::PolicyBase<Policy> policy,
                                                  Device)

  {
    using DispatcherType = vtkm::worklet::DispatcherPointNeighborhood<VoxelMean, Device>;

    vtkm::cont::ArrayHandle<vtkm::Float32> out;
    vtkm::cont::ArrayHandle<vtkm::Float32> in;

    //get the coordinate system we are using for the 2D area
    const vtkm::cont::DynamicCellSet& cells = input.GetCellSet(this->GetActiveCellSetIndex());

    //get the previous state of the game
    input.GetField("scalar", vtkm::cont::Field::ASSOC_POINTS).GetData().CopyTo(in);

    // Create scatter array
    std::vector<vtkm::Id> ids;
    for (int k = 0; k < this->InputDimensions[2]; k += 2)
    {
      for (int j = 0; j < this->InputDimensions[1]; j += 2)
      {
        for (int i = 0; i < this->InputDimensions[0]; i += 2)
        {
          ids.push_back(i + (j * this->InputDimensions[0]) + (k * this->InputDimensions[0] * this->InputDimensions[1]));
        }
      }
    }

    auto counting = vtkm::cont::make_ArrayHandle(ids);

    //Update the game state
    VoxelMean worklet(counting);
    DispatcherType(worklet).Invoke(vtkm::filter::ApplyPolicy(cells, policy), in, out);

    //save the results
    vtkm::cont::DataSet output;
    output.AddCellSet(input.GetCellSet(this->GetActiveCellSetIndex()));
    output.AddCoordinateSystem(input.GetCoordinateSystem(this->GetActiveCoordinateSystemIndex()));

    vtkm::cont::Field outputField("scalar", vtkm::cont::Field::ASSOC_POINTS, out);
    output.AddField(outputField);

    // vtkm::cont::printSummary_ArrayHandle(out, std::cout);

    return vtkm::filter::ResultDataSet(output);
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
  MPI_Comm comm, const std::string& workingDirectory, int reductionFactor)
{
  this->Communicator = comm;

  vtkNew<vtkMPICommunicator> vtkComm;
  vtkMPICommunicatorOpaqueComm h(&this->Communicator);
  vtkComm->InitializeExternal(&h);

  vtkNew<vtkMPIController> con;
  con->SetCommunicator(vtkComm.GetPointer());
  vtkMultiProcessController::SetGlobalController(con.GetPointer());
  con->Register(NULL); // Keep ref

  this->Helper = new CinemaHelper();
  this->Helper->SetWorkingDirectory(workingDirectory);
  this->Helper->SetExportType("vtk-volume");
}

//-----------------------------------------------------------------------------
bool VTKmVolumeReductionAnalysis::Execute(DataAdaptor* data)
{
  timer::MarkEvent mark("VTKmVolumeReductionAnalysis::execute");
  vtkMultiProcessController* controller = vtkMultiProcessController::GetGlobalController();

  this->Helper->AddTimeEntry();

  vtkDataObject* mesh = data->GetMesh(/*structure_only*/true);
  bool dataError = !data->AddArray(mesh, vtkDataObject::FIELD_ASSOCIATION_CELLS, "data");

  vtkMultiBlockDataSet* blocks = vtkMultiBlockDataSet::SafeDownCast(mesh);
  vtkImageData* originalImageData = nullptr;
  for(int i = 0; i < blocks->GetNumberOfBlocks(); i++)
    {
    vtkImageData* block = vtkImageData::SafeDownCast(blocks->GetBlock(i));
    originalImageData = block ? block : originalImageData;
    }
  if (originalImageData == nullptr) {
    return true;
  }

  timer::MarkStartEvent("VTKm reduction");
  // Data preparation
  vtkFloatArray* dataArray = vtkFloatArray::SafeDownCast(originalImageData->GetCellData()->GetScalars());
  float* inFloat = dataArray->GetPointer(0);
  int* ijkSize = originalImageData->GetDimensions();
  ijkSize[0]--; // cell data to point data
  ijkSize[1]--; // cell data to point data
  ijkSize[2]--; // cell data to point data
  int size = ijkSize[0] * ijkSize[1] * ijkSize[2];

  // --- vtk-m filtering ---
  // - create vtkm dataset
  vtkm::cont::ArrayHandle<vtkm::Float32> handle = vtkm::cont::make_ArrayHandle(inFloat, size);
  vtkm::cont::DataSetBuilderUniform builder;
  vtkm::cont::DataSet dataset = builder.Create(vtkm::Id3(ijkSize[0], ijkSize[1], ijkSize[2]));
  vtkm::cont::Field scalarField("scalar", vtkm::cont::Field::ASSOC_POINTS, handle);
  dataset.AddField(scalarField);

  // - create and execute filter
  ImageReduction filter;
  filter.SetInputDimensions(ijkSize[0], ijkSize[1], ijkSize[2]);
  vtkm::filter::ResultDataSet rdata = filter.Execute(dataset, ImageReductionPolicy());

  // - recover data from vtkm
  vtkm::cont::ArrayHandle<vtkm::Float32> tmp;
  rdata.GetDataSet().GetField("scalar", vtkm::cont::Field::ASSOC_POINTS).GetData().CopyTo(tmp);
  float* t = tmp.GetStorage().GetArray();

  // - rebuild regular vtkImageData
  int newDims[3] = { int((ijkSize[0] + 1) / 2), int((ijkSize[1] + 1) / 2), int((ijkSize[2] + 1) / 2) };
  int reducedSize = newDims[0] * newDims[1] * newDims[2];
  // std::cout << "New dimensions: "<< newDims[0] << ", " << newDims[1] << ", " << newDims[2] << std::endl;

  vtkNew<vtkFloatArray> scalars;
  scalars->SetName("scalars");
  scalars->SetArray(t, reducedSize, 1);

  vtkNew<vtkImageData> outputDataSet;
  outputDataSet->SetDimensions(newDims[0], newDims[1], newDims[2]);
  outputDataSet->GetPointData()->SetScalars(scalars);
  timer::MarkEndEvent("VTKm reduction");

  // -----------------------

  timer::MarkStartEvent("Cinema Volume export");
  this->Helper->WriteVolume(outputDataSet.GetPointer());
  this->Helper->WriteMetadata();
  timer::MarkEndEvent("Cinema Volume export");

  return true;
}

}
