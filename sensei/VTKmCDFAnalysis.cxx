#include "VTKmCDFAnalysis.h"

#include "CDFReducer.h"
#include "CinemaHelper.h"
#include "DataAdaptor.h"
#include <Timer.h>
#include <Error.h>

#include <vtkDataSet.h>
#include <vtkCellData.h>
#include <vtkIntArray.h>
#include <vtkSortDataArray.h>
#include <vtkDoubleArray.h>
#include <vtkImageData.h>
#ifdef ENABLE_VTK_MPI
#  include <vtkMPICommunicator.h>
#  include <vtkMPIController.h>
#endif
#include <vtkMultiProcessController.h>
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
senseiNewMacro(VTKmCDFAnalysis);

//-----------------------------------------------------------------------------
VTKmCDFAnalysis::VTKmCDFAnalysis()
  : Communicator(MPI_COMM_WORLD)
  , Helper(nullptr)
  , NumberOfQuantiles(10)
  , RequestSize(10)
{
}

//-----------------------------------------------------------------------------
VTKmCDFAnalysis::~VTKmCDFAnalysis()
{
    delete this->Helper;
}

//-----------------------------------------------------------------------------
void VTKmCDFAnalysis::Initialize(
  const std::string& meshName,
  const std::string& fieldName,
  const std::string& fieldAssoc,
  const std::string& workingDirectory,
  int numberOfQuantiles,
  int requestSize,
  MPI_Comm comm
  )
{
  this->MeshName = meshName;
  this->FieldName = fieldName;
  this->FieldAssoc = fieldAssoc == "cell" ?
    vtkm::cont::Field::Association::CELL_SET :
    vtkm::cont::Field::Association::POINTS;
  this->Communicator = comm;
  this->NumberOfQuantiles = numberOfQuantiles;
  this->RequestSize = requestSize;

#ifdef ENABLE_VTK_MPI
  vtkNew<vtkMPIController> con;
  con->Initialize(0, 0, 1); // initialized externally
  vtkMultiProcessController::SetGlobalController(con.GetPointer());
  con->Register(NULL); // Keep ref
#endif

  this->Helper = new CinemaHelper();
  this->Helper->SetWorkingDirectory(workingDirectory);
  this->Helper->SetExportType("cdf");
  this->Helper->SetSampleSize(this->NumberOfQuantiles);
}

//-----------------------------------------------------------------------------
bool VTKmCDFAnalysis::Execute(DataAdaptor* data)
{
  Timer::MarkEvent mark("VTKmCDFAnalysis::execute");
  this->Helper->AddTimeEntry();

  // Get the mesh from the simulation:
  vtkDataObject* mesh = nullptr;
  if (data->GetMesh(this->MeshName, /*structure_only*/true, mesh) || !mesh)
  {
    return false;
  }
  // Tell the simulation to add the array we want:
  data->AddArray(
    mesh, this->MeshName,
    this->FieldAssoc == vtkm::cont::Field::Association::POINTS ?
    vtkDataObject::POINT : vtkDataObject::CELL,
    this->FieldName);

  // Now ask the mesh for the array:
  vtkDataArray* array = nullptr;

  if (mesh && this->FieldAssoc == vtkm::cont::Field::Association::WHOLE_MESH)
  {
    array = mesh->GetFieldData()->GetArray(this->FieldName.c_str());
  }

  auto dataset = vtkDataSet::SafeDownCast(mesh);
  if (!array && dataset)
  {
    if (this->FieldAssoc == vtkm::cont::Field::Association::POINTS)
    {
      array = dataset->GetPointData()->GetArray(this->FieldName.c_str());
    }
    else if (this->FieldAssoc == vtkm::cont::Field::Association::CELL_SET)
    {
      array = dataset->GetCellData()->GetArray(this->FieldName.c_str());
    }
  }

  if (!array)
  {
    vtkMultiBlockDataSet* blocks = vtkMultiBlockDataSet::SafeDownCast(mesh);
    for(unsigned int i = 0; !array && i < blocks->GetNumberOfBlocks(); i++)
    {
      auto dobj = blocks->GetBlock(i);
      if (dobj && this->FieldAssoc == vtkm::cont::Field::Association::WHOLE_MESH)
      {
        array = dobj->GetFieldData()->GetArray(this->FieldName.c_str());
        if (array)
        {
          break;
        }
      }
      auto dataset = vtkDataSet::SafeDownCast(dobj);
      if (dataset && this->FieldAssoc == vtkm::cont::Field::Association::POINTS)
      {
        array = dataset->GetPointData()->GetArray(this->FieldName.c_str());
      }
      else if (dataset && this->FieldAssoc == vtkm::cont::Field::Association::CELL_SET)
      {
        array = dataset->GetCellData()->GetArray(this->FieldName.c_str());
      }
    }
  }

  if (!array)
  {
    SENSEI_ERROR("Could not obtain array \"" << this->FieldName << "\" from data adaptor.");
    return false;
  }

  if (array->GetNumberOfComponents() > 1)
  {
    SENSEI_ERROR("Cannot compute CDF of multi-component (vector, non-scalar)  array.");
    return false;
  }

  if (this->NumberOfQuantiles <= 0)
  {
    SENSEI_ERROR("Invalid CDF request (bad number of quantiles).");
    return false;
  }

  Timer::MarkStartEvent("VTKm CDF");
  vtkNew<vtkDoubleArray> sorted;
  sorted->DeepCopy(array);
  vtkSortDataArray::SortArrayByComponent(sorted, 0);

#ifdef ENABLE_VTK_MPI
  vtkNew<vtkMPIController> controller;
#else
  vtkNew<vtkMultiProcessController> controller;
#endif
  CDFReducer reducer(controller);
  reducer.SetBufferSize(this->RequestSize);

  double* cdf = reducer.Compute(sorted->GetPointer(0), sorted->GetNumberOfTuples(), this->NumberOfQuantiles);
  Timer::MarkEndEvent("VTKm CDF");

  Timer::MarkStartEvent("Cinema CDF export");
  this->Helper->WriteCDF(this->NumberOfQuantiles, cdf);
  this->Helper->WriteMetadata();
  Timer::MarkEndEvent("Cinema CDF export");

  return true;
}

}
