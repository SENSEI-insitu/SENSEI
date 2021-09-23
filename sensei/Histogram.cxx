#include "Histogram.h"
#include "DataAdaptor.h"
#include "MeshMetadata.h"
#include "MeshMetadataMap.h"
#include "Profiler.h"
#include "HistogramInternals.h"
#include "VTKUtils.h"
#include "Error.h"

#include <vtkCompositeDataIterator.h>
#include <vtkCompositeDataSet.h>
#include <vtkDataObject.h>
#include <vtkDataSetAttributes.h>
#include <vtkObjectFactory.h>
#include <vtkSmartPointer.h>
#include <vtkUnsignedCharArray.h>

#include <algorithm>
#include <vector>

namespace
{
// **************************************************************************
int Write(const std::string &fileName, int step, double time,
  const std::string &meshName, const std::string &arrayName,
  sensei::Histogram::Data &result)
{
  // write the histogram to a file
  char fname[1024] = {'\0'};

  snprintf(fname, 1024, "%s_%s_%s_%d.txt", fileName.c_str(),
    meshName.c_str(), arrayName.c_str(), step);

  FILE *file = fopen(fname, "w");
  if (!file)
    {
    char *estr = strerror(errno);
    SENSEI_ERROR("Failed to open \"" << fname << "\"" << std::endl << estr)
    return -1;
    }

  fprintf(file, "step : %d\n", step);
  fprintf(file, "time : %0.6g\n", time);
  fprintf(file, "num bins : %d\n", result.NumberOfBins);
  fprintf(file, "range : %0.6g %0.6g\n", result.BinMin, result.BinMax);
  fprintf(file, "bin edges : ");
  for (int i = 0; i < result.NumberOfBins + 1; ++i)
    fprintf(file, "%0.6g ", result.BinMin + i*result.BinWidth);
  fprintf(file, "\n");
  fprintf(file, "counts : ");
  for (int i = 0; i < result.NumberOfBins; ++i)
    fprintf(file, "%d ", result.Histogram[i]);
  fprintf(file, "\n");
  fclose(file);

  return 0;
}

// **************************************************************************
int Write(int step, double time, const std::string &meshName,
  const std::string &arrayName, sensei::Histogram::Data &result)
{
  // write the histogram to std::cout
  int origPrec = cout.precision();
  std::cout.precision(4);

  std::cout << "Histogram mesh \"" << meshName << "\" data array \""
    << arrayName << "\" step " << step << " time " << time << std::endl;

  for (int i = 0; i < result.NumberOfBins; ++i)
    {
    const int wid = 15;
    std::cout << std::scientific << std::setw(wid) << std::right << result.BinMin + i*result.BinWidth
      << " - " << std::setw(wid) << std::left << result.BinMin + (i+1)*result.BinWidth
      << ": " << std::fixed << result.Histogram[i] << std::endl;
    }

  std::cout.precision(origPrec);

  return 0;
}
}

namespace sensei
{
//-----------------------------------------------------------------------------
senseiNewMacro(Histogram);

//-----------------------------------------------------------------------------
Histogram::Histogram() : NumberOfBins(0),
  Association(vtkDataObject::FIELD_ASSOCIATION_POINTS)
{
}

//-----------------------------------------------------------------------------
Histogram::~Histogram()
{
}

//-----------------------------------------------------------------------------
void Histogram::Initialize(int bins, const std::string &meshName,
  int association, const std::string& arrayName, const std::string &fileName)
{
  this->NumberOfBins = bins;
  this->MeshName = meshName;
  this->ArrayName = arrayName;
  this->Association = association;
  this->FileName = fileName;
}

//-----------------------------------------------------------------------------
const char *Histogram::GetGhostArrayName()
{
#if VTK_MAJOR_VERSION == 6 && VTK_MINOR_VERSION == 1
    return "vtkGhostType";
#else
    return vtkDataSetAttributes::GhostArrayName();
#endif
}

//-----------------------------------------------------------------------------
bool Histogram::Execute(DataAdaptor* data)
{
  TimeEvent<128> mark("Histogram::Execute");

  // see what the simulation is providing
  MeshMetadataMap mdMap;
  if (mdMap.Initialize(data))
    {
    SENSEI_ERROR("Failed to get metadata")
    return false;
    }

  // get the mesh metadata object
  MeshMetadataPtr mmd;
  if (mdMap.GetMeshMetadata(this->MeshName, mmd))
    {
    SENSEI_ERROR("Failed to get metadata for mesh \"" << this->MeshName << "\"")
    return false;
    }

  // get the mesh object
  vtkDataObject *dobj = nullptr;
  if (data->GetMesh(this->MeshName, true, dobj))
    {
    SENSEI_ERROR("Failed to get mesh \"" << this->MeshName << "\"")
    return false;
    }

  int rank = 0;
  MPI_Comm comm = this->GetCommunicator();
  MPI_Comm_rank(comm, &rank);

  // TODO : this lets one laod balance across multiple GPU's and CPU's
  // set -1 to execute on the CPU and 0 to N_CUDA_DEVICES -1 to specify
  // the specific GPU to run on.
#if defined(ENABLE_CUDA)
  const char *aDevId = getenv("HISTOGRAM_DEVICE_ID");
  aDevId = aDevId ? aDevId : "0";
  int deviceId = atoi(aDevId);
#else
  // run on the CPU
  int deviceId = -1;
  const char *aDevId = "";
#endif

  // get the current time and step
  int step = data->GetDataTimeStep();
  double time = data->GetDataTime();

  if (rank == 0)
    {
    SENSEI_STATUS("Step = " << step << " Time = " << time
      << " Computing the histogram on mesh \""
      << this->MeshName << "\" array \"" << this->ArrayName
      << "\" using " << (deviceId < 0 ? "the CPU" : "CUDA GPU ")
      << aDevId)
    }


  // create a new histogram computation. this class does all the work.
  std::shared_ptr<sensei::HistogramInternals>
    internals(new sensei::HistogramInternals(comm, deviceId, this->NumberOfBins));

  if (!dobj)
    {
    // it is not an necessarilly an error if all ranks do not have
    // a dataset to process. However, all ranks must participate due
    // to the use of MPI collectives.
    internals->Initialize();
    internals->ComputeHistogram();
    internals->Clear();
    return true;
    }

  // fetch the array that the hiostogram will be computed on
  if (data->AddArray(dobj, this->MeshName, this->Association, this->ArrayName))
    {
    SENSEI_ERROR(<< data->GetClassName() << " failed to add "
      << (this->Association == vtkDataObject::POINT ? "point" : "cell")
      << " data array \""  << this->ArrayName << "\"")

    // abort to avoid deadlocks in collective calls
    MPI_Abort(comm, -1);
    return false;
    }

  // add the ghost zones
  if ((mmd->NumGhostCells || VTKUtils::AMR(mmd)) &&
    data->AddGhostCellsArray(dobj, this->MeshName))
    {
    SENSEI_ERROR(<< data->GetClassName() << " failed to add ghost cells.")
    // abort to avoid deadlocks in collective calls
    MPI_Abort(comm, -1);
    return false;
    }

  if (mmd->NumGhostNodes && data->AddGhostNodesArray(dobj, this->MeshName))
    {
    SENSEI_ERROR(<< data->GetClassName() << " failed to add ghost nodes.")
    // abort to avoid deadlocks in collective calls
    MPI_Abort(comm, -1);
    return false;
    }

  // add all blocks of data
  vtkCompositeDataSetPtr mesh = VTKUtils::AsCompositeData(comm, dobj, true);
  vtkSmartPointer<vtkCompositeDataIterator> iter;
  iter.TakeReference(mesh->NewIterator());
  for (iter->InitTraversal(); !iter->IsDoneWithTraversal(); iter->GoToNextItem())
    {
    // get the local mesh
    vtkDataObject *curObj = iter->GetCurrentDataObject();

    // get the array to compute histogram for
    vtkDataArray* array = this->GetArray(curObj, this->ArrayName);
    if (!array)
      {
      SENSEI_WARNING("Data block " << iter->GetCurrentFlatIndex()
        << " of mesh \"" << this->MeshName << " has no array named \""
        << this->ArrayName << "\"")
      continue;
      }

    // and get the ghost cell array
    vtkUnsignedCharArray *ghostArray = dynamic_cast<vtkUnsignedCharArray*>(
      this->GetArray(curObj, this->GetGhostArrayName()));

    // add this blocks contribution to the calculation
    if (internals->AddLocalData(array, ghostArray))
      {
      SENSEI_ERROR("Failed to add array \"" << this->ArrayName
        << "\" data block " << iter->GetCurrentFlatIndex() << " of mesh \""
        << this->MeshName << "\"")
      // abort to prevent deadlock in collective calls
      MPI_Abort(comm, -1);
      }
    }

  // compute the histogram. this is an MPI collective, all MPI ranks must participate.
  // after this call returns MPI rank 0 holds the histogram
  if (internals->ComputeHistogram())
    {
    SENSEI_ERROR("Failed to compute the histogram for array \""
      << this->ArrayName << "\" of mesh \"" << this->MeshName << "\"")
    // abort to prevent deadlock in collective calls
    MPI_Abort(comm, -1);
    }

  // store a copy of the histogram. this can be acccessed from scripts ofr
  // regression testing etc.
  Histogram::Data result;

  internals->GetHistogram(result.NumberOfBins, result.BinMin,
    result.BinMax, result.BinWidth, result.Histogram);

  this->LastResult = result;

  // write the results if on MPI rank 0
  if (rank == 0)
    {
    if (this->FileName.empty())
      {
      ::Write(step, time, this->MeshName, this->ArrayName, result);
      }
    else
      {
      if (::Write(this->FileName, step, time, this->MeshName, this->ArrayName, result))
        {
        SENSEI_ERROR("Failed to write histogram.")
        return false;
        }
      }
    }

  internals->Clear();

  return true;
}

//-----------------------------------------------------------------------------
vtkDataArray* Histogram::GetArray(vtkDataObject* dobj, const std::string& arrayname)
{
  if (vtkFieldData* fd = dobj->GetAttributesAsFieldData(this->Association))
    {
    return fd->GetArray(arrayname.c_str());
    }
  return nullptr;
}

//-----------------------------------------------------------------------------
int Histogram::GetHistogram(Histogram::Data &result)
{
  result = this->LastResult;
  return 0;
}

//-----------------------------------------------------------------------------
int Histogram::Finalize()
{
  return 0;
}

}
