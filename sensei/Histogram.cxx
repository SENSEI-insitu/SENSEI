#include "Histogram.h"
#include "DataAdaptor.h"
#include "MeshMetadata.h"
#include "MeshMetadataMap.h"
#include "Timer.h"
#include "VTKHistogram.h"
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

namespace sensei
{

//-----------------------------------------------------------------------------
senseiNewMacro(Histogram);

//-----------------------------------------------------------------------------
Histogram::Histogram() : Bins(0),
  Association(vtkDataObject::FIELD_ASSOCIATION_POINTS), Internals(nullptr)
{
}

//-----------------------------------------------------------------------------
Histogram::~Histogram()
{
  delete this->Internals;
}

//-----------------------------------------------------------------------------
void Histogram::Initialize(int bins, const std::string &meshName,
  int association, const std::string& arrayName, const std::string &fileName)
{
  this->Bins = bins;
  this->MeshName = meshName;
  this->ArrayName = arrayName;
  this->Association = association;
  this->FileName = fileName;
}

//-----------------------------------------------------------------------------
const char *Histogram::GetGhostArrayName()
{
// TODO -- fix the version logic below, what rev was
// vtkDataSetAttributes::GhostArrayName introduced in?
#if VTK_MAJOR_VERSION == 6 && VTK_MINOR_VERSION == 1
    return "vtkGhostType";
#else
    return vtkDataSetAttributes::GhostArrayName();
#endif
}

//-----------------------------------------------------------------------------
bool Histogram::Execute(DataAdaptor* data)
{
  Timer::MarkEvent mark("Histogram::Execute");

  // see what the simulation is providing
  MeshMetadataMap mdMap;
  if (mdMap.Initialize(data))
    {
    SENSEI_ERROR("Failed to get metadata")
    return false;
    }

  delete this->Internals;
  this->Internals = new VTKHistogram;

  // get the current time and step
  int step = data->GetDataTimeStep();
  double time = data->GetDataTime();

  // get the mesh metadata object
  MeshMetadataPtr mmd;
  if (mdMap.GetMeshMetadata(this->MeshName, mmd))
    {
    SENSEI_ERROR("Failed to get metadata for mesh \"" << this->MeshName << "\"")
    return false;
    }

  // get the mesh object
  vtkDataObject* mesh = nullptr;
  if (data->GetMesh(this->MeshName, true, mesh))
    {
    SENSEI_ERROR("Failed to get mesh \"" << this->MeshName << "\"")
    return false;
    }

  if (!mesh)
    {
    // it is not an necessarilly an error if all ranks do not have
    // a dataset to process
    this->Internals->PreCompute(this->GetCommunicator(), this->Bins);

    this->Internals->PostCompute(this->GetCommunicator(), this->Bins,
      step, time, this->MeshName, this->ArrayName, this->FileName);

    return true;
    }

  // add the array
  if (data->AddArray(mesh, this->MeshName, this->Association, this->ArrayName))
    {
    // it is an error if we try to compute a histogram over a non
    // existant array
    SENSEI_ERROR(<< data->GetClassName() << " failed to add "
      << (this->Association == vtkDataObject::POINT ? "point" : "cell")
      << " data array \""  << this->ArrayName << "\"")

    this->Internals->PreCompute(this->GetCommunicator(), this->Bins);

    this->Internals->PostCompute(this->GetCommunicator(), this->Bins,
      step, time, this->MeshName, this->ArrayName, this->FileName);

    return false;
    }

  // add the ghost zones
  if (mmd->NumGhostCells && data->AddGhostCellsArray(mesh, this->MeshName))
    {
    SENSEI_ERROR(<< data->GetClassName() << " failed to add ghost cells.")
    return false;
    }

  if (mmd->NumGhostNodes && data->AddGhostNodesArray(mesh, this->MeshName))
    {
    SENSEI_ERROR(<< data->GetClassName() << " failed to add ghost nodes.")
    return false;
    }

  if (vtkCompositeDataSet* cd = dynamic_cast<vtkCompositeDataSet*>(mesh))
    {
    vtkSmartPointer<vtkCompositeDataIterator> iter;
    iter.TakeReference(cd->NewIterator());

    for (iter->InitTraversal(); !iter->IsDoneWithTraversal(); iter->GoToNextItem())
      {
      // get the local mesh
      vtkDataObject *curObj = iter->GetCurrentDataObject();

      // get the array to compute histogram for
      vtkDataArray* array = this->GetArray(curObj, this->ArrayName);
      if (!array)
        {
        SENSEI_WARNING("Dataset " << iter->GetCurrentFlatIndex()
          << " has no array named \"" << this->ArrayName << "\"")
        continue;
        }

      // and get the ghost cell array
      vtkUnsignedCharArray *ghostArray = dynamic_cast<vtkUnsignedCharArray*>(
        this->GetArray(curObj, this->GetGhostArrayName()));

      // compute local histogram range
      this->Internals->AddRange(array, ghostArray);
      }

    // compute global histogram range
    this->Internals->PreCompute(this->GetCommunicator(), this->Bins);

    for (iter->InitTraversal(); !iter->IsDoneWithTraversal(); iter->GoToNextItem())
      {
      // get the local mesh
      vtkDataObject *curObj = iter->GetCurrentDataObject();
      // get the array to compute histogram for
      vtkDataArray* array = this->GetArray(curObj, this->ArrayName);
      if (!array)
        {
        SENSEI_WARNING("Dataset " << iter->GetCurrentFlatIndex()
          << " has no array named \"" << this->ArrayName << "\"")
        continue;
        }

      // and get the ghost cell array
      vtkUnsignedCharArray *ghostArray = dynamic_cast<vtkUnsignedCharArray*>(
        this->GetArray(curObj, this->GetGhostArrayName()));

      // compute local histogram
      this->Internals->Compute(array, ghostArray);
      }

    // compute the global histogram
    this->Internals->PostCompute(this->GetCommunicator(), this->Bins,
      step, time, this->MeshName, this->ArrayName, this->FileName);
    }
  else
    {
    vtkDataArray* array = this->GetArray(mesh, this->ArrayName);
    if (!array)
      {
      int rank = 0;
      MPI_Comm_rank(this->GetCommunicator(), &rank);

      SENSEI_WARNING("Dataset " << rank << " has no array named \""
        << this->ArrayName << "\"")

      this->Internals->PreCompute(this->GetCommunicator(), this->Bins);

      this->Internals->PostCompute(this->GetCommunicator(), this->Bins,
        step, time, this->MeshName, this->ArrayName, this->FileName);
      }
    else
      {
      vtkUnsignedCharArray *ghostArray = dynamic_cast<vtkUnsignedCharArray*>(
        this->GetArray(mesh, this->GetGhostArrayName()));

      this->Internals->AddRange(array, ghostArray);
      this->Internals->PreCompute(this->GetCommunicator(), this->Bins);
      this->Internals->Compute(array, ghostArray);

      this->Internals->PostCompute(this->GetCommunicator(), this->Bins,
        step, time, this->MeshName, this->ArrayName, this->FileName);
      }
    }

  mesh->Delete();
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
int Histogram::GetHistogram(double &min, double &max,
  std::vector<unsigned int> &bins)
{
  if (!this->Internals)
    return -1;

  return this->Internals->GetHistogram(this->GetCommunicator(), min, max, bins);
}

//-----------------------------------------------------------------------------
int Histogram::Finalize()
{
  delete this->Internals;
  this->Internals = nullptr;
  return 0;
}

}
