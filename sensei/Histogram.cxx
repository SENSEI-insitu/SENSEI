#include "Histogram.h"
#include "DataAdaptor.h"
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
Histogram::Histogram() : Communicator(MPI_COMM_WORLD), Bins(0),
  Association(vtkDataObject::FIELD_ASSOCIATION_POINTS)
{
}

//-----------------------------------------------------------------------------
Histogram::~Histogram()
{
}

//-----------------------------------------------------------------------------
void Histogram::Initialize(
  MPI_Comm comm, int bins, int association, const std::string& arrayname)
{
  this->Communicator = comm;
  this->Bins = bins;
  this->ArrayName = arrayname;
  this->Association = association;
}

const char *
Histogram::GetGhostArrayName()
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
  timer::MarkEvent mark("histogram::execute");

  VTKHistogram histogram;
  vtkDataObject* mesh = data->GetMesh(/*structure_only*/true);
  if (!mesh)
    {
    // it is not an necessarilly an error if all ranks do not have
    // a dataset to process
    histogram.PreCompute(this->Communicator, this->Bins);
    histogram.PostCompute(this->Communicator, this->Bins, this->ArrayName);
    return true;
    }

  if (!data->AddArray(mesh, this->Association, this->ArrayName.c_str()))
    {
    // it is an error if we try to compute a histogram over a non
    // existant array
    SENSEI_ERROR(<< data->GetClassName() << " faild to add "
      << (this->Association == vtkDataObject::POINT ? "point" : "cell")
      << " data array \""  << this->ArrayName << "\"")

    histogram.PreCompute(this->Communicator, this->Bins);
    histogram.PostCompute(this->Communicator, this->Bins, this->ArrayName);
    return false;
    }

//  mesh->Print(cerr);

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
      histogram.AddRange(array, ghostArray);
      }

    // compute global histogram range
    histogram.PreCompute(this->Communicator, this->Bins);

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
      histogram.Compute(array, ghostArray);
      }

    // compute the global histogram
    histogram.PostCompute(this->Communicator, this->Bins, this->ArrayName);
    }
  else
    {
    vtkDataArray* array = this->GetArray(mesh, this->ArrayName);
    if (!array)
      {
      int rank = 0;
      MPI_Comm_rank(this->Communicator, &rank);
      SENSEI_WARNING("Dataset " << rank << " has no array named \""
        << this->ArrayName << "\"")
      histogram.PreCompute(this->Communicator, this->Bins);
      histogram.PostCompute(this->Communicator, this->Bins, this->ArrayName);
      }
    else
      {
      vtkUnsignedCharArray *ghostArray = dynamic_cast<vtkUnsignedCharArray*>(
        this->GetArray(mesh, this->GetGhostArrayName()));
      histogram.AddRange(array, ghostArray);
      histogram.PreCompute(this->Communicator, this->Bins);
      histogram.Compute(array, ghostArray);
      histogram.PostCompute(this->Communicator, this->Bins, this->ArrayName);
      }
    }
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

}
