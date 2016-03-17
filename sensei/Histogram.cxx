#include "Histogram.h"

#include <vtkCompositeDataIterator.h>
#include <vtkCompositeDataSet.h>
#include <vtkDataArray.h>
#include <vtkDataObject.h>
#include <vtkFieldData.h>
#include <vtkObjectFactory.h>
#include <vtkSmartPointer.h>

#ifdef VTK_HAS_GENERIC_ARRAYS
# include "HistogramInternals-GenericArrays.h"
#else
# include "HistogramInternals-NoGenericArrays.h"
#endif

#include "DataAdaptor.h"

#include <timer/Timer.h>

#include <algorithm>
#include <vector>

namespace sensei
{

#if VTK_MAJOR_VERSION == 6 && VTK_MINOR_VERSION == 1
Histogram *Histogram::New() { return new Histogram; }
#else
vtkStandardNewMacro(Histogram);
#endif

//-----------------------------------------------------------------------------
Histogram::Histogram() :
  Communicator(MPI_COMM_WORLD),
  Bins(0),
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

//-----------------------------------------------------------------------------
bool Histogram::Execute(sensei::DataAdaptor* data)
{
  timer::MarkEvent mark("histogram::execute");

  vtkHistogram histogram;
  vtkDataObject* mesh = data->GetMesh(/*structure_only*/true);
  if (mesh == NULL || !data->AddArray(mesh, this->Association, this->ArrayName.c_str()))
    {
    histogram.PreCompute(this->Communicator, this->Bins);
    histogram.PostCompute(this->Communicator, this->Bins, this->ArrayName);
    return true;
    }

  if (vtkCompositeDataSet* cd = vtkCompositeDataSet::SafeDownCast(mesh))
    {
    vtkSmartPointer<vtkCompositeDataIterator> iter;
    iter.TakeReference(cd->NewIterator());
    for (iter->InitTraversal(); !iter->IsDoneWithTraversal(); iter->GoToNextItem())
      {
      vtkDataArray* array = this->GetArray(iter->GetCurrentDataObject());
      histogram.AddRange(array);
      }
    histogram.PreCompute(this->Communicator, this->Bins);
    for (iter->InitTraversal(); !iter->IsDoneWithTraversal(); iter->GoToNextItem())
      {
      vtkDataArray* array = this->GetArray(iter->GetCurrentDataObject());
      histogram.Compute(array);
      }
    histogram.PostCompute(this->Communicator, this->Bins, this->ArrayName);
    }
  else
    {
    vtkDataArray* array = this->GetArray(mesh);
    histogram.AddRange(array);
    histogram.PreCompute(this->Communicator, this->Bins);
    histogram.Compute(array);
    histogram.PostCompute(this->Communicator, this->Bins, this->ArrayName);
    }
  return true;
}

//-----------------------------------------------------------------------------
vtkDataArray* Histogram::GetArray(vtkDataObject* dobj)
{
  assert(dobj != NULL && vtkCompositeDataSet::SafeDownCast(dobj) == NULL);
  if (vtkFieldData* fd = dobj->GetAttributesAsFieldData(this->Association))
    {
    return fd->GetArray(this->ArrayName.c_str());
    }
  return NULL;
}

} // end of namespace sensei
