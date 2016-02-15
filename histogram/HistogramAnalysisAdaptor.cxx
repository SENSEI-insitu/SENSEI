#include "HistogramAnalysisAdaptor.h"

#include "vtkCompositeDataIterator.h"
#include "vtkCompositeDataSet.h"
#include "vtkDataObject.h"
#include "vtkFieldData.h"
#include "vtkInsituDataAdaptor.h"
#include "vtkObjectFactory.h"
#include "vtkSmartPointer.h"

#include "vtk_histogram.h"

vtkStandardNewMacro(HistogramAnalysisAdaptor);
//-----------------------------------------------------------------------------
HistogramAnalysisAdaptor::HistogramAnalysisAdaptor() :
  Communicator(MPI_COMM_WORLD),
  Bins(0),
  Association(vtkDataObject::FIELD_ASSOCIATION_POINTS)
{
}

//-----------------------------------------------------------------------------
HistogramAnalysisAdaptor::~HistogramAnalysisAdaptor()
{
}

//-----------------------------------------------------------------------------
void HistogramAnalysisAdaptor::Initialize(
  MPI_Comm comm, int bins, int association, const std::string& arrayname)
{
  this->Communicator = comm;
  this->Bins = bins;
  this->ArrayName = arrayname;
  this->Association = association;
}

//-----------------------------------------------------------------------------
bool HistogramAnalysisAdaptor::Execute(vtkInsituDataAdaptor* data)
{
  bool retval = false;
  vtkDataObject* mesh = data->GetMesh(/*structure_only*/true);
  if (mesh && data->AddArray(mesh, this->Association, this->ArrayName.c_str()))
    {
    if (vtkCompositeDataSet* cd = vtkCompositeDataSet::SafeDownCast(mesh))
      {
      vtkSmartPointer<vtkCompositeDataIterator> iter;
      iter.TakeReference(cd->NewIterator());
      for (iter->InitTraversal(); !iter->IsDoneWithTraversal(); iter->GoToNextItem())
        {
        retval = this->Execute(iter->GetCurrentDataObject()) || retval;
        }
      }
    else
      {
      retval = this->Execute(mesh);
      }
    }
  return retval;
}

//-----------------------------------------------------------------------------
bool HistogramAnalysisAdaptor::Execute(vtkDataObject* mesh)
{
  if (vtkFieldData* fd = mesh->GetAttributesAsFieldData(this->Association))
    {
    if (vtkDataArray* da = fd->GetArray(this->ArrayName.c_str()))
      {
      vtk_histogram(this->Communicator, da, this->Bins);
      }
    return true;
    }
  return false;
}

