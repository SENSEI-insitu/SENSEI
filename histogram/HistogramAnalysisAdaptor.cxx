#include "HistogramAnalysisAdaptor.h"

#include "vtkDataObject.h"
#include "vtkFieldData.h"
#include "vtkInsituDataAdaptor.h"
#include "vtkObjectFactory.h"

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
  vtkDataObject* mesh = data->GetMesh(/*structure_only*/true);
  if (mesh && data->AddArray(mesh, this->Association, this->ArrayName.c_str()))
    {
    if (vtkFieldData* fd = mesh->GetAttributesAsFieldData(this->Association))
      {
      vtk_histogram(this->Communicator, fd->GetArray(this->ArrayName.c_str()), this->Bins);
      return true;
      }
    }
  return false;
}
