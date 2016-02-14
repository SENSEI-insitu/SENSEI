#include "HistogramAnalysisAdaptor.h"

#include "vtkDataArray.h"
#include "vtkDataObject.h"
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
  vtk_histogram(this->Communicator, vtkDataArray::SafeDownCast(
      data->GetArray(this->Association, this->ArrayName.c_str())), this->Bins);
  return true;
}
