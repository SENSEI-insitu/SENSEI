#include "AnalysisAdaptor.h"

namespace sensei
{

//----------------------------------------------------------------------------
AnalysisAdaptor::AnalysisAdaptor() : Verbose(0)
{
  MPI_Comm_dup(MPI_COMM_WORLD, &this->Comm);
}

//----------------------------------------------------------------------------
AnalysisAdaptor::~AnalysisAdaptor()
{
  MPI_Comm_free(&this->Comm);
}

//----------------------------------------------------------------------------
int AnalysisAdaptor::SetCommunicator(MPI_Comm comm)
{
  MPI_Comm_free(&this->Comm);
  MPI_Comm_dup(comm, &this->Comm);
  return 0;
}

//----------------------------------------------------------------------------
void AnalysisAdaptor::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

}
