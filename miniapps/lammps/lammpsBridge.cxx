#include "lammpsBridge.h"
#include "lammpsDataAdaptor.h"
#include <vtkSmartPointer.h>
#include <ConfigurableAnalysis.h>

namespace lammpsBridge
{
  static vtkSmartPointer<senseiLammps::lammpsDataAdaptor>     GlobalDataAdaptor;
  static vtkSmartPointer<sensei::ConfigurableAnalysis> GlobalAnalysisAdaptor;

void Initialize(MPI_Comm world, const std::string& config_file)
{ 
  GlobalDataAdaptor = vtkSmartPointer<senseiLammps::lammpsDataAdaptor>::New();
  GlobalDataAdaptor->Initialize();
  GlobalDataAdaptor->SetCommunicator(world);

  GlobalAnalysisAdaptor = vtkSmartPointer<sensei::ConfigurableAnalysis>::New();
  GlobalAnalysisAdaptor->Initialize(config_file);
}

void SetData(long ntimestep, int nlocal, int *id, 
             int nghost, int *type, double **x, 
             double xsublo, double xsubhi, 
             double ysublo, double ysubhi, 
             double zsublo, double zsubhi )
{
  GlobalDataAdaptor->AddLAMMPSData( ntimestep, nlocal, id, nghost, type, x, xsublo, xsubhi, ysublo, ysubhi, zsublo, zsubhi);
}

void Analyze()
{
  GlobalAnalysisAdaptor->Execute(GlobalDataAdaptor.GetPointer());
  GlobalDataAdaptor->ReleaseData();
}

//-----------------------------------------------------------------------------
void Finalize()
{
  GlobalAnalysisAdaptor = NULL;
  GlobalDataAdaptor = NULL;  
}

}	// namespace lammpsBridge
