#include "lammpsBridge.h"
#include "lammpsDataAdaptor.h"
#include <vtkSmartPointer.h>
#include <ConfigurableAnalysis.h>

#include <vtkDataObject.h>

namespace lammpsBridge
{
  static vtkSmartPointer<senseiLammps::lammpsDataAdaptor>  DataAdaptor;
  static vtkSmartPointer<sensei::ConfigurableAnalysis>     AnalysisAdaptor;

void Initialize(MPI_Comm world, const std::string& config_file)
{ 
  DataAdaptor = vtkSmartPointer<senseiLammps::lammpsDataAdaptor>::New();
  DataAdaptor->Initialize();
  DataAdaptor->SetCommunicator(world);
  DataAdaptor->SetDataTimeStep(-1);

  AnalysisAdaptor = vtkSmartPointer<sensei::ConfigurableAnalysis>::New();
  AnalysisAdaptor->Initialize(config_file);
}

void SetData(long ntimestep, int nlocal, int *id, 
             int nghost, int *type, double **x, 
             double xsublo, double xsubhi, 
             double ysublo, double ysubhi, 
             double zsublo, double zsubhi )
{
  DataAdaptor->AddLAMMPSData( ntimestep, nlocal, id, nghost, type, x, \
                              xsublo, xsubhi, ysublo, ysubhi, zsublo, zsubhi);
  DataAdaptor->SetDataTimeStep(ntimestep);
  DataAdaptor->SetDataTime(ntimestep);
}

void Analyze()
{
  AnalysisAdaptor->Execute(DataAdaptor.GetPointer());

  //vtkDataObject *mesh;
  //DataAdaptor->GetMesh("atoms", false, mesh);
  //DataAdaptor->AddArray(mesh, "atoms", vtkDataObject::FIELD_ASSOCIATION_POINTS, "type");

  DataAdaptor->ReleaseData();
}

//-----------------------------------------------------------------------------
void Finalize()
{
  AnalysisAdaptor->Finalize();
  AnalysisAdaptor = nullptr;
  DataAdaptor = nullptr;
}

}	// namespace lammpsBridge
