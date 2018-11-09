#include "lammpsBridge.h"
#include "lammpsDataAdaptor.h"
#include <vtkSmartPointer.h>
#include <ConfigurableAnalysis.h>
#include <libIS/is_sim.h>

namespace lammpsBridge
{
  static vtkSmartPointer<senseiLammps::lammpsDataAdaptor>     GlobalDataAdaptor;
  static vtkSmartPointer<sensei::ConfigurableAnalysis> GlobalAnalysisAdaptor;

void Initialize(MPI_Comm world, const std::string& config_file)
{ 
  GlobalDataAdaptor = vtkSmartPointer<senseiLammps::lammpsDataAdaptor>::New();
  GlobalDataAdaptor->Initialize();
  GlobalDataAdaptor->SetCommunicator(world);
  GlobalDataAdaptor->SetDataTimeStep(-1);

  GlobalAnalysisAdaptor = vtkSmartPointer<sensei::ConfigurableAnalysis>::New();
  GlobalAnalysisAdaptor->Initialize(config_file);

  const int port = 29374;
  libISInit(world, port);
}

void SetData(long ntimestep, int nlocal, int *id, 
             int nghost, int *type, double **x, 
             double xsublo, double xsubhi, 
             double ysublo, double ysubhi, 
             double zsublo, double zsubhi )
{
  GlobalDataAdaptor->AddLAMMPSData( ntimestep, nlocal, id, nghost, type, x, xsublo, xsubhi, ysublo, ysubhi, zsublo, zsubhi);
  GlobalDataAdaptor->SetDataTimeStep(ntimestep);
  GlobalDataAdaptor->SetDataTime(ntimestep);

  //update libIS
  libISBox3f bounds=libISMakeBox3f();
  libISVec3f islo{ (float)xsublo, (float)ysublo, (float)zsublo};
  libISVec3f ishi{ (float)xsubhi, (float)ysubhi, (float)zsubhi};
  libISBoxExtend(&bounds, &islo);
  libISBoxExtend(&bounds, &ishi);

  libISBox3f ghostBounds = bounds;
  // TODO: Which side is better to convert on? We have
  // to convert here anyway since LAMMPS is SoA and we're
  // expecting AoS data coming in to libIS.
  struct LammpsAtom {
	  float x, y, z;
	  int type;
  };
  std::vector<LammpsAtom> atoms;
  atoms.reserve(nlocal + nghost);
  for (int i = 0; i < nlocal + nghost; ++i) {
	  atoms.push_back(LammpsAtom{ (float)(*x)[i * 3], 
                                  (float)(*x)[i * 3 + 1],
			                      (float)(*x)[i * 3 + 2], 
                                  type[i]});
	  libISVec3f pos{ (float)(*x)[i * 3], 
                      (float)(*x)[i * 3 + 1],
			          (float)(*x)[i * 3 + 2]};
	  libISBoxExtend(&ghostBounds, &pos);
  }

  libISSimState *state = libISMakeSimState();
  // TODO: Set world bounds as well
  libISSetLocalBounds(state, bounds);
  libISSetGhostBounds(state, ghostBounds);
  libISSetParticles(state, nlocal, nghost, sizeof(LammpsAtom), atoms.data());
  libISProcess(state);
  libISFreeSimState(state);

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
  libISFinalize();   
}

}	// namespace lammpsBridge
