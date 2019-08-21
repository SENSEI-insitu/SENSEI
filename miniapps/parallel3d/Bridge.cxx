#include "Bridge.h"
#include "DataAdaptor.h"

#include <sensei/ConfigurableAnalysis.h>
#include <vtkNew.h>
#include <vtkDataObject.h>
#include <vector>

#include "Timer.h"

namespace bridge
{
  static parallel3d::DataAdaptor *DataAdaptor = nullptr;
  static sensei::ConfigurableAnalysis *AnalysisAdaptor = nullptr;
}

//-----------------------------------------------------------------------------
int bridge_initialize(const char* config_file, int g_nx, int g_ny, int g_nz,
  uint64_t offs_x, uint64_t offs_y, uint64_t offs_z, int l_nx, int l_ny, int l_nz,
  double *pressure, double* temperature, double* density)
{
  sensei::Timer::Initialize();

  // configure the analysis. this can be an expensive operation
  // hence we only want to do it once per run
  bridge::AnalysisAdaptor = sensei::ConfigurableAnalysis::New();

  if (bridge::AnalysisAdaptor->Initialize(config_file))
    {
    std::cerr << "Failed to initialize the analysis using \""
      << config_file << "\"" << std::endl;
    return -1;
    }

  // we can do mesh construction here because in this mini-app niether
  // the mesh geometry nor domain decomposition evolves in time and we
  // are going to use zero copy for arrays. in a simulation that has time
  // evolving geometry or domain decomposition, or where you are not using
  // zero copy, this code would be in bridge_update and be called before
  // every analysis.
  bridge::DataAdaptor = parallel3d::DataAdaptor::New();

  bridge::DataAdaptor->UpdateGeometry(0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
    g_nx, g_ny, g_nz, offs_x, offs_y, offs_z, l_nx, l_ny, l_nz);

  bridge::DataAdaptor->UpdateArrays(pressure, temperature, density);

  return 0;
}

//-----------------------------------------------------------------------------
void bridge_update(int tstep, double time)
{
  // we've cached our data adaptor because niether our mesh nor domain decomp
  // evolves in time, and we are using zero copy to pass data. here, all we
  // need to do is update the time and step
  bridge::DataAdaptor->SetDataTime(time);
  bridge::DataAdaptor->SetDataTimeStep(tstep);

  // invoke the analysis
  bridge::AnalysisAdaptor->Execute(bridge::DataAdaptor);

  bridge::DataAdaptor->ReleaseData();
}

//-----------------------------------------------------------------------------
void bridge_finalize()
{
  // tear down the analysis. this is especially important for the larger
  // infrastructures who might make MPI calls in their tear down.
  bridge::AnalysisAdaptor->Finalize();

  // release our memory
  bridge::AnalysisAdaptor->Delete();
  bridge::DataAdaptor->Delete();

  sensei::Timer::Finalize();
}
