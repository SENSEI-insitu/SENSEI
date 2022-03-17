#include "bridge.h"
#include "Oscillator.h"
#include "DataAdaptor.h"

#include <ConfigurableAnalysis.h>

#include <Profiler.h>
using sensei::TimeEvent;

#include <svtkDataObject.h>
#include <svtkNew.h>
#include <svtkSmartPointer.h>

namespace bridge
{
static svtkSmartPointer<oscillators::DataAdaptor> DataAdaptor;
static svtkSmartPointer<sensei::ConfigurableAnalysis> AnalysisAdaptor;

//-----------------------------------------------------------------------------
int initialize(size_t nblocks, size_t n_local_blocks,
  float *origin, float *spacing, int domain_shape_x, int domain_shape_y,
  int domain_shape_z, int *gid, int *from_x, int *from_y, int *from_z,
  int *to_x, int *to_y, int *to_z, int *shape, int ghostLevels,
  const std::string &config_file)
{
  TimeEvent<128> event("bridge::initialize");

  DataAdaptor = svtkSmartPointer<oscillators::DataAdaptor>::New();

  DataAdaptor->Initialize(nblocks, n_local_blocks, origin, spacing,
    domain_shape_x, domain_shape_y, domain_shape_z, gid, from_x, from_y,
    from_z, to_x, to_y, to_z, shape, ghostLevels);

  AnalysisAdaptor = svtkSmartPointer<sensei::ConfigurableAnalysis>::New();
  if (AnalysisAdaptor->Initialize(config_file))
    {
    std::cerr << "Failed to initialize the analysis adaptor" << std::endl;
    return -1;
    }

  return 0;
}

//-----------------------------------------------------------------------------
void set_data(int gid, float* data)
{
  DataAdaptor->SetBlockData(gid, data);
}

//-----------------------------------------------------------------------------
void set_particles(int gid, const std::vector<Particle> &particles)
{
  DataAdaptor->SetParticleData(gid, particles);
}

//-----------------------------------------------------------------------------
void set_oscillators(const OscillatorArray &oscillators)
{
  DataAdaptor->SetOscillators(oscillators);
}

//-----------------------------------------------------------------------------
void execute(long step, float time, sensei::DataAdaptor **dataOut)
{
  TimeEvent<128> event("bridge::Execute");

  DataAdaptor->SetDataTimeStep(step);
  DataAdaptor->SetDataTime(time);

  AnalysisAdaptor->Execute(DataAdaptor.GetPointer(), dataOut);

  DataAdaptor->ReleaseData();
}

//-----------------------------------------------------------------------------
void finalize()
{
  TimeEvent<128> event("bridge::finalize");

  AnalysisAdaptor->Finalize();

  AnalysisAdaptor = nullptr;
  DataAdaptor = nullptr;
}

}
