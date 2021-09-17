#include "Catalyst2AnalysisAdaptor.h"

#include "DataAdaptor.h"
#include "Error.h"
#include "MeshMetadata.h"
#include "Profiler.h"
#include "VTKUtils.h"

#include <catalyst.h>
#include <catalyst_conduit.hpp>

namespace sensei
{

//-----------------------------------------------------------------------------
senseiNewMacro(Catalyst2AnalysisAdaptor);

//-----------------------------------------------------------------------------
Catalyst2AnalysisAdaptor::Catalyst2AnalysisAdaptor()
{
  this->Initialize();
}

//-----------------------------------------------------------------------------
Catalyst2AnalysisAdaptor::~Catalyst2AnalysisAdaptor()
{
  this->Finalize();
}

//-----------------------------------------------------------------------------
void Catalyst2AnalysisAdaptor::Initialize() {}

//-----------------------------------------------------------------------------
void Catalyst2AnalysisAdaptor::AddPythonScriptPipeline(const std::string& fileName)
{
#ifdef ENABLE_CATALYST_PYTHON
  conduit_cpp::Node node;
  node["catalyst/scripts/script"].set_string(fileName);
  auto error_code = catalyst_initialize(conduit_cpp::c_node(&node));
  if (error_code != catalyst_status_ok)
  {
    // you are in trouble young man
  }
#else
  (void)fileName;
  SENSEI_ERROR("Failed to add Python script pipeline. "
               "Re-compile with ENABLE_CATALYST_PYTHON=ON")
#endif
}

//----------------------------------------------------------------------------
bool Catalyst2AnalysisAdaptor::Execute(DataAdaptor* dataAdaptor)
{
  TimeEvent<128> mark("Catalyst2AnalysisAdaptor::Execute");

  // Get a description of the simulation metadata
  unsigned int nMeshes = 0;
  if (dataAdaptor->GetNumberOfMeshes(nMeshes))
  {
    SENSEI_ERROR("Failed to get the number of meshes")
    return false;
  }

  std::vector<MeshMetadataPtr> metadata(nMeshes);
  for (unsigned int i = 0; i < nMeshes; ++i)
  {
    MeshMetadataPtr mmd = MeshMetadata::New();

    if (dataAdaptor->GetMeshMetadata(i, mmd))
    {
      SENSEI_ERROR("Failed to get metadata for mesh " << i << " of " << nMeshes)
      return false;
    }
    metadata[i] = mmd;
  }

  // Conduit node to fill
  conduit_cpp::Node exec_params;

  // Time description
  double time = dataAdaptor->GetDataTime();
  int timeStep = dataAdaptor->GetDataTimeStep();
  auto state = exec_params["catalyst/state"];
  state["timestep"].set(long(timeStep));
  state["time"].set(time);

  return true;
}

//-----------------------------------------------------------------------------
int Catalyst2AnalysisAdaptor::Finalize()
{
  return 0;
}

//-----------------------------------------------------------------------------
void Catalyst2AnalysisAdaptor::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

}
