#include "Catalyst2AnalysisAdaptor.h"

#include "Catalyst2DataAdaptor.h"
#include "Error.h"
#include "MeshMetadata.h"
#include "Profiler.h"
#include "VTKUtils.h"

#include <catalyst.h>
#include <catalyst_conduit.hpp>

#include <vtkDataObject.h>
#include <vtkDataObjectToConduit.h>
#include <vtkDataObjectTreeRange.h>
#include <vtkPartitionedDataSet.h>
#include <vtkRange.h>

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
    std::cerr << "catalyst initialize failed with code: " << error_code << std::endl;
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

  // Casting into the concrete Catalyst2DataAdaptor is not ideal,
  // but it remove the need to GetMesh - Mesh to conduit.
  // (still need testing, may need conduit to conduit adapt)
  auto c2dataAdaptor = Catalyst2DataAdaptor::SafeDownCast(dataAdaptor);
  if (c2dataAdaptor == nullptr)
  {
    SENSEI_ERROR("The Catalyst2AnalysisAdaptor requires a Catalyst2DataAdaptor to do 0-Copy.");
    return false;
  }
  exec_params = c2dataAdaptor->GetNode(0);

  // Time description
  double time = dataAdaptor->GetDataTime();
  int timeStep = dataAdaptor->GetDataTimeStep();
  auto state = exec_params["catalyst/state"];
  state["timestep"].set(long(timeStep));
  state["time"].set(time);

  catalyst_execute(conduit_cpp::c_node(&exec_params));

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
