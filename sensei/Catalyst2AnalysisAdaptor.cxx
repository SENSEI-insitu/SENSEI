#include "Catalyst2AnalysisAdaptor.h"

#include "Catalyst2DataAdaptor.h"
#include "Error.h"
#include "MeshMetadata.h"
#include "Profiler.h"
#include "VTKUtils.h"

#include <catalyst.h>
#include <catalyst_conduit.hpp>

#include <vtkDataObject.h>
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
  conduit_cpp::Node node;
  node["catalyst/scripts/script"].set_string(fileName);
  auto error_code = catalyst_initialize(conduit_cpp::c_node(&node));
  if (error_code != catalyst_status_ok)
  {
    // you are in trouble young man
    std::cerr << "catalyst initialize failed with code: " << error_code << std::endl;
  }
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

  auto c2dataAdaptor = Catalyst2DataAdaptor::SafeDownCast(dataAdaptor);
  if (c2dataAdaptor)
  {
    // fastpath, no need to translate data
    unsigned int nbNodes;
    c2dataAdaptor->GetNumberOfMeshes(nbNodes);
    if (nbNodes > 1)
    {
      SENSEI_WARNING("The mesh has several nodes, only the first one will be processed.");
    }

    exec_params = c2dataAdaptor->GetNode(0);
  }
  else
  {
    // translate data to conduit using VTK Module
    // TODO: Update the CMake to depends on the VTK conduit module
    for (auto meta : metadata)
    {
      const char* meshName = meta->MeshName.c_str();
      vtkDataObject* dobj = nullptr;
      if (dataAdaptor->GetMesh(meshName, false, dobj))
      {
        SENSEI_ERROR("Failed to get mesh \"" << meshName << "\"")
        return -1;
      }
      if (dobj)
      {
        if (auto* pds = vtkPartitionedDataSet::SafeDownCast(dobj))
        {
          for (auto node : vtk::Range(pds))
          {
            // TODO
            // if (node->IsA("vtkDataObject"))
            //   vtkDataObjectToConduit::FillConduitNode(node, exec_params);
            // else
            //   std::cout << "ingore: " << node->GetClassName() << std::endl;
          }
        }
        else
        {
          // TODO
          // vtkDataObjectToConduit::FillConduitNode(dobj, exec_params);
        }
      }
    }
  }

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
