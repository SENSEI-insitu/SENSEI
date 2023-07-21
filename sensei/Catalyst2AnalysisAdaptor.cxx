#include "Catalyst2AnalysisAdaptor.h"

#include "DataAdaptor.h"
#include "Error.h"
#include "MeshMetadata.h"
#include "Profiler.h"
#include "SVTKUtils.h"

#include <catalyst.h>
#include <catalyst_conduit.hpp>

#include <svtkDataObject.h>
#include <svtkCompositeDataSetRange.h>
#include <svtkDataObjectToConduit.h>
#include <svtkDataSet.h>
#include <svtkMultiBlockDataSet.h>
#include <svtkPointData.h>
#include <svtkRange.h>

namespace sensei
{

// #define Catalyst2DebugMacro(x) std::cerr x << std::endl;
#define Catalyst2DebugMacro(x)

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
bool Catalyst2AnalysisAdaptor::Execute(DataAdaptor* dataAdaptor, DataAdaptor**)
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

  for (auto meta : metadata)
  {
    const char* meshName = meta->MeshName.c_str();
    svtkDataObject* dobj = nullptr;
    if (dataAdaptor->GetMesh(meshName, false, dobj))
    {
      SENSEI_ERROR("Failed to get mesh \"" << meshName << "\"");
        return -1;
    }

    for (int j = 0; j < meta->NumArrays; ++j)
    {
      int assoc = meta->ArrayCentering[j];
      const char *arrayName = meta->ArrayName[j].c_str();

      if (dataAdaptor->AddArray(dobj, meshName, assoc, arrayName))
      {
        SENSEI_ERROR("Failed to add "
                     << SVTKUtils::GetAttributesName(assoc)
                     << " data array \"" << arrayName << "\" to mesh \""
                     << meshName << "\"");
          return -1;
      }
    }

    if (dobj)
    {
      MPI_Comm comm = this->GetCommunicator();
      if (auto mbds = SVTKUtils::AsCompositeData(comm, dobj, false))
      {
        for (auto node : svtk::Range(mbds))
        {
          auto ds = svtkDataSet::SafeDownCast(node);
          if (ds)
          {
            Catalyst2DebugMacro(
              << "node: " << ds->GetClassName() << " # points: " << ds->GetNumberOfPoints()
              << " # cells: " << ds->GetNumberOfCells());
          }
          auto channel = exec_params[std::string("catalyst/channels/") + meshName];
          channel["type"].set("mesh");
          auto mesh = channel["data"];
          if (! svtkDataObjectToConduit::FillConduitNode(node, mesh))
          {
            std::cerr << "ignore: " << node->GetClassName() << std::endl;
          }
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
  Catalyst2DebugMacro( << "time: " << time);
  catalyst_execute(conduit_cpp::c_node(&exec_params));

  return true;
}

//-----------------------------------------------------------------------------
int Catalyst2AnalysisAdaptor::Finalize()
{
  return 0;
}

//-----------------------------------------------------------------------------
void Catalyst2AnalysisAdaptor::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

}
