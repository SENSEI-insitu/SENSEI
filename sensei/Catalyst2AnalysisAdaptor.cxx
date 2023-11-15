#include "Catalyst2AnalysisAdaptor.h"

#include "DataAdaptor.h"
#include "Error.h"
#include "MeshMetadata.h"
#include "MeshMetadataMap.h"

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
#include <svtkSmartPointer.h>

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

//-----------------------------------------------------------------------------
int Catalyst2AnalysisAdaptor::SetDataRequirements(const DataRequirements &reqs)
{
  this->Requirements = reqs;
  return 0;
}

//----------------------------------------------------------------------------
bool Catalyst2AnalysisAdaptor::Execute(DataAdaptor* dataAdaptor, DataAdaptor**)
{
  TimeEvent<128> mark("Catalyst2AnalysisAdaptor::Execute");

  // see what the simulation is providing
  MeshMetadataFlags flags;
  flags.SetBlockDecomp();
  flags.SetBlockSize();

  MeshMetadataMap mdMap;
  if (mdMap.Initialize(dataAdaptor, flags))
  {
    SENSEI_ERROR("Failed to get metadata")
      return false;
  }
  // if no dataIn requirements are given, push all the data
  // fill in the requirements with every thing
  if (this->Requirements.Empty())
  {
    if (this->Requirements.Initialize(dataAdaptor, false))
    {
      SENSEI_ERROR("Failed to initialze dataIn description")
        return false;
    }

    if (this->GetVerbose())
      SENSEI_WARNING("No subset specified. Writing all available data");
  }
  MeshRequirementsIterator mit = this->Requirements.GetMeshRequirementsIterator();


  // Conduit node to fill
  conduit_cpp::Node exec_params;
  // Store the simulation data so that it is kept until catalyst_execute and
  // it is deleted at the end of the function
  std::vector<svtkCompositeDataSetPtr> simulation_data;
  while(mit)
  {
    //const char* meshName = mmd->MeshName.c_str();
    const std::string &meshName = mit.MeshName();
    // get the metadta
    MeshMetadataPtr mmd;
    if (mdMap.GetMeshMetadata(meshName, mmd))
    {
      SENSEI_ERROR("Failed to get metadata for mesh \"" << meshName << "\"")
        return false;
    }

    svtkDataObject* dobj = nullptr;
    if (dataAdaptor->GetMesh(meshName, false/*structureOnly*/, dobj))
    {
      SENSEI_ERROR("Failed to get mesh \"" << meshName << "\"");
      return -1;
    }

    // add the ghost cell arrays to the mesh
    if ((mmd->NumGhostCells || SVTKUtils::AMR(mmd)) &&
        dataAdaptor->AddGhostCellsArray(dobj, meshName))
    {
      SENSEI_ERROR("Failed to get ghost cells for mesh \"" << meshName << "\"")
        return false;
    }

    // add the ghost node arrays to the mesh
    if (mmd->NumGhostNodes && dataAdaptor->AddGhostNodesArray(dobj, meshName))
    {
      SENSEI_ERROR("Failed to get ghost nodes for mesh \"" << meshName << "\"")
        return false;
    }

    // add the required arrays
    ArrayRequirementsIterator ait =
      this->Requirements.GetArrayRequirementsIterator(meshName);

    while (ait)
    {
      int assoc = ait.Association();
      std::string arrayName = ait.Array();

      if (dataAdaptor->AddArray(dobj, meshName, assoc, arrayName))
      {
        SENSEI_ERROR("Failed to add "
                     << SVTKUtils::GetAttributesName(assoc)
                     << " data array \"" << arrayName << "\" to mesh \""
                     << meshName << "\"");
        return -1;
      }
      ++ait;
    }

    MPI_Comm comm = this->GetCommunicator();
    if (svtkCompositeDataSetPtr mbds = SVTKUtils::AsCompositeData(comm, dobj, true/*take*/))
    {
      simulation_data.push_back(mbds);
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
    ++mit;
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
  conduit_cpp::Node node;
  auto error_code = catalyst_finalize(conduit_cpp::c_node(&node));
  if (error_code != catalyst_status_ok)
  {
    // you are in trouble young man
    std::cerr << "catalyst finalize failed with code: " << error_code << std::endl;
    return 1;
  }
  return 0;
}

//-----------------------------------------------------------------------------
void Catalyst2AnalysisAdaptor::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

}
