#include <cassert>
#include <sstream>

#include <catalyst.h>
#include <catalyst_conduit.hpp>

#include <vtkConduitSource.h>
#include <vtkDataObjectToConduit.h> // Not in the 5.9, may requires ParaView 5.10
#include <vtkNew.h>
#include <vtkObjectFactory.h>

#include "Catalyst2DataAdaptor.h"
#include "Error.h"

namespace sensei
{

//-----------------------------------------------------------------------------
senseiNewMacro(Catalyst2DataAdaptor);

//-----------------------------------------------------------------------------
Catalyst2DataAdaptor::Catalyst2DataAdaptor() {}

//-----------------------------------------------------------------------------
Catalyst2DataAdaptor::~Catalyst2DataAdaptor() {}

//-----------------------------------------------------------------------------
void Catalyst2DataAdaptor::PrintSelf(ostream &os, vtkIndent indent)
{
  // can't print conduit_cpp::node on custom ostream yet
  // use cout for debug
  for (auto n : this->Nodes) {
    std::cout << indent;
    n.print();
    std::cout << std::endl;
  }
  Superclass::PrintSelf(os, indent);
}

//-----------------------------------------------------------------------------
void Catalyst2DataAdaptor::SetNode(const conduit_cpp::Node& node)
{
  this->Nodes.clear();
  this->AddNode(node);
}

//-----------------------------------------------------------------------------
void Catalyst2DataAdaptor::AddNode(const conduit_cpp::Node& node)
{
  this->Nodes.emplace_back(node);
}

//-----------------------------------------------------------------------------
int Catalyst2DataAdaptor::GetMesh(
  const std::string& meshName, bool /*structureOnly*/, vtkDataObject*& mesh)
{
  conduit_cpp::Node meshNode;
  for (auto n : this->Nodes)
  {
    std::cout << " meshName " << meshName << std::endl;
    if (n.has_path(meshName))
    {
      std::cout << "has it!" << std::endl;
      meshNode = n;
      break;
    }
  }
  if (meshNode.number_of_children() == 0)
  {
    SENSEI_ERROR("GetMesh: Mesh " << meshName << " Cannot Be Found");
    return -1;
  }

  vtkNew<vtkConduitSource> conduitToVTK;
  conduitToVTK->SetNode(conduit_cpp::c_node(&meshNode));
  conduitToVTK->Update();
  if (vtkDataObject* res = conduitToVTK->GetOutputDataObject(0))
  {
    mesh = res;
  }
  else
  {
    return 1;
  }
  return 0;
}
//-----------------------------------------------------------------------------
int Catalyst2DataAdaptor::GetNumberOfMeshes(unsigned int& numberOfMeshes)
{
  numberOfMeshes = this->Nodes.size();
  return 0;
}
//-----------------------------------------------------------------------------
int Catalyst2DataAdaptor::GetMeshMetadata(unsigned int id, sensei::MeshMetadataPtr& metadata)
{
  metadata->MeshName = this->Nodes.at(id)["catalyst/channels"].path();
  return 0;
}

//-----------------------------------------------------------------------------
int Catalyst2DataAdaptor::AddArray(vtkDataObject* mesh, const std::string& meshName,
  int /*association*/, const std::string& arrayname)
{
  // TODO
  std::cerr << "not supported yet: add " << meshName << " to " << arrayname << std::endl;
  return 0;
}

//-----------------------------------------------------------------------------
int Catalyst2DataAdaptor::ReleaseData()
{
  this->Nodes.clear();
  return 0;
}

}
