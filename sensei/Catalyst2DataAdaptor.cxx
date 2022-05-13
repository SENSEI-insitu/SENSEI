#include <cassert>
#include <sstream>

#include <catalyst.h>
#include <catalyst_conduit.hpp>

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
const conduit_cpp::Node& Catalyst2DataAdaptor::GetNode(unsigned int id) const
{
  return this->Nodes.at(id);
}

//-----------------------------------------------------------------------------
int Catalyst2DataAdaptor::GetMesh(
  const std::string& meshName, bool /*structureOnly*/, vtkDataObject*& mesh)
{
  for (auto n : this->Nodes)
  {
    if (n.has_path("catalyst/channels/"+meshName))
    {
      conduit_cpp::Node meshNode(n["catalyst/channels/"+meshName+"/data"]);
      // TODO
      // vtkNew<vtkConduitSource> conduitToVTK;
      // conduitToVTK->SetNode(conduit_cpp::c_node(&meshNode));
      // conduitToVTK->Update();
      // if (vtkDataObject* res = conduitToVTK->GetOutputDataObject(0))
      // {
      //   mesh = res;
      //   mesh->Register(this);
      //   return 0;
      // }
      // else
      // {
      //   SENSEI_ERROR("Error while creating the VTK mesh for " << meshName);
      //   return 1;
      // }
    }
  }
  SENSEI_ERROR("GetMesh: Mesh " << meshName << " Cannot Be Found");
  return -1;
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

  if (this->Nodes.at(id)["catalyst/channels/"].number_of_children() < 1)
  {
    SENSEI_ERROR("Node " << id << " is ill-formed or has no mesh assiciated.");
    return -1;
  }
  metadata->MeshName = this->Nodes.at(id)["catalyst/channels/"].child(0).name();
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
