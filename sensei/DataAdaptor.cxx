#include "DataAdaptor.h"
#include "VTKUtils.h"
#include "Error.h"

#include <vtkDataObject.h>
#include <vtkInformation.h>
#include <vtkInformationIntegerKey.h>
#include <vtkObjectFactory.h>

#include <map>
#include <vector>
#include <string>
#include <utility>

namespace sensei
{

using AssocArrayMapType = std::map<int, std::vector<std::string>>;
using MeshArrayMapType = std::map<std::string, AssocArrayMapType>;

struct DataAdaptor::InternalsType
{
  InternalsType() : Information(vtkInformation::New()) {}
  ~InternalsType(){ this->Information->Delete(); }

  void Clear()
  {
    this->Information->Delete();
    this->Information = vtkInformation::New();
    this->MeshNames.clear();
    this->MeshArrayMap.clear();
  }

  std::vector<std::string> MeshNames;
  MeshArrayMapType MeshArrayMap;
  vtkInformation *Information;
};

//----------------------------------------------------------------------------
vtkInformationKeyMacro(DataAdaptor, DATA_TIME_STEP_INDEX, Integer);

//----------------------------------------------------------------------------
DataAdaptor::DataAdaptor()
{
  this->Internals = new InternalsType;
}

//----------------------------------------------------------------------------
DataAdaptor::~DataAdaptor()
{
  delete this->Internals;
}

//----------------------------------------------------------------------------
vtkInformation* DataAdaptor::GetInformation()
{
  return this->Internals->Information;
}

//----------------------------------------------------------------------------
double DataAdaptor::GetDataTime()
{
  return this->GetDataTime(this->Internals->Information);
}

//----------------------------------------------------------------------------
void DataAdaptor::SetDataTime(double time)
{
  this->SetDataTime(this->Internals->Information, time);
}

//----------------------------------------------------------------------------
int DataAdaptor::GetDataTimeStep()
{
  return this->GetDataTimeStep(this->Internals->Information);
}

//----------------------------------------------------------------------------
double DataAdaptor::GetDataTime(vtkInformation* info)
{
  return info->Has(vtkDataObject::DATA_TIME_STEP()) ?
    info->Get(vtkDataObject::DATA_TIME_STEP()) : 0.0;
}

//----------------------------------------------------------------------------
void DataAdaptor::SetDataTime(vtkInformation* info, double time)
{
  info->Set(vtkDataObject::DATA_TIME_STEP(), time);
}

//----------------------------------------------------------------------------
void DataAdaptor::SetDataTimeStep(int index)
{
  this->SetDataTimeStep(this->Internals->Information, index);
}

//----------------------------------------------------------------------------
int DataAdaptor::GetDataTimeStep(vtkInformation* info)
{
  return info->Has(DataAdaptor::DATA_TIME_STEP_INDEX()) ?
    info->Get(DataAdaptor::DATA_TIME_STEP_INDEX()) : 0;
}

//----------------------------------------------------------------------------
void DataAdaptor::SetDataTimeStep(vtkInformation* info, int index)
{
  info->Set(DataAdaptor::DATA_TIME_STEP_INDEX(), index);
}

//----------------------------------------------------------------------------
int DataAdaptor::GetMeshNames(std::vector<std::string> &meshNames)
{
  if (!this->Internals->MeshNames.empty())
    {
    meshNames = this->Internals->MeshNames;
    return 0;
    }

  unsigned int nMeshes = 0;
  if (this->GetNumberOfMeshes(nMeshes))
    {
    SENSEI_ERROR("Failed to get the number of meshes")
    return -1;
    }

  for (unsigned int i = 0; i < nMeshes; ++i)
    {
    std::string meshName;
    if (this->GetMeshName(i, meshName))
      {
      SENSEI_ERROR("Failed to get the mesh name at " << i)
      return -1;
      }
    this->Internals->MeshNames.push_back(meshName);
    }

  meshNames = this->Internals->MeshNames;
  return 0;
}

//----------------------------------------------------------------------------
int DataAdaptor::GetArrayNames(const std::string &meshName, int association,
    std::vector<std::string> &arrayNames)
{
  MeshArrayMapType::iterator it =
    this->Internals->MeshArrayMap.find(meshName);

  if (it != this->Internals->MeshArrayMap.end())
    {
    AssocArrayMapType::iterator ait = it->second.find(association);
    if (ait != it->second.end())
      {
      arrayNames = ait->second;
      return 0;
      }
    }

  unsigned int nArrays = 0;
  if (this->GetNumberOfArrays(meshName, association, nArrays))
    {
    SENSEI_ERROR("Failed to get number of "
      << VTKUtils::GetAttributesName(association)
      << " data arrays for mesh " << meshName)
    return -1;
    }

  for (unsigned int i = 0; i < nArrays; ++i)
    {
    std::string arrayName;
    if (this->GetArrayName(meshName, association, i, arrayName))
      {
      SENSEI_ERROR("Failed to get the name of "
        << VTKUtils::GetAttributesName(association)
        << " data array " << i << " on mesh \"" << meshName << "\"")
      return -1;
      }
    arrayNames.push_back(arrayName);
    }

  // cache the list
  this->Internals->MeshArrayMap[meshName][association] = arrayNames;
  return 0;
}

//----------------------------------------------------------------------------
int DataAdaptor::AddArrays(vtkDataObject* mesh, const std::string &meshName,
    int association, const std::vector<std::string> &arrayNames)
{
  unsigned int nArrays = arrayNames.size();
  for (unsigned int i = 0; i < nArrays; ++i)
    {
    if (this->AddArray(mesh, meshName, association, arrayNames[i]))
      {
      SENSEI_ERROR("Failed to add "
        << VTKUtils::GetAttributesName(association) << " data array \""
        << arrayNames[i] << " to mesh \"" << meshName << "\"")
      return -1;
      }
    }
  return 0;
}

//----------------------------------------------------------------------------
int DataAdaptor::AddArrays(vtkDataObject* mesh, const std::string &meshName,
    int association)
{
  std::vector<std::string> arrayNames;
  if (this->GetArrayNames(meshName, association, arrayNames))
    {
    SENSEI_ERROR("Failed to get "
       << VTKUtils::GetAttributesName(association)
       << " data array names for mesh \"" << meshName << "\"")
    return -1;
    }

  if (this->AddArrays(mesh, meshName, association, arrayNames))
    return -1;

  return 0;
}

//----------------------------------------------------------------------------
int DataAdaptor::GetCompleteMesh(const std::string &meshName,
  bool structureOnly, vtkDataObject *&mesh)
{
  mesh = nullptr;
  if (this->GetMesh(meshName, structureOnly, mesh))
    {
    SENSEI_ERROR("failed to get mesh \"" << meshName << "\"")
    return -1;
    }
  if (mesh == nullptr)
    {
    SENSEI_ERROR("data adaptor returned null mesh for \"" << meshName << "\"")
    return -1;
    }

  if (this->AddArrays(mesh, meshName, vtkDataObject::CELL))
    {
    SENSEI_ERROR("Failed to add cell arrays to mesh \"" << meshName << "\"")
    return -1;
    }

  if (this->AddArrays(mesh, meshName, vtkDataObject::POINT))
    {
    SENSEI_ERROR("Failed to add point arrays to mesh \"" << meshName << "\"")
    return -1;
    }

  int nLayers = 0;
  if (this->GetMeshHasGhostNodes(meshName, nLayers) == 0)
    {
    if(nLayers > 0)
      {
      if(this->AddGhostNodesArray(mesh, meshName))
        {
        SENSEI_ERROR("Failed to add ghost nodes to mesh \"" << meshName << "\"")
        return -1;
        }
      }
    }

  nLayers = 0;
  if (this->GetMeshHasGhostCells(meshName, nLayers) == 0)
    {
    if(nLayers > 0)
      {
      if(this->AddGhostCellsArray(mesh, meshName))
        {
        SENSEI_ERROR("Failed to add ghost cells to mesh \"" << meshName << "\"")
        return -1;
        }
      }
    }

  return 0;
}

//----------------------------------------------------------------------------
int DataAdaptor::GetMeshHasGhostNodes(const std::string &/*meshName*/, 
  int &nLayers)
{
  nLayers = 0;
  return 0;
}

//----------------------------------------------------------------------------
int DataAdaptor::AddGhostNodesArray(vtkDataObject*, const std::string &)
{
  return 0;
}

//----------------------------------------------------------------------------
int DataAdaptor::GetMeshHasGhostCells(const std::string &/*meshName*/, 
  int &nLayers)
{
  nLayers = 0;
  return 0;
}

//----------------------------------------------------------------------------
int DataAdaptor::AddGhostCellsArray(vtkDataObject*, const std::string &)
{
  return 0;
}

//----------------------------------------------------------------------------
void DataAdaptor::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

}
