#include "ProgrammableDataAdaptor.h"

#include "senseiConfig.h"
#include "MeshMetadata.h"
#include "Error.h"

#include <vtkObjectFactory.h>

namespace sensei
{
//-----------------------------------------------------------------------------
senseiNewMacro(ProgrammableDataAdaptor);

//----------------------------------------------------------------------------
ProgrammableDataAdaptor::ProgrammableDataAdaptor() :
 GetNumberOfMeshesCallback(nullptr), GetMeshMetadataCallback(nullptr),
 GetMeshCallback(nullptr), AddArrayCallback(nullptr),
 ReleaseDataCallback(nullptr)
{
}

//----------------------------------------------------------------------------
ProgrammableDataAdaptor::~ProgrammableDataAdaptor()
{
}
//----------------------------------------------------------------------------
void ProgrammableDataAdaptor::SetGetNumberOfMeshesCallback(
  const GetNumberOfMeshesFunction &callback)
{
  this->GetNumberOfMeshesCallback = callback;
}

//----------------------------------------------------------------------------
void ProgrammableDataAdaptor::SetGetMeshMetadataCallback(
  const GetMeshMetadataFunction &callback)
{
  this->GetMeshMetadataCallback = callback;
}

//----------------------------------------------------------------------------
void ProgrammableDataAdaptor::SetGetMeshCallback(
  const GetMeshFunction &callback)
{
  this->GetMeshCallback = callback;
}

//----------------------------------------------------------------------------
void ProgrammableDataAdaptor::SetAddArrayCallback(
 const AddArrayFunction &callback)
{
  this->AddArrayCallback = callback;
}

//----------------------------------------------------------------------------
void ProgrammableDataAdaptor::SetReleaseDataCallback(
  const ReleaseDataFunction &callback)
{
  this->ReleaseDataCallback = callback;
}

//----------------------------------------------------------------------------
int ProgrammableDataAdaptor::GetNumberOfMeshes(unsigned int &numMeshes)
{
  numMeshes = 0;

  if (!this->GetNumberOfMeshesCallback)
    {
    SENSEI_ERROR("No GetNumberOfMeshesCallback has been provided")
    return -1;
    }

  return this->GetNumberOfMeshesCallback(numMeshes);
}

//----------------------------------------------------------------------------
int ProgrammableDataAdaptor::GetMeshMetadata(unsigned int id,
  MeshMetadataPtr &metadata)
{
  if (!this->GetMeshMetadataCallback)
    {
    SENSEI_ERROR("No GetMeshMetadataCallback has been provided")
    return -1;
    }

  return this->GetMeshMetadataCallback(id, metadata);

}

//----------------------------------------------------------------------------
int ProgrammableDataAdaptor::GetMesh(const std::string &meshName,
  bool structureOnly, vtkDataObject *&mesh)
{
  mesh = nullptr;

  if (!this->GetMeshCallback)
    {
    SENSEI_ERROR("No GetMeshCallback has been provided")
    return -1;
    }

  return this->GetMeshCallback(meshName, structureOnly, mesh);
}

//----------------------------------------------------------------------------
int ProgrammableDataAdaptor::AddArray(vtkDataObject* mesh,
  const std::string &meshName, int association, const std::string &arrayName)
{
  if (!this->AddArrayCallback)
    {
    SENSEI_ERROR("No AddArrayCallback has been provided")
    return -1;
    }

  return this->AddArrayCallback(mesh, meshName, association, arrayName);
}

//----------------------------------------------------------------------------
int ProgrammableDataAdaptor::ReleaseData()
{
  if (this->ReleaseDataCallback)
    return this->ReleaseDataCallback();

  return -1;
}

}
