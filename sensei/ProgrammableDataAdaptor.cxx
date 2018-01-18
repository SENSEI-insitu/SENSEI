#include "ProgrammableDataAdaptor.h"

#include "senseiConfig.h"
#include "Error.h"

#include <vtkObjectFactory.h>

namespace sensei
{
//-----------------------------------------------------------------------------
senseiNewMacro(ProgrammableDataAdaptor);

//----------------------------------------------------------------------------
ProgrammableDataAdaptor::ProgrammableDataAdaptor() :
 GetMeshCallback(nullptr), AddArrayCallback(nullptr),
 GetNumberOfArraysCallback(nullptr), GetArrayNameCallback(nullptr),
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
void ProgrammableDataAdaptor::SetGetMeshNameCallback(
  const GetMeshNameFunction &callback)
{
  this->GetMeshNameCallback = callback;
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
void ProgrammableDataAdaptor::SetGetNumberOfArraysCallback(
  const GetNumberOfArraysFunction &callback)
{
  this->GetNumberOfArraysCallback = callback;
}

//----------------------------------------------------------------------------
void ProgrammableDataAdaptor::SetGetArrayNameCallback(
  const GetArrayNameFunction &callback)
{
  this->GetArrayNameCallback = callback;
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
int ProgrammableDataAdaptor::GetMeshName(unsigned int id, std::string &meshName)
{
  meshName.clear();

  if (!this->GetMeshNameCallback)
    {
    SENSEI_ERROR("No GetMeshNameCallback has been provided")
    return -1;
    }

  return this->GetMeshNameCallback(id, meshName);

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
int ProgrammableDataAdaptor::GetNumberOfArrays(const std::string &meshName,
  int association, unsigned int &numberOfArrays)
{
  numberOfArrays = 0;

  if (!this->GetNumberOfArraysCallback)
    {
    SENSEI_ERROR("No GetNumberOfArraysCallback has been provided")
    return -1;
    }

  return this->GetNumberOfArraysCallback(meshName,
    association, numberOfArrays);
}

//----------------------------------------------------------------------------
int ProgrammableDataAdaptor::GetArrayName(const std::string &meshName,
  int association, unsigned int index, std::string &arrayName)

{
  arrayName.clear();

  if (!this->GetArrayNameCallback)
    {
    SENSEI_ERROR("No GetArrayNameCallback has been provided")
    return -1;
    }

  return this->GetArrayNameCallback(meshName, association, index, arrayName);
}

//----------------------------------------------------------------------------
int ProgrammableDataAdaptor::ReleaseData()
{
  if (this->ReleaseDataCallback)
    return this->ReleaseDataCallback();

  return -1;
}

}
