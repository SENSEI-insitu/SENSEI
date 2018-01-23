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
void ProgrammableDataAdaptor::SetGetMeshCallback(
  const GetMeshCallbackType &callback)
{
  this->GetMeshCallback = callback;
}

//----------------------------------------------------------------------------
void ProgrammableDataAdaptor::SetAddArrayCallback(
 const AddArrayCallbackType &callback)
{
  this->AddArrayCallback = callback;
}

//----------------------------------------------------------------------------
void ProgrammableDataAdaptor::SetGetNumberOfArraysCallback(
  const GetNumberOfArraysCallbackType &callback)
{
  this->GetNumberOfArraysCallback = callback;
}

//----------------------------------------------------------------------------
void ProgrammableDataAdaptor::SetGetArrayNameCallback(
  const GetArrayNameCallbackType &callback)
{
  this->GetArrayNameCallback = callback;
}

//----------------------------------------------------------------------------
void ProgrammableDataAdaptor::SetReleaseDataCallback(
  const ReleaseDataCallbackType &callback)
{
  this->ReleaseDataCallback = callback;
}

//----------------------------------------------------------------------------
vtkDataObject *ProgrammableDataAdaptor::GetMesh(bool structure_only)
{
  if (!this->GetMeshCallback)
    {
    SENSEI_ERROR("No GetMeshCallback has been provided")
    return nullptr;
    }

  return this->GetMeshCallback(structure_only);
}

//----------------------------------------------------------------------------
bool ProgrammableDataAdaptor::AddArray(vtkDataObject* mesh, int association,
  const std::string& arrayname)
{
  if (!this->AddArrayCallback)
    return true;

  return this->AddArrayCallback(mesh, association, arrayname);
}

//----------------------------------------------------------------------------
unsigned int ProgrammableDataAdaptor::GetNumberOfArrays(int association)
{
  if (!this->GetNumberOfArraysCallback)
    return 0;

  return this->GetNumberOfArraysCallback(association);
}

//----------------------------------------------------------------------------
std::string ProgrammableDataAdaptor::GetArrayName(int association,
  unsigned int index)
{
  if (!this->GetArrayNameCallback)
    return "";

  return this->GetArrayNameCallback(association, index);
}

//----------------------------------------------------------------------------
void ProgrammableDataAdaptor::ReleaseData()
{
  if (this->ReleaseDataCallback)
    this->ReleaseDataCallback();
}

}
