#include "MeshMetadataMap.h"
#include "DataAdaptor.h"

namespace sensei
{

// --------------------------------------------------------------------------
int MeshMetadataMap::Initialize(DataAdaptor *da, MeshMetadataFlags flags)
{
  this->Clear();

  unsigned int nMeshes = 0;
  if (da->GetNumberOfMeshes(nMeshes))
    {
    SENSEI_ERROR("Failed to get the number of meshes")
    return -1;
    }

  this->Metadata.resize(nMeshes);

  for (unsigned int i = 0; i < nMeshes; ++i)
    {
    MeshMetadataPtr md = MeshMetadata::New();
    md->Flags = flags;

    if (da->GetMeshMetadata(i, md))
      {
      SENSEI_ERROR("Failed to get metadata for data object " << i)
      return -1;
      }

    if (md->Validate(da->GetCommunicator(), flags))
      {
      SENSEI_ERROR("The requested metadata was not provided for data object " << i)
      return -1;
      }

    this->Metadata[i] = md;
    this->IdMap[md->MeshName] = i;
    }

  return 0;
}

// --------------------------------------------------------------------------
int MeshMetadataMap::SetMeshMetadata(unsigned int id, MeshMetadataPtr &md)
{
  unsigned int n = this->Metadata.size();
  if (id >= n)
    {
    SENSEI_ERROR("Id " << id << " is out of bounds in map of size " << n)
    return -1;
    }

  this->Metadata[id] = md;
  this->IdMap[md->MeshName] = id;
  return 0;
}

// --------------------------------------------------------------------------
void MeshMetadataMap::PushBack(MeshMetadataPtr &md)
{
  unsigned int id = this->Metadata.size();
  this->IdMap[md->MeshName] = id;

  this->Metadata.push_back(md);
}

// --------------------------------------------------------------------------
void MeshMetadataMap::Clear()
{
  this->Metadata.clear();
  this->IdMap.clear();
}

// --------------------------------------------------------------------------
int MeshMetadataMap::GetMeshId(const std::string &name, unsigned int &id) const
{
  std::map<std::string, unsigned int>::const_iterator it = this->IdMap.find(name);
  if (it == this->IdMap.end())
    {
    SENSEI_ERROR("No mesh named \"" << name << "\" in map")
    return -1;
    }

  id = it->second;
  return 0;
}

// --------------------------------------------------------------------------
int MeshMetadataMap::GetMeshMetadata(const std::string &name, MeshMetadataPtr &md)
{
  unsigned int id = 0;
  if (this->GetMeshId(name, id))
    return -1;

  md = this->Metadata[id];
  return 0;
}

// --------------------------------------------------------------------------
int MeshMetadataMap::GetMeshMetadata(unsigned int id, MeshMetadataPtr &md)
{
  unsigned int n = this->Metadata.size();
  if (id >= n)
    {
    SENSEI_ERROR("Id " << id << " is out of bounds in map of size " << n)
    return -1;
    }

  md = this->Metadata[id];
  return 0;
}

}
