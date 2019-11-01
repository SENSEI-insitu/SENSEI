#ifndef MeshMetadataMap_h
#define MeshMetadataMap_h

#include "MeshMetadata.h"

#include <vector>
#include <map>
#include <string>

namespace sensei
{
class DataAdaptor;

// An associative container mapping mesh names to metadata
// and data object id
class MeshMetadataMap
{
public:
  // initialize the map by getting metadata for all of the
  // meshes provided by the simulation.
  int Initialize(DataAdaptor *da, MeshMetadataFlags flags = MeshMetadataFlags());

  void PushBack(MeshMetadataPtr &md);

  unsigned int Size() const { return this->Metadata.size(); }

  void Resize(unsigned int n){ this->Metadata.resize(n); }

  void Clear();

  // get the id of the object by name
  int GetMeshId(const std::string &name, unsigned int &id) const;

  // get the metadata associated with the named object
  int GetMeshMetadata(const std::string &name, MeshMetadataPtr &md);

  // set/get the i'th object's metadata
  int GetMeshMetadata(unsigned int i, MeshMetadataPtr &md);
  int SetMeshMetadata(unsigned int i, MeshMetadataPtr &md);

private:
  // a vector of metadata for each data object provided by
  // the simulation
  std::vector<MeshMetadataPtr> Metadata;

  // a mapping between mesh names and index into the the above
  // vector
  std::map<std::string, unsigned int> IdMap;
};

}

#endif
