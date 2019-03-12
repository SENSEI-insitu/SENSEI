#ifndef sensei_ConfigurablePartitioner_h
#define sensei_ConfigurablePartitioner_h

#include "Partitioner.h"

namespace pugi { class xml_node; }

namespace sensei
{

class ConfigurablePartitioner : public Partitioner
{
public:
  ConfigurablePartitioner();
  ~ConfigurablePartitioner();

  // given an existing partitioning of data passed in the first MeshMetadata
  // argument,return a new partittioning in the second MeshMetadata argument.
  int GetPartition(MPI_Comm comm, const MeshMetadataPtr &in,
    MeshMetadataPtr &out) override;

  // initialize the partitioner from the XML node.  recognizes the following
  // Partitioner's: block, cyclic, planar, and mapped. The XML schema is as
  // follows:
  //
  // <partitioner type="..." ... >
  //   ...
  // </partitioner>
  //
  // where type is one of block, cyclic, planar, or mapped. See Parititioner
  // sub-classes for documentation on the specific XML recognized by each.
  virtual int Initialize(pugi::xml_node &) override;

private:
  struct InternalsType;
  InternalsType *Internals;
};

}

#endif
