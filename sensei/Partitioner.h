#ifndef sensei_Partitioner_h
#define sensei_Partitioner_h

#include "MeshMetadata.h"
#include "Error.h"

#include <memory>
#include <mpi.h>

namespace pugi { class xml_node; }

namespace sensei
{
class Partitioner;
using PartitionerPtr = std::shared_ptr<sensei::Partitioner>;

/// @class Partitioner
/// @brief represents the way data is partitioned for in-transit operation mode.
///
/// given a collection of data and set of ranks, partition  the data to a new
/// set of ranks. the collection of data and its assigmment to ranks is
/// described by MeshMetadata. The partitioner is not responsible for moving
/// data, only deciding how it should be layed out and distributed after the
/// move.
class SENSEI_EXPORT Partitioner
{
public:
  // return the name of the class
  virtual const char *GetClassName() = 0;

  // given an existing partitioning of data passed in the first MeshMetadata
  // argument,return a new partittioning in the second MeshMetadata argument.
  virtual int GetPartition(MPI_Comm comm, const sensei::MeshMetadataPtr &in,
    sensei::MeshMetadataPtr &out) = 0;

  // initialize the partitioner from the XML node.
  virtual int Initialize(pugi::xml_node &)
  {
      SENSEI_STATUS("Configured " << this->GetClassName())
      return 0;
  }

  // enable/disable generation of debugging output
  virtual void SetVerbose(int val){ this->Verbose = val; }
  virtual int GetVerbose(){ return this->Verbose; }

  Partitioner() : Verbose(0) {}
  virtual ~Partitioner() {}

protected:
  int Verbose;
};

}

#endif
