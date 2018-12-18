#ifndef sensei_InTransitDataAdaptor_h
#define sensei_InTransitDataAdaptor_h

#include "DataAdaptor.h"
#include <pugixml.hpp>

namespace sensei
{
/// @class InTransitDataAdaptor
/// @brief InTransitDataAdaptor is an abstract base class that defines the data interface.
///
/// InTransitDataAdaptor defines the data interface. Any simulation code that interfaces with
/// Sensei needs to provide an implementation for this interface. Analysis routines
/// (via AnalysisAdator) use the InTransitDataAdaptor implementation to access simulation data.
class InTransitDataAdaptor : public sensei::DataAdaptor
{
public:
  senseiBaseTypeMacro(InTransitDataAdaptor, sensei::DataAdaptor);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  // New API that enables run-time user based control of how data lands for
  // sensei::AnalysisAdaptor's which do not need explicit control. The specific
  // transport layers will implement this, and may support different options.
  // However, they will all support the 'partitioner' attribute and the following
  // partitioning modes:
  //
  //     block  The block distribution method will distribute blocks to a rank
  //            such that consecutive blocks share a rank.
  //
  //     cyclic The cyclic distribution method will distribute blocks to a rank
  //            such that consecutive blocks are distributed over consecutive
  //            ranks (in a round-robin fashion).
  //
  //     planar The  blocks are distributed in blocks of a specified size.
  //            The size is specified in the 'plane_size' attribute. Note
  //            block is a special case of planar with a plane_size of 1
  //
  //     mapped The mapped method of distribution will allocate blocks
  //            in-order as listed in a nested 'block_owner' and 'block_id'
  //            elements.  each entry in the block element has a
  //            corresponding entry in the proc element naming the mpi rank
  //            where the block lands
  //
  // Note, that these are core partitioning supported in SENSEI 3, specific
  // InTransitDataAdaptor instances are free to support other partitionings
  // but not required to do so.
  //
  // Illustrative example of the XML:
  //
  // <sensei>
  //   <data_adaptor type="adios_2" partitioner="block" ... >
  //     ...
  //   </data_adaptor>
  //   <analysis type="histogram" ... >
  //     ...
  //   </analysis>
  // </sensei>
  //
  // For more information on the 'analysis element' see sensei::ConfigurableAnalysis.
  // For more information on the 'data_adaptor' 'type' attribute see
  // sensei::InTransitAdaptorFactory
  virtual int Initialize(pugi::xml_node parent) = 0;

  // New API that enables one to peek at how the data is partitioned on the
  // simulation/remote side. Analyses that need control over how data lands
  // can use this to see what data is available, associated metadata such as
  // block bounds and array metadata and how it's laid out on the sender side.
  virtual int GetSenderMeshMetadata(unsigned int id, MeshMetadataPtr &metadata) = 0;

  // New API that enables one to specify how the data is partitioned on the
  // analysis/local side. Analyses that need control over how data lands
  // can use this to say where data lands. The metadata object passed here
  // will be returned to the Analysis, and the transport layer will use it
  // to move blocks onto the correct ranks. Care, should be taken as there
  // will be variablility in terms of what various transport layers support.
  // The requirement for SENSEI 3.0 is that blocks are elemental. In other
  // words given M ranks and P blocks on the sender/simulation side, a partitioning
  // with N ranks and P blocks on the receiver/analysis side is supported.
  // A transport may support more sophistocated partitioning, but it's not
  // required. An analysis need not use this API, in that case the default
  // is handled by the transport layer. See comments in InTransitDataAdaptor::Initialize
  // for the universal partioning options as well as comments in the specific
  // transport's implementation.
  virtual int SetReceiverMeshMetadata(unsigned int id, MeshMetadataPtr metadata) = 0;

  // Enables an analysis adaptor to programmatically select one of the default
  // partitioners.
  void SetPartitioner(const std::string &part);

  // Query the current partitioner
  enum {PARTITION_BLOCK, PARTITION_CYCLIC, PARTITION_PLANAR, PARTITION_MAPPED};
  int GetPartitioner();

  // New API that is called before the application is brought down
  virtual int Finalize() = 0;

  // Control API
  virtual int OpenStream() = 0;
  virtual int CloseStream() = 0;
  virtual int AdvanceStream() = 0;
  virtual int StreamGood() = 0;

  // Default partitioners
  int GetBlockPartition(sensei::MeshMetadataPtr &remote, sensei::MeshMetadataPtr &local);
  int GetCyclicPartition(sensei::MeshMetadataPtr &remote, sensei::MeshMetadataPtr &local);
  int GetPlanePartition(sensei::MeshMetadataPtr &remote, sensei::MeshMetadataPtr &local);
  int GetMappedPartition(sensei::MeshMetadataPtr &remote, sensei::MeshMetadataPtr &local);

protected:
  InTransitDataAdaptor();
  ~InTransitDataAdaptor();

  InTransitDataAdaptor(const InTransitDataAdaptor&) = delete;
  void operator=(const InTransitDataAdaptor&) = delete;

  struct InternalsType;
  InternalsType *Internals;
};

}
#endif
