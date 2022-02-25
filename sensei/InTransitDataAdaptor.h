#ifndef sensei_InTransitDataAdaptor_h
#define sensei_InTransitDataAdaptor_h

#include "DataAdaptor.h"
#include "Partitioner.h"

/// @cond
namespace pugi { class xml_node; }
/// @endcond

namespace sensei
{

/** Defines the control API for in transit data movement.  The
 * InTransitDataAdaptor layers a control API onto the sensei::DataAdaptor API.
 * In what follows the simulation is the sender of data and the end point and
 * or sensei::AnalysisAdaptor is the receiver of data. The InTransitDataAdaptor
 * control API gives end point control over how data lands. A data receiver may
 * epxlicitly specifiy how data lands (see SetReceiverMeshMetadata) or use one
 * of a number of common paritioning strategies (see sensei::Partitioner and
 * derived classes). Typically by an AnalysisAdaptor which needs explicit
 * control over how data is partitioned will use SetReceiverMeshMetadata. When
 * no receiver MeshMetadata has been provided a sensei::Partitioner is used.
 * The partioner may be specified in XML, and if it is not, then the default is
 * sensei::BlockPartitioner.
 */
class SENSEI_EXPORT InTransitDataAdaptor : public sensei::DataAdaptor
{
public:
  senseiBaseTypeMacro(InTransitDataAdaptor, sensei::DataAdaptor);

  /** Pass in a string containing transport specific connection information.
   * This is optional, as XML may be used to specify connection as well.
   * When used the details will be specific to the transport, for instance
   * ADIOS uses a file to negotiate the connection, hence for ADIOS
   * connection info will be a path to that file.
   */
  virtual int SetConnectionInfo(const std::string &info);

  /// Return the current connection info.
  virtual const std::string &GetConnectionInfo() const;

  /** Initialize the adaptor from an XML node. The default implementation
   * handles initializing a sensei::ConfigurablePartitioner. If the
   * ConfigurablePartitioner fails to initialize, then a we fall back to a
   * default initialized sensei::BlockPartitioner.
   */
  virtual int Initialize(pugi::xml_node &node);

  /// Get metadta object describing the data that is available in the simulation.
  virtual int GetSenderMeshMetadata(unsigned int id, MeshMetadataPtr &metadata) = 0;

  /** This API that enables one to specify how the data is partitioned on the
   * analysis/local side. Analyses that need control over how data lands can
   * use this to say where data lands. The metadata object passed here will be
   * returned to the Analysis, and the transport layer will use it to move
   * blocks onto the correct ranks. Care, should be taken as there will be
   * variablility in terms of what various transport layers support.  The
   * requirement for SENSEI 3.0 is that blocks are elemental. In other words
   * given M ranks and P blocks on the sender/simulation side, a partitioning
   * with N ranks and P blocks on the receiver/analysis side is supported.  A
   * transport may support more sophistocated partitioning, but it's not
   * required. An analysis need not use this API, in that case the default is
   * handled by the transport layer. See comments in
   * InTransitDataAdaptor::Initialize for the universal partioning options as
   * well as comments in the specific transport's implementation.
   *
   * The default implementation manages the metadata objects, derived classes
   * must handle the details of initiallizing these objects. Get calls will
   * return -1 if no object has been set for a given id.
   */
  virtual int SetReceiverMeshMetadata(unsigned int id, MeshMetadataPtr &metadata);

  /// Returns the current receiver mesh metadata.
  virtual int GetReceiverMeshMetadata(unsigned int id, MeshMetadataPtr &metadata);

  /**  Set/get the partitioner. The partitioner is used when no receiver mesh
   *  metadata has been set. The Initialize method will initialize an instance
   *  of a ConfigurablePartitioner using user provided XML, if that fails will
   *  fall back to a default initialized instance of BlockPartitioner.
   */
  virtual void SetPartitioner(const sensei::PartitionerPtr &partitioner);

  /// Return the current partitioner.
  virtual sensei::PartitionerPtr GetPartitioner();

  /// Opens a stream and connects to the simulation.
  virtual int OpenStream() = 0;

  /// Closes a stream and disconnects from the simulation.
  virtual int CloseStream() = 0;

  /// Signals the that we are finished with this time step.
  virtual int AdvanceStream() = 0;

  /// Returns true while there is more data to process.
  virtual int StreamGood() = 0;

  /// Called before the application is brought down. It is safe to use MPI here.
  virtual int Finalize() = 0;

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
