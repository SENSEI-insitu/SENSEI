#ifndef sensei_ConfigurableInTransitDataAdaptor_h
#define sensei_ConfigurableInTransitDataAdaptor_h

#include "InTransitDataAdaptor.h"

#include "senseiConfig.h"
#include "svtkObjectBase.h"

#include <vector>
#include <string>
#include <memory>

class svtkDataObject;

namespace sensei
{

/** The ConfigurableInTransitDataAdaptor implements the InTransitDataAdaptor
 * interface, provides a factory for creating a concrete instance of the
 * InTransitDataAdpator from an XML configuration, and delegates in coming calls
 * through the InTransitDataAdapotor/DataAdaptor API to the instance.  The
 * purpose of this class is to provide run-time configurability of the
 * concrete InTransitDataAdaptors.
 *
 * Configurartion shold be placed in an element of type `transport`, the
 * `type` attribute names the concrete class to create and initialize.
 * Each concrete class will have a number of attributes used for configuration.
 *
 * The supported transport types are:
 *
 *   adios_1, adios_2, hdf5, libis
 *
 * Illustrative example of the XML:
 *
 * ```xml
 * <sensei>
 *   <transport type="adios_1" file_name="test.bp" read_method="FLEXPATH">
 *     <paritioner type="block"/>
 *   </transport>
 * <sensei>
 * ```
 */
class SENSEI_EXPORT ConfigurableInTransitDataAdaptor : public sensei::InTransitDataAdaptor
{
public:
  static ConfigurableInTransitDataAdaptor *New();
  senseiTypeMacro(ConfigurableInTransitDataAdaptor, InTransitDataAdaptor);

  int Initialize(const std::string &fileName);

  int SetConnectionInfo(const std::string &info) override;
  const std::string &GetConnectionInfo() const override;

  int Initialize(pugi::xml_node &node) override;

  int GetSenderMeshMetadata(unsigned int id,
    MeshMetadataPtr &metadata) override;

  int GetReceiverMeshMetadata(unsigned int id,
    MeshMetadataPtr &metadata) override;

  int SetReceiverMeshMetadata(unsigned int id,
     MeshMetadataPtr &metadata) override;

  void SetPartitioner(const sensei::PartitionerPtr &partitioner) override;
  sensei::PartitionerPtr GetPartitioner() override;

  int OpenStream() override;
  int CloseStream() override;
  int AdvanceStream() override;
  int StreamGood() override;
  int Finalize() override;

  int GetNumberOfMeshes(unsigned int &numMeshes) override;
  int GetMeshMetadata(unsigned int id, MeshMetadataPtr &metadata) override;

  int GetMesh(const std::string &meshName,
    bool structureOnly, svtkDataObject *&mesh) override;

  int AddGhostNodesArray(svtkDataObject* mesh,
    const std::string &meshName) override;

  int AddGhostCellsArray(svtkDataObject* mesh,
    const std::string &meshName) override;

  int AddArray(svtkDataObject* mesh, const std::string &meshName,
    int association, const std::string &arrayName) override;

  int AddArrays(svtkDataObject* mesh, const std::string &meshName,
    int association, const std::vector<std::string> &arrayName) override;

  int ReleaseData() override;
  double GetDataTime() override;
  void SetDataTime(double time) override;
  long GetDataTimeStep() override;
  void SetDataTimeStep(long index) override;

protected:
  ConfigurableInTransitDataAdaptor();
  ~ConfigurableInTransitDataAdaptor();

  ConfigurableInTransitDataAdaptor(const ConfigurableInTransitDataAdaptor&) = delete;
  void operator=(const ConfigurableInTransitDataAdaptor&) = delete;

  struct InternalsType;
  InternalsType *Internals;
};

}
#endif
