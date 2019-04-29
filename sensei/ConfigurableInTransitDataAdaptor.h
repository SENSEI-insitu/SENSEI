#ifndef sensei_ConfigurableInTransitDataAdaptor_h
#define sensei_ConfigurableInTransitDataAdaptor_h

#include "InTransitDataAdaptor.h"

#include "senseiConfig.h"
#include "vtkObjectBase.h"

#include <vector>
#include <string>
#include <memory>

class vtkDataObject;

namespace sensei
{

// @class ConfigurableInTransitDataAdaptor
// The ConfigurableInTransitDataAdaptor implements the InTransitDataAdaptor
// inteface, provides a factory for creating a concrete instance of the
// InTransitDataAdpator from an XML configuration, and delagtes in coming calls
// through the InTransitDataAdapotor/DataAdaptor API to the instance.  The
// puprose of this class is to provide run-time configurablility of the
// concrete InTransitDataAdaptors.
//
// @section XML
// Configurartion shold be placed in an element of type `transport`, the
// `type` attribute names the concrete class to create and initialize.
// Each concrete class will have a number of attributes used for configuration.
//
// The supported transport types are:
//
//   adios_1, adios_2, hdf5, libis
//
// Illustrative example of the XML:
//
// <sensei>
//   <transport type="adios_1" file_name="test.bp" read_method="FLEXPATH">
//     <paritioner type="block"/>
//   </transport>
// <sensei>
//
class ConfigurableInTransitDataAdaptor : public sensei::InTransitDataAdaptor
{
public:
  static ConfigurableInTransitDataAdaptor *New();
  senseiTypeMacro(ConfigurableInTransitDataAdaptor, InTransitDataAdaptor);

  int Initialize(const std::string &fileName);

  // sensei::InTransitDataAdaptor API
  int Initialize(pugi::xml_node &node) override;

  int GetSenderMeshMetadata(unsigned int id,
    MeshMetadataPtr &metadata) override;

  int GetReceiverMeshMetadata(unsigned int id,
    MeshMetadataPtr &metadata) override;

  int SetReceiverMeshMetadata(unsigned int id,
     MeshMetadataPtr &metadata) override;

  void SetPartitioner(sensei::PartitionerPtr &partitioner) override;
  sensei::PartitionerPtr GetPartitioner() override;

  int OpenStream() override;
  int CloseStream() override;
  int AdvanceStream() override;
  int StreamGood() override;
  int Finalize() override;

  // sensei::DataAdaptor API
  int GetNumberOfMeshes(unsigned int &numMeshes) override;
  int GetMeshMetadata(unsigned int id, MeshMetadataPtr &metadata) override;

  int GetMesh(const std::string &meshName,
    bool structureOnly, vtkDataObject *&mesh) override;

  int GetMesh(const std::string &meshName,
    bool structureOnly, vtkCompositeDataSet *&mesh) override;

  int AddGhostNodesArray(vtkDataObject* mesh,
    const std::string &meshName) override;

  int AddGhostCellsArray(vtkDataObject* mesh,
    const std::string &meshName) override;

  int AddArray(vtkDataObject* mesh, const std::string &meshName,
    int association, const std::string &arrayName) override;

  int AddArrays(vtkDataObject* mesh, const std::string &meshName,
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
