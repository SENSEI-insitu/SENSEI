#ifndef CONDUIT_DATAADAPTOR_H
#define CONDUIT_DATAADAPTOR_H

#include <vector>
#include <vtkDataArray.h>
#include <conduit.hpp>

#include "DataAdaptor.h"


namespace sensei
{

class ConduitDataAdaptor : public sensei::DataAdaptor
{
public:
  static ConduitDataAdaptor* New();
  senseiTypeMacro(ConduitDataAdaptor, sensei::DataAdaptor);
  void PrintSelf(ostream &os, vtkIndent indent) override;

  void SetNode(conduit::Node* node);
  void UpdateFields();

  // SENSEI DataAdaptor API.
  int GetNumberOfMeshes(unsigned int &numMeshes) override;

  int GetMeshMetadata(unsigned int id, sensei::MeshMetadataPtr &metadata) override;

  int GetMesh(const std::string &meshName, bool structureOnly, vtkDataObject *&mesh) override;

  //int AddGhostNodesArray(vtkDataObject* mesh, const std::string &meshName) override;
  //int AddGhostCellsArray(vtkDataObject* mesh, const std::string &meshName) override;

  int AddArray(vtkDataObject* mesh, const std::string &meshName, int association, const std::string &arrayName) override;

  int ReleaseData() override;

protected:
  ConduitDataAdaptor();
  ~ConduitDataAdaptor();

  typedef std::map<std::string, std::vector<std::string>> Fields;
  Fields FieldNames;
  int *GlobalBlockDistribution;

private:
  ConduitDataAdaptor(const ConduitDataAdaptor&) = delete; // not implemented.
  void operator=(const ConduitDataAdaptor&) = delete; // not implemented.

  conduit::Node* Node;
};

} // namespace sensei

#endif
