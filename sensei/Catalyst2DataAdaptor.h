#ifndef CATALYST2_DATAADAPTOR_H
#define CATALYST2_DATAADAPTOR_H

#include <catalyst_conduit.hpp>
#include <vector>

#include "DataAdaptor.h"

namespace sensei
{

class Catalyst2DataAdaptor : public sensei::DataAdaptor
{
public:
  static Catalyst2DataAdaptor* New();
  senseiTypeMacro(Catalyst2DataAdaptor, sensei::DataAdaptor);
  void PrintSelf(ostream &os, vtkIndent indent) override;

  // Accessor
  // --------

  // reset the list of node to a single node
  void SetNode(const conduit_cpp::Node& node);
  // add a new node
  void AddNode(const conduit_cpp::Node& node);

  // DataAdaptor API

  int GetNumberOfMeshes(unsigned int &numMeshes) override;
  int GetMeshMetadata(unsigned int id, sensei::MeshMetadataPtr &metadata) override;
  int GetMesh(const std::string &meshName, bool structureOnly, vtkDataObject *&mesh) override;
  int AddArray(vtkDataObject* mesh, const std::string &meshName, int association, const std::string &arrayName) override;
  int ReleaseData() override;

protected:
  Catalyst2DataAdaptor();
  virtual ~Catalyst2DataAdaptor();

  std::vector<conduit_cpp::Node> Nodes;

private:
  Catalyst2DataAdaptor(const Catalyst2DataAdaptor&) = delete; // not implemented.
  void operator=(const Catalyst2DataAdaptor&) = delete; // not implemented.

};

} // namespace sensei

#endif // Catalyst2DataAdaptor_h_INCLUDED

