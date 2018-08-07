#ifndef CONDUIT_DATAADAPTOR_H
#define CONDUIT_DATAADAPTOR_H

#include <vector>
#include <vtkDataArray.h>
#include <vtkSmartPointer.h>
#include <vtkMultiBlockDataSet.h>
#include <sensei/DataAdaptor.h>
#include <conduit.hpp>


namespace sensei
{

class ConduitDataAdaptor : public sensei::DataAdaptor
{
public:
  static ConduitDataAdaptor* New();
  senseiTypeMacro(ConduitDataAdaptor, sensei::DataAdaptor);

  /// @brief Initialize the data adaptor.
  ///
  /// This initializes the data adaptor. This must be called once per simulation run.
  /// @param node contains the current data that will be visualized. 
  void Initialize(conduit::Node* node);

  void SetNode(conduit::Node* node);

  int GetMeshName(unsigned int id, std::string &meshName) override;
  
  int GetNumberOfMeshes(unsigned int &numMeshes) override;

  int GetMesh(const std::string &meshName, 
              bool structureOnly, 
              vtkDataObject* &mesh) override;
  int AddArray(vtkDataObject* mesh, 
               const std::string &meshName, 
               int association, 
               const std::string& arrayname) override;
  int GetNumberOfArrays(const std::string &meshName, 
                        int association, 
                        unsigned int &numberOfArrays) override;
  int GetArrayName(const std::string &meshName, 
                   int association, 
                   unsigned int index,
                   std::string &arrayName) override;
  int ReleaseData() override;

  void UpdateFields();

protected:
  ConduitDataAdaptor();
  ~ConduitDataAdaptor();
  typedef std::map<std::string, std::vector<std::string>> Fields;
  Fields FieldNames;
  int *GlobalBlockDistribution;

private:
  ConduitDataAdaptor(const ConduitDataAdaptor&)=delete; // not implemented.
  void operator=(const ConduitDataAdaptor&)=delete; // not implemented.
  conduit::Node* Node;
};

}
#endif
