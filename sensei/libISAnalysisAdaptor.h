#pragma once

#include "AnalysisAdaptor.h"
#include "DataRequirements.h"
#include "MeshMetadata.h"

#include <vector>
#include <string>
#include <mpi.h>

namespace senseilibIS { class DataObjectCollectionSchema; }
class vtkDataObject;
class vtkCompositeDataSet;

namespace sensei
{

/// The write side of the libIS transport
class libISAnalysisAdaptor : public AnalysisAdaptor
{
public:
  static libISAnalysisAdaptor* New();
  senseiTypeMacro(libISAnalysisAdaptor, AnalysisAdaptor);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  /// @brief Set the TCP port.
  ///
  void SetPort(int value)
  { this->port = value; }
 
  /// data requirements tell the adaptor what to push
  /// if none are given then all data is pushed.
  int SetDataRequirements(const DataRequirements &reqs);

  int AddDataRequirement(const std::string &meshName,
    int association, const std::vector<std::string> &arrays);

  // SENSEI AnalysisAdaptor API
  bool Execute(DataAdaptor* data) override;
  int Finalize() override;

protected:
  libISAnalysisAdaptor();
  ~libISAnalysisAdaptor();

  // intializes libIS
  // fixme: may not need metadata
  int InitializelibIS(const std::vector<MeshMetadataPtr> &metadata);

  // writes the data collection
  int WriteTimestep(unsigned long timeStep, double time,
    const std::vector<MeshMetadataPtr> &metadata,
    const std::vector<vtkCompositeDataSet*> &dobjects);

  // shuts down libIS
  int FinalizelibIS();

  senseilibIS::DataObjectCollectionSchema *Schema;
  sensei::DataRequirements Requirements;
  int port;  
  int64_t GroupHandle;

private:
  libISAnalysisAdaptor(const libISAnalysisAdaptor&) = delete;
  void operator=(const libISAnalysisAdaptor&) = delete;
};

}
