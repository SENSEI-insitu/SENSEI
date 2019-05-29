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

/// The write side of the ADIOS 1 transport
class libISAnalysisAdaptor : public AnalysisAdaptor
{
public:
  static libISAnalysisAdaptor* New();
  senseiTypeMacro(libISAnalysisAdaptor, AnalysisAdaptor);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  /// Sets the maximum buffer allocated by libIS in MB
  /// takes affect on first Execute
  //void SetMaxBufferSize(unsigned int size)
  //{ this->MaxBufferSize = size; }

  /// @brief Set the libIS method e.g. MPI, FLEXPATH etc.
  ///
  /// Default value is "MPI".
  //void SetMethod(const std::string &method)
  //{ this->Method = method; }

  //std::string GetMethod() const
  //{ return this->Method; }

  /// @brief Set the filename.
  ///
  /// Default value is "sensei.bp"
  //void SetFileName(const std::string &filename)
  //{ this->FileName = filename; }

  //std::string GetFileName() const
  //{ return this->FileName; }

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

  // intializes libIS in no-xml mode, allocate buffers, and declares a group
  int InitializelibIS(const std::vector<MeshMetadataPtr> &metadata);

  // writes the data collection
  int WriteTimestep(unsigned long timeStep, double time,
    const std::vector<MeshMetadataPtr> &metadata,
    const std::vector<vtkCompositeDataSet*> &dobjects);

  // shuts down libIS
  int FinalizelibIS();

  unsigned int MaxBufferSize;
  senseilibIS::DataObjectCollectionSchema *Schema;
  sensei::DataRequirements Requirements;
  std::string Method;
  std::string FileName;
  int64_t GroupHandle;

private:
  libISAnalysisAdaptor(const libISAnalysisAdaptor&) = delete;
  void operator=(const libISAnalysisAdaptor&) = delete;
};

}


