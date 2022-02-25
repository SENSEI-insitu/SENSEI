#ifndef ADIOS1AnalysisAdaptor_h
#define ADIOS1AnalysisAdaptor_h

#include "AnalysisAdaptor.h"
#include "DataRequirements.h"
#include "MeshMetadata.h"

#include <vector>
#include <string>
#include <mpi.h>

namespace senseiADIOS1 { class DataObjectCollectionSchema; }
class svtkDataObject;
class svtkCompositeDataSet;

namespace sensei
{

/// The write side of the ADIOS 1 transport
class SENSEI_EXPORT ADIOS1AnalysisAdaptor : public AnalysisAdaptor
{
public:
  static ADIOS1AnalysisAdaptor* New();
  senseiTypeMacro(ADIOS1AnalysisAdaptor, AnalysisAdaptor);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /// Sets the maximum buffer allocated by ADIOS1 in MB
  /// takes affect on first Execute
  void SetMaxBufferSize(unsigned int size)
  { this->MaxBufferSize = size; }

  /// @brief Set the ADIOS1 method e.g. MPI, FLEXPATH etc.
  ///
  /// Default value is "MPI".
  void SetMethod(const std::string &method)
  { this->Method = method; }

  std::string GetMethod() const
  { return this->Method; }

  /// @brief Set the filename.
  ///
  /// Default value is "sensei.bp"
  void SetFileName(const std::string &filename)
  { this->FileName = filename; }

  std::string GetFileName() const
  { return this->FileName; }

  /// data requirements tell the adaptor what to push
  /// if none are given then all data is pushed.
  int SetDataRequirements(const DataRequirements &reqs);

  int AddDataRequirement(const std::string &meshName,
    int association, const std::vector<std::string> &arrays);

  // SENSEI AnalysisAdaptor API
  bool Execute(DataAdaptor* data, DataAdaptor*& result) override;
  int Finalize() override;

protected:
  ADIOS1AnalysisAdaptor();
  ~ADIOS1AnalysisAdaptor();

  // intializes ADIOS1 in no-xml mode, allocate buffers, and declares a group
  int InitializeADIOS1(const std::vector<MeshMetadataPtr> &metadata);

  // writes the data collection
  int WriteTimestep(unsigned long timeStep, double time,
    const std::vector<MeshMetadataPtr> &metadata,
    const std::vector<svtkCompositeDataSet*> &dobjects);

  // shuts down ADIOS1
  int FinalizeADIOS1();

  unsigned int MaxBufferSize;
  senseiADIOS1::DataObjectCollectionSchema *Schema;
  sensei::DataRequirements Requirements;
  std::string Method;
  std::string FileName;
  int64_t GroupHandle;

private:
  ADIOS1AnalysisAdaptor(const ADIOS1AnalysisAdaptor&) = delete;
  void operator=(const ADIOS1AnalysisAdaptor&) = delete;
};

}

#endif
