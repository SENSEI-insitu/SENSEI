#ifndef ADIOS2AnalysisAdaptor_h
#define ADIOS2AnalysisAdaptor_h

#include "AnalysisAdaptor.h"
#include "DataRequirements.h"
#include "MeshMetadata.h"

#include <ADIOS2Schema.h>

#include <vector>
#include <string>
#include <mpi.h>

namespace senseiADIOS2 { class AdiosHandle; class DataObjectCollectionSchema; }
class vtkDataObject;
class vtkCompositeDataSet;

namespace sensei
{
/// The write side of the ADIOS 2 transport
class ADIOS2AnalysisAdaptor : public AnalysisAdaptor
{
public:
  static ADIOS2AnalysisAdaptor* New();
  senseiTypeMacro(ADIOS2AnalysisAdaptor, AnalysisAdaptor);
  void PrintSelf(ostream& os, vtkIndent indent) override;


  /// @brief Set the ADIOS2 engine
  void SetEngineName(const std::string &engineName)
  { this->EngineName = engineName; }

  std::string GetEngineName() const
  { return this->EngineName; }

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
  bool Execute(DataAdaptor* data) override;
  int Finalize() override;

protected:
  ADIOS2AnalysisAdaptor();
  ~ADIOS2AnalysisAdaptor();

  // intializes ADIOS2 in no-xml mode, allocate buffers, and declares a group
  int InitializeADIOS2(const std::vector<MeshMetadataPtr> &metadata);

  // writes the data collection
  int WriteTimestep(unsigned long timeStep, double time,
    const std::vector<MeshMetadataPtr> &metadata,
    const std::vector<vtkCompositeDataSet*> &dobjects);

  // shuts down ADIOS2
  int FinalizeADIOS2();

  senseiADIOS2::DataObjectCollectionSchema *Schema;
  sensei::DataRequirements Requirements;
  std::string EngineName;
  std::string FileName;
  senseiADIOS2::AdiosHandle Handles;
  adios2_adios *Adios;

private:
  ADIOS2AnalysisAdaptor(const ADIOS2AnalysisAdaptor&) = delete;
  void operator=(const ADIOS2AnalysisAdaptor&) = delete;
};

}

#endif
