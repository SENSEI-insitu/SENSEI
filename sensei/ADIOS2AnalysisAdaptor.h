#ifndef ADIOS2AnalysisAdaptor_h
#define ADIOS2AnalysisAdaptor_h

#include "AnalysisAdaptor.h"
#include "DataRequirements.h"
#include "MeshMetadata.h"

#include <ADIOS2Schema.h>

#include <vector>
#include <string>
#include <mpi.h>

namespace senseiADIOS2
{
struct AdiosHandle;
class DataObjectCollectionSchema;
}

class vtkDataObject;
class vtkCompositeDataSet;

namespace pugi { class xml_node; }

namespace sensei
{
/// The write side of the ADIOS 2 transport
class ADIOS2AnalysisAdaptor : public AnalysisAdaptor
{
public:
  static ADIOS2AnalysisAdaptor* New();
  senseiTypeMacro(ADIOS2AnalysisAdaptor, AnalysisAdaptor);

  /// initialize from an XML representation
  int Initialize(pugi::xml_node &parent);

  /// Add name value pairs to be passed to ADIOS
  void AddParameter(const std::string &key, const std::string &value);

  /// @brief Set the ADIOS2 engine
  void SetEngineName(const std::string &engineName)
  { this->EngineName = engineName; }

  std::string GetEngineName() const
  { return this->EngineName; }

  /// @brief Set the filename.
  /// Default value is "sensei.bp" which is suitable for use with streams or
  /// transport engines such as SST. When writing files to disk using the BP4
  /// engine one could SetStepsPerFile to prevent all steps being accumulated in
  /// a single file. In this case one should also use a printf like format
  /// specifier compatible with an int type in the file name. For example
  /// "sensei_%04d.bp".
  void SetFileName(const std::string &filename)
  { this->FileName = filename; }

  std::string GetFileName() const
  { return this->FileName; }

  /// @brief Set the number of time steps to store in each file.  The default
  /// value is 0 which results in all the steps landing in a single file. If set
  /// to non-zero then multiple files per run are created each with this number
  /// of steps. An ordinal file index is incorporated in the file name. See
  /// notes in SetFileName for details on specifying the format specifier.
  void SetStepsPerFile(long steps)
  { this->StepsPerFile = steps; }

  /// @brief Enable/disable debugging output
  /// Default value is 0
  void SetDebugMode(int mode)
  { this->DebugMode = mode; }

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

  // intializes ADIOS2 in no-xml mode
  int InitializeADIOS2();

  // tells ADIOS what we will write
  int DefineVariables(const std::vector<MeshMetadataPtr> &metadata);

  // initializes the output stream, and in the case of writing
  // file series advances to the next file in the series.
  int UpdateStream();

  // writes the data collection
  int WriteTimestep(unsigned long timeStep, double time,
    const std::vector<MeshMetadataPtr> &metadata,
    const std::vector<vtkCompositeDataSet*> &dobjects);

  // shuts down ADIOS2
  int FinalizeADIOS2();

  // fetch meshes and metadata objects from the simulation
  int FetchFromProducer(sensei::DataAdaptor *da,
    std::vector<vtkCompositeDataSet*> &objects,
    std::vector<MeshMetadataPtr> &metadata);

  senseiADIOS2::DataObjectCollectionSchema *Schema;
  sensei::DataRequirements Requirements;
  std::string EngineName;
  std::string FileName;
  senseiADIOS2::AdiosHandle Handles;
  adios2_adios *Adios;
  std::vector<std::pair<std::string,std::string>> Parameters;
  int DebugMode;
  long StepsPerFile;
  long StepIndex;
  long FileIndex;

private:
  ADIOS2AnalysisAdaptor(const ADIOS2AnalysisAdaptor&) = delete;
  void operator=(const ADIOS2AnalysisAdaptor&) = delete;
};

}

#endif
