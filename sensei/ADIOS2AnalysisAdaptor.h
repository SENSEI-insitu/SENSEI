#ifndef ADIOS2AnalysisAdaptor_h
#define ADIOS2AnalysisAdaptor_h

#include "AnalysisAdaptor.h"
#include "DataRequirements.h"
#include "MeshMetadata.h"

#include <ADIOS2Schema.h>

#include <vector>
#include <string>
#include <mpi.h>

/// @cond
namespace senseiADIOS2
{
struct AdiosHandle;
class DataObjectCollectionSchema;
}

class svtkDataObject;
class svtkCompositeDataSet;

namespace pugi { class xml_node; }
/// @endcond

namespace sensei
{
/// The write side of the ADIOS2 transport
class SENSEI_EXPORT ADIOS2AnalysisAdaptor : public AnalysisAdaptor
{
public:
  /// constructs a new ADIOS2AnalysisAdaptor instance.
  static ADIOS2AnalysisAdaptor* New();

  senseiTypeMacro(ADIOS2AnalysisAdaptor, AnalysisAdaptor);

  /// @name runtime configuration
  /// @{

  /// initialize from an XML representation
  int Initialize(pugi::xml_node &parent);

  /// Add name value pairs to be passed to ADIOS
  void AddParameter(const std::string &key, const std::string &value);

  /// Set the ADIOS2 engine.
  void SetEngineName(const std::string &engineName)
  { this->EngineName = engineName; }

  /// Get the ADIOS2 engine.
  std::string GetEngineName() const
  { return this->EngineName; }

  /** Set the filename.  Default value is "sensei.bp" which is suitable for use
   * with streams or transport engines such as SST. When writing files to disk
   * using the BP4 engine one could SetStepsPerFile to prevent all steps being
   * accumulated in a single file. In this case one should also use a printf
   * like format specifier compatible with an int type in the file name. For
   * example "sensei_%04d.bp".
   */
  void SetFileName(const std::string &filename)
  { this->FileName = filename; }

  /// Returns the filename.
  std::string GetFileName() const
  { return this->FileName; }

  /** Set the number of time steps to store in each file.  The default value is
   * 0 which results in all the steps landing in a single file. If set to
   * non-zero then multiple files per run are created each with this number of
   * steps. An ordinal file index is incorporated in the file name. See notes
   * in SetFileName for details on specifying the format specifier.
   */
  void SetStepsPerFile(long steps)
  { this->StepsPerFile = steps; }


  /** Adds a set of sensei::DataRequirements, typically this will come from an XML
   * configuratiopn file. Data requirements tell the adaptor what to fetch from
   * the simulation and write to disk. If none are given then all available
   * data is fetched and written.
   */
  int SetDataRequirements(const DataRequirements &reqs);

  /** Add an indivudal data requirement. Data requirements tell the adaptor
   * what to fetch from the simulation and write to disk. If none are given
   * then all available data is fetched and written.

   * @param[in] meshName    the name of the mesh to fetch and write
   * @param[in] association the type of data array to fetch and write
   *                        vtkDataObject::POINT or vtkDataObject::CELL
   * @param[in] arrays      a list of arrays to fetch and write
   * @returns zero if successful.
   */
  int AddDataRequirement(const std::string &meshName,
    int association, const std::vector<std::string> &arrays);

  /** Controls how many calls to Execute do nothing between actual I/O and
   * streaming.
   */
  int SetFrequency(long frequency);

  /// @}

  /// Invokes ADIOS2 based I/O or streaming.
  bool Execute(DataAdaptor* data, DataAdaptor** result) override;

  /// Shut down and clean up including flushing and closing all streams and files.
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
    const std::vector<svtkCompositeDataSetPtr> &dobjects);

  // shuts down ADIOS2
  int FinalizeADIOS2();

  // fetch meshes and metadata objects from the simulation
  int FetchFromProducer(sensei::DataAdaptor *da,
    std::vector<svtkCompositeDataSetPtr> &objects,
    std::vector<MeshMetadataPtr> &metadata);

  senseiADIOS2::DataObjectCollectionSchema *Schema;
  sensei::DataRequirements Requirements;
  std::string EngineName;
  std::string FileName;
  senseiADIOS2::AdiosHandle Handles;
  adios2_adios *Adios;
  std::vector<std::pair<std::string,std::string>> Parameters;
  long StepsPerFile;
  long StepIndex;
  long FileIndex;
  long Frequency;

private:
  ADIOS2AnalysisAdaptor(const ADIOS2AnalysisAdaptor&) = delete;
  void operator=(const ADIOS2AnalysisAdaptor&) = delete;
};

}

#endif
