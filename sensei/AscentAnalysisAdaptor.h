#ifndef sensei_AscentAnalysisAdaptor_h
#define sensei_AscentAnalysisAdaptor_h

#include "AnalysisAdaptor.h"
#include "DataRequirements.h"

#include <conduit.hpp>
#include <ascent.hpp>
#include <string>

namespace sensei
{

/// An analysis adaptor for ascent-based analysis pipelines.
class SENSEI_EXPORT AscentAnalysisAdaptor : public AnalysisAdaptor
{
public:
  /// Creates an AscentAnalysisAdaptor instance.
  static AscentAnalysisAdaptor* New();

  senseiTypeMacro(AscentAnalysisAdaptor, AnalysisAdaptor);

  /// @name Run time configuration
  /// @{

  /// Initialize the Ascent library using Ascent specific json configurations.
  int Initialize(const std::string &json_file_path,
    const std::string &options_file_path);

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

  /// @}

  /// Invoke in situ processing using Ascent
  bool Execute(DataAdaptor* data, DataAdaptor*&) override;

  /// Shut down and clean up the Ascent library.
  int Finalize() override;

protected:
  AscentAnalysisAdaptor();
  ~AscentAnalysisAdaptor();

  AscentAnalysisAdaptor(const AscentAnalysisAdaptor&) = delete;
  void operator=(const AscentAnalysisAdaptor&) = delete;

private:
  bool Execute_original(DataAdaptor* data);
  bool Execute_new(DataAdaptor* data);

  ascent::Ascent _ascent;
  conduit::Node optionsNode;    // Ascent options from json file.
  conduit::Node actionsNode;    // Ascent actions from json file.

  void GetFieldsFromActions();
  std::set<std::string> Fields;

  DataRequirements Requirements;
};

}

#endif
