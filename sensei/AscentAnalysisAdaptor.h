#ifndef sensei_AscentAnalysisAdaptor_h
#define sensei_AscentAnalysisAdaptor_h

#include "AnalysisAdaptor.h"
#include "DataRequirements.h"

#include <conduit.hpp>
#include <ascent.hpp>
#include <string>

// 2/5/2021 wes. created an ascent-2021 branch for doign code dev work
// to modernize the AscentAnalysisAdaptor


namespace sensei
{

/// @brief Analysis adaptor for ascent-based analysis pipelines.
///
/// AscentAnalysisAdaptor is a subclass of sensei::AnalysisAdaptor that
/// can be used as the superclass for all analysis that uses libsim.
class AscentAnalysisAdaptor : public AnalysisAdaptor
{
public:
  static AscentAnalysisAdaptor* New();
  senseiTypeMacro(AscentAnalysisAdaptor, AnalysisAdaptor);

  int Initialize(const std::string &json_file_path,
    const std::string &options_file_path);

  bool Execute(DataAdaptor* data) override;

  int Finalize() override;

  /// data requirements tell the adaptor what to process
  /// currently data requiremetns must be specified
  int SetDataRequirements(const DataRequirements &reqs);

  int AddDataRequirement(const std::string &meshName,
    int association, const std::vector<std::string> &arrays);

protected:
  AscentAnalysisAdaptor();
  ~AscentAnalysisAdaptor();

  AscentAnalysisAdaptor(const AscentAnalysisAdaptor&) = delete;
  void operator=(const AscentAnalysisAdaptor&) = delete;

private:
  ascent::Ascent _ascent;
  conduit::Node optionsNode;    // Ascent options from json file.
  conduit::Node actionsNode;    // Ascent actions from json file.

  void GetFieldsFromActions();
  std::set<std::string> Fields;

  DataRequirements Requirements;
};

}

#endif
