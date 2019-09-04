#ifndef sensei_AscentAnalysisAdaptor_h
#define sensei_AscentAnalysisAdaptor_h

#include <conduit.hpp>
#include <ascent.hpp>
#include <string>

#include "AnalysisAdaptor.h"

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

  void Initialize(const std::string &json_file_path, const std::string &options_file_path);

  bool Execute(DataAdaptor* data) override;

  int Finalize() override;

protected:
  AscentAnalysisAdaptor();
  ~AscentAnalysisAdaptor();

private:
  AscentAnalysisAdaptor(const AscentAnalysisAdaptor&)=delete; // Not implemented.
  void operator=(const AscentAnalysisAdaptor&)=delete; // Not implemented.

  void GetFieldsFromActions();

  ascent::Ascent _ascent;
  conduit::Node optionsNode;    // Ascent options from json file.
  conduit::Node actionsNode;    // Ascent actions from json file.
  std::set<std::string> Fields;
};

} // namespace sensei

#endif
