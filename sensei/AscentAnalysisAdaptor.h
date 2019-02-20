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

  void Initialize(conduit::Node xml_actions, conduit::Node setup);
  void Initialize(std::string json_file_path, conduit::Node setup);

  bool Execute(DataAdaptor* data) override;

  int Finalize() override;

protected:
  AscentAnalysisAdaptor();
  ~AscentAnalysisAdaptor();

private:
  AscentAnalysisAdaptor(const AscentAnalysisAdaptor&)=delete; // Not implemented.
  void operator=(const AscentAnalysisAdaptor&)=delete; // Not implemented.

  void GetFieldsFromActions();

  ascent::Ascent a;
  conduit::Node actionNode;
  std::set<std::string> Fields;
};

} // namespace sensei

#endif
