#ifndef Catalyst2AnalysisAdaptor_h
#define Catalyst2AnalysisAdaptor_h

#include "AnalysisAdaptor.h"
#include "DataRequirements.h"
#include "MeshMetadata.h"

#include <memory>
#include <vector>
#include <set>

namespace sensei
{

/// @brief Analysis adaptor for Catalyst2-based analysis pipelines.
///
/// AnalysisAdaptor is a subclass of AnalysisAdaptor that is
/// that can be used as the superclass for all analysis that uses Catalyst 2.
class Catalyst2AnalysisAdaptor : public AnalysisAdaptor
{
public:
  static Catalyst2AnalysisAdaptor* New();
  senseiTypeMacro(Catalyst2AnalysisAdaptor, AnalysisAdaptor);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /// Adds a pipeline initialized from a Catalyst python script
  virtual void AddPythonScriptPipeline(const std::string &fileName);
  /** Adds a set of sensei::DataRequirements, typically this will come from an XML
   * configuratiopn file. Data requirements tell the adaptor what to fetch from
   * the simulation and write to disk. If none are given then all available
   * data is fetched and written.
   */
  int SetDataRequirements(const DataRequirements &reqs);

  bool Execute(DataAdaptor* data, DataAdaptor** dataOut = nullptr) override;

  int Finalize() override;

protected:
  Catalyst2AnalysisAdaptor();
  ~Catalyst2AnalysisAdaptor() override;

  void Initialize();
  /// meshes needed by the analysis. If empty, all meshes are needed.
  DataRequirements Requirements;

private:
  Catalyst2AnalysisAdaptor(const Catalyst2AnalysisAdaptor&); // Not implemented.
  void operator=(const Catalyst2AnalysisAdaptor&); // Not implemented.
};

}

#endif
