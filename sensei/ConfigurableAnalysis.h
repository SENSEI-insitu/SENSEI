#ifndef sensei_ConfigurableAnalysis_h
#define sensei_ConfigurableAnalysis_h

#include "AnalysisAdaptor.h"

#include <string>
#include <mpi.h>

namespace sensei
{

/// @brief ConfigurableAnalysis is all-in-one analysis adaptor that
/// can execute all available analysis adaptors.
class ConfigurableAnalysis : public AnalysisAdaptor
{
public:
  static ConfigurableAnalysis* New();
  senseiTypeMacro(ConfigurableAnalysis, AnalysisAdaptor);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  /// @brief Initialize the adaptor using the configuration specified.
  bool Initialize(MPI_Comm world, const std::string& filename);

  bool Execute(DataAdaptor* data) override;

  int Finalize() override;

protected:
  ConfigurableAnalysis();
  ~ConfigurableAnalysis();

private:
  ConfigurableAnalysis(const ConfigurableAnalysis&); // Not implemented.
  void operator=(const ConfigurableAnalysis&); // Not implemented.

  class InternalsType;
  InternalsType* Internals;
};

}

#endif
