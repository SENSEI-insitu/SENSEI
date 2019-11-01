#ifndef sensei_ConfigurableAnalysis_h
#define sensei_ConfigurableAnalysis_h

#include "AnalysisAdaptor.h"

#include <string>
#include <mpi.h>

namespace pugi { class xml_node; }

namespace sensei
{

/// @brief ConfigurableAnalysis is all-in-one analysis adaptor that
/// can execute all available analysis adaptors.
class ConfigurableAnalysis : public AnalysisAdaptor
{
public:
  static ConfigurableAnalysis *New();
  senseiTypeMacro(ConfigurableAnalysis, AnalysisAdaptor);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  /// @brief Set the communicator used by the adaptor.
  /// The default communicator is a duplicate of MPI_COMMM_WORLD, giving
  /// each adaptor a unique communication space. Users wishing to override
  /// this should set the communicator before doing anything else. Derived
  /// classes should use the communicator returned by GetCommunicator.
  int SetCommunicator(MPI_Comm comm) override;

  /// @brief Initialize the adaptor using the configuration specified.
  int Initialize(const std::string &filename);
  int Initialize(const pugi::xml_node &root);

  bool Execute(DataAdaptor *data) override;

  int Finalize() override;

protected:
  ConfigurableAnalysis();
  ~ConfigurableAnalysis();

  ConfigurableAnalysis(const ConfigurableAnalysis&) = delete;
  void operator=(const ConfigurableAnalysis&) = delete;

private:
  struct InternalsType;
  InternalsType *Internals;
};

}

#endif
