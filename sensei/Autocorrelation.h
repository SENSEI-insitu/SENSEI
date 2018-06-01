#ifndef sensei_Autocorrelation_h
#define sensei_Autocorrelation_h

#include "AnalysisAdaptor.h"
#include <mpi.h>
#include <string>

namespace sensei
{
/// @class autocorrelation::AnalysisAdaptor
/// @brief AnalysisAdaptor subclass for autocorrelation.
///
/// Autocorrelation is an analysis adaptor that performs
/// autocorrelation on the dataset.
class Autocorrelation : public AnalysisAdaptor
{
public:
  static Autocorrelation* New();
  senseiTypeMacro(Autocorrelation, AnalysisAdaptor);

  /// @brief Initialize the adaptor.
  ///
  /// This method must be called to initialize the adaptor with configuration
  /// parameters for the analysis to perform.
  ///
  /// @param window analysis window in timestep count.
  /// @param name of mesh containing the array to process
  /// @param association together with \c arrayname, identifies the array to
  ///         compute autocorrelation for.
  /// @param arrayname together with \c association, identifies the array to
  ///         compute autocorrelation for.
  /// @param k_max number of strongest autocorrelations to report
  void Initialize(size_t window, const std::string &meshName,
    int association, const std::string &arrayname, size_t k_max);

  bool Execute(DataAdaptor* data) override;

  int Finalize() override;

protected:
  Autocorrelation();
  ~Autocorrelation();

  void PrintResults(size_t k_max);
private:
  Autocorrelation(const Autocorrelation&); // not implemented.
  void operator=(const Autocorrelation&);

  class AInternals;
  AInternals* Internals;
};

}
#endif
