#ifndef sensei_Autocorrelation_h
#define sensei_Autocorrelation_h

#include "AnalysisAdaptor.h"
#include <mpi.h>
#include <string>

namespace sensei
{
/// Performs a temporal autocorrelation on the simulation data
class Autocorrelation : public AnalysisAdaptor
{
public:
  /// Allocate a new Autocorrelation instance
  static Autocorrelation* New();

  senseiTypeMacro(Autocorrelation, AnalysisAdaptor);

  /** Initialize the adaptor.  This method must be called to initialize the
   * adaptor with configuration parameters for the analysis to perform.
   *
   * @param window analysis window in timestep count.
   * @param name of mesh containing the array to process
   * @param association together with \c arrayname, identifies the array to
   *         compute autocorrelation for.
   * @param arrayname together with \c association, identifies the array to
   *         compute autocorrelation for.
   * @param kMax number of strongest autocorrelations to report
   * @param numThreads number of threads in sdiy's thread pool
   */
  void Initialize(size_t window, const std::string &meshName,
    int association, const std::string &arrayname, size_t kMax,
    int numThreads = 1);

  /// Incrementally computes autocorrelation on the current simulation state
  bool Execute(DataAdaptor* data) override;

  /// Finishes the calculation and dumps the results
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
