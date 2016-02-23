#ifndef sensei_Autocorrelation_h
#define sensei_Autocorrelation_h

#include "AnalysisAdaptor.h"
#include <mpi.h>

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
  vtkTypeMacro(Autocorrelation, AnalysisAdaptor);

  /// @brief Initialize the adaptor.
  ///
  /// This method must be called to initialize the adaptor with configuration
  /// parameters for the analysis to perform.
  ///
  /// @param world MPI communicator to use.
  /// @param window analysis window in timestep count.
  /// @param association together with \c arrayname, identifies the array to
  ///         compute autocorrelation for.
  /// @param arrayname together with \c association, identifies the array to
  ///         compute autocorrelation for.
  /// @param k_max number of strongest autocorrelations to report
  void Initialize(MPI_Comm world,
    size_t window, int association, const char* arrayname, size_t k_max);

  virtual bool Execute(DataAdaptor* data);
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
