#ifndef AUTOCORRELATIONANALYSISADAPTOR_H
#define AUTOCORRELATIONANALYSISADAPTOR_H

#include "vtkInsituAnalysisAdaptor.h"
#include <mpi.h>

/// @brief vtkInsituAnalysisAdaptor subclass for autocorrelation.
///
/// AutocorrelationAnalysisAdaptor is an analysis adaptor that performs
/// autocorrelation on the dataset.
class AutocorrelationAnalysisAdaptor : public vtkInsituAnalysisAdaptor
{
public:
  static AutocorrelationAnalysisAdaptor* New();
  vtkTypeMacro(AutocorrelationAnalysisAdaptor, vtkInsituAnalysisAdaptor);

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

  virtual bool Execute(vtkInsituDataAdaptor* data);
protected:
  AutocorrelationAnalysisAdaptor();
  ~AutocorrelationAnalysisAdaptor();

  void PrintResults(size_t k_max);
private:
  AutocorrelationAnalysisAdaptor(const AutocorrelationAnalysisAdaptor&); // not implemented.
  void operator=(const AutocorrelationAnalysisAdaptor&);

  class AInternals;
  AInternals* Internals;
};
#endif
