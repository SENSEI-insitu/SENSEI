#ifndef AUTOCORRELATIONANALYSISADAPTOR_H
#define AUTOCORRELATIONANALYSISADAPTOR_H

#include "vtkInsituAnalysisAdaptor.h"
#include <mpi.h>

class AutocorrelationAnalysisAdaptor : public vtkInsituAnalysisAdaptor
{
public:
  static AutocorrelationAnalysisAdaptor* New();
  vtkTypeMacro(AutocorrelationAnalysisAdaptor, vtkInsituAnalysisAdaptor);


  void Initialize(MPI_Comm world,
    size_t window, int association, const char* arrayname);
  void PrintResults(size_t k_max);

  virtual bool Execute(vtkInsituDataAdaptor* data);
protected:
  AutocorrelationAnalysisAdaptor();
  ~AutocorrelationAnalysisAdaptor();

private:
  AutocorrelationAnalysisAdaptor(const AutocorrelationAnalysisAdaptor&); // not implemented.
  void operator=(const AutocorrelationAnalysisAdaptor&);

  class AInternals;
  AInternals* Internals;
};
#endif
