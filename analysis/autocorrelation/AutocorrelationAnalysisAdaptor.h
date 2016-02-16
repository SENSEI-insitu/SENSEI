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
                  size_t window,
                  size_t n_local_blocks,
                  int domain_shape_x, int domain_shape_y, int domain_shape_z,
                  int* gid,
                  int* from_x, int* from_y, int* from_z,
                  int* to_x,   int* to_y,   int* to_z);
  virtual bool Execute(vtkInsituDataAdaptor* data);

  void PrintResults(size_t k_max, size_t nblocks);

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
