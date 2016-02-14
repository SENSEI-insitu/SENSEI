#ifndef HISTOGRAMANALYSISADAPTOR_H
#define HISTOGRAMANALYSISADAPTOR_H


#include "vtkInsituAnalysisAdaptor.h"
#include <mpi.h>

/// HistogramAnalysisAdaptor is a vtkInsituAnalysisAdaptor specialization for
/// histogram analysis.
class HistogramAnalysisAdaptor : public vtkInsituAnalysisAdaptor
{
public:
  static HistogramAnalysisAdaptor* New();
  vtkTypeMacro(HistogramAnalysisAdaptor, vtkInsituAnalysisAdaptor);

  void Initialize(MPI_Comm comm, int bins,
    int association, const std::string& arrayname);

  virtual bool Execute(vtkInsituDataAdaptor* data);

protected:
  HistogramAnalysisAdaptor();
  virtual ~HistogramAnalysisAdaptor();

  MPI_Comm Communicator;
  int Bins;
  std::string ArrayName;
  int Association;
private:
  HistogramAnalysisAdaptor(const HistogramAnalysisAdaptor&);
  void operator=(const HistogramAnalysisAdaptor&);
};

#endif
