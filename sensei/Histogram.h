#ifndef sensei_Histogram_h
#define sensei_Histogram_h

#include "AnalysisAdaptor.h"
#include <mpi.h>

class vtkDataObject;
class vtkDataArray;

namespace sensei
{

/// @class Histogram
/// @brief sensei::Histogram is a AnalysisAdaptor specialization for histogram analysis.
///
/// This class demonstrates how a custom analysis code may be written within the
/// Sensei infrastructure.
class Histogram : public AnalysisAdaptor
{
public:
  static Histogram* New();
  senseiTypeMacro(Histogram, AnalysisAdaptor);

  void Initialize(MPI_Comm comm, int bins,
    int association, const std::string& arrayname);

  bool Execute(DataAdaptor* data) override;

protected:
  Histogram();
  ~Histogram();

  static const char *GetGhostArrayName();
  vtkDataArray* GetArray(vtkDataObject* dobj, const std::string& arrayname);

  MPI_Comm Communicator;
  int Bins;
  std::string ArrayName;
  int Association;

private:
  Histogram(const Histogram&);
  void operator=(const Histogram&);
};

}

#endif
