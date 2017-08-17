#ifndef sensei_VTKmVolumeReductionAnalysis_h
#define sensei_VTKmVolumeReductionAnalysis_h

#include "AnalysisAdaptor.h"
#include <mpi.h>

namespace sensei
{
class CinemaHelper;

class VTKmVolumeReductionAnalysis : public AnalysisAdaptor
{
public:
  static VTKmVolumeReductionAnalysis* New();
  senseiTypeMacro(VTKmVolumeReductionAnalysis, AnalysisAdaptor);

  void Initialize(MPI_Comm comm, const std::string& workingDirectory, int reductionFactor);

  bool Execute(DataAdaptor* data) override;

protected:
  VTKmVolumeReductionAnalysis();
  ~VTKmVolumeReductionAnalysis();

  MPI_Comm Communicator;
  CinemaHelper* Helper;
  int Reduction;
private:
  VTKmVolumeReductionAnalysis(const VTKmVolumeReductionAnalysis&);
  void operator=(const VTKmVolumeReductionAnalysis&);
};

}

#endif
