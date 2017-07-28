#ifndef sensei_VTKmContourCompositeAnalysis_h
#define sensei_VTKmContourCompositeAnalysis_h

#include "AnalysisAdaptor.h"
#include <mpi.h>

namespace sensei
{
class CinemaHelper;
struct PipelineHandler;

class VTKmContourCompositeAnalysis : public AnalysisAdaptor
{
public:
  static VTKmContourCompositeAnalysis* New();
  senseiTypeMacro(VTKmContourCompositeAnalysis, AnalysisAdaptor);

  void Initialize(MPI_Comm comm, const std::string& workingDirectory,
                  int* imageSize, const std::string& contours,
                  const std::string& camera);

  bool Execute(DataAdaptor* data) override;

protected:
  VTKmContourCompositeAnalysis();
  ~VTKmContourCompositeAnalysis();

  void AddContour(double value);

  MPI_Comm Communicator;
  CinemaHelper* Helper;
  PipelineHandler* Pipeline;
private:
  VTKmContourCompositeAnalysis(const VTKmContourCompositeAnalysis&);
  void operator=(const VTKmContourCompositeAnalysis&);
};

}

#endif
