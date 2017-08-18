#ifndef sensei_VTKOnlyContourCompositeAnalysis_h
#define sensei_VTKOnlyContourCompositeAnalysis_h

#include "AnalysisAdaptor.h"
#include <mpi.h>

namespace sensei
{
class CinemaHelper;
struct PipelineHandler;

class VTKOnlyContourCompositeAnalysis : public AnalysisAdaptor
{
public:
  static VTKOnlyContourCompositeAnalysis* New();
  senseiTypeMacro(VTKOnlyContourCompositeAnalysis, AnalysisAdaptor);

  void Initialize(MPI_Comm comm, const std::string& workingDirectory,
                  int* imageSize, const std::string& contours,
                  const std::string& camera);

  bool Execute(DataAdaptor* data) override;

protected:
  VTKOnlyContourCompositeAnalysis();
  ~VTKOnlyContourCompositeAnalysis();

  void AddContour(double value);

  MPI_Comm Communicator;
  CinemaHelper* Helper;
  PipelineHandler* Pipeline;
private:
  VTKOnlyContourCompositeAnalysis(const VTKOnlyContourCompositeAnalysis&);
  void operator=(const VTKOnlyContourCompositeAnalysis&);
};

}

#endif
