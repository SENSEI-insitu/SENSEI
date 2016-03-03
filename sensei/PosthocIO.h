#ifndef sensei_PosthocIO_h
#define sensei_PosthocIO_h

#include "AnalysisAdaptor.h"

#include <mpi.h>
#include <vector>
#include <string>

class vtkInformation;
class vtkCompositeDataSet;

namespace sensei
{
/// @class PosthocIO
/// brief sensei::PosthocIO is a AnalysisAdaptor that writes
/// the data to disk for a posthoc analysis.
class PosthocIO : public AnalysisAdaptor
{
public:
  static PosthocIO* New();

  vtkTypeMacro(PosthocIO, AnalysisAdaptor);

  // modes.
  enum {MpiIO=1, FilePerProc=2};

  void Initialize(MPI_Comm comm, const std::string &outputDir,
    const std::string &headerFile, const std::string &blockExt,
    const std::vector<std::string> &cellArrays,
    const std::vector<std::string> &pointArrays, int mode,
    int period);

  bool Execute(sensei::DataAdaptor* data) override;

  int WriteBOVHeader(vtkInformation *info);

protected:
  PosthocIO();
  ~PosthocIO();

private:
  int WriteBOVHeader(const std::string &fileName,
    const std::vector<std::string> &arrays, const int *wholeExtent);

  int WriteBOV(vtkCompositeDataSet *cd,
    vtkInformation *info, int timeStep);

  int WriteXMLP(vtkCompositeDataSet *cd,
    vtkInformation *info, int timeStep);

private:
  MPI_Comm Comm;
  int CommRank;
  std::string OutputDir;
  std::string HeaderFile;
  std::string BlockExt;
  std::vector<std::string> CellArrays;
  std::vector<std::string> PointArrays;
  bool HaveHeader;
  int Mode;
  int Period;

private:
  PosthocIO(const PosthocIO&);
  void operator=(const PosthocIO&);
};

}
#endif
