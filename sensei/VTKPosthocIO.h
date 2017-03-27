#ifndef sensei_VTKPosthocIO_h
#define sensei_VTKPosthocIO_h

#include "AnalysisAdaptor.h"

#include <mpi.h>
#include <vector>
#include <string>

class vtkInformation;
class vtkCompositeDataSet;

namespace sensei
{
/// @class VTKPosthocIO
/// brief sensei::VTKPosthocIO is a AnalysisAdaptor that writes
/// the data to disk for a posthoc analysis. This adaptor supports
/// writing to a VTK(PXML), VisIt(.visit) or ParaView(.pvd) compatible
/// format and can be used to obtain representative dataset for
/// configuring in situ rendering with Libsim or Catalyst.
class VTKPosthocIO : public AnalysisAdaptor
{
public:
  static VTKPosthocIO* New();
  senseiTypeMacro(VTKPosthocIO, AnalysisAdaptor);

  enum {MODE_PARAVIEW=0, MODE_VISIT=1};

  void Initialize(MPI_Comm comm, const std::string &outputDir,
    const std::string &headerFile, const std::vector<std::string> &cellArrays,
    const std::vector<std::string> &pointArrays, int mode, int period);

  bool Execute(DataAdaptor* data) override;

  bool Finalize();

protected:
  VTKPosthocIO();
  ~VTKPosthocIO();

private:
  int WriteXMLP(vtkCompositeDataSet *cd,
    vtkInformation *info, int timeStep);

private:
  MPI_Comm Comm;
  std::string OutputDir;
  std::string FileName;
  std::vector<std::string> CellArrays;
  std::vector<std::string> PointArrays;
  std::vector<double> Time;
  std::vector<long> TimeStep;
  std::vector<long> NumBlocks;
  std::vector<long> BlockStarts;
  std::string BlockExt;
  long FileId;
  int Mode;
  int Period;
  int HaveBlockInfo;

private:
  VTKPosthocIO(const VTKPosthocIO&);
  void operator=(const VTKPosthocIO&);
};

}
#endif
