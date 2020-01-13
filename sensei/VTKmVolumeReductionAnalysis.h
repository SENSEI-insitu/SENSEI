#ifndef sensei_VTKmVolumeReductionAnalysis_h
#define sensei_VTKmVolumeReductionAnalysis_h

#include "AnalysisAdaptor.h"
#include <vtkm/cont/Field.h>
#include <mpi.h>

namespace sensei
{
class CinemaHelper;

class VTKmVolumeReductionAnalysis : public AnalysisAdaptor
{
public:
  static VTKmVolumeReductionAnalysis* New();
  senseiTypeMacro(VTKmVolumeReductionAnalysis, AnalysisAdaptor);

  void Initialize(
    const std::string& meshName,
    const std::string& fieldName,
    const std::string& fieldAssoc,
    const std::string& workingDirectory,
    int reductionFactor,
    MPI_Comm comm);

  bool Execute(DataAdaptor* data, DataAdaptor*&) override;

  int Finalize() override { return 0; }

protected:
  VTKmVolumeReductionAnalysis();
  ~VTKmVolumeReductionAnalysis();

  std::string MeshName;
  std::string FieldName;
  vtkm::cont::Field::Association FieldAssoc;
  MPI_Comm Communicator;
  CinemaHelper* Helper;
  int Reduction;

private:
  VTKmVolumeReductionAnalysis(const VTKmVolumeReductionAnalysis&);
  void operator=(const VTKmVolumeReductionAnalysis&);
};

}

#endif
