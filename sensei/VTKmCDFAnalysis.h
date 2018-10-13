#ifndef sensei_VTKmCDFAnalysis_h
#define sensei_VTKmCDFAnalysis_h

#include "AnalysisAdaptor.h"
#include <vtkm/cont/Field.h>
#include <mpi.h>

namespace sensei
{
class CinemaHelper;

class VTKmCDFAnalysis : public AnalysisAdaptor
{
public:
  static VTKmCDFAnalysis* New();
  senseiTypeMacro(VTKmCDFAnalysis, AnalysisAdaptor);

  void Initialize(
    const std::string& meshName,
    const std::string& fieldName,
    const std::string& fieldAssoc,
    const std::string& workingDirectory,
    int numberOfQuantiles,
    int requestSize,
    MPI_Comm comm);

  bool Execute(DataAdaptor* data) override;

  int Finalize() override { return 0; }

protected:
  VTKmCDFAnalysis();
  ~VTKmCDFAnalysis();

  std::string MeshName;
  std::string FieldName;
  vtkm::cont::Field::Association FieldAssoc;
  MPI_Comm Communicator;
  CinemaHelper* Helper;
  int NumberOfQuantiles;
  int RequestSize;

private:
  VTKmCDFAnalysis(const VTKmCDFAnalysis&);
  void operator=(const VTKmCDFAnalysis&);
};

}

#endif
