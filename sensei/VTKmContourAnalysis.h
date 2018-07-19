#ifndef sensei_VTKmContourAnalysis_h
#define sensei_VTKmContourAnalysis_h

#include "AnalysisAdaptor.h"
#include <mpi.h>

class vtkDataObject;
class vtkDataArray;

namespace sensei
{

/// @class VTKmContourAnalysis
/// @brief sensei::VTKmContourAnalysis is a AnalysisAdaptor specialization for contouring.
///
class VTKmContourAnalysis : public AnalysisAdaptor
{
public:
  static VTKmContourAnalysis* New();
  vtkTypeMacro(VTKmContourAnalysis, AnalysisAdaptor);

  void Initialize(const std::string& meshName, const std::string& arrayname,
    double value, bool writeOutput);

  bool Execute(sensei::DataAdaptor* data) override;

  int Finalize() override;

protected:
  VTKmContourAnalysis();
  virtual ~VTKmContourAnalysis();

  std::string MeshName;
  std::string ArrayName;
  double Value;
  bool WriteOutput;

private:
  VTKmContourAnalysis(const VTKmContourAnalysis&);
  void operator=(const VTKmContourAnalysis&);
};

}

#endif
