#ifndef sensei_CatalystCinema_h
#define sensei_CatalystCinema_h

#include "vtkCPPipeline.h"

namespace sensei
{

class CatalystCinema : public vtkCPPipeline
{
public:
  static CatalystCinema* New();
  vtkTypeMacro(CatalystCinema, vtkCPPipeline);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  /// @brief Set parameters for saving rendering result.
  ///
  /// If not specified, the pipeline will not do any rendering.
  void SetImageParameters(const std::string& basepath, int width, int height);

  /// @brief Set parameters for camera navigation.
  void SetCameraConfiguration(const std::string& config);

  /// @brief Configure cinema database type
  void SetExportType(const std::string& exportType);

  int RequestDataDescription(vtkCPDataDescription* dataDesc) override;
  int CoProcess(vtkCPDataDescription* dataDesc) override;
  int Finalize() override;

protected:
  CatalystCinema();
  ~CatalystCinema();

private:
  CatalystCinema(const CatalystCinema&);
  void operator=(const CatalystCinema&);

  class vtkInternals;
  vtkInternals* Internals;
};

}

#endif
