#ifndef vtkCatalystSlicePipeline_h
#define vtkCatalystSlicePipeline_h

#include "vtkCPPipeline.h"

class vtkCatalystSlicePipeline : public vtkCPPipeline
{
public:
  static vtkCatalystSlicePipeline* New();
  vtkTypeMacro(vtkCatalystSlicePipeline, vtkCPPipeline);
  void PrintSelf(ostream& os, vtkIndent indent);

  /// @brief Set slice plane origin.
  void SetSliceOrigin(double x, double y, double z);

  /// @brief Set slice plane normal.
  void SetSliceNormal(double i, double j, double k);

  /// @brief Set array to color with.
  ///
  /// Set array to color with. If arrayname is NULL, coloring will be disabled.
  void ColorBy(int association, const char* arrayname);

  virtual int RequestDataDescription(vtkCPDataDescription* dataDesc);
  virtual int CoProcess(vtkCPDataDescription* dataDesc);
  virtual int Finalize();

protected:
  vtkCatalystSlicePipeline();
  ~vtkCatalystSlicePipeline();

private:
  vtkCatalystSlicePipeline(const vtkCatalystSlicePipeline&); // Not implemented.
  void operator=(const vtkCatalystSlicePipeline&); // Not implemented.

  class vtkInternals;
  vtkInternals* Internals;
};

#endif

