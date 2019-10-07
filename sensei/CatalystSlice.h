#ifndef sensei_CatalystSlice_h
#define sensei_CatalystSlice_h

#include "vtkCPPipeline.h"

namespace sensei
{

class CatalystSlice : public vtkCPPipeline
{
public:
  static CatalystSlice* New();
  vtkTypeMacro(CatalystSlice, vtkCPPipeline);

  /// @brief Set the mesh on which the slice should operate.
  void SetInputMesh(const std::string& meshName);

  /// @brief Set slice plane origin.
  void SetSliceOrigin(double x, double y, double z);

  /// @brief Set slice plane normal.
  void SetSliceNormal(double i, double j, double k);

  /// @brief When set to true, the slice will be repositioned to the center of
  /// the domain on each iteration. Default: true.
  void SetAutoCenter(bool val);

  /// @brief Set parameters for saving rendering result.
  ///
  /// If not specified, the pipeline will not do any rendering.
  void SetImageParameters(const std::string& filename, int width, int height);

  /// @brief Set array to color with.
  ///
  /// Set array to color with. If arrayname is NULL, coloring will be disabled.
  void ColorBy(int association, const std::string& arrayname);

  /// @brief Set whether to automatically determine color range per time iteration.
  ///
  /// When set to true, this analysis adaptor will compute and set the color range per
  /// timestep. Otherwise, it will use the value specified using SetColorRange().
  void SetAutoColorRange(bool val);
  bool GetAutoColorRange() const;

  /// @brief Set the color range to use when AutoColorRange is false.
  ///
  /// Set the color range to use when AutoColorRange is false.
  void SetColorRange(double min, double max);
  const double* GetColorRange() const;

  /// @brief Set whether to use log scale for coloring.
  void SetUseLogScale(bool val);
  bool GetUseLogScale() const;

  int RequestDataDescription(vtkCPDataDescription* dataDesc) override;
  int CoProcess(vtkCPDataDescription* dataDesc) override;
  int Finalize() override;

protected:
  CatalystSlice();
  ~CatalystSlice();

private:
  CatalystSlice(const CatalystSlice&); // Not implemented.
  void operator=(const CatalystSlice&); // Not implemented.

  class vtkInternals;
  vtkInternals* Internals;
};

}

#endif
