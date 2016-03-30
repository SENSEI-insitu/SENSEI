#ifndef sensei_catalyst_Slice_h
#define sensei_catalyst_Slice_h

#include "vtkCPPipeline.h"

namespace sensei
{
namespace catalyst
{
class Slice : public vtkCPPipeline
{
public:
  static Slice* New();
  vtkTypeMacro(Slice, vtkCPPipeline);
  void PrintSelf(ostream& os, vtkIndent indent);

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
  const bool GetUseLogScale() const;

  virtual int RequestDataDescription(vtkCPDataDescription* dataDesc);
  virtual int CoProcess(vtkCPDataDescription* dataDesc);
  virtual int Finalize();
protected:
  Slice();
  ~Slice();

private:
  Slice(const Slice&); // Not implemented.
  void operator=(const Slice&); // Not implemented.

  class vtkInternals;
  vtkInternals* Internals;
};

} // catalyst
} // sensei

#endif
