#ifndef sensei_CatalystParticle_h
#define sensei_CatalystParticle_h

#include "vtkCPPipeline.h"

namespace sensei
{

class CatalystParticle : public vtkCPPipeline
{
public:
  static CatalystParticle* New();
  vtkTypeMacro(CatalystParticle, vtkCPPipeline);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  /// @brief Set the mesh on which the slice should operate.
  void SetInputMesh(const std::string& meshName);

  /// @brief Set particle glyph type.
  void SetParticleGlyph(const std::string& glyphType);

  /// @brief Set parameters for saving rendering result.
  ///
  /// If not specified, the pipeline will not do any rendering.
  void SetImageParameters(const std::string& filename, int width, int height);

  /// @brief Choose how particles are rendered.
  ///
  /// This must be one of the following strings:
  /// + "Gaussian Blur"
  /// + "Sphere"
  /// + "Black-edged circle"
  /// + "Plain circle"
  /// + "Triangle"
  /// + "Square Outline"
  ///
  /// The default is "Sphere".
  void SetParticleStyle(const std::string& style);

  /// @brief Set a scale factor to apply to each particle's Gaussian radius.
  ///
  /// The default is 1.0.
  void SetParticleRadius(float radius);

  /// @brief Set the camera location
  ///
  /// If unset, the camera will be placed at (1,1,1) and reset at each time-step.
  /// Note that resetting the camera may cause discontinuities in the camera
  /// motion if the bounds of the data change abruptly with time.
  void SetCameraPosition(const double posn[3]);

  /// @brief Set the camera's focal point.
  ///
  /// If unset, the camera will focus on the origin, (0,0,0).
  void SetCameraFocus(const double focus[3]);

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
  CatalystParticle();
  ~CatalystParticle();

private:
  CatalystParticle(const CatalystParticle&); // Not implemented.
  void operator=(const CatalystParticle&); // Not implemented.

  class vtkInternals;
  vtkInternals* Internals;
};

}

#endif
