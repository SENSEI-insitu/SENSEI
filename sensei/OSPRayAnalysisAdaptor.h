/**
 *
 */

#ifndef OSPRayAnalysisAdaptor_h
#define OSPRayAnalysisAdaptor_h

// SENSEI
#include <AnalysisAdaptor.h>

// VTK
class svtkDataObject;
#include <svtkObject.h>
#include <svtkSetGet.h>
#include <svtkSmartPointer.h>

#include <vector>

namespace sensei {

/// An analysis adaptor for OSPRay based rendering pipelines
class OSPRayAnalysisAdaptor : public AnalysisAdaptor {
public:
  static OSPRayAnalysisAdaptor *New();
  senseiTypeMacro(OSPRayAnalysisAdaptor, AnalysisAdaptor);

  /// Initialize OSPRay library
  int Initialize();
  // Run in situ OSPRay Rendering
  bool Execute(sensei::DataAdaptor *, sensei::DataAdaptor **) override;
  /// Shut down and clean up OSPRay
  int Finalize() override;

  /// one of "SPHERES", "UGRID", "SGRID"
  virtual void SetRenderAs(const char *_arg);

  virtual void SetMeshName(const char *_arg) {
    this->MeshName = _arg;
  }

  /// point or cell data
  virtual void SetAssociation(const char *_arg) {
    this->Association = _arg;
  }
  /// data array name
  virtual void SetArrayName(const char *_arg) {
    this->ArrayName = _arg;
  }

  virtual void SetDirectory(const char *_arg) {
    this->Directory = _arg;
  }
  virtual void SetFileName(const char *_arg) {
    this->FileName = _arg;
  }

  virtual void SetWidth(int _arg) {
    this->Width = _arg;
  }

  virtual void SetHeight(int _arg) {
    this->Height = _arg;
  }

  void SetUseD3(bool _arg) {
    this->UseD3 = _arg;
  }
  // linearized colormap of colors and opacities.  Colors should be 3x opacities
  void SetColormap(const std::vector<float>& colors, const std::vector<float>& opacities) {
    this->Colors = colors;
    this->Opacities = opacities;
  }

  void SetColormapRange(float v1, float v2) {
    this->TFRange[0] = v1;
    this->TFRange[1] = v2;
  }

  void SetParticleRadius(float radius) {
    this->ParticleRadius = radius;
  }

  void SetParticleColor(float color[3]) {
    this->ParticleColor[0] = color[0];
    this->ParticleColor[1] = color[1];
    this->ParticleColor[2] = color[2];
    this->ParticleColorSet = true;
  }

  void SetBackgroundColor(float color[4]) {
    this->BackgroundColor[0] = color[0];
    this->BackgroundColor[1] = color[1];
    this->BackgroundColor[2] = color[2];
    this->BackgroundColor[3] = color[3];
  }

  /// OSPRay specific camera data
  struct Camera {
    std::vector<double> Position = {30.0, 30.0, 150.0};
    std::vector<double> Direction = {0.0, 0.0, -1.0};
    std::vector<double> Up = {0.0, 1.0, 0.0};
    double Fovy = 35.0;
    double FocusDistance = 0.0;
  };

  void SetCamera(OSPRayAnalysisAdaptor::Camera camera) {
    this->CameraData = camera;
  }

private:
  int RenderAs = 0;
  std::string MeshName;
  std::string Association;
  std::string ArrayName;

  std::string Directory = ".";
  std::string FileName = "output";
  int Width;
  int Height;
  std::vector<float> Colors;
  std::vector<float> Opacities;
  float TFRange[2] = {0.f, 0.5f};
  OSPRayAnalysisAdaptor::Camera CameraData;
  float ParticleRadius = 0.7f;
  float ParticleColor[3] {0.f, 0.f, 0.f};
  bool ParticleColorSet = false;
  float BackgroundColor[4] {0.f, 0.f, 0.f, 1.f};

  bool UseD3{false};

  struct InternalsType;
  InternalsType *Internals;

  void RenderAsSpheresImpl(svtkDataObject *, int);
  void RenderAsUGridImpl(svtkDataObject *, int);
  void RenderAsSGridImpl(svtkDataObject *, int);
};

} /* namespace sensei */

#endif /* OSPRayAnalysisAdaptor_h */
