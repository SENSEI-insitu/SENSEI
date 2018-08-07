#ifndef sensei_CinemaHelper_h
#define sensei_CinemaHelper_h

class vtkSMViewProxy;
class vtkSMRenderViewProxy;
class vtkSMRepresentationProxy;
class vtkActor;
class vtkCamera;
class vtkRenderer;
class vtkRenderWindow;
class vtkIceTCompositePass;
class vtkCameraPass;
class vtkLightingMapPass;
class vtkImageData;

#include <string>

namespace sensei
{

class CinemaHelper
{
public:
    CinemaHelper();
    ~CinemaHelper();

    // Metadata handling
    void SetImageSize(int width, int height);
    void SetWorkingDirectory(const std::string& path);
    void SetExportType(const std::string& exportType);
    void AddTimeEntry();
    void WriteMetadata();

    // Camera handling
    void SetCameraConfig(const std::string& config);
    int GetNumberOfCameraPositions();
    void ApplyCameraPosition(vtkSMViewProxy* view, int cameraPositionIndex);
    void ApplyCameraPosition(vtkCamera* camera, int cameraPositionIndex);

    // Contours handling
    int GetNumberOfContours();
    double GetContourValue(int idx);
    void SetContours(const std::string& values);

    // Composite dataset handling
    // => FIXME: Assume => intensity + constant coloring
    int RegisterLayer(const std::string& name, vtkSMRepresentationProxy* representation, double scalarValue);
    int RegisterLayer(const std::string& name, vtkActor* actor, double scalarValue);
    void CaptureSortedCompositeData(vtkSMRenderViewProxy* view);
    void CaptureSortedCompositeData(vtkRenderWindow* renderWindow, vtkRenderer* renderer, vtkIceTCompositePass* compositePass);
    void Render(vtkRenderWindow* renderWindow);
    vtkImageData* CaptureWindow(vtkRenderWindow* renderWindow);

    // Image handling
    void CaptureImage(vtkSMViewProxy* view, const std::string fileName, const std::string writerName, double scale = 1, bool createDirectory = true);

    // Generic Methods
    void Capture(vtkSMViewProxy* view);

    // Volume handling
    void WriteVolume(vtkImageData* image);

private:
    struct Internals;
    Internals *Data;
};

}

#endif
