#ifndef sensei_CinemaHelper_h
#define sensei_CinemaHelper_h

class vtkSMViewProxy;
class vtkSMRenderViewProxy;
class vtkSMRepresentationProxy;

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

    // Composite dataset handling
    // => FIXME: Assume => intensity + constant coloring
    int RegisterLayer(const std::string& name, vtkSMRepresentationProxy* representation, double scalarValue);
    void CaptureSortedCompositeData(vtkSMRenderViewProxy* view);

    // Image handling
    void CaptureImage(vtkSMViewProxy* view, const std::string fileName, const std::string writerName, double scale = 1, bool createDirectory = true);

    // Generic Methods
    void Capture(vtkSMViewProxy* view);

private:
    struct Internals;
    Internals *Data;
};

}

#endif
