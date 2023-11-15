#ifndef sensei_CinemaHelper_h
#define sensei_CinemaHelper_h

#include "senseiConfig.h"

class vtkSMViewProxy;
class vtkSMRenderViewProxy;
class vtkSMRepresentationProxy;
class vtkActor;
class vtkCamera;
class vtkRenderer;
class vtkRenderWindow;
class vtkIceTCompositePass;
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
    void SetSampleSize(int ssz);
    void AddTimeEntry();
    void WriteMetadata();

    // Camera handling
    void SetCameraConfig(const std::string& config);
    int GetNumberOfCameraPositions();
#ifdef SENSEI_ENABLE_CATALYST
    void ApplyCameraPosition(vtkSMViewProxy* view, int cameraPositionIndex);
#endif
#if defined(SENSEI_ENABLE_CATALYST) || defined (SENSEI_ENABLE_VTK_RENDERING)
    void ApplyCameraPosition(vtkCamera* camera, int cameraPositionIndex);
#endif

    // Scalar selection
    void SetScalarAssociation(const std::string& assoc); // "points" or "cells"
    std::string GetScalarAssociation() const;
    void SetScalarName(const std::string& name);
    std::string GetScalarName() const;

    // Contours handling
    int GetNumberOfContours();
    double GetContourValue(int idx);
    void SetContours(const std::string& values);

    // Composite dataset handling
    // => FIXME: Assume => intensity + constant coloring
#ifdef SENSEI_ENABLE_CATALYST
    int RegisterLayer(const std::string& name, vtkSMRepresentationProxy* representation, double scalarValue);
#endif
#if defined(SENSEI_ENABLE_CATALYST) || defined (SENSEI_ENABLE_VTK_RENDERING)
    int RegisterLayer(const std::string& name, vtkActor* actor, double scalarValue);
#endif
#ifdef SENSEI_ENABLE_CATALYST
    void CaptureSortedCompositeData(vtkSMRenderViewProxy* view);
    void CaptureSortedCompositeData(vtkRenderWindow* renderWindow, vtkRenderer* renderer, vtkIceTCompositePass* compositePass);
#endif
#if defined(SENSEI_ENABLE_CATALYST) || defined (SENSEI_ENABLE_VTK_RENDERING)
    void Render(vtkRenderWindow* renderWindow);
    vtkImageData* CaptureWindow(vtkRenderWindow* renderWindow);
#endif

#ifdef SENSEI_ENABLE_CATALYST
    // Image handling
    void CaptureImage(vtkSMViewProxy* view, const std::string fileName, const std::string writerName, double scale = 1, bool createDirectory = true);

    // Generic Methods
    void Capture(vtkSMViewProxy* view);
#endif

    // Volume handling
    void WriteVolume(vtkImageData* image);

    // CDF handling
    void WriteCDF(long long totalArraySize, const double* cdfValues);

private:
    struct Internals;
    Internals *Data;
};

}

#endif
