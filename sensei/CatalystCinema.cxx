#include "CatalystCinema.h"
#include "CatalystUtilities.h"
#include <Timer.h>

// Debug
#include <vtkSMDoubleVectorProperty.h>

#include <vtkCellData.h>
#include <vtkCommunicator.h>
#include <vtkCPDataDescription.h>
#include <vtkCPInputDataDescription.h>
#include <vtkDataArray.h>
#include <vtkDataObject.h>
#include <vtkMultiBlockDataSet.h>
#include <vtkMultiProcessController.h>
#include <vtkObjectFactory.h>
#include <vtkPVArrayInformation.h>
#include <vtkPVDataInformation.h>
#include <vtkPVDataSetAttributesInformation.h>
#include <vtkPVTrivialProducer.h>
#include <vtkSMProperty.h>
#include <vtkSMPropertyHelper.h>
#include <vtkSMProxyListDomain.h>
#include <vtkSMPVRepresentationProxy.h>
#include <vtkSMRenderViewProxy.h>
#include <vtkSMRepresentationProxy.h>
#include <vtkSMTransferFunctionProxy.h>
#include <vtkSMViewProxy.h>
#include <vtkImageData.h>

#include <CinemaHelper.h>

#include <sstream>
#include <cassert>
#include <vector>

namespace sensei
{

class CatalystCinema::vtkInternals
{
public:
  vtkSmartPointer<vtkSMSourceProxy> TrivialProducer;
  std::vector<vtkSmartPointer<vtkSMSourceProxy>> Sources;
  std::vector<vtkSmartPointer<vtkSMRepresentationProxy>> Representations;
  vtkSmartPointer<vtkSMViewProxy> RenderView;
  bool PipelineCreated;
  std::string BasePath;
  int ImageSize[2];
  CinemaHelper Helper;

  vtkInternals() : PipelineCreated(false)
  {
    this->ImageSize[0] = this->ImageSize[1] = 512;
  }

  bool EnableRendering() const
  {
    return !this->BasePath.empty();
  }

  void AddContour(double value)
  {
    vtkSmartPointer<vtkSMSourceProxy> cellToPoint = catalyst::CreatePipelineProxy("filters", "CellDataToPointData", this->TrivialProducer);
    this->Sources.push_back(cellToPoint);

    vtkSmartPointer<vtkSMSourceProxy> contour = catalyst::CreatePipelineProxy("filters", "Contour", cellToPoint);
    this->Sources.push_back(contour);

    // Configure contour
    vtkSMPropertyHelper(contour, "ContourValues").Set(value);
    contour->UpdateVTKObjects();

    if (this->EnableRendering() && this->RenderView)
        {
        std::cout << "Add representation for contour on " << value << std::endl;
        vtkSmartPointer<vtkSMRepresentationProxy> representation = catalyst::Show(contour, this->RenderView);
        this->Representations.push_back(representation);
        }
  }

  void UpdatePipeline(vtkDataObject* data, int timestep, double time)
  {
    this->Helper.AddTimeEntry();

    if (!this->PipelineCreated)
      {
      this->TrivialProducer = catalyst::CreatePipelineProxy("sources", "PVTrivialProducer");
      vtkPVTrivialProducer *tp = vtkPVTrivialProducer::SafeDownCast(
        this->TrivialProducer->GetClientSideObject());
      tp->SetOutput(data, time);

      if (this->EnableRendering())
        {
        this->RenderView = catalyst::CreateViewProxy("views", "RenderView");
        vtkSMPropertyHelper(this->RenderView, "CenterAxesVisibility").Set(0);
        vtkSMPropertyHelper(this->RenderView, "OrientationAxesVisibility").Set(0);
        vtkSMPropertyHelper(this->RenderView, "ViewSize").Set(this->ImageSize, 2);
        this->RenderView->UpdateVTKObjects();

        vtkSmartPointer<vtkSMRepresentationProxy> representation = catalyst::Show(this->TrivialProducer, this->RenderView);
        this->Representations.push_back(representation);
        vtkSMPropertyHelper(representation, "Representation").Set("Outline");
        representation->UpdateVTKObjects();

        this->Helper.RegisterLayer("Outline", representation, -1);
        }

      // Add countours
      for (double value = 0.1; value < 0.99; value += 0.1)
        {
        this->AddContour(value);
        this->Helper.RegisterLayer("Contour", this->Representations.back(), value);
        }

      this->PipelineCreated = true;
      }
    else
      {
      vtkPVTrivialProducer *tp = vtkPVTrivialProducer::SafeDownCast(
        this->TrivialProducer->GetClientSideObject());
      tp->SetOutput(data, time);
      }

    // Update pipeline
    int size = this->Sources.size();
    for (int i = 0; i < size; i++)
        {
        this->Sources[i]->UpdatePipeline(time);
        }

    if (this->EnableRendering())
      {
      vtkSMPropertyHelper(this->RenderView, "ViewTime").Set(time);
      this->RenderView->UpdateVTKObjects();

      // vtkSMRenderViewProxy::SafeDownCast(this->RenderView)->ResetCamera();
      // this->RenderView->UpdatePropertyInformation();
      // std::cout
      //   << vtkSMDoubleVectorProperty::SafeDownCast(this->RenderView->GetProperty("CameraFocalPoint"))->GetElement(0) << ", "
      //   << vtkSMDoubleVectorProperty::SafeDownCast(this->RenderView->GetProperty("CameraFocalPoint"))->GetElement(1) << ", "
      //   << vtkSMDoubleVectorProperty::SafeDownCast(this->RenderView->GetProperty("CameraFocalPoint"))->GetElement(2)
      //   << std::endl;
      // std::cout
      //   << vtkSMDoubleVectorProperty::SafeDownCast(this->RenderView->GetProperty("CameraPosition"))->GetElement(0) << ", "
      //   << vtkSMDoubleVectorProperty::SafeDownCast(this->RenderView->GetProperty("CameraPosition"))->GetElement(1) << ", "
      //   << vtkSMDoubleVectorProperty::SafeDownCast(this->RenderView->GetProperty("CameraPosition"))->GetElement(2)
      //   << std::endl;
      // std::cout
      //   << vtkSMDoubleVectorProperty::SafeDownCast(this->RenderView->GetProperty("CameraViewUp"))->GetElement(0) << ", "
      //   << vtkSMDoubleVectorProperty::SafeDownCast(this->RenderView->GetProperty("CameraViewUp"))->GetElement(1) << ", "
      //   << vtkSMDoubleVectorProperty::SafeDownCast(this->RenderView->GetProperty("CameraViewUp"))->GetElement(2)
      //   << std::endl;
      // this->Helper.CaptureImage(this->RenderView, "capture.jpg", "vtkJPEGWriter");

      int nbCameraIdx = this->Helper.GetNumberOfCameraPositions();
      for (int cameraIdx = 0; cameraIdx < nbCameraIdx; cameraIdx++)
        {
        this->Helper.ApplyCameraPosition(this->RenderView, cameraIdx);
        this->Helper.Capture(this->RenderView);
        }
      }

    this->Helper.WriteMetadata();
  }
};

//----------------------------------------------------------------------------
vtkStandardNewMacro(CatalystCinema);

//----------------------------------------------------------------------------
CatalystCinema::CatalystCinema()
{
  this->Internals = new vtkInternals();
}

//----------------------------------------------------------------------------
CatalystCinema::~CatalystCinema()
{
  vtkInternals& internals = (*this->Internals);
  catalyst::DeletePipelineProxy(internals.TrivialProducer);
  while (internals.Sources.size())
    {
    catalyst::DeletePipelineProxy(internals.Sources.back());
    internals.Sources.pop_back();
    }
  delete this->Internals;
}

//----------------------------------------------------------------------------
void CatalystCinema::SetImageParameters(const std::string& basepath, int width, int height)
{
  vtkInternals& internals = (*this->Internals);

  // to-be-removed ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  internals.BasePath = basepath;
  internals.ImageSize[0] = width;
  internals.ImageSize[1] = height;
  // to-be-removed ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  internals.Helper.SetImageSize(width, height);
  internals.Helper.SetWorkingDirectory(basepath);
}

//----------------------------------------------------------------------------
void CatalystCinema::SetCameraConfiguration(const std::string& config)
{
  vtkInternals& internals = (*this->Internals);
  internals.Helper.SetCameraConfig(config);
}

//----------------------------------------------------------------------------
void CatalystCinema::SetExportType(const std::string& exportType)
{
  vtkInternals& internals = (*this->Internals);
  internals.Helper.SetExportType(exportType);
}

//----------------------------------------------------------------------------
int CatalystCinema::RequestDataDescription(vtkCPDataDescription* dataDesc)
{
  dataDesc->GetInputDescription(0)->GenerateMeshOn();
  dataDesc->GetInputDescription(0)->AllFieldsOn();
  return 1;
}

//----------------------------------------------------------------------------
int CatalystCinema::CoProcess(vtkCPDataDescription* dataDesc)
{
  timer::MarkEvent mark("catalyst::cinema");
  vtkInternals& internals = (*this->Internals);
  internals.UpdatePipeline(dataDesc->GetInputDescription(0)->GetGrid(),
    dataDesc->GetTimeStep(), dataDesc->GetTime());
  return 1;
}

//----------------------------------------------------------------------------
int CatalystCinema::Finalize()
{
  return 1;
}

//----------------------------------------------------------------------------
void CatalystCinema::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

}
