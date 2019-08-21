#include "CatalystParticle.h"
#include "CatalystUtilities.h"
#include <Timer.h>
#include <Error.h>

#include <vtkCPDataDescription.h>
#include <vtkCPInputDataDescription.h>
#include <vtkObjectFactory.h>
#include <vtkPVDataInformation.h>
#include <vtkPVDataSetAttributesInformation.h>
#include <vtkPVTrivialProducer.h>
#include <vtkSMProperty.h>
#include <vtkSMProxyListDomain.h>
#include <vtkSMRepresentationProxy.h>
#include <vtkSMTransferFunctionProxy.h>
#include <vtkPVArrayInformation.h>
#include <vtkMultiProcessController.h>
#include <vtkCommunicator.h>
#include <vtkSMViewProxy.h>
#include <vtkSMPropertyHelper.h>
#include <vtkSMPVRepresentationProxy.h>
#include <vtkSMRenderViewProxy.h>
#include <vtkVector.h>

#include <iomanip>
#include <sstream>
#include <cassert>

namespace sensei
{

class CatalystParticle::vtkInternals
{
public:
  vtkSmartPointer<vtkSMSourceProxy> TrivialProducer;
  vtkSmartPointer<vtkSMSourceProxy> Particle;
  // vtkSmartPointer<vtkSMProxy> ParticlePlane;
  vtkSmartPointer<vtkSMViewProxy> RenderView;
  vtkSmartPointer<vtkSMRepresentationProxy> ParticleRepresentation;
  bool PipelineCreated;
  int ColorAssociation;
  std::string ColorArrayName;
  std::string ParticleStyle;
  float ParticleRadius;
  vtkVector3d CameraPosition;
  vtkVector3d CameraFocus;
  bool ShouldResetCamera;
  bool UseLogScale;
  std::string Mesh;

  double ColorRange[2];
  bool AutoColorRange;

  std::string ImageFileName;
  int ImageSize[2];

  vtkInternals() : PipelineCreated(false), ColorAssociation(0),
    ParticleStyle("Sphere"), ParticleRadius(1.0f),
    CameraPosition(1.f,1.f,1.f), CameraFocus(0.f,0.f,0.f),
    ShouldResetCamera(true),
    UseLogScale(false), AutoColorRange(true)
  {
    this->ColorRange[0] = 0; this->ColorRange[1] = 1.0;
    this->ImageSize[0] = this->ImageSize[1] = 800;
  }

  bool EnableRendering() const
  {
    return !this->ImageFileName.empty();
  }

  void UpdatePipeline(vtkDataObject* data, int timestep, double time)
  {
    if (!this->PipelineCreated)
      {
      this->TrivialProducer = catalyst::CreatePipelineProxy("sources", "PVTrivialProducer");
      vtkPVTrivialProducer* tp = vtkPVTrivialProducer::SafeDownCast(
        this->TrivialProducer->GetClientSideObject());
      tp->SetOutput(data, time);

      if (this->EnableRendering())
        {
        this->RenderView = catalyst::CreateViewProxy("views", "RenderView");
        // uncomment below if we want to see the frame rate and other annotation information
        //vtkSMPropertyHelper(this->RenderView, "ShowAnnotation", true).Set(1);
        vtkSMPropertyHelper(this->RenderView, "ViewTime").Set(time);
        vtkSMPropertyHelper(this->RenderView, "ViewSize").Set(this->ImageSize, 2);
        this->RenderView->UpdateVTKObjects();

        this->ParticleRepresentation = catalyst::Show(this->TrivialProducer, this->RenderView);
        vtkSMPropertyHelper(this->ParticleRepresentation, "Representation").Set("Point Gaussian"); // .Set("3D Glyphs");
        vtkSMPropertyHelper(this->ParticleRepresentation, "ShaderPreset").Set(this->ParticleStyle.c_str());
        vtkSMPropertyHelper(this->ParticleRepresentation, "GaussianRadius").Set(this->ParticleRadius);
        }

      this->Particle = nullptr;
      this->PipelineCreated = true;
      }
    else
      {
      vtkPVTrivialProducer *tp = vtkPVTrivialProducer::SafeDownCast(
        this->TrivialProducer->GetClientSideObject());
      tp->SetOutput(data, time);

      this->Particle = catalyst::CreatePipelineProxy("filters", "Glyph", this->TrivialProducer);
      // vtkSMPropertyHelper(this->Particle, "GlyphType").Set("Sphere");
      this->Particle->UpdateVTKObjects();
      this->PipelineCreated = true;
      }

    vtkMultiProcessController* controller = vtkMultiProcessController::GetGlobalController();

    if (this->EnableRendering())
      {
      vtkSMPropertyHelper(this->RenderView, "ViewTime").Set(time);
      this->RenderView->UpdateVTKObjects();
      vtkSMPVRepresentationProxy::SetScalarColoring(
        this->ParticleRepresentation, this->ColorArrayName.c_str(), this->ColorAssociation);
      if (vtkSMPVRepresentationProxy::GetUsingScalarColoring(this->ParticleRepresentation))
        {
        // Request an explicit update to ensure representation gives us valid data information.
        this->RenderView->Update();

        double range[2] = {VTK_DOUBLE_MAX, VTK_DOUBLE_MIN};
        if (this->AutoColorRange)
          {
          // Here, we use RepresentedDataInformation so that we get the range
          // for the geometry after ghost elements have been pruned.
          if (vtkPVArrayInformation* ai =
            this->ParticleRepresentation->GetRepresentedDataInformation()->
            GetArrayInformation(this->ColorArrayName.c_str(), this->ColorAssociation))
            {
            ai->GetComponentRange(-1, range);
            }
          range[0] *= -1; // make range[0] negative to simplify reduce.
          double grange[2];
          controller->AllReduce(range, grange, 2, vtkCommunicator::MAX_OP);
          grange[0] *= -1;
          std::copy(grange, grange+2, range);
          }
        else
          {
          std::copy(this->ColorRange, this->ColorRange+2, range);
          }
        vtkSMTransferFunctionProxy* lut = vtkSMTransferFunctionProxy::SafeDownCast(
          vtkSMPropertyHelper(this->ParticleRepresentation, "LookupTable").GetAsProxy());
        lut->RescaleTransferFunction(range[0], range[1]);

        bool use_log_scale = this->UseLogScale && (range[0] > 0.0);
        vtkSMPropertyHelper ulsHelper(lut, "UseLogScale");
        if (ulsHelper.GetAsInt() != (use_log_scale? 1: 0))
          {
          lut->MapControlPointsToLogSpace(/*inverse=*/!use_log_scale);
          ulsHelper.Set(use_log_scale? 1 : 0);
          }
        }
      vtkSMRenderViewProxy* renderViewProxy = vtkSMRenderViewProxy::SafeDownCast(this->RenderView);
      vtkSMPropertyHelper(renderViewProxy, "CameraPosition").Set(this->CameraPosition.GetData(), 3);
      vtkSMPropertyHelper(renderViewProxy, "CameraFocalPoint").Set(this->CameraFocus.GetData(), 3);

      if (this->ShouldResetCamera)
      {
        renderViewProxy->ResetCamera();
      }

      std::string filename = this->ImageFileName;

      // replace "%ts" with timestep in filename
      std::ostringstream ts_stream;
      ts_stream << setfill('0') << setw(4) << timestep;
      std::string::size_type pos = filename.find("%ts");
      while (pos != std::string::npos)
        {
        filename.replace(pos, 3, ts_stream.str());
        pos = filename.find("%ts");
        }
      // replace "%t" with time in filename
      std::ostringstream t_stream;
      t_stream << time;
      pos = filename.find("%t");
      while (pos != std::string::npos)
        {
        filename.replace(pos, 2, t_stream.str());
        pos = filename.find("%t");
        }
      this->RenderView->WriteImage(filename.c_str(), "vtkPNGWriter", 1);
      }
  }
};

//----------------------------------------------------------------------------
vtkStandardNewMacro(CatalystParticle);

//----------------------------------------------------------------------------
CatalystParticle::CatalystParticle()
{
  this->Internals = new vtkInternals();
}

//----------------------------------------------------------------------------
CatalystParticle::~CatalystParticle()
{
  vtkInternals& internals = (*this->Internals);
  catalyst::DeletePipelineProxy(internals.Particle);
  catalyst::DeletePipelineProxy(internals.TrivialProducer);
  delete this->Internals;
}

//----------------------------------------------------------------------------
void CatalystParticle::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

//----------------------------------------------------------------------------
void CatalystParticle::SetInputMesh(const std::string& meshName)
{
  vtkInternals& internals = (*this->Internals);
  internals.Mesh = meshName;
}

//----------------------------------------------------------------------------
void CatalystParticle::SetParticleStyle(const std::string& style)
{
  vtkInternals& internals = (*this->Internals);
  internals.ParticleStyle = style;
}

//----------------------------------------------------------------------------
void CatalystParticle::SetParticleRadius(float radius)
{
  vtkInternals& internals = (*this->Internals);
  internals.ParticleRadius = radius;
}

//----------------------------------------------------------------------------
void CatalystParticle::SetCameraPosition(const double posn[3])
{
  vtkInternals& internals = (*this->Internals);
  internals.CameraPosition = vtkVector3d(posn);
  internals.ShouldResetCamera = false;
}

//----------------------------------------------------------------------------
void CatalystParticle::SetCameraFocus(const double focus[3])
{
  vtkInternals& internals = (*this->Internals);
  internals.CameraFocus = vtkVector3d(focus);
}

//----------------------------------------------------------------------------
void CatalystParticle::ColorBy(int association, const std::string& arrayname)
{
  vtkInternals& internals = (*this->Internals);
  internals.ColorArrayName = arrayname;
  internals.ColorAssociation = association;
}

//----------------------------------------------------------------------------
void CatalystParticle::SetImageParameters(const std::string& filename, int width, int height)
{
  vtkInternals& internals = (*this->Internals);
  internals.ImageFileName = filename;
  internals.ImageSize[0] = width;
  internals.ImageSize[1] = height;
}

//----------------------------------------------------------------------------
int CatalystParticle::RequestDataDescription(vtkCPDataDescription* dataDesc)
{
  vtkInternals& internals = (*this->Internals);
  auto dd = internals.Mesh.empty() ?
    dataDesc->GetInputDescription(0) :
    dataDesc->GetInputDescriptionByName(internals.Mesh.c_str());
  if (dd)
  {
    dd->GenerateMeshOn();
    dd->AllFieldsOn();
    return 1;
  }
  if (internals.Mesh.empty())
  {
    SENSEI_ERROR("Unable to obtain default dataset");
  }
  else
  {
    SENSEI_ERROR("Unable to obtain dataset \"" << internals.Mesh.c_str() << "\"");
  }
  return 0;
}

//----------------------------------------------------------------------------
int CatalystParticle::CoProcess(vtkCPDataDescription* dataDesc)
{
  Timer::MarkEvent mark("catalyst::slice");
  vtkInternals& internals = (*this->Internals);
  auto dd = internals.Mesh.empty() ?
    dataDesc->GetInputDescription(0) :
    dataDesc->GetInputDescriptionByName(internals.Mesh.c_str());
  if (dd)
  {
    internals.UpdatePipeline(dd->GetGrid(),
      dataDesc->GetTimeStep(), dataDesc->GetTime());
    return 1;
  }
  return 0;
}

//----------------------------------------------------------------------------
int CatalystParticle::Finalize()
{
  return 1;
}

//----------------------------------------------------------------------------
void CatalystParticle::SetAutoColorRange(bool val)
{
  vtkInternals& internals = (*this->Internals);
  internals.AutoColorRange = val;
}

//----------------------------------------------------------------------------
void CatalystParticle::SetColorRange(double min, double max)
{
  assert(min <= max);
  vtkInternals& internals = (*this->Internals);
  internals.ColorRange[0] = min;
  internals.ColorRange[1] = max;
}

//----------------------------------------------------------------------------
bool CatalystParticle::GetAutoColorRange() const
{
  vtkInternals& internals = (*this->Internals);
  return internals.AutoColorRange;
}

//----------------------------------------------------------------------------
const double* CatalystParticle::GetColorRange() const
{
  vtkInternals& internals = (*this->Internals);
  return internals.ColorRange;
}

//----------------------------------------------------------------------------
void CatalystParticle::SetUseLogScale(bool val)
{
  vtkInternals& internals = (*this->Internals);
  internals.UseLogScale = val;
}

//----------------------------------------------------------------------------
bool CatalystParticle::GetUseLogScale() const
{
  vtkInternals& internals = (*this->Internals);
  return internals.UseLogScale;
}

}
