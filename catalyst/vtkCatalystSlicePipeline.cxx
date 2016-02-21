#include "vtkCatalystSlicePipeline.h"

#include <vtkCPDataDescription.h>
#include <vtkCPInputDataDescription.h>
#include <vtkObjectFactory.h>
#include <vtkPVDataInformation.h>
#include <vtkPVDataSetAttributesInformation.h>
#include <vtkPVTrivialProducer.h>
#include <vtkSMProperty.h>
#include <vtkSMProxyListDomain.h>
#include <vtkSMTransferFunctionProxy.h>
#include <vtkPVArrayInformation.h>
#include <vtkMultiProcessController.h>
#include <vtkCommunicator.h>

#include "vtkCatalystUtilities.h"

class vtkCatalystSlicePipeline::vtkInternals
{
public:
  vtkSmartPointer<vtkSMSourceProxy> TrivialProducer;
  vtkSmartPointer<vtkSMSourceProxy> Slice;
  vtkSmartPointer<vtkSMProxy> SlicePlane;
  vtkSmartPointer<vtkSMViewProxy> RenderView;
  vtkSmartPointer<vtkSMProxy> SliceRepresentation;
  double Origin[3];
  double Normal[3];
  bool PipelineCreated;
  int ColorAssociation;
  std::string ColorArrayName;

  vtkInternals() : PipelineCreated(false), ColorAssociation(0)
  {
  }

  void UpdatePipeline(vtkDataObject* data, double time)
    {
    if (!this->PipelineCreated)
      {
      this->TrivialProducer = catalyst::CreatePipelineProxy("sources", "PVTrivialProducer");
      vtkPVTrivialProducer *tp = vtkPVTrivialProducer::SafeDownCast(
        this->TrivialProducer->GetClientSideObject());
      tp->SetOutput(data, time);

      this->Slice = catalyst::CreatePipelineProxy("filters", "Cut", this->TrivialProducer);
      vtkSMProxyListDomain* pld = vtkSMProxyListDomain::SafeDownCast(
        this->Slice->GetProperty("CutFunction")->FindDomain("vtkSMProxyListDomain"));
      this->SlicePlane = pld->FindProxy("implicit_functions", "Plane");
      vtkSMPropertyHelper(this->Slice, "CutFunction").Set(this->SlicePlane);
      this->Slice->UpdateVTKObjects();

      this->RenderView = catalyst::CreateViewProxy("views", "RenderView");
      vtkSMPropertyHelper(this->RenderView, "ShowAnnotation", true).Set(1);
      vtkSMPropertyHelper(this->RenderView, "ViewTime").Set(time);
      this->RenderView->UpdateVTKObjects();

      this->SliceRepresentation = catalyst::Show(this->Slice, this->RenderView);
      this->PipelineCreated = true;
      }
    else
      {
      vtkPVTrivialProducer *tp = vtkPVTrivialProducer::SafeDownCast(
        this->TrivialProducer->GetClientSideObject());
      tp->SetOutput(data, time);
      }

    vtkSMPropertyHelper(this->SlicePlane, "Origin").Set(this->Origin, 3);
    vtkSMPropertyHelper(this->SlicePlane, "Normal").Set(this->Normal, 3);
    this->SlicePlane->UpdateVTKObjects();

    vtkSMPropertyHelper(this->RenderView, "ViewTime").Set(time);
    this->RenderView->UpdateVTKObjects();

    this->Slice->UpdatePipeline(time);


    vtkSMPVRepresentationProxy::SetScalarColoring(
      this->SliceRepresentation, this->ColorArrayName.c_str(), this->ColorAssociation);
    if (vtkPVArrayInformation* ai = vtkSMPVRepresentationProxy::GetArrayInformationForColorArray(
      this->SliceRepresentation))
      {
      double range[2], grange[2];
      ai->GetComponentRange(-1, range);
      range[0] *= -1; // make range[0] negative to simplify reduce.
      vtkMultiProcessController::GetGlobalController()->AllReduce(range, grange, 2, vtkCommunicator::MAX_OP);
      grange[0] *= -1;
      vtkSMTransferFunctionProxy::RescaleTransferFunction(
        vtkSMPropertyHelper(this->SliceRepresentation, "LookupTable").GetAsProxy(), grange[0], grange[1]);
      }
    vtkSMRenderViewProxy::SafeDownCast(this->RenderView)->ResetCamera();
    this->RenderView->StillRender();
    }

};

vtkStandardNewMacro(vtkCatalystSlicePipeline);
//----------------------------------------------------------------------------
vtkCatalystSlicePipeline::vtkCatalystSlicePipeline()
{
  this->Internals = new vtkInternals();
}

//----------------------------------------------------------------------------
vtkCatalystSlicePipeline::~vtkCatalystSlicePipeline()
{
  vtkInternals& internals = (*this->Internals);
  catalyst::DeletePipelineProxy(internals.Slice);
  catalyst::DeletePipelineProxy(internals.TrivialProducer);
  delete this->Internals;
}

//----------------------------------------------------------------------------
void vtkCatalystSlicePipeline::SetSliceOrigin(double x, double y, double z)
{
  vtkInternals& internals = (*this->Internals);
  this->Internals->Origin[0] = x;
  this->Internals->Origin[1] = y;
  this->Internals->Origin[2] = z;
}

//----------------------------------------------------------------------------
void vtkCatalystSlicePipeline::SetSliceNormal(double x, double y, double z)
{
  vtkInternals& internals = (*this->Internals);
  this->Internals->Normal[0] = x;
  this->Internals->Normal[1] = y;
  this->Internals->Normal[2] = z;
}

//----------------------------------------------------------------------------
void vtkCatalystSlicePipeline::ColorBy(int association, const char* arrayname)
{
  vtkInternals& internals = (*this->Internals);
  internals.ColorArrayName = arrayname? arrayname : "";
  internals.ColorAssociation = association;
}

//----------------------------------------------------------------------------
int vtkCatalystSlicePipeline::RequestDataDescription(vtkCPDataDescription* dataDesc)
{
  dataDesc->GetInputDescription(0)->GenerateMeshOn();
  dataDesc->GetInputDescription(0)->AllFieldsOn();
  return 1;
}

//----------------------------------------------------------------------------
int vtkCatalystSlicePipeline::CoProcess(vtkCPDataDescription* dataDesc)
{
  vtkInternals& internals = (*this->Internals);
  internals.UpdatePipeline(dataDesc->GetInputDescription(0)->GetGrid(), dataDesc->GetTime());
  return 1;
}

//----------------------------------------------------------------------------
int vtkCatalystSlicePipeline::Finalize()
{
  return 1;
}

//----------------------------------------------------------------------------
void vtkCatalystSlicePipeline::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}
