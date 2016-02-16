#include "vtkCatalystSlicePipeline.h"

#include <vtkCPDataDescription.h>
#include <vtkCPInputDataDescription.h>
#include <vtkObjectFactory.h>
#include <vtkPVDataInformation.h>
#include <vtkPVDataSetAttributesInformation.h>
#include <vtkPVTrivialProducer.h>
#include <vtkSMProperty.h>
#include <vtkSMProxyListDomain.h>

#include "vtkCatalystUtilities.h"

class vtkCatalystSlicePipeline::vtkInternals
{
public:
  vtkSmartPointer<vtkSMSourceProxy> TrivialProducer;
  vtkSmartPointer<vtkSMSourceProxy> Slice;
  vtkSmartPointer<vtkSMProxy> SlicePlane;
  vtkSmartPointer<vtkSMProxy> RenderView;

  double Origin[3];
  double Normal[3];

  void UpdatePipeline(vtkDataObject* data, double time)
    {
    if (!this->TrivialProducer)
      {
      this->TrivialProducer = catalyst::CreatePipelineProxy("sources", "PVTrivialProducer");
      }
    vtkPVTrivialProducer *tp = vtkPVTrivialProducer::SafeDownCast(
      this->TrivialProducer->GetClientSideObject());
    tp->SetOutput(data, time);
    if (!this->Slice)
      {
      this->Slice = catalyst::CreatePipelineProxy("filters", "Cut", this->TrivialProducer);
      vtkSMProxyListDomain* pld = vtkSMProxyListDomain::SafeDownCast(
        this->Slice->GetProperty("CutFunction")->FindDomain("vtkSMProxyListDomain"));
      this->SlicePlane = pld->FindProxy("implicit_functions", "Plane");
      vtkSMPropertyHelper(this->Slice, "CutFunction").Set(this->SlicePlane);
      this->Slice->UpdateVTKObjects();
      }
    vtkSMPropertyHelper(this->SlicePlane, "Origin").Set(this->Origin, 3);
    vtkSMPropertyHelper(this->SlicePlane, "Normal").Set(this->Normal, 3);
    this->SlicePlane->UpdateVTKObjects();

    if (!this->RenderView)
      {
      this->RenderView = catalyst::CreateViewProxy("views", "RenderView");
      }
    catalyst::ShowAndRender(this->Slice, this->RenderView, time);
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
}

//----------------------------------------------------------------------------
void vtkCatalystSlicePipeline::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}
