#include "vtkCatalystAnalysisAdaptor.h"

#include <vtkInsituDataAdaptor.h>

#include <vtkCPAdaptorAPI.h>
#include <vtkCPDataDescription.h>
#include <vtkCPInputDataDescription.h>
#include <vtkCPProcessor.h>
#include <vtkDataObject.h>
#include <vtkObjectFactory.h>

namespace
{
  static size_t vtkCPAdaptorAPIInitializationCounter = 0;
}

vtkStandardNewMacro(vtkCatalystAnalysisAdaptor);
//-----------------------------------------------------------------------------
vtkCatalystAnalysisAdaptor::vtkCatalystAnalysisAdaptor()
{
  if (vtkCPAdaptorAPIInitializationCounter == 0)
    {
    vtkCPAdaptorAPI::CoProcessorInitialize();
    }
  vtkCPAdaptorAPIInitializationCounter++;
}

//-----------------------------------------------------------------------------
vtkCatalystAnalysisAdaptor::~vtkCatalystAnalysisAdaptor()
{
  vtkCPAdaptorAPIInitializationCounter--;
  if (vtkCPAdaptorAPIInitializationCounter == 0)
    {
    vtkCPAdaptorAPI::CoProcessorFinalize();
    }
}

//-----------------------------------------------------------------------------
void vtkCatalystAnalysisAdaptor::AddPipeline(vtkCPPipeline* pipeline)
{
  if (pipeline)
    {
    vtkCPAdaptorAPI::GetCoProcessor()->AddPipeline(pipeline);
    }
}

//-----------------------------------------------------------------------------
bool vtkCatalystAnalysisAdaptor::Execute(vtkInsituDataAdaptor* dataAdaptor)
{
  double time = dataAdaptor->GetDataTime();
  int timeStep = dataAdaptor->GetDataTimeStep();
  int coprocessThisTimeStep;

  vtkCPInputDataDescription* inputDesc =
    vtkCPAdaptorAPI::GetCoProcessorData()->GetInputDescription(0);
  if (!this->FillDataDescriptionWithMetaData(dataAdaptor, inputDesc))
    {
    return false;
    }
  vtkCPAdaptorAPI::RequestDataDescription(&timeStep, &time, &coprocessThisTimeStep);
  if (coprocessThisTimeStep == 1)
    {
    if (!this->FillDataDescriptionWithData(dataAdaptor, inputDesc))
      {
      return false;
      }
    vtkCPAdaptorAPI::CoProcess();
    vtkCPAdaptorAPI::GetCoProcessorData()->ResetAll();
    }
  return true;
}

//-----------------------------------------------------------------------------
bool vtkCatalystAnalysisAdaptor::FillDataDescriptionWithMetaData(
  vtkInsituDataAdaptor* dA, vtkCPInputDataDescription* desc)
{
  desc->Reset();
  for (unsigned int cc=0, max=dA->GetNumberOfArrays(vtkDataObject::POINT); cc<max;++cc)
    {
    desc->AddPointField(dA->GetArrayName(vtkDataObject::POINT, cc));
    }
  for (unsigned int cc=0, max=dA->GetNumberOfArrays(vtkDataObject::CELL); cc<max;++cc)
    {
    desc->AddCellField(dA->GetArrayName(vtkDataObject::CELL, cc));
    }

  // XXX(todo): Add whole extent, if available.
  return true;
}

//-----------------------------------------------------------------------------
bool vtkCatalystAnalysisAdaptor::FillDataDescriptionWithData(
  vtkInsituDataAdaptor* dA, vtkCPInputDataDescription* desc)
{
  bool structure_only = desc->GetGenerateMesh()? false : true;
  vtkDataObject* mesh = dA->GetMesh(structure_only);

  for (int attr=vtkDataObject::POINT; attr<=vtkDataObject::CELL; ++attr)
    {
    for (unsigned int cc=0, max=dA->GetNumberOfArrays(attr); cc < max; ++cc)
      {
      const char* aname = dA->GetArrayName(attr, cc);
      if (desc->GetAllFields() || desc->IsFieldNeeded(aname))
        {
        dA->AddArray(mesh, attr, aname);
        }
      }
    }
  desc->SetGrid(mesh);
  return true;
}

//-----------------------------------------------------------------------------
void vtkCatalystAnalysisAdaptor::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}
