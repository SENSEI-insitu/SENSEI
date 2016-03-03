#include "AnalysisAdaptor.h"

#include <sensei/DataAdaptor.h>
#include <timer/Timer.h>

#include <vtkCPAdaptorAPI.h>
#include <vtkCPDataDescription.h>
#include <vtkCPInputDataDescription.h>
#include <vtkCPProcessor.h>
#include <vtkDataObject.h>
#include <vtkObjectFactory.h>

namespace sensei
{
namespace catalyst
{

static size_t vtkCPAdaptorAPIInitializationCounter = 0;

vtkStandardNewMacro(AnalysisAdaptor);
//-----------------------------------------------------------------------------
AnalysisAdaptor::AnalysisAdaptor()
{
  if (vtkCPAdaptorAPIInitializationCounter == 0)
    {
    timer::MarkEvent mark("catalyst::initialize");
    vtkCPAdaptorAPI::CoProcessorInitialize();
    }
  vtkCPAdaptorAPIInitializationCounter++;
}

//-----------------------------------------------------------------------------
AnalysisAdaptor::~AnalysisAdaptor()
{
  vtkCPAdaptorAPIInitializationCounter--;
  if (vtkCPAdaptorAPIInitializationCounter == 0)
    {
    timer::MarkEvent mark("catalyst::finalize");
    vtkCPAdaptorAPI::CoProcessorFinalize();
    }
}

//-----------------------------------------------------------------------------
void AnalysisAdaptor::AddPipeline(vtkCPPipeline* pipeline)
{
  if (pipeline)
    {
    vtkCPAdaptorAPI::GetCoProcessor()->AddPipeline(pipeline);
    }
}

//-----------------------------------------------------------------------------
bool AnalysisAdaptor::Execute(DataAdaptor* dataAdaptor)
{
  timer::MarkEvent mark("catalyst::execute");
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
bool AnalysisAdaptor::FillDataDescriptionWithMetaData(
  DataAdaptor* dA, vtkCPInputDataDescription* desc)
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
bool AnalysisAdaptor::FillDataDescriptionWithData(
  DataAdaptor* dA, vtkCPInputDataDescription* desc)
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
void AnalysisAdaptor::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

} // catalyst
} // sensei
