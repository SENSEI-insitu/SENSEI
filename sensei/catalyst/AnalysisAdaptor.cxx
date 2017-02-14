#include "AnalysisAdaptor.h"

#include <sensei/DataAdaptor.h>
#include <timer/Timer.h>

#include <vtkCommunicator.h>
#include <vtkCPAdaptorAPI.h>
#include <vtkCPDataDescription.h>
#include <vtkCPInputDataDescription.h>
#include <vtkCPProcessor.h>
#include <vtkDataObject.h>
#include <vtkImageData.h>
#include <vtkMultiProcessController.h>
#include <vtkObjectFactory.h>
#include <vtkRectilinearGrid.h>
#include <vtkStructuredGrid.h>

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
    desc->AddPointField(dA->GetArrayName(vtkDataObject::POINT, cc).c_str());
    }
  for (unsigned int cc=0, max=dA->GetNumberOfArrays(vtkDataObject::CELL); cc<max;++cc)
    {
    desc->AddCellField(dA->GetArrayName(vtkDataObject::CELL, cc).c_str());
    }

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
      std::string aname = dA->GetArrayName(attr, cc);
      if (desc->GetAllFields() || desc->IsFieldNeeded(aname.c_str()))
        {
        dA->AddArray(mesh, attr, aname);
        }
      }
    }

  desc->SetGrid(mesh);

  if (mesh->IsA("vtkImageData") || mesh->IsA("vtkRectilinearGrid") ||
      mesh->IsA("vtkStructuredGrid") )
    {
    int wholeExtent[6], localExtent[6];
    if (vtkImageData* id = vtkImageData::SafeDownCast(mesh))
      {
      id->GetExtent(localExtent);
      }
    else if(vtkRectilinearGrid* rg = vtkRectilinearGrid::SafeDownCast(mesh))
      {
      rg->GetExtent(localExtent);
      }
    else if(vtkStructuredGrid* sg = vtkStructuredGrid::SafeDownCast(mesh))
      {
      sg->GetExtent(localExtent);
      }
    vtkMultiProcessController* c =
      vtkMultiProcessController::GetGlobalController();
    localExtent[0] = -localExtent[0];
    localExtent[2] = -localExtent[2];
    localExtent[4] = -localExtent[4];
    c->AllReduce(localExtent, wholeExtent, 6, vtkCommunicator::MAX_OP);
    wholeExtent[0] = -wholeExtent[0];
    wholeExtent[2] = -wholeExtent[2];
    wholeExtent[4] = -wholeExtent[4];
    desc->SetWholeExtent(wholeExtent);
    }

  return true;
}

//-----------------------------------------------------------------------------
void AnalysisAdaptor::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

} // catalyst
} // sensei
