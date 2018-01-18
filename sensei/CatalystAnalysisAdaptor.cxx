#include "CatalystAnalysisAdaptor.h"

#include "DataAdaptor.h"
#include "DataRequirements.h"
#include "VTKUtils.h"
#include "Error.h"
#include "Timer.h"

#include <vtkSmartPointer.h>
#include <vtkNew.h>
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
#ifdef ENABLE_CATALYST_PYTHON
#include <vtkCPPythonScriptPipeline.h>
#endif

namespace sensei
{

static int vtkCPAdaptorAPIInitializationCounter = 0;

//-----------------------------------------------------------------------------
senseiNewMacro(CatalystAnalysisAdaptor);

//-----------------------------------------------------------------------------
CatalystAnalysisAdaptor::CatalystAnalysisAdaptor()
{
  this->Initialize();
}

//-----------------------------------------------------------------------------
CatalystAnalysisAdaptor::~CatalystAnalysisAdaptor()
{
  this->Finalize();
}

//-----------------------------------------------------------------------------
void CatalystAnalysisAdaptor::Initialize()
{
  if (vtkCPAdaptorAPIInitializationCounter == 0)
    {
    timer::MarkEvent mark("CatalystAnalysisAdaptor::Initialize");
    vtkCPAdaptorAPI::CoProcessorInitialize();
    }
  vtkCPAdaptorAPIInitializationCounter++;
}

//-----------------------------------------------------------------------------
void CatalystAnalysisAdaptor::AddPipeline(vtkCPPipeline* pipeline)
{
  if (pipeline)
    {
    vtkCPAdaptorAPI::GetCoProcessor()->AddPipeline(pipeline);
    }
}

//-----------------------------------------------------------------------------
void CatalystAnalysisAdaptor::AddPythonScriptPipeline(
  const std::string &fileName)
{
#ifdef ENABLE_CATALYST_PYTHON
  vtkNew<vtkCPPythonScriptPipeline> pythonPipeline;
  pythonPipeline->Initialize(fileName.c_str());
  this->AddPipeline(pythonPipeline.GetPointer());
#else
  (void)fileName;
  SENSEI_ERROR("Failed to add Python script pipeline. "
    "Re-compile with ENABLE_CATALYST_PYTHON=ON")
#endif
}

// ---------------------------------------------------------------------------
int CatalystAnalysisAdaptor::DescribeData(int timeStep, double time,
  const DataRequirements &reqs, vtkCPDataDescription *dataDesc)
{
  dataDesc->SetTimeData(time, timeStep);

  // pass metadata from sim into Catalyst
  MeshRequirementsIterator mit =
    reqs.GetMeshRequirementsIterator();

  for (; mit; ++mit)
    {
    // add the mesh
    std::string meshName = mit.MeshName();

    dataDesc->AddInput(meshName.c_str());

    vtkCPInputDataDescription *inDesc =
      dataDesc->GetInputDescriptionByName(meshName.c_str());

    // add the available arrays
    ArrayRequirementsIterator ait =
      reqs.GetArrayRequirementsIterator(meshName);

    for (; ait; ++ait)
      {
      int assoc = ait.Association();
      if (assoc == vtkDataObject::POINT)
        inDesc->AddPointField(ait.Array().c_str());
      else if (assoc == vtkDataObject::CELL)
        inDesc->AddCellField(ait.Array().c_str());
      else
        SENSEI_WARNING("Unknown association " << assoc)
      }

    // let Catalyst tell us which arrays are needed
    inDesc->AllFieldsOff();
    }

  return 0;
}

// ---------------------------------------------------------------------------
int CatalystAnalysisAdaptor::SelectData(DataAdaptor *dataAdaptor,
  const DataRequirements &reqs, vtkCPDataDescription *dataDesc)
{
  MeshRequirementsIterator mit = reqs.GetMeshRequirementsIterator();
  for (; mit; ++mit)
    {
    std::string meshName = mit.MeshName();

    vtkCPInputDataDescription *inDesc =
      dataDesc->GetInputDescriptionByName(meshName.c_str());

    if (inDesc->GetIfGridIsNecessary())
      {
      // get the mesh
      vtkDataObject* dobj = nullptr;
      if (dataAdaptor->GetMesh(meshName, mit.StructureOnly(), dobj))
        {
        SENSEI_ERROR("Failed to get mesh \"" << mit.MeshName() << "\"")
        return -1;
        }

      // add the requested arrays
      ArrayRequirementsIterator ait =
        reqs.GetArrayRequirementsIterator(meshName);

      for (; ait; ++ait)
        {
        int assoc = ait.Association();
        std::string arrayName = ait.Array();

        if (inDesc->IsFieldNeeded(arrayName.c_str()))
          {
          if (dataAdaptor->AddArray(dobj, meshName, assoc, arrayName))
            {
            SENSEI_ERROR("Failed to add "
              << VTKUtils::GetAttributesName(assoc)
              << " data array \"" << arrayName << "\" to mesh \""
              << meshName << "\"")
            return -1;
            }
          }
        }

      inDesc->SetGrid(dobj);
      this->SetWholeExtent(dobj, inDesc);
      }
    }
  return 0;
}

//----------------------------------------------------------------------------
int CatalystAnalysisAdaptor::SetWholeExtent(vtkDataObject *dobj,
  vtkCPInputDataDescription *desc)
{
  int localExtent[6] = {0};

  if (vtkImageData *id = dynamic_cast<vtkImageData*>(dobj))
    id->GetExtent(localExtent);
  else if (vtkRectilinearGrid *rg = dynamic_cast<vtkRectilinearGrid*>(dobj))
    rg->GetExtent(localExtent);
  else if (vtkStructuredGrid *sg = dynamic_cast<vtkStructuredGrid*>(dobj))
    sg->GetExtent(localExtent);
  else
    return 0;

  localExtent[0] = -localExtent[0];
  localExtent[2] = -localExtent[2];
  localExtent[4] = -localExtent[4];

  int wholeExtent[6] = {0};

  vtkMultiProcessController* controller =
    vtkMultiProcessController::GetGlobalController();

  controller->AllReduce(localExtent, wholeExtent, 6, vtkCommunicator::MAX_OP);

  wholeExtent[0] = -wholeExtent[0];
  wholeExtent[2] = -wholeExtent[2];
  wholeExtent[4] = -wholeExtent[4];

  desc->SetWholeExtent(wholeExtent);

  return 0;
}


//----------------------------------------------------------------------------
bool CatalystAnalysisAdaptor::Execute(DataAdaptor* dataAdaptor)
{
  timer::MarkEvent mark("CatalystAnalysisAdaptor::Execute");

  // Get a description of the simulation metadata
  DataRequirements reqs;
  if (reqs.Initialize(dataAdaptor))
    {
    SENSEI_ERROR("Failed to initialze data requirements")
    return false;
    }

  double time = dataAdaptor->GetDataTime();
  int timeStep = dataAdaptor->GetDataTimeStep();

  vtkSmartPointer<vtkCPDataDescription> dataDesc =
    vtkSmartPointer<vtkCPDataDescription>::New();

  if (this->DescribeData(timeStep, time, reqs, dataDesc.GetPointer()))
    {
    SENSEI_ERROR("Failed to describe simulation data")
    return false;
    }

  vtkCPProcessor *proc = vtkCPAdaptorAPI::GetCoProcessor();
  if (proc->RequestDataDescription(dataDesc.GetPointer()))
    {
    // Querry Catalyst for what data is required, fetch from the sim
    if (this->SelectData(dataAdaptor, reqs, dataDesc.GetPointer()))
      {
      SENSEI_ERROR("Failed to selct data")
      return false;
      }

    // transfer control to Catalyst
    proc->CoProcess(dataDesc.GetPointer());
    }

  return true;
}

//-----------------------------------------------------------------------------
int CatalystAnalysisAdaptor::Finalize()
{
  vtkCPAdaptorAPIInitializationCounter--;
  if (vtkCPAdaptorAPIInitializationCounter == 0)
    {
    timer::MarkEvent mark("CatalystAnalysisAdaptor::Finalize");
    vtkCPAdaptorAPI::CoProcessorFinalize();
    }
  return 0;
}

//-----------------------------------------------------------------------------
void CatalystAnalysisAdaptor::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

}
