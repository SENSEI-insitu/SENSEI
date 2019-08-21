#include "CatalystAnalysisAdaptor.h"

#include "DataAdaptor.h"
#include "MeshMetadata.h"
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
#include <vtkPVConfig.h>
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
    Timer::MarkEvent mark("CatalystAnalysisAdaptor::Initialize");
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
  const std::vector<MeshMetadataPtr> &metadata, vtkCPDataDescription *dataDesc)
{
  dataDesc->SetTimeData(time, timeStep);

  unsigned int nMeshes = metadata.size();
  for (unsigned int i = 0; i < nMeshes; ++i)
    {
    // add the mesh
    const char *meshName = metadata[i]->MeshName.c_str();

    dataDesc->AddInput(meshName);

    vtkCPInputDataDescription *inDesc =
      dataDesc->GetInputDescriptionByName(meshName);

    // add the arrays
    for (int j = 0; j < metadata[i]->NumArrays; ++j)
      {
      int assoc = metadata[i]->ArrayCentering[j];
      const char *arrayName = metadata[i]->ArrayName[j].c_str();

#if (PARAVIEW_VERSION_MAJOR == 5 && PARAVIEW_VERSION_MINOR >= 6) || PARAVIEW_VERSION_MAJOR > 5
      inDesc->AddField(arrayName, assoc);
#else
      if (assoc == vtkDataObject::POINT)
        inDesc->AddPointField(arrayName);
      else if (assoc == vtkDataObject::CELL)
        inDesc->AddCellField(arrayName);
      else
        SENSEI_WARNING("Unknown association " << assoc)
#endif
      }

    // let Catalyst tell us which arrays are needed
    inDesc->AllFieldsOff();
    }

  return 0;
}

// ---------------------------------------------------------------------------
int CatalystAnalysisAdaptor::SelectData(DataAdaptor *dataAdaptor,
  const std::vector<MeshMetadataPtr> &metadata, vtkCPDataDescription *dataDesc)
{
  unsigned int nMeshes = metadata.size();
  for (unsigned int i = 0; i < nMeshes; ++i)
    {
    // add the mesh
    const char *meshName = metadata[i]->MeshName.c_str();

    vtkCPInputDataDescription *inDesc =
      dataDesc->GetInputDescriptionByName(meshName);

    if (inDesc->GetIfGridIsNecessary())
      {
      // get the mesh
      vtkDataObject* dobj = nullptr;
      if (dataAdaptor->GetMesh(meshName, false, dobj))
        {
        SENSEI_ERROR("Failed to get mesh \"" << meshName << "\"")
        return -1;
        }

      // add the requested arrays
      for (int j = 0; j < metadata[i]->NumArrays; ++j)
        {
        int assoc = metadata[i]->ArrayCentering[j];
        const char *arrayName = metadata[i]->ArrayName[j].c_str();

#if (PARAVIEW_VERSION_MAJOR == 5 && PARAVIEW_VERSION_MINOR >= 6) || PARAVIEW_VERSION_MAJOR > 5
        if (inDesc->IsFieldNeeded(arrayName, assoc))
#else
        if (inDesc->IsFieldNeeded(arrayName))
#endif
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

      // add ghost zones
      if ((metadata[i]->NumGhostNodes > 0) &&
        dataAdaptor->AddGhostNodesArray(dobj, meshName))
        {
        SENSEI_ERROR("Failed to get ghost nodes array for mesh \""
          << meshName << "\"")
        }

      if ((metadata[i]->NumGhostCells > 0) &&
        dataAdaptor->AddGhostCellsArray(dobj, meshName))
        {
        SENSEI_ERROR("Failed to get ghost nodes array for mesh \""
          << meshName << "\"")
        }

      inDesc->SetGrid(dobj);
      dobj->Delete();

      // we could get this info from metadata, however if there
      // is not advantage to doing so we might as well get it
      // from the data itself
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
  Timer::MarkEvent mark("CatalystAnalysisAdaptor::Execute");

  // Get a description of the simulation metadata
  unsigned int nMeshes = 0;
  if (dataAdaptor->GetNumberOfMeshes(nMeshes))
    {
    SENSEI_ERROR("Failed to get the number of meshes")
    return false;
    }

  std::vector<MeshMetadataPtr> metadata(nMeshes);
  for (unsigned int i = 0; i < nMeshes; ++i)
    {
    MeshMetadataPtr mmd = MeshMetadata::New();

    // for now, rather than querry metadata for whole extent
    // use data object itself
    //mmd->Flags.SetBlockExtents();

    if (dataAdaptor->GetMeshMetadata(i, mmd))
      {
      SENSEI_ERROR("Failed to get metadata for mesh " << i << " of " << nMeshes)
      return false;
      }
    metadata[i] = mmd;
    }

  double time = dataAdaptor->GetDataTime();
  int timeStep = dataAdaptor->GetDataTimeStep();

  vtkSmartPointer<vtkCPDataDescription> dataDesc =
    vtkSmartPointer<vtkCPDataDescription>::New();

  if (this->DescribeData(timeStep, time, metadata, dataDesc.GetPointer()))
    {
    SENSEI_ERROR("Failed to describe simulation data")
    return false;
    }

  vtkCPProcessor *proc = vtkCPAdaptorAPI::GetCoProcessor();
  if (proc->RequestDataDescription(dataDesc.GetPointer()))
    {
    // Querry Catalyst for what data is required, fetch from the sim
    if (this->SelectData(dataAdaptor, metadata, dataDesc.GetPointer()))
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
    Timer::MarkEvent mark("CatalystAnalysisAdaptor::Finalize");
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
