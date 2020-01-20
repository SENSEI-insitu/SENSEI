#include "CatalystAnalysisAdaptor.h"

#include "DataAdaptor.h"
#include "MeshMetadata.h"
#include "SVTKUtils.h"
#include "Error.h"
#include "MeshMetadata.h"
#include "Profiler.h"
#include "VTKDataAdaptor.h"
#include "VTKUtils.h"

#include <svtkDataObject.h>
#include <svtkImageData.h>
#include <svtkObjectFactory.h>
#include <svtkRectilinearGrid.h>
#include <svtkStructuredGrid.h>

#include <vtkAlgorithm.h>
#include <vtkCommunicator.h>
#include <vtkCPAdaptorAPI.h>
#include <vtkCPDataDescription.h>
#include <vtkCPInputDataDescription.h>
#include <vtkMultiProcessController.h>
#include <vtkCPProcessor.h>
#include <vtkPVConfig.h>
#include <vtkSmartPointer.h>
#include <vtkSMProxyManager.h>
#include <vtkSMSessionProxyManager.h>
#include <vtkSMSourceProxy.h>
#ifdef ENABLE_CATALYST_PYTHON
#include <vtkCPPythonScriptPipeline.h>
#endif

#include <cassert>

namespace sensei
{

#ifdef ENABLE_CATALYST_PYTHON
/// This vtkCPPythonScriptPipeline subclass helps us pass back data
/// from producer identified in the configuration as result using a
/// DataAdaptor.
class CatalystScriptPipeline : public vtkCPPythonScriptPipeline
{
public:
  static CatalystScriptPipeline* New();
  vtkTypeMacro(CatalystScriptPipeline, vtkCPPythonScriptPipeline);

  //@{
  /// Get/Set result producer algorithm's registration name.
  vtkSetStringMacro(ResultProducer);
  vtkGetStringMacro(ResultProducer);
  //@}

  //@{
  /// Get/Set result mesh name.
  vtkSetStringMacro(ResultMesh);
  vtkGetStringMacro(ResultMesh);
  //@}

  vtkDataObject* GetResultData() { return this->ResultData.GetPointer(); }

  /// helper function to find the first CatalystScriptPipeline know to the
  /// processor to return the result from.
  static std::pair<std::string, vtkDataObject*> GetResultData(vtkCPProcessor* processor)
  {
    for (int cc=0, max = processor->GetNumberOfPipelines(); cc < max; ++cc)
      {
      if (auto csp = CatalystScriptPipeline::SafeDownCast(processor->GetPipeline(cc)))
        {
        if (csp->GetResultData() != nullptr)
          return std::make_pair(std::string(csp->GetResultMesh()), csp->GetResultData());
        }
      }
    return std::pair<std::string, vtkDataObject*>();
  }

protected:
  CatalystScriptPipeline()  : ResultProducer(nullptr), ResultMesh(nullptr) {}
  ~CatalystScriptPipeline() override
  {
    this->SetResultProducer(nullptr);
    this->SetResultMesh(nullptr);
  }

  int CoProcess(vtkCPDataDescription* dataDescription) override
  {
    this->ResultData = nullptr;

    const auto status = this->Superclass::CoProcess(dataDescription);
    if (!status || this->ResultProducer == nullptr || this->ResultProducer[0] == '\0')
      {
      return status;
      }


    // find `ResultProducer` proxy and update it and get its data.
    auto pxm = vtkSMProxyManager::GetProxyManager()->GetActiveSessionProxyManager();
    assert(pxm != nullptr);

    auto producer = vtkSMSourceProxy::SafeDownCast(pxm->GetProxy("sources", this->ResultProducer));
    if (!producer)
    {
      SENSEI_ERROR("Failed to locate producer '" << this->ResultProducer << "'. "
        "Please check the configuration or the Catalyst Python script for errors.");
      return 0;
    }
    producer->UpdatePipeline(dataDescription->GetTime());
    if (auto result = vtkAlgorithm::SafeDownCast(producer->GetClientSideObject())->GetOutputDataObject(0))
    {
      this->ResultData.TakeReference(result->NewInstance());
      this->ResultData->ShallowCopy(result);
    }
    return 1;
  }


private:
  CatalystScriptPipeline(const CatalystScriptPipeline&) = delete;
  void operator=(const CatalystScriptPipeline&) = delete;

  char* ResultProducer;
  char* ResultMesh;
  vtkSmartPointer<vtkDataObject> ResultData;
};
vtkStandardNewMacro(CatalystScriptPipeline);
#endif // ENABLE_CATALYST_PYTHON

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
  TimeEvent<128> mark("CatalystAnalysisAdaptor::Initialize");
  if (vtkCPAdaptorAPIInitializationCounter == 0)
    {
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
void CatalystAnalysisAdaptor::AddPythonScriptPipeline(const std::string &fileName,
  const std::string& resultProducer, const std::string& resultMesh)
{
#ifdef ENABLE_CATALYST_PYTHON
#if PARAVIEW_VERSION_MAJOR > 5 || (PARAVIEW_VERSION_MAJOR == 5 && PARAVIEW_VERSION_MINOR >= 9)
  (void) resultMesh;
  (void) resultProducer;
  // TODO -- update the bi directional work for PV 5.9.0
  this->AddPipeline(vtkCPPythonPipeline::CreateAndInitializePipeline(fileName.c_str()));
#else
  // 5.8.0 version of bi-directional work
  vtkNew<vtkCPPythonScriptPipeline> pythonPipeline;
  vtkNew<sensei::CatalystScriptPipeline> pythonPipeline;
  pythonPipeline->SetResultProducer(!resultProducer.empty() ? resultProducer.c_str() : nullptr);
  pythonPipeline->SetResultMesh(!resultMesh.empty() ? resultMesh.c_str() : nullptr);
  pythonPipeline->Initialize(fileName.c_str());
  this->AddPipeline(pythonPipeline.GetPointer());
#endif
#else
  (void)fileName;
  (void)resultProducer;
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
      if (assoc == svtkDataObject::POINT)
        inDesc->AddPointField(arrayName);
      else if (assoc == svtkDataObject::CELL)
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
      svtkDataObject* dobj = nullptr;
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
              << SVTKUtils::GetAttributesName(assoc)
              << " data array \"" << arrayName << "\" to mesh \""
              << meshName << "\"")
            return -1;
            }
          }
        }

      // add ghost zones
      if ((metadata[i]->NumGhostCells || SVTKUtils::AMR(metadata[i])) &&
        dataAdaptor->AddGhostNodesArray(dobj, meshName))
        {
        SENSEI_ERROR("Failed to get ghost nodes array for mesh \""
          << meshName << "\"")
        }

      if (metadata[i]->NumGhostCells &&
        dataAdaptor->AddGhostCellsArray(dobj, meshName))
        {
        SENSEI_ERROR("Failed to get ghost nodes array for mesh \""
          << meshName << "\"")
        }

      SENSEI_ERROR("TODO conversion from SVTK data set to VTK data set")
      // TODO inDesc->SetGrid(dobj);

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
int CatalystAnalysisAdaptor::SetWholeExtent(svtkDataObject *dobj,
  vtkCPInputDataDescription *desc)
{
  int localExtent[6] = {0};

  if (svtkImageData *id = dynamic_cast<svtkImageData*>(dobj))
    id->GetExtent(localExtent);
  else if (svtkRectilinearGrid *rg = dynamic_cast<svtkRectilinearGrid*>(dobj))
    rg->GetExtent(localExtent);
  else if (svtkStructuredGrid *sg = dynamic_cast<svtkStructuredGrid*>(dobj))
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
int CatalystAnalysisAdaptor::SetFrequency(unsigned int frequency)
{
  this->Frequency = frequency;
  return 0;
}

//----------------------------------------------------------------------------
bool CatalystAnalysisAdaptor::Execute(DataAdaptor* dataAdaptor, DataAdaptor*& result)
{
  long step = dataAdaptor->GetDataTimeStep();

  if(this->Frequency > 0 && step % this->Frequency != 0)
    {
    return true;
    }
  TimeEvent<128> mark("CatalystAnalysisAdaptor::Execute");

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
    if (proc->CoProcess(dataDesc.GetPointer()))
      {
        auto data = sensei::CatalystScriptPipeline::GetResultData(proc);
        if (data.second != nullptr)
        {
          VTKDataAdaptor* vtkresult = VTKDataAdaptor::New();
          vtkresult->SetDataObject(data.first, data.second);
          result = vtkresult;
        }
      }
    }

  return true;
}

//-----------------------------------------------------------------------------
int CatalystAnalysisAdaptor::Finalize()
{
  TimeEvent<128> mark("CatalystAnalysisAdaptor::Finalize");
  vtkCPAdaptorAPIInitializationCounter--;
  if (vtkCPAdaptorAPIInitializationCounter == 0)
    {
    vtkCPAdaptorAPI::CoProcessorFinalize();
    }
  return 0;
}

//-----------------------------------------------------------------------------
void CatalystAnalysisAdaptor::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

}
