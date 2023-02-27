#include "CatalystAnalysisAdaptor.h"

#include "DataAdaptor.h"
#include "MeshMetadata.h"
#include "Error.h"
#include "MeshMetadata.h"
#include "Profiler.h"
#include "SVTKDataAdaptor.h"
#include "SVTKUtils.h"

#include <svtkDataObject.h>
#include <svtkImageData.h>
#include <svtkObjectFactory.h>
#include <svtkRectilinearGrid.h>
#include <svtkStructuredGrid.h>

#include <vtkArrayDispatch.h>
#include <vtkObjectFactory.h>
#include <vtkDataObject.h>
#include <vtkAlgorithm.h>
#include <vtkCommunicator.h>
#include <vtkCPAdaptorAPI.h>
#include <vtkCPDataDescription.h>
#include <vtkCPInputDataDescription.h>
#include <vtkDataArrayAccessor.h>
#include <vtkFieldData.h>
#include <vtkMultiBlockDataSet.h>
#include <vtkMultiProcessController.h>
#include <vtkCPProcessor.h>
#include <vtkPointSet.h>
#include <vtkPVConfig.h>
#include <vtkPVXMLElement.h>
#include <vtkSmartPointer.h>
#include <vtkSMDoubleVectorProperty.h>
#include <vtkSMProxyManager.h>
#include <vtkSMSessionProxyManager.h>
#include <vtkSMSourceProxy.h>
#include <vtkSMIdTypeVectorProperty.h>
#include <vtkSMIntVectorProperty.h>
#include <vtkSMPluginManager.h>
#ifdef ENABLE_CATALYST_PYTHON
#include <vtkCPPythonScriptPipeline.h>
#endif

#include <cassert>

namespace sensei
{
#ifdef ENABLE_CATALYST_PYTHON

template <typename PropertyType>
struct PropertyCopier
{
  PropertyType* SMProperty = nullptr;

  template <typename ArrayType>
  void operator()(ArrayType* array)
  {
    const vtkIdType numTuples = array->GetNumberOfTuples();
    const int numComponents = array->GetNumberOfComponents();
    this->SMProperty->SetNumberOfElements(static_cast<unsigned int>(numTuples*numComponents));
    vtkDataArrayAccessor<ArrayType> a(array);
    for (vtkIdType cc=0; cc < numTuples; ++cc)
      {
      for (int comp=0; comp < numComponents; ++comp)
        {
        this->SMProperty->SetElement(cc*numComponents + comp, a.Get(cc, comp));
        }
      }
  }
};

template <typename T>
void SetPropertyValue(T* prop, vtkDataArray* array)
{
  PropertyCopier<T> copier;
  copier.SMProperty = prop;
  using Dispatcher = vtkArrayDispatch::DispatchByValueType<vtkArrayDispatch::AllTypes>;
  if (!Dispatcher::Execute(array, copier))
    {
    copier(array);
    }
}

/// Handles `<SenseiInitializePropertiesWithMesh />` hints on a proxy to
/// initialize it's properties using data values.
static bool HandleSenseiInitializePropertiesWithMesh(
    vtkSMProxy* source, vtkCPDataDescription* dataDescription)
{
  if (source == nullptr) { return false; }
  auto hints = source->GetHints() ?
    source->GetHints()->FindNestedElementByName("SenseiInitializePropertiesWithMesh") : nullptr;
  if (!hints) { return true; }

  const char* meshname = hints->GetAttribute("mesh");
  if (!meshname)
    {
    SENSEI_ERROR("`SenseiInitializePropertiesWithMesh` missing 'mesh' attribute.");
    return false;
    }

  // initialize properties on the `source` using the mesh data.
  auto ipDesc = dataDescription->GetInputDescriptionByName(meshname);
  if (!ipDesc)
    {
    SENSEI_ERROR("No mesh named '"<< meshname << "' present.");
    return false;
    }

  auto grid = ipDesc->GetGrid();
  if (auto mb = vtkMultiBlockDataSet::SafeDownCast(grid))
    {
    // this may need some rethinking. For now, I am keeping it simply to just use
    // 1st block in case of vtkMultiBlockDataSet.
    grid = mb->GetBlock(0);
    }

  if (!grid)
    {
    SENSEI_WARNING("Empty grid received for mesh named '" << meshname << "'.");
    return true;
    }

  for (unsigned int cc=0, max=hints->GetNumberOfNestedElements(); cc < max; ++cc)
    {
    auto child = hints->GetNestedElement(cc);
    if (child == nullptr || child->GetName() == nullptr || strcmp(child->GetName(), "Property") != 0)
      {
      continue;
      }
    if (!child->GetAttribute("name") || !child->GetAttribute("array"))
      {
      SENSEI_WARNING("Missing required attribute on `Property` element. Skipping.");
      continue;
      }

    auto property = source->GetProperty(child->GetAttribute("name"));
    if (!property)
      {
      SENSEI_WARNING("No property named '" << child->GetAttribute("name") << "' "
          "present on proxy. Skipping.");
      continue;
      }

    int assoc = 0;
    if (SVTKUtils::GetAssociation(child->GetAttributeOrDefault("association", "point"), assoc))
      {
      SENSEI_WARNING("Invalid 'association' specified. Skipping.");
      continue;
      }
    auto arrayname = child->GetAttribute("array");
    auto fd = grid->GetAttributesAsFieldData(assoc);
    vtkDataArray* array = fd ? fd->GetArray(arrayname) : nullptr;
    if (strcmp(arrayname, "coords") == 0 && vtkPointSet::SafeDownCast(grid) &&
        assoc == vtkDataObject::POINT)
      {
      array = vtkPointSet::SafeDownCast(grid)->GetPoints()->GetData();
      }
    if (!array)
      {
      SENSEI_WARNING("No array named '" << arrayname << "' present. Skipping.");
      continue;
      }

    if (array)
      {
        if (auto dp = vtkSMDoubleVectorProperty::SafeDownCast(property))
          {
          SetPropertyValue<vtkSMDoubleVectorProperty>(dp, array);
          }
        else if (auto ip = vtkSMIntVectorProperty::SafeDownCast(property))
          {
          SetPropertyValue<vtkSMIntVectorProperty>(ip, array);
          }
        else if (auto idp = vtkSMIdTypeVectorProperty::SafeDownCast(property))
          {
          SetPropertyValue<vtkSMIdTypeVectorProperty>(idp, array);
          }
        else
        {
          SENSEI_WARNING("Properties of type '" << property->GetClassName() << "' "
              "are not supported. Skipping.");
          continue;
        }
      }
    }
  return true;
}
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
  /// Get/Set steerable proxy name.
  vtkSetStringMacro(SteerableSourceType);
  vtkGetStringMacro(SteerableSourceType);
  //@}

  //@{
  /// Get/Set result mesh name.
  vtkSetStringMacro(ResultMesh);
  vtkGetStringMacro(ResultMesh);
  //@}

  svtkDataObject* GetResultData() { return this->ResultData.GetPointer(); }

  /// helper function to find the first CatalystScriptPipeline know to the
  /// processor to return the result from.
  static std::pair<std::string, svtkDataObject*> GetResultData(vtkCPProcessor* processor)
  {
    for (int cc=0, max = processor->GetNumberOfPipelines(); cc < max; ++cc)
      {
      if (auto csp = CatalystScriptPipeline::SafeDownCast(processor->GetPipeline(cc)))
        {
        if (csp->GetResultData() != nullptr)
          return std::make_pair(std::string(csp->GetResultMesh()), csp->GetResultData());
        }
      }
    return std::pair<std::string, svtkDataObject*>();
  }

protected:
  CatalystScriptPipeline() : ResultProducer(nullptr),
    SteerableSourceType(nullptr), ResultMesh(nullptr) {}

  ~CatalystScriptPipeline() override
  {
    this->SetResultProducer(nullptr);
    this->SetSteerableSourceType(nullptr);
    this->SetResultMesh(nullptr);
  }

  int CoProcess(vtkCPDataDescription* dataDescription) override
  {
    this->ResultData = nullptr;

    if (this->SteerableSourceType != nullptr)
    {
      this->InitializeSteerableSource(dataDescription);
    }

    const auto status = this->Superclass::CoProcess(dataDescription);
    if (!status)
    {
      return status;
    }


    // find `ResultProducer` proxy and update it and get its data.
    auto pxm = vtkSMProxyManager::GetProxyManager()->GetActiveSessionProxyManager();
    assert(pxm != nullptr);

    vtkSMSourceProxy* producer = nullptr;
    if (this->ResultProducer != nullptr && this->ResultProducer[0] != '\0')
    {
      // find `ResultProducer` proxy and update it and get its data.
      producer = vtkSMSourceProxy::SafeDownCast(pxm->GetProxy("sources", this->ResultProducer));
      if (!producer)
      {
        SENSEI_ERROR("Failed to locate producer '" << this->ResultProducer << "'. "
          "Please check the configuration or the Catalyst Python script for errors.");
        return 0;
      }
    }
    else if (this->SteerableSource)
    {
      producer = vtkSMSourceProxy::SafeDownCast(this->SteerableSource);
    }

    if (producer)
    {
      producer->UpdatePipeline(dataDescription->GetTime());
      if (vtkDataObject *result = vtkAlgorithm::SafeDownCast(producer->GetClientSideObject())->GetOutputDataObject(0))
      {
        svtkDataObject *sResult = SVTKUtils::SVTKObjectFactory::New(result);
        this->ResultData.TakeReference(sResult->NewInstance());
        this->ResultData->ShallowCopy(sResult);
      }
    }
    return 1;
  }

private:
  CatalystScriptPipeline(const CatalystScriptPipeline&) = delete;
  void operator=(const CatalystScriptPipeline&) = delete;

  // Creates steerable source proxy, if needed and then initializes it with
  // current property values.
  void InitializeSteerableSource(vtkCPDataDescription* dataDescription)
  {
    auto pxm = vtkSMProxyManager::GetProxyManager()->GetActiveSessionProxyManager();
    assert(pxm != nullptr);

    vtkSmartPointer<vtkSMProxy> source = this->SteerableSource;
    if (source == nullptr)
    {
      source.TakeReference(pxm->NewProxy("sources", this->SteerableSourceType));
      if (!source)
      {
        SENSEI_ERROR("Failed to create source for steering (sources, " << this->SteerableSourceType << ").");
        // set type to null so we don't attempt this again.
        this->SetSteerableSourceType(nullptr);
        return;
      }
    }

    if (!HandleSenseiInitializePropertiesWithMesh(source, dataDescription))
    {
      // set type to null so we don't attempt this again.
      this->SteerableSource = nullptr;
      this->SetSteerableSourceType(nullptr);
      return;
    }
    source->UpdateVTKObjects();
    if (this->SteerableSource == nullptr)
    {
      pxm->RegisterProxy("sources", "SteeringParameters", source);
      this->SteerableSource = source;
    }
  }

  char* ResultProducer;
  char* SteerableSourceType;
  char* ResultMesh;
  vtkSmartPointer<vtkSMProxy> SteerableSource;
  svtkSmartPointer<svtkDataObject> ResultData;
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
void CatalystAnalysisAdaptor::AddPythonScriptPipeline(
  const std::string& fileName,
  const std::string& resultProducer,
  const std::string& steerableSourceType,
  const std::string& resultMesh)
{
#ifdef ENABLE_CATALYST_PYTHON
#if PARAVIEW_VERSION_MAJOR > 5 || (PARAVIEW_VERSION_MAJOR == 5 && PARAVIEW_VERSION_MINOR >= 9)
  // detect if we are given a Catalyst 1 or 2 script
  vtkSmartPointer<vtkCPPythonPipeline> pythonPipeline =
    vtkCPPythonPipeline::CreateAndInitializePipeline(fileName.c_str());

  // if we have a catalyst 1 script, we can create a pipeline with steering options
  if(auto catalyst1Pipeline = vtkCPPythonScriptPipeline::SafeDownCast(pythonPipeline))
  {
    vtkNew<sensei::CatalystScriptPipeline> steerablePipeline;
    steerablePipeline->SetResultProducer(!resultProducer.empty() ? resultProducer.c_str() : nullptr);
    steerablePipeline->SetResultMesh(!resultMesh.empty() ? resultMesh.c_str() : nullptr);
    steerablePipeline->SetSteerableSourceType(
      !steerableSourceType.empty() ? steerableSourceType.c_str() : nullptr);
    steerablePipeline->Initialize(fileName.c_str());
    this->AddPipeline(steerablePipeline.GetPointer());
  }
  else if(pythonPipeline)
  {
    // we currently do not support steering with this code path for Catalyst 2
    this->AddPipeline(pythonPipeline.GetPointer());
  }
#else
  // we only have access to Catalyst 1, so we can use the steerable pipeline
  vtkNew<sensei::CatalystScriptPipeline> steerablePipeline;
  steerablePipeline->SetResultProducer(!resultProducer.empty() ? resultProducer.c_str() : nullptr);
  steerablePipeline->SetResultMesh(!resultMesh.empty() ? resultMesh.c_str() : nullptr);
  steerablePipeline->SetSteerableSourceType(
    !steerableSourceType.empty() ? steerableSourceType.c_str() : nullptr);
  steerablePipeline->Initialize(fileName.c_str());
  this->AddPipeline(steerablePipeline.GetPointer());
#endif

#else
  (void)fileName;
  (void)resultProducer;
  SENSEI_ERROR("Failed to add Python script pipeline. "
    "Re-compile with ENABLE_CATALYST_PYTHON=ON")
#endif
}

// ---------------------------------------------------------------------------
void CatalystAnalysisAdaptor::AddPluginXML(const std::string& fileName)
{
  auto plmgr = vtkSMProxyManager::GetProxyManager()->GetPluginManager();
  if (!plmgr->LoadLocalPlugin(fileName.c_str()))
    {
    SENSEI_ERROR("Failed to load plugin xml " << fileName.c_str());
    }
  else
    {
    SENSEI_STATUS("Catalyst plugin loaded: " << fileName.c_str());
    }
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

      // convert from SVTK to VTK
      vtkDataObject *vdobj = SVTKUtils::VTKObjectFactory::New(dobj);

      inDesc->SetGrid(vdobj);

      vdobj->Delete();

      // we could get this info from metadata, however if there
      // is not advantage to doing so we might as well get it
      // from the data itself
      this->SetWholeExtent(dobj, inDesc);

      dobj->Delete();
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
bool CatalystAnalysisAdaptor::Execute(DataAdaptor* dataIn, DataAdaptor** dataOut)
{
  long step = dataIn->GetDataTimeStep();

  if(this->Frequency > 0 && step % this->Frequency != 0)
    {
    return true;
    }
  TimeEvent<128> mark("CatalystAnalysisAdaptor::Execute");

  // Get a description of the simulation metadata
  unsigned int nMeshes = 0;
  if (dataIn->GetNumberOfMeshes(nMeshes))
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

    if (dataIn->GetMeshMetadata(i, mmd))
      {
      SENSEI_ERROR("Failed to get metadata for mesh " << i << " of " << nMeshes)
      return false;
      }
    metadata[i] = mmd;
    }

  double time = dataIn->GetDataTime();
  int timeStep = dataIn->GetDataTimeStep();

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
    // Query Catalyst for what data is required, fetch from the sim
    if (this->SelectData(dataIn, metadata, dataDesc.GetPointer()))
      {
      SENSEI_ERROR("Failed to select data")
      return false;
      }

    // transfer control to Catalyst
    if (proc->CoProcess(dataDesc.GetPointer()))
      {
      auto data = sensei::CatalystScriptPipeline::GetResultData(proc);
      if (data.second != nullptr)
        {
        if (dataOut)
          {
          SVTKDataAdaptor* result = SVTKDataAdaptor::New();
          result->SetDataObject(data.first, data.second);
          *dataOut = result;
          }
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
