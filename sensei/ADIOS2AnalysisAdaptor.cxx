#include "ADIOS2AnalysisAdaptor.h"

#include "ADIOS2Schema.h"
#include "DataAdaptor.h"
#include "MeshMetadataMap.h"
#include "SVTKUtils.h"
#include "MPIUtils.h"
#include "XMLUtils.h"
#include "Profiler.h"
#include "Error.h"

#include <svtkCellTypes.h>
#include <svtkCellData.h>
#include <svtkCompositeDataIterator.h>
#include <svtkCompositeDataSet.h>
#include <svtkDataSetAttributes.h>
#include <svtkDoubleArray.h>
#include <svtkFloatArray.h>
#include <svtkIntArray.h>
#include <svtkUnsignedIntArray.h>
#include <svtkLongArray.h>
#include <svtkUnsignedLongArray.h>
#include <svtkCharArray.h>
#include <svtkUnsignedCharArray.h>
#include <svtkIdTypeArray.h>
#include <svtkCellArray.h>
#include <svtkUnstructuredGrid.h>
#include <svtkPolyData.h>
#include <svtkImageData.h>
#include <svtkObjectFactory.h>
#include <svtkPointData.h>
#include <svtkSmartPointer.h>

#include <mpi.h>
#include <vector>
#include <regex>
#include <pugixml.hpp>

using senseiADIOS2::adios2_strerror;

namespace sensei
{

//----------------------------------------------------------------------------
senseiNewMacro(ADIOS2AnalysisAdaptor);

//----------------------------------------------------------------------------
ADIOS2AnalysisAdaptor::ADIOS2AnalysisAdaptor() :
    Schema(nullptr), FileName("sensei.bp"),
    StepsPerFile(0), StepIndex(0), FileIndex(0), Frequency(1)
{
  this->Handles.io = nullptr;
  this->Handles.engine = nullptr;
}

//----------------------------------------------------------------------------
ADIOS2AnalysisAdaptor::~ADIOS2AnalysisAdaptor()
{
  delete this->Schema;
}

//-----------------------------------------------------------------------------
int ADIOS2AnalysisAdaptor::SetDataRequirements(const DataRequirements &reqs)
{
  this->Requirements = reqs;
  return 0;
}

//-----------------------------------------------------------------------------
int ADIOS2AnalysisAdaptor::AddDataRequirement(const std::string &meshName,
  int association, const std::vector<std::string> &arrays)
{
  this->Requirements.AddRequirement(meshName, association, arrays);
  return 0;
}

//-----------------------------------------------------------------------------
int ADIOS2AnalysisAdaptor::FetchFromProducer(
  sensei::DataAdaptor *dataAdaptor,
  std::vector<svtkCompositeDataSetPtr> &objects,
  std::vector<MeshMetadataPtr> &metadata)
{
  // figure out what the simulation can provide. include the full
  // suite of metadata for the end-point partitioners
  MeshMetadataFlags flags;
  flags.SetBlockDecomp();
  flags.SetBlockSize();
  flags.SetBlockBounds();
  flags.SetBlockExtents();
  flags.SetBlockArrayRange();

  MeshMetadataMap mdm;
  if (mdm.Initialize(dataAdaptor, flags))
    {
    SENSEI_ERROR("Failed to get metadata")
    return false;
    }

  // loop over the required meshes and arrays subsetting
  // in the process. only the required meshes and arrays
  // need be buffered and presented to the consumer
  MeshRequirementsIterator mit =
    this->Requirements.GetMeshRequirementsIterator();

  while (mit)
    {
    // get metadata
    MeshMetadataPtr mdIn;
    if (mdm.GetMeshMetadata(mit.MeshName(), mdIn))
      {
      SENSEI_ERROR("Failed to get mesh metadata for mesh \""
        << mit.MeshName() << "\"")
      return false;
      }

    // copy the metadata and prepare for subsetting by array
    MeshMetadataPtr mdOut = mdIn->NewCopy();
    mdOut->ClearArrayInfo();

    // get the mesh
    svtkDataObject *dobj = nullptr;
    if (dataAdaptor->GetMesh(mit.MeshName(), mit.StructureOnly(), dobj))
      {
      SENSEI_ERROR("Failed to get mesh \"" << mit.MeshName() << "\"")
      return false;
      }

    // add the ghost cell arrays to the mesh
    if ((mdIn->NumGhostCells || SVTKUtils::AMR(mdIn)) &&
        dataAdaptor->AddGhostCellsArray(dobj, mit.MeshName()))
      {
      SENSEI_ERROR("Failed to get ghost cells for mesh \"" << mit.MeshName() << "\"")
      return false;
      }

    // add the ghost node arrays to the mesh
    if (mdIn->NumGhostNodes && dataAdaptor->AddGhostNodesArray(dobj, mit.MeshName()))
      {
      SENSEI_ERROR("Failed to get ghost nodes for mesh \"" << mit.MeshName() << "\"")
      return false;
      }

    // add the required arrays
    ArrayRequirementsIterator ait =
      this->Requirements.GetArrayRequirementsIterator(mit.MeshName());

    while (ait)
      {
      // add the array and its metadata
      const std::string arrayName = ait.Array();
      if (mdOut->CopyArrayInfo(mdIn, arrayName)
        || dataAdaptor->AddArray(dobj, mit.MeshName(),
         ait.Association(), arrayName))
        {
        SENSEI_ERROR("Failed to add "
          << SVTKUtils::GetAttributesName(ait.Association())
          << " data array \"" << arrayName << "\" to mesh \""
          << mit.MeshName() << "\"")
        return false;
        }

      ++ait;
      }

    // generate a global view of the metadata. everything we do from here
    // on out depends on having the global view.
    MPI_Comm comm = this->GetCommunicator();
    mdOut->GlobalizeView(comm);

    // ensure a composite data object
    svtkCompositeDataSetPtr cds = sensei::SVTKUtils::AsCompositeData(comm, dobj);

    // add to the collection
    objects.push_back(cds);
    metadata.push_back(mdOut);

    ++mit;
    }

  return 0;
}

//----------------------------------------------------------------------------
int ADIOS2AnalysisAdaptor::SetFrequency(long frequency)
{
  this->Frequency = std::max(1l, frequency);
  return 0;
}

//----------------------------------------------------------------------------
bool ADIOS2AnalysisAdaptor::Execute(DataAdaptor* dataAdaptor, DataAdaptor**daOut)
{
  TimeEvent<128> mark("ADIOS2AnalysisAdaptor::Execute");

  // we currently do not return anything
  if (daOut)
    {
    daOut = nullptr;
    }

  long step = dataAdaptor->GetDataTimeStep();

  if(this->Frequency > 0 && step % this->Frequency != 0)
    {
    return true;
    }

  // if no dataAdaptor requirements are given, push all the data
  // fill in the requirements with every thing
  if (this->Requirements.Empty())
    {
    if (this->Requirements.Initialize(dataAdaptor, false))
      {
      SENSEI_ERROR("Failed to initialze dataAdaptor description")
      return false;
      }
    SENSEI_WARNING("No subset specified. Writing all available data")
    }

  // collect the specified data objects and metadata
  std::vector<svtkCompositeDataSetPtr> objects;
  std::vector<MeshMetadataPtr> metadata;

  if (this->FetchFromProducer(dataAdaptor, objects, metadata))
    {
    SENSEI_ERROR("Failed to fetch data from the producer")
    return false;
    }

  // set everything up the first time through
  if (!this->Schema)
    this->InitializeADIOS2();

  unsigned long timeStep = dataAdaptor->GetDataTimeStep();
  double time = dataAdaptor->GetDataTime();

  if (this->DefineVariables(metadata) ||
    this->WriteTimestep(timeStep, time, metadata, objects))
    return false;

  if (this->GetVerbose())
      SENSEI_STATUS("ADIOS2AnalysisAdaptor wrote step " << timeStep)

  return true;
}

//----------------------------------------------------------------------------
int ADIOS2AnalysisAdaptor::Initialize(pugi::xml_node &node)
{
  TimeEvent<128> mark("ADIOS2AnalysisAdaptor::Initialize");

  if (XMLUtils::RequireAttribute(node, "engine") ||
    XMLUtils::RequireAttribute(node, "filename"))
    {
    SENSEI_ERROR("Failed to initialize ADIOS2AnalysisAdaptor");
    return -1;
    }

  std::string filename = node.attribute("filename").value();
  this->SetFileName(filename);

  std::string engine = node.attribute("engine").value();
  this->SetEngineName(engine);

  // buffer size is given in the units of number of time steps,
  // 0 means buffer all steps
  std::string bufferSize = node.attribute("buffer_size").as_string("");
  if (!bufferSize.empty())
    this->AddParameter("QueueLimit", bufferSize);

  // valid modes are Block or Discard
  std::string bufferMode = node.attribute("buffer_mode").as_string("");
  if (!bufferMode.empty())
    this->AddParameter("QueueFullPolicy", bufferMode);

  // enable file series for file based engines
  this->SetStepsPerFile(node.attribute("steps_per_file").as_int(0));

  // pass a group of engine parameters
  pugi::xml_node params = node.child("engine_parameters");
  if (params)
    {
    std::vector<std::string> name;
    std::vector<std::string> value;
    XMLUtils::ParseNameValuePairs(params, name, value);
    size_t n = name.size();
    for (size_t i = 0; i < n; ++i)
        this->AddParameter(name[i], value[i]);
    }

  // set the data requirements
  DataRequirements req;
  if (req.Initialize(node))
    {
    SENSEI_ERROR("Failed to initialize ADIOS 2.")
    return -1;
    }
  this->SetDataRequirements(req);

  SENSEI_STATUS("Configured ADIOSAnalysisAdaptor filename=\""
    << filename << "\" engine=" << engine
    << (!bufferMode.empty() ? "buffer_mode=" : "")
    << (!bufferMode.empty() ? bufferMode.c_str() : "")
    << (!bufferSize.empty() ? "buffer_size=" : "")
    << (!bufferSize.empty() ? bufferSize.c_str() : ""))

  return 0;
}

//----------------------------------------------------------------------------
int ADIOS2AnalysisAdaptor::InitializeADIOS2()
{
  TimeEvent<128> mark("ADIOS2AnalysisAdaptor::IntializeADIOS2");

  if (this->StepsPerFile > 0)
    {
    // look for for a decimal format specifier in the file name.
    // if it's not present, the same file will be over written
    // and data will be lost
    std::regex decFmtSpec("%[0-9]*[diuoxX]", std::regex_constants::basic);
    if (!std::regex_search(this->FileName.c_str(), decFmtSpec))
      {
      SENSEI_ERROR("Use of StepsPerFile requires a "
          "printf integer format specifier to prevent "
          "repeatedly overwriting the same file. Hint: "
          "try something like \"sensei_%04d.bp\"");
      return -1;
      }
    }

  // initialize adios2
#if ADIOS2_VERSION_MAJOR > 2 || (ADIOS2_VERSION_MAJOR == 2 && ADIOS2_VERSION_MINOR >= 9)
  // adios2_init()'s signature changed in version 2.9.0
  this->Adios = adios2_init(this->GetCommunicator());
#else
  this->Adios = adios2_init(this->GetCommunicator(),
    adios2_debug_mode(this->DebugMode));
#endif

  if (this->Adios == nullptr)
    {
    SENSEI_ERROR("adios2_init failed")
    return -1;
    }

  // Open the io handle
  this->Handles.io = adios2_declare_io(this->Adios, "SENSEI");
  if (this->Handles.io == nullptr)
    {
    SENSEI_ERROR("adios2_declare_io failed")
    return -1;
    }

  // create space for ADIOS2 variables
  this->Schema = new senseiADIOS2::DataObjectCollectionSchema;

  // Open the engine
  if (adios2_set_engine(this->Handles.io, this->EngineName.c_str()))
    {
    SENSEI_ERROR("adios2_set_engine failed")
    return -1;
    }

  // If the user set additional parameters, add them now to ADIOS2
  for (unsigned int j = 0; j < this->Parameters.size(); j++)
    {
    adios2_set_parameter(this->Handles.io,
                         this->Parameters[j].first.c_str(),
                         this->Parameters[j].second.c_str());
    }

  return 0;
}

//----------------------------------------------------------------------------
int ADIOS2AnalysisAdaptor::DefineVariables(
  const std::vector<MeshMetadataPtr> &metadata)
{
  // On subsequent ts we need to clear the existing variables so we don't try to
  // redefine existing variables
  adios2_error aerr = adios2_error_none;

  if ((aerr = adios2_remove_all_variables(this->Handles.io)))
    {
    SENSEI_ERROR("adios2_remove_all_variables failed. " << adios2_strerror(aerr))
    return -1;
    }

  // (re)define variables to support meshes that evovle in time
  if (this->Schema->DefineVariables(this->GetCommunicator(),
    this->Handles, metadata))
    {
    SENSEI_ERROR("Failed to define variables")
    return -1;
    }

  return 0;
}

//----------------------------------------------------------------------------
int ADIOS2AnalysisAdaptor::FinalizeADIOS2()
{
  int ierr = 0;
  if (this->Schema)
    {
    // handle the case where new file in file series is needed
    if (this->UpdateStream())
      return -1;

    adios2_error aerr = adios2_error_none;

    // mark the stream as completed by sending INT_MAX in the timestep
    // this is used for series of BP4 files and not needed for SST or
    // stream based processing from a single file.
    if ((aerr = adios2_remove_all_variables(this->Handles.io)))
      {
      SENSEI_ERROR("adios2_remove_all_variables failed. "
        << adios2_strerror(aerr))
      return -1;
      }

    if (!adios2_define_variable(this->Handles.io, "time_step",
      adios2_type_uint64_t, 0, NULL, NULL, NULL, adios2_constant_dims_true))
      {
      SENSEI_ERROR("adios2_define_variable time_step failed")
      return -1;
      }

    adios2_step_status status;
    if ((aerr = adios2_begin_step(this->Handles.engine,
      adios2_step_mode_append, -1, &status)))
      {
      SENSEI_ERROR("adios2_begin_step failed. " << adios2_strerror(aerr))
      return -1;
      }

    uint64_t time_step = std::numeric_limits<uint64_t>::max();
    if ((aerr = adios2_put_by_name(this->Handles.engine, "time_step",
      &time_step, adios2_mode_sync)))
      {
      SENSEI_ERROR("adios_put_by_name time_step failed. "
        << adios2_strerror(aerr))
      return -1;
      }

    if ((aerr = adios2_perform_puts(this->Handles.engine)))
      {
      SENSEI_ERROR("adios2_perform_puts failed. " << adios2_strerror(aerr))
      return -1;
      }

    if ((aerr = adios2_end_step(this->Handles.engine)))
      {
      SENSEI_ERROR("adios2_end_step failed. " << adios2_strerror(aerr))
      return -1;
      }

    if ((aerr = adios2_close(this->Handles.engine)))
      {
      SENSEI_ERROR("adios2_close failed. " << adios2_strerror(aerr))
      --ierr;
      }

    this->Handles.io = nullptr;
    this->Handles.engine = nullptr;

    if ((aerr = adios2_finalize(this->Adios)))
      {
      SENSEI_ERROR("adios2_finalize failed. " << adios2_strerror(aerr))
      --ierr;
      }

    delete this->Schema;
    this->Schema = nullptr;
    }

  return ierr;
}

//----------------------------------------------------------------------------
int ADIOS2AnalysisAdaptor::Finalize()
{
  TimeEvent<128> mark("ADIOS2AnalysisAdaptor::Finalize");
  return this->FinalizeADIOS2();
}

//----------------------------------------------------------------------------
int ADIOS2AnalysisAdaptor::WriteTimestep(unsigned long timeStep,
  double time, const std::vector<MeshMetadataPtr> &metadata,
  const std::vector<svtkCompositeDataSetPtr> &objects)
{
  TimeEvent<128> mark("ADIOS2AnalysisAdaptor::WriteTimestep");

  int ierr = 0;
  adios2_error aerr = adios2_error_none;

  if (this->UpdateStream())
    return -1;

  adios2_step_status status;
  if ((aerr = adios2_begin_step(this->Handles.engine,
    adios2_step_mode_append, -1, &status)))
    {
    SENSEI_ERROR("adios2_begin_step failed. " << adios2_strerror(aerr))
    return -1;
    }

  if (this->Schema->Write(this->GetCommunicator(),
    this->Handles, timeStep, time, metadata, objects))
    {
    SENSEI_ERROR("Failed to write step " << timeStep
      << " to \"" << this->FileName << "\"")
    ierr = -1;
    }

  if ((aerr = adios2_perform_puts(this->Handles.engine)))
    {
    SENSEI_ERROR("adios2_perform_puts failed. " << adios2_strerror(aerr))
    return -1;
    }

  if ((aerr = adios2_end_step(this->Handles.engine)))
    {
    SENSEI_ERROR("adios2_end_step failed. " << adios2_strerror(aerr))
    return -1;
    }

  ++this->StepIndex;

  return ierr;
}

//----------------------------------------------------------------------------
void ADIOS2AnalysisAdaptor::AddParameter(const std::string &key,
  const std::string &value)
{
  this->Parameters.emplace_back(key, value);
}

//----------------------------------------------------------------------------
int ADIOS2AnalysisAdaptor::UpdateStream()
{
  adios2_error aerr = adios2_error_none;

  // close the existing file
  if ((this->StepsPerFile > 0) && this->Handles.engine
    && (this->StepIndex % this->StepsPerFile == 0))
    {
    if ((aerr = adios2_close(this->Handles.engine)))
      {
      SENSEI_ERROR("adios2_close failed. " << adios2_strerror(aerr))
      return -1;
      }

    this->Handles.engine = nullptr;
    }

  // open a new file/stream
  if (!this->Handles.engine)
    {
    // format the file name
    char buffer[1024];
    buffer[1023] = '\0';
    if (this->StepsPerFile > 0)
      {
      snprintf(buffer, 1023, this->FileName.c_str(), this->FileIndex);
      ++this->FileIndex;
      }
    else
      {
      strncpy(buffer, this->FileName.c_str(), 1023);
      }

    // open
    if (!(this->Handles.engine = adios2_open(this->Handles.io,
        buffer, adios2_mode_write)))
      {
      SENSEI_ERROR("Failed to open \"" << buffer << "\" for writing")
      return -1;
      }
    }

  return 0;
}

}
