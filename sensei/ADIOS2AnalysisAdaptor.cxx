#include "ADIOS2AnalysisAdaptor.h"

#include "ADIOS2Schema.h"
#include "DataAdaptor.h"
#include "MeshMetadataMap.h"
#include "VTKUtils.h"
#include "MPIUtils.h"
#include "Profiler.h"
#include "Error.h"

#include <vtkCellTypes.h>
#include <vtkCellData.h>
#include <vtkCompositeDataIterator.h>
#include <vtkCompositeDataSet.h>
#include <vtkDataSetAttributes.h>
#include <vtkDoubleArray.h>
#include <vtkFloatArray.h>
#include <vtkIntArray.h>
#include <vtkUnsignedIntArray.h>
#include <vtkLongArray.h>
#include <vtkUnsignedLongArray.h>
#include <vtkCharArray.h>
#include <vtkUnsignedCharArray.h>
#include <vtkIdTypeArray.h>
#include <vtkCellArray.h>
#include <vtkUnstructuredGrid.h>
#include <vtkPolyData.h>
#include <vtkImageData.h>
#include <vtkObjectFactory.h>
#include <vtkPointData.h>
#include <vtkSmartPointer.h>

#include <mpi.h>
#include <vector>

namespace sensei
{

//----------------------------------------------------------------------------
senseiNewMacro(ADIOS2AnalysisAdaptor);

//----------------------------------------------------------------------------
ADIOS2AnalysisAdaptor::ADIOS2AnalysisAdaptor() : Schema(nullptr),
    FileName("sensei.bp"), DebugMode(0)
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

//----------------------------------------------------------------------------
bool ADIOS2AnalysisAdaptor::Execute(DataAdaptor* dataAdaptor)
{
  TimeEvent<128> mark("ADIOS2AnalysisAdaptor::Execute");

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
  std::vector<vtkCompositeDataSet*> objects;
  std::vector<MeshMetadataPtr> metadata;

  MeshRequirementsIterator mit =
    this->Requirements.GetMeshRequirementsIterator();

  while (mit)
    {
    // get metadata
    MeshMetadataPtr md;
    if (mdm.GetMeshMetadata(mit.MeshName(), md))
      {
      SENSEI_ERROR("Failed to get mesh metadata for mesh \""
        << mit.MeshName() << "\"")
      return false;
      }

    // get the mesh
    vtkCompositeDataSet *dobj = nullptr;
    if (dataAdaptor->GetMesh(mit.MeshName(), mit.StructureOnly(), dobj))
      {
      SENSEI_ERROR("Failed to get mesh \"" << mit.MeshName() << "\"")
      return false;
      }

    // add the ghost cell arrays to the mesh
    if ((md->NumGhostCells || VTKUtils::AMR(md)) &&
        dataAdaptor->AddGhostCellsArray(dobj, mit.MeshName()))
      {
      SENSEI_ERROR("Failed to get ghost cells for mesh \"" << mit.MeshName() << "\"")
      return false;
      }

    // add the ghost node arrays to the mesh
    if (md->NumGhostNodes && dataAdaptor->AddGhostNodesArray(dobj, mit.MeshName()))
      {
      SENSEI_ERROR("Failed to get ghost nodes for mesh \"" << mit.MeshName() << "\"")
      return false;
      }

    // add the required arrays
    ArrayRequirementsIterator ait =
      this->Requirements.GetArrayRequirementsIterator(mit.MeshName());

    while (ait)
      {
      if (dataAdaptor->AddArray(dobj, mit.MeshName(),
         ait.Association(), ait.Array()))
        {
        SENSEI_ERROR("Failed to add "
          << VTKUtils::GetAttributesName(ait.Association())
          << " data array \"" << ait.Array() << "\" to mesh \""
          << mit.MeshName() << "\"")
        return false;
        }

      ++ait;
      }

    // generate a global view of the metadata. everything we do from here
    // on out depends on having the global view.
    md->GlobalizeView(this->GetCommunicator());

    // add to the collection
    objects.push_back(dobj);
    metadata.push_back(md);

    ++mit;
    }

  unsigned long timeStep = dataAdaptor->GetDataTimeStep();
  double time = dataAdaptor->GetDataTime();

  if (this->InitializeADIOS2(metadata) ||
    this->WriteTimestep(timeStep, time, metadata, objects))
    return false;

  unsigned int n_objects = objects.size();
  for (unsigned int i = 0; i < n_objects; ++i)
    objects[i]->Delete();

  return true;
}

//----------------------------------------------------------------------------
int ADIOS2AnalysisAdaptor::InitializeADIOS2(
  const std::vector<MeshMetadataPtr> &metadata)
{
  TimeEvent<128> mark("ADIOS2AnalysisAdaptor::IntializeADIOS2");

  if (!this->Schema)
    {
    // initialize adios2
    this->Adios = adios2_init(this->GetCommunicator(),
      adios2_debug_mode(this->DebugMode));

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

    // Open the engine now variables are declared
    if (adios2_set_engine(this->Handles.io, this->EngineName.c_str()))
      {
      SENSEI_ERROR("adios2_set_engine failed")
      return -1;
      }
    }

  // On subsequent ts we need to clear the existing variables so we don't try to
  // redefine existing variables
  adios2_error clearErr = adios2_remove_all_variables(this->Handles.io);
  if (clearErr != 0)
    {
    SENSEI_ERROR("adios2_remove_all_variables failed " << clearErr )
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
  adios2_error err = adios2_finalize(this->Adios);
  if (err != 0)
    {
    SENSEI_ERROR("ADIOS2 error on adios2_finalize call, error code enum: " << err )
    return -1;
    }
  return 0;
}

//----------------------------------------------------------------------------
int ADIOS2AnalysisAdaptor::Finalize()
{
  TimeEvent<128> mark("ADIOS2AnalysisAdaptor::Finalize");

  if (this->Schema)
    {
    adios2_error err = adios2_close(this->Handles.engine);
    if (err != 0)
      {
      SENSEI_ERROR("ADIOS2 error on adios2_close call, error code enum: " << err )
      return -1;
      }

    this->FinalizeADIOS2();
    }


  delete this->Schema;
  this->Schema = nullptr;
  this->Handles.io = nullptr;
  this->Handles.engine = nullptr;

  return 0;
}

//----------------------------------------------------------------------------
int ADIOS2AnalysisAdaptor::WriteTimestep(unsigned long timeStep,
  double time, const std::vector<MeshMetadataPtr> &metadata,
  const std::vector<vtkCompositeDataSet*> &objects)
{
  TimeEvent<128> mark("ADIOS2AnalysisAdaptor::WriteTimestep");

  int ierr = 0;
  if (!this->Handles.engine)
    {
    // If the user set additional parameters, add them now to ADIOS2
    for (unsigned int j = 0; j < this->Parameters.size(); j++)
      {
      adios2_set_parameter(this->Handles.io,
                           this->Parameters[j].first.c_str(),
                           this->Parameters[j].second.c_str());
      }

    this->Handles.engine = adios2_open(this->Handles.io,
        this->FileName.c_str(), adios2_mode_write);

    if (!this->Handles.engine)
      {
      SENSEI_ERROR("Failed to open \"" << this->FileName << "\" for writing")
      return -1;
      }
    }

  adios2_step_status status;
  adios2_error err = adios2_begin_step(this->Handles.engine,
    adios2_step_mode_append, -1, &status);

  if (err != 0)
    {
    SENSEI_ERROR("ADIOS2 advance time step error, error code\"" << status
      << "\" see adios2_c_types.h for the adios2_step_status enum for details.")
    return -1;
    }


  if (this->Schema->Write(this->GetCommunicator(),
    this->Handles, timeStep, time, metadata, objects))
    {
    SENSEI_ERROR("Failed to write step " << timeStep
      << " to \"" << this->FileName << "\"")
    ierr = -1;
    }

  adios2_perform_puts(this->Handles.engine);

  adios2_error endErr = adios2_end_step(this->Handles.engine);
  if (endErr != 0)
    {
    SENSEI_ERROR("ADIOS2 error on adios2_end_step call, error code enum: " << endErr )
    return -1;
    }

  return ierr;
}

//----------------------------------------------------------------------------
void ADIOS2AnalysisAdaptor::AddParameter(const std::string &key,
  const std::string &value)
{
  this->Parameters.emplace_back(key, value);
}

}
