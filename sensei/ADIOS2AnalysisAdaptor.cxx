#include "ADIOS2AnalysisAdaptor.h"

#include "ADIOS2Schema.h"
#include "DataAdaptor.h"
#include "MeshMetadataMap.h"
#include "VTKUtils.h"
#include "MPIUtils.h"
#include "Timer.h"
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
#include <adios.h>
#include <vector>

namespace sensei
{

//----------------------------------------------------------------------------
senseiNewMacro(ADIOS2AnalysisAdaptor);

//----------------------------------------------------------------------------
ADIOS2AnalysisAdaptor::ADIOS2AnalysisAdaptor() : Schema(nullptr), FileName("sensei.bp")
{
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
  timer::MarkEvent mark("ADIOS2AnalysisAdaptor::Execute");

  // figure out what the simulation can provide. include the full
  // suite of metadata for the end-point partitioners
  MeshMetadataFlags flags;
  flags.SetBlockDecomp();
  flags.SetBlockSize();
  flags.SetBlockBounds();
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
    if (md->NumGhostCells && dataAdaptor->AddGhostCellsArray(dobj, mit.MeshName()))
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
  timer::MarkEvent mark("ADIOS2AnalysisAdaptor::IntializeADIOS2");

  if (!this->Schema)
    {
    // initialize adios2
    // args  0: comm
    //       1: debug mode
    //TODO turn off debug/expose to xml
    this->Adios = adios2_init(this->GetCommunicator(), adios2_debug_mode_on);

    // Open the io handle
    this->Handles.io = adios2_declare_io(this->Adios, "SENSEI");

    // create space for ADIOS2 variables
    this->Schema = new senseiADIOS2::DataObjectCollectionSchema;

    // Open the engine now variables are declared
    adios2_set_engine(this->Handles.io, this->EngineName.c_str());
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
  adios2_finalize(this->Adios);
  return 0;
}

//----------------------------------------------------------------------------
int ADIOS2AnalysisAdaptor::Finalize()
{
  timer::MarkEvent mark("ADIOS2AnalysisAdaptor::Finalize");

  if (this->Schema)
    {
    adios2_close(this->Handles.engine);
    this->FinalizeADIOS2();
    }


  delete this->Schema;
  this->Schema = nullptr;

  return 0;
}

//----------------------------------------------------------------------------
int ADIOS2AnalysisAdaptor::WriteTimestep(unsigned long timeStep,
  double time, const std::vector<MeshMetadataPtr> &metadata,
  const std::vector<vtkCompositeDataSet*> &objects)
{
  timer::MarkEvent mark("ADIOS2AnalysisAdaptor::WriteTimestep");

  int ierr = 0;
  if(!this->Handles.engine)
    {
    //engine type should have already been set
    this->Handles.engine = adios2_open(this->Handles.io, this->FileName.c_str(), adios2_mode_write);
    }

  if (this->Schema->Write(this->GetCommunicator(),
    this->Handles, timeStep, time, metadata, objects))
    {
    SENSEI_ERROR("Failed to write step " << timeStep
      << " to \"" << this->FileName << "\"")
    ierr = -1;
    }

  adios2_close(this->Handles.engine);

  return ierr;
}

//----------------------------------------------------------------------------
void ADIOS2AnalysisAdaptor::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

}
