#include "libISAnalysisAdaptor.h"

#include "libISSchema.h"
#include "libis/is_sim.h"
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
#include <vector>

namespace sensei
{

//----------------------------------------------------------------------------
senseiNewMacro(libISAnalysisAdaptor);

//----------------------------------------------------------------------------
libISAnalysisAdaptor::libISAnalysisAdaptor() : 
    Schema(nullptr), port(29374), GroupHandle(0)
{
}

//----------------------------------------------------------------------------
libISAnalysisAdaptor::~libISAnalysisAdaptor()
{
  delete this->Schema;
}

//-----------------------------------------------------------------------------
int libISAnalysisAdaptor::SetDataRequirements(const DataRequirements &reqs)
{
  this->Requirements = reqs;
  return 0;
}

//-----------------------------------------------------------------------------
int libISAnalysisAdaptor::AddDataRequirement(const std::string &meshName,
  int association, const std::vector<std::string> &arrays)
{
  this->Requirements.AddRequirement(meshName, association, arrays);
  return 0;
}

//----------------------------------------------------------------------------
bool libISAnalysisAdaptor::Execute(DataAdaptor* dataAdaptor)
{
  timer::MarkEvent mark("libISAnalysisAdaptor::Execute");

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
      SENSEI_ERROR("Failed to initialize dataAdaptor description")
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
    // fixme. DEBUG only
    else
      {
      SENSEI_STATUS("got mesh metadata for mesh \""
        << mit.MeshName() << "\"")
      }


    // get the mesh
    vtkCompositeDataSet *dobj = nullptr;
    if (dataAdaptor->GetMesh(mit.MeshName(), mit.StructureOnly(), dobj))
      {
      SENSEI_ERROR("Failed to get mesh \"" << mit.MeshName() << "\"")
      return false;
      }
    // fixme. DEBUG only
    else
      {
      SENSEI_STATUS("got mesh \""
        << mit.MeshName() << "\"")
      }


    // add the ghost cell arrays to the mesh
    if (md->NumGhostCells && dataAdaptor->AddGhostCellsArray(dobj, mit.MeshName()))
      {
      SENSEI_ERROR("Failed to get ghost cells for mesh \"" << mit.MeshName() << "\"")
      return false;
      }
    // fixme. DEBUG only
    else
      {
      SENSEI_STATUS("got ghost cell for mesh \""
        << mit.MeshName() << "\"")
      }

    // add the ghost node arrays to the mesh
    if (md->NumGhostNodes && dataAdaptor->AddGhostNodesArray(dobj, mit.MeshName()))
      {
      SENSEI_ERROR("Failed to get ghost nodes for mesh \"" << mit.MeshName() << "\"")
      return false;
      }
    // fixme. DEBUG only
    else
      {
      SENSEI_STATUS("got ghost nodes for mesh \""
        << mit.MeshName() << "\"")
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

  if (this->InitializelibIS(metadata) ||
    this->WriteTimestep(timeStep, time, metadata, objects))
    return false;

  unsigned int n_objects = objects.size();
  for (unsigned int i = 0; i < n_objects; ++i)
    objects[i]->Delete();

  return true;
}

//----------------------------------------------------------------------------
int libISAnalysisAdaptor::InitializelibIS(
  const std::vector<MeshMetadataPtr> &metadata)
{
  timer::MarkEvent mark("libISAnalysisAdaptor::IntializelibIS");

  if (!this->Schema)
    {

    //libISInit(this->GetCommunicator(), this->port);

    // define libIS variables
    this->Schema = new senseilibIS::DataObjectCollectionSchema;
    }

  // (re)define variables to support meshes that evolve in time
  if (this->Schema->DefineVariables(this->GetCommunicator(),
    this->GroupHandle, metadata))
    {
    SENSEI_ERROR("Failed to define variables")
    return -1;
    }

  return 0;
}

//----------------------------------------------------------------------------
int libISAnalysisAdaptor::FinalizelibIS()
{
  libISFinalize();
  return 0;
}

//----------------------------------------------------------------------------
int libISAnalysisAdaptor::Finalize()
{
  timer::MarkEvent mark("libISAnalysisAdaptor::Finalize");

  if (this->Schema)
    this->FinalizelibIS();

  delete this->Schema;
  this->Schema = nullptr;

  return 0;
}

//----------------------------------------------------------------------------
int libISAnalysisAdaptor::WriteTimestep(unsigned long timeStep,
  double time, const std::vector<MeshMetadataPtr> &metadata,
  const std::vector<vtkCompositeDataSet*> &objects)
{
  timer::MarkEvent mark("libISAnalysisAdaptor::WriteTimestep");

  int ierr = 0;
  int64_t handle = 0;

  //adios_open(&handle, "sensei", this->FileName.c_str(),
  //  timeStep == 0 ? "w" : "a", this->GetCommunicator());

  // TODO -- what are the implications of not setting the group
  // size? it's a lot of work to calculate the size. user manual
  // indicates that it is optional. would setting a fixed size
  // like 200MB be better than not setting the size?
  //
  /*uint64_t group_size = this->Schema->GetSize(
    this->GetCommunicator(), metadata, objects);
  adios_group_size(handle, group_size, &group_size);*/

  //libISSimState *libis_state = libISMakeSimState();

//fixme

  // may need to get size from schema here


  /////////////////////////////////////////////////////////////////
  // from libIS example, just to make sure the code builds for now

  /*************
  struct Particle {
	//vec3<float> pos;
	int attrib;
  };

  size_t NUM_PARTICLES = 2000;

  std::vector<float> field_one;
  std::vector<uint8_t> field_two;
  const std::array<uint64_t, 3> field_dims({32, 32, 32});

  std::vector<Particle> particle;

  libISBox3f bounds;
  ***************/




  /////////////////////////////////////////////////////////////////



  // fixme
  // after the call to libISMakeBox3f(), bounds are set to
  // std::numeric_limits<float>::infinity() 
  // and -std::numeric_limits<float>::infinity()
  // I will not change those values for the moment

  //libISBox3f world_bounds = libISMakeBox3f();
  //libISBoxExtend(&world_bounds, &world_min);
  //libISBoxExtend(&world_bounds, &world_max); 
  //libISSetWorldBounds(libis_state, world_bounds);







  //libISSetLocalBounds(libis_state, bounds);
  //libISSetGhostBounds(libis_state, bounds);

  // Setup the shared pointers to our particle and field data
  //libISSetParticles(libis_state, NUM_PARTICLES, 0, sizeof(Particle), particle.data());
  //libISSetField(libis_state, "field_one", field_dims.data(), FLOAT, field_one.data());
  //libISSetField(libis_state, "field_two", field_dims.data(), UINT8, field_two.data());


  /*
  if (this->Schema->Write(this->GetCommunicator(),
    handle, timeStep, time, metadata, objects))
    {
    SENSEI_ERROR("Failed to write step " << timeStep)
      //<< " to \"" << this->FileName << "\"")
    ierr = -1;
    }
  */

  //adios_close(handle);
  //libISFreeSimState(libis_state);

  return ierr;
}

//----------------------------------------------------------------------------
void libISAnalysisAdaptor::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

}
