#include "HDF5AnalysisAdaptor.h"

#include "DataAdaptor.h"
#include "Error.h"
#include "HDF5Schema.h"
#include "MPIUtils.h"
#include "MeshMetadataMap.h"
#include "Profiler.h"
#include "SVTKUtils.h"

#include <svtkCellArray.h>
#include <svtkCellData.h>
#include <svtkCellTypes.h>
#include <svtkCharArray.h>
#include <svtkCompositeDataIterator.h>
#include <svtkCompositeDataSet.h>
#include <svtkDataSetAttributes.h>
#include <svtkDoubleArray.h>
#include <svtkFloatArray.h>
#include <svtkIdTypeArray.h>
#include <svtkImageData.h>
#include <svtkInformation.h>
#include <svtkIntArray.h>
#include <svtkLongArray.h>
#include <svtkObjectFactory.h>
#include <svtkPointData.h>
#include <svtkPolyData.h>
#include <svtkSmartPointer.h>
#include <svtkUnsignedCharArray.h>
#include <svtkUnsignedIntArray.h>
#include <svtkUnsignedLongArray.h>
#include <svtkUnstructuredGrid.h>

#include <mpi.h>
#include <vector>

namespace sensei
{

//----------------------------------------------------------------------------
senseiNewMacro(HDF5AnalysisAdaptor);

//----------------------------------------------------------------------------
HDF5AnalysisAdaptor::HDF5AnalysisAdaptor()
  : m_FileName("no.file")
  , m_HDF5Writer(nullptr)
{
}

//----------------------------------------------------------------------------
HDF5AnalysisAdaptor::~HDF5AnalysisAdaptor()
{
  delete m_HDF5Writer;
}

//-----------------------------------------------------------------------------
int HDF5AnalysisAdaptor::SetDataRequirements(const DataRequirements& reqs)
{
  this->Requirements = reqs;
  return 0;
}

//-----------------------------------------------------------------------------
int HDF5AnalysisAdaptor::AddDataRequirement(
  const std::string& meshName,
  int association,
  const std::vector<std::string>& arrays)
{
  this->Requirements.AddRequirement(meshName, association, arrays);
  return 0;
}

//----------------------------------------------------------------------------
bool HDF5AnalysisAdaptor::Execute(DataAdaptor* dataAdaptor, DataAdaptor** daOut)
{
  TimeEvent<128> mark("HDF5AnalysisAdaptor::Execute");

  // we currently do not return anything
  if (daOut)
    {
    daOut = nullptr;
    }

  // figure out what the simulation can provide
  MeshMetadataFlags flags;
  flags.SetBlockDecomp();
  flags.SetBlockSize();

  MeshMetadataMap mdm;
  if (mdm.Initialize(dataAdaptor, flags))
    {
      SENSEI_ERROR("Failed to get metadata");
      return false;
    }

  //
  // (usage is from beambeam3d)
  // we set block extent if images
  // cannt set it with SetBlockDecomp() etc
  // error is thrown for non image meshes
  //
  for (unsigned int i = 0; i < mdm.Size(); ++i)
    {
      MeshMetadataPtr older;
      mdm.GetMeshMetadata(i, older);

      if (older->BlockType == SVTK_IMAGE_DATA) 
      {
	MeshMetadataPtr curr = sensei::MeshMetadata::New();;
	flags.SetBlockExtents();
	curr->Flags = flags;

	if (dataAdaptor->GetMeshMetadata(i, curr))
	{
	  SENSEI_ERROR("Failed to get metadata with block extent for data object " << i)
	    return -1;
	}

	if (curr->Validate(dataAdaptor->GetCommunicator(), flags))
	{
	  SENSEI_ERROR("The requested metadata was not provided for data object " << i)
	    return -1;
	}
	mdm.SetMeshMetadata(i, curr);
      }
    }


  // if no dataAdaptor requirements are given, push all the data
  // fill in the requirements with every thing
  if (this->Requirements.Empty())
    {
      if (this->Requirements.Initialize(dataAdaptor, false))
        {
          SENSEI_ERROR("Failed to initialze dataAdaptor description");
          return false;
        }
      SENSEI_WARNING("No subset specified. Writing all available data");
    }

  if (!this->InitializeHDF5())
    return false;

  unsigned long timeStep = dataAdaptor->GetDataTimeStep();
  double time = dataAdaptor->GetDataTime();

  if (!this->m_HDF5Writer->AdvanceTimeStep(timeStep, time))
    return false;

  // senseiHDF5::HDF5GroupGuard g(this->m_HDF5Writer->m_TimeStepGroupId);

  // collect the specified data objects and metadata
  // std::vector<svtkCompositeDataSet*> objects;

  MeshRequirementsIterator mit =
    this->Requirements.GetMeshRequirementsIterator();

  // unsigned int meshCounter = 0;

  while (mit)
    {
      // get metadata
      MeshMetadataPtr md;
      if (mdm.GetMeshMetadata(mit.MeshName(), md))
        {
          SENSEI_ERROR("Failed to get mesh metadata for mesh \""
                       << mit.MeshName() << "\"");
          return false;
        }

      // get the mesh
      svtkCompositeDataSet* dobj = nullptr;
      if (dataAdaptor->GetMesh(mit.MeshName(), mit.StructureOnly(), dobj))
        {
          SENSEI_ERROR("Failed to get mesh \"" << mit.MeshName() << "\"");
          return false;
        }

      // add the ghost cell arrays to the mesh
      if ((md->NumGhostCells || SVTKUtils::AMR(md)) &&
          dataAdaptor->AddGhostCellsArray(dobj, mit.MeshName()))
        {
          SENSEI_ERROR("Failed to get ghost cells for mesh \"" << mit.MeshName()
                                                               << "\"")
          return false;
        }

      // add the ghost node arrays to the mesh
      if (md->NumGhostNodes &&
          dataAdaptor->AddGhostNodesArray(dobj, mit.MeshName()))
        {
          SENSEI_ERROR("Failed to get ghost nodes for mesh \"" << mit.MeshName()
                                                               << "\"");
          return false;
        }

      // add the required arrays
      ArrayRequirementsIterator ait =
        this->Requirements.GetArrayRequirementsIterator(mit.MeshName());

      while (ait)
        {
          if (dataAdaptor->AddArray(
                dobj, mit.MeshName(), ait.Association(), ait.Array()))
            {
              SENSEI_ERROR("Failed to add "
                           << SVTKUtils::GetAttributesName(ait.Association())
                           << " data array \"" << ait.Array() << "\" to mesh \""
                           << mit.MeshName() << "\"");
              return false;
            }

          ++ait;
        }

      // generate a global view of the metadata. everything we do from here
      // on out depends on having the global view.
      if (!md->GlobalView)
        {
          MPI_Comm comm = this->GetCommunicator();
          sensei::MPIUtils::GlobalViewV(comm, md->BlockOwner);
          sensei::MPIUtils::GlobalViewV(comm, md->BlockIds);
          sensei::MPIUtils::GlobalViewV(comm, md->BlockNumPoints);
          sensei::MPIUtils::GlobalViewV(comm, md->BlockNumCells);
          sensei::MPIUtils::GlobalViewV(comm, md->BlockCellArraySize);
          sensei::MPIUtils::GlobalViewV(comm, md->BlockExtents);
          md->GlobalView = true;
        }

      this->m_HDF5Writer->WriteMesh(md, dobj);

      dobj->Delete();

      ++mit;
    }

  return true;
}

//----------------------------------------------------------------------------
bool HDF5AnalysisAdaptor::InitializeHDF5()
{
  TimeEvent<128> mark("HDF5AnalysisAdaptor::IntializeHDF5");

  if (!this->m_HDF5Writer)
    {
      this->m_HDF5Writer =
        new senseiHDF5::WriteStream(this->GetCommunicator(), m_DoStreaming);
      if (!this->m_HDF5Writer->Init(this->m_FileName))
        {
          return -1;
        }
    }
  return true;
}

//----------------------------------------------------------------------------
int HDF5AnalysisAdaptor::Finalize()
{
  TimeEvent<128> mark("HDF5AnalysisAdaptor::Finalize");

  if (this->m_HDF5Writer)
    delete this->m_HDF5Writer;

  this->m_HDF5Writer = nullptr;

  return 0;
}

/*
//----------------------------------------------------------------------------
bool HDF5AnalysisAdaptor::WriteTimestep(unsigned long timeStep, double time,
                                      const std::vector<svtkCompositeDataSet*>
&objects)
{
TimeEvent<128> mark("HDF5AnalysisAdaptor::WriteTimestep");

int ierr = 0;

TimeEvent<128> mark("HDF5AnalysisAdaptor::WriteTimestep");

this->m_HDF5Writer->AdvanceTimeStep(timestep, time);

if (this->m_HDF5Writer->Write(objects))
  {
    SENSEI_ERROR("Failed to write step " << timeStep
                 << " to \"" << this->FileName << "\"");
    return -1;
  }

return ierr;
}
*/
//----------------------------------------------------------------------------
void HDF5AnalysisAdaptor::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

} // namespace sensei
