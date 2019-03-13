#include "HDF5DataAdaptor.h"

#include "Error.h"
#include "Timer.h"

#include "VTKUtils.h"
#include "MeshMetadata.h"

#include <vtkCompositeDataIterator.h>
#include <vtkDataSetAttributes.h>
#include <vtkInformation.h>
#include <vtkMultiBlockDataSet.h>
#include <vtkObjectFactory.h>
#include <vtkSmartPointer.h>
#include <vtkDataSet.h>

#include <sstream>

namespace sensei
{


//----------------------------------------------------------------------------
senseiNewMacro(HDF5DataAdaptor);

//----------------------------------------------------------------------------
  HDF5DataAdaptor::HDF5DataAdaptor():m_HDF5Reader(nullptr)
{
}

//----------------------------------------------------------------------------
HDF5DataAdaptor::~HDF5DataAdaptor()
{
  delete m_HDF5Reader;
}

//----------------------------------------------------------------------------
  int HDF5DataAdaptor::Open(const std::string& fileName)			  
{
  timer::MarkEvent mark("HDF5DataAdaptor::Open");
  
  if (this->m_HDF5Reader == NULL) {
    this->m_HDF5Reader = new senseiHDF5::ReadStream(this->GetCommunicator(), m_Streaming);
  }
      
  if (!this->m_HDF5Reader->Init(fileName)) {
    SENSEI_ERROR("Failed to open \"" << fileName << "\"");
    return -1;
  }
  
  // initialize the time step                                                                                                                                                                               
  if (this->UpdateTimeStep())
    return -1;

  return 0;
}


//----------------------------------------------------------------------------
int HDF5DataAdaptor::Close()
{
  timer::MarkEvent mark("HDF5DataAdaptor::Close");
  int m_Rank;
  MPI_Comm_rank(GetCommunicator(), &m_Rank);

  if (this->m_HDF5Reader != nullptr) {
    delete this->m_HDF5Reader;
    this->m_HDF5Reader = nullptr;
  }

  return 0;
}

//----------------------------------------------------------------------------
int HDF5DataAdaptor::Advance()
{
  timer::MarkEvent mark("HDF5DataAdaptor::Advance");

  return this->UpdateTimeStep();
}

//----------------------------------------------------------------------------
int HDF5DataAdaptor::UpdateTimeStep()
{
  timer::MarkEvent mark("HDF5DataAdaptor::UpdateTimeStep");

  // update data object time and time step
  unsigned long timeStep = 0;
  double time = 0.0;
  
  if (!this->m_HDF5Reader->AdvanceTimeStep(timeStep, time))
    {
      //SENSEI_ERROR("Failed to update time step");
      return -1;
    }

  this->SetDataTimeStep(timeStep);
  this->SetDataTime(time);

  // read metadata

  unsigned int nMeshes = 0;
  if (!this->m_HDF5Reader->ReadMetadata(nMeshes)) {
    SENSEI_ERROR("Failed to read metadata at timestep: "<<timeStep);
      return -1;
  }

  if (nMeshes == 0)  {    
      SENSEI_ERROR("No Mesh at this timestep found");
      return -1;
  }
  return 0;
}

//----------------------------------------------------------------------------
int HDF5DataAdaptor::GetNumberOfMeshes(unsigned int &numMeshes)
{
  if (this->m_HDF5Reader) {
    numMeshes =  this->m_HDF5Reader->GetNumberOfMeshes();
    return 0;
  }
    
  return  -1;
}

//----------------------------------------------------------------------------
int HDF5DataAdaptor::GetMeshMetadata(unsigned int id, MeshMetadataPtr &metadata)
{
  if (this->m_HDF5Reader->ReadMeshMetaData(id, metadata))
    return 0;
  else 
    {
      SENSEI_ERROR("Failed to get metadata for object " << id);
      return -1;
    }
}

//----------------------------------------------------------------------------
int HDF5DataAdaptor::GetMesh(const std::string &meshName,
			     bool structureOnly, vtkDataObject *&mesh)
{
  timer::MarkEvent mark("HDF5DataAdaptor::GetMesh");

  mesh = nullptr;

  // other wise we need to read the mesh at the current time step
  if (!this->m_HDF5Reader->ReadMesh(meshName, mesh, structureOnly))
  {
    SENSEI_ERROR("Failed to read mesh \"" << meshName << "\"");
    return -1;
  }

  return 0;
}

//----------------------------------------------------------------------------
int HDF5DataAdaptor::AddGhostNodesArray(vtkDataObject *mesh,
					const std::string &meshName)
{
  return AddArray(mesh, meshName, vtkDataObject::POINT, "vtkGhostType");
}

//----------------------------------------------------------------------------
int HDF5DataAdaptor::AddGhostCellsArray(vtkDataObject *mesh,
					const std::string &meshName)
{
  return AddArray(mesh, meshName, vtkDataObject::CELL, "vtkGhostType");
}

//----------------------------------------------------------------------------
int HDF5DataAdaptor::AddArray(vtkDataObject* mesh,
			      const std::string &meshName, 
			      int association, 
			      const std::string& arrayName)
{
  timer::MarkEvent mark("HDF5DataAdaptor::AddArray");

  // the mesh should never be null. there must have been an error
  // upstream.
  if (!mesh) {
    SENSEI_ERROR("Invalid mesh object");
    return -1;
  }

  if (!this->m_HDF5Reader->ReadInArray(meshName, association, arrayName, mesh))					
    {
      SENSEI_ERROR("Failed to read " << VTKUtils::GetAttributesName(association)
		   << " data array \"" << arrayName << "\" from mesh \"" << meshName << "\"");
      return -1;
    }

  return 0;
}

//----------------------------------------------------------------------------
int HDF5DataAdaptor::ReleaseData()
{
  timer::MarkEvent mark("HDF5DataAdaptor::ReleaseData");
  return 0;
}

//----------------------------------------------------------------------------
void HDF5DataAdaptor::PrintSelf(ostream& os, vtkIndent indent)
{
  this->sensei::DataAdaptor::PrintSelf(os, indent);
}

} // namespace
