#include "DataAdaptor.h"
#include "MeshMetadata.h"
#include "VTKUtils.h"
#include "Error.h"

#include <vtkDataObject.h>
#include <vtkCompositeDataSet.h>
#include <vtkObjectFactory.h>

#include <map>
#include <vector>
#include <string>
#include <utility>

namespace sensei
{

struct DataAdaptor::InternalsType
{
  InternalsType() : Time(0.0), TimeStep(0) {}
  ~InternalsType() {}

  MeshMetadataFlags Flags;
  std::vector<MeshMetadataPtr> Metadata;
  double Time;
  long TimeStep;
};

//----------------------------------------------------------------------------
DataAdaptor::DataAdaptor()
{
  MPI_Comm_dup(MPI_COMM_WORLD, &this->Comm);
  this->Internals = new InternalsType;
}

//----------------------------------------------------------------------------
DataAdaptor::~DataAdaptor()
{
  MPI_Comm_free(&this->Comm);
  delete this->Internals;
}

//----------------------------------------------------------------------------
int DataAdaptor::SetCommunicator(MPI_Comm comm)
{
  MPI_Comm_free(&this->Comm);
  MPI_Comm_dup(comm, &this->Comm);
  return 0;
}

//----------------------------------------------------------------------------
double DataAdaptor::GetDataTime()
{
  return this->Internals->Time;
}

//----------------------------------------------------------------------------
void DataAdaptor::SetDataTime(double time)
{
  this->Internals->Time = time;
}

//----------------------------------------------------------------------------
long DataAdaptor::GetDataTimeStep()
{
  return this->Internals->TimeStep;
}

//----------------------------------------------------------------------------
void DataAdaptor::SetDataTimeStep(long index)
{
  this->Internals->TimeStep = index;
}

//----------------------------------------------------------------------------
int DataAdaptor::GetMesh(const std::string &meshName, bool structureOnly,
    vtkCompositeDataSet *&mesh)
{
  mesh = nullptr;

  // get the object from the simulation
  vtkDataObject *dobj = nullptr;
  if (this->GetMesh(meshName, structureOnly, dobj))
    {
    SENSEI_ERROR("Failed to get mesh \"" << meshName << "\"")
    return -1;
    }

  vtkCompositeDataSetPtr meshptr = VTKUtils::AsCompositeData(
    this->GetCommunicator(), dobj, true);

  mesh = meshptr.GetPointer();
  mesh->Register(nullptr);

  return 0;
}

//----------------------------------------------------------------------------
int DataAdaptor::AddArrays(vtkDataObject* mesh, const std::string &meshName,
    int association, const std::vector<std::string> &arrayNames)
{
  unsigned int nArrays = arrayNames.size();
  for (unsigned int i = 0; i < nArrays; ++i)
    {
    if (this->AddArray(mesh, meshName, association, arrayNames[i]))
      {
      SENSEI_ERROR("Failed to add "
        << VTKUtils::GetAttributesName(association) << " data array \""
        << arrayNames[i] << " to mesh \"" << meshName << "\"")
      return -1;
      }
    }
  return 0;
}

//----------------------------------------------------------------------------
int DataAdaptor::AddGhostNodesArray(vtkDataObject*, const std::string &)
{
  return 0;
}

//----------------------------------------------------------------------------
int DataAdaptor::AddGhostCellsArray(vtkDataObject*, const std::string &)
{
  return 0;
}

//----------------------------------------------------------------------------
void DataAdaptor::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

}
