#include "DataAdaptor.h"
#include "MeshMetadata.h"
#include "VTKUtils.h"
#include "Error.h"

#include <vtkDataObject.h>
#include <vtkInformation.h>
#include <vtkInformationIntegerKey.h>
#include <vtkObjectFactory.h>

#include <map>
#include <vector>
#include <string>
#include <utility>

namespace sensei
{

struct DataAdaptor::InternalsType
{
  InternalsType() : Information(vtkInformation::New()) {}
  ~InternalsType(){ this->Information->Delete(); }

  void Clear()
  {
    this->Information->Delete();
    this->Information = vtkInformation::New();
  }

  MeshMetadataFlags Flags;
  std::vector<MeshMetadataPtr> Metadata;
  vtkInformation *Information;
};

//----------------------------------------------------------------------------
vtkInformationKeyMacro(DataAdaptor, DATA_TIME_STEP_INDEX, Integer);

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
vtkInformation* DataAdaptor::GetInformation()
{
  return this->Internals->Information;
}

//----------------------------------------------------------------------------
double DataAdaptor::GetDataTime()
{
  return this->GetDataTime(this->Internals->Information);
}

//----------------------------------------------------------------------------
void DataAdaptor::SetDataTime(double time)
{
  this->SetDataTime(this->Internals->Information, time);
}

//----------------------------------------------------------------------------
int DataAdaptor::GetDataTimeStep()
{
  return this->GetDataTimeStep(this->Internals->Information);
}

//----------------------------------------------------------------------------
double DataAdaptor::GetDataTime(vtkInformation* info)
{
  return info->Has(vtkDataObject::DATA_TIME_STEP()) ?
    info->Get(vtkDataObject::DATA_TIME_STEP()) : 0.0;
}

//----------------------------------------------------------------------------
void DataAdaptor::SetDataTime(vtkInformation* info, double time)
{
  info->Set(vtkDataObject::DATA_TIME_STEP(), time);
}

//----------------------------------------------------------------------------
void DataAdaptor::SetDataTimeStep(int index)
{
  this->SetDataTimeStep(this->Internals->Information, index);
}

//----------------------------------------------------------------------------
int DataAdaptor::GetDataTimeStep(vtkInformation* info)
{
  return info->Has(DataAdaptor::DATA_TIME_STEP_INDEX()) ?
    info->Get(DataAdaptor::DATA_TIME_STEP_INDEX()) : 0;
}

//----------------------------------------------------------------------------
void DataAdaptor::SetDataTimeStep(vtkInformation* info, int index)
{
  info->Set(DataAdaptor::DATA_TIME_STEP_INDEX(), index);
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
