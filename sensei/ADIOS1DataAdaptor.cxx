#include "ADIOS1DataAdaptor.h"

#include "Error.h"
#include "Timer.h"
#include "ADIOS1Schema.h"
#include "VTKUtils.h"
#include "MeshMetadata.h"

#include <vtkCompositeDataIterator.h>
#include <vtkDataSetAttributes.h>
#include <vtkMultiBlockDataSet.h>
#include <vtkObjectFactory.h>
#include <vtkSmartPointer.h>
#include <vtkDataSet.h>

#include <sstream>

namespace sensei
{
struct ADIOS1DataAdaptor::InternalsType
{
  InternalsType() : Stream() {}

  senseiADIOS1::InputStream Stream;
  senseiADIOS1::DataObjectCollectionSchema Schema;
};

//----------------------------------------------------------------------------
senseiNewMacro(ADIOS1DataAdaptor);

//----------------------------------------------------------------------------
ADIOS1DataAdaptor::ADIOS1DataAdaptor() : Internals(nullptr)
{
  this->Internals = new InternalsType;
}

//----------------------------------------------------------------------------
ADIOS1DataAdaptor::~ADIOS1DataAdaptor()
{
  delete this->Internals;
}

//----------------------------------------------------------------------------
int ADIOS1DataAdaptor::Open(const std::string &method,
  const std::string& filename)
{
  size_t n = method.size();
  std::string lcase_method(n, ' ');
  for (size_t i = 0; i < n; ++i)
    lcase_method[i] = tolower(method[i]);

  std::map<std::string, ADIOS_READ_METHOD> readMethods;
  readMethods["bp"] = ADIOS_READ_METHOD_BP;
  readMethods["bp_aggregate"] = ADIOS_READ_METHOD_BP_AGGREGATE;
  readMethods["dataspaces"] = ADIOS_READ_METHOD_DATASPACES;
  readMethods["dimes"] = ADIOS_READ_METHOD_DIMES;
  readMethods["flexpath"] = ADIOS_READ_METHOD_FLEXPATH;

  std::map<std::string, ADIOS_READ_METHOD>::iterator it =
    readMethods.find(lcase_method);

  if (it == readMethods.end())
    {
    SENSEI_ERROR("Unsupported read method requested \"" << method << "\"")
    return -1;
    }

  return this->Open(it->second, filename);
}

//----------------------------------------------------------------------------
int ADIOS1DataAdaptor::Open(ADIOS_READ_METHOD method, const std::string& fileName)
{
  timer::MarkEvent mark("ADIOS1DataAdaptor::Open");

  if (this->Internals->Stream.Open(this->GetCommunicator(), method, fileName))
    {
    SENSEI_ERROR("Failed to open \"" << fileName << "\"")
    return -1;
    }

  // initialize the time step
  if (this->UpdateTimeStep())
    return -1;

  return 0;
}

//----------------------------------------------------------------------------
int ADIOS1DataAdaptor::Close()
{
  timer::MarkEvent mark("ADIOS1DataAdaptor::Close");

  this->Internals->Stream.Close();

  return 0;
}

//----------------------------------------------------------------------------
int ADIOS1DataAdaptor::Advance()
{
  timer::MarkEvent mark("ADIOS1DataAdaptor::Advance");

  if (this->Internals->Stream.AdvanceTimeStep())
    return -1;

  if (this->UpdateTimeStep())
    return -1;

  return 0;
}

//----------------------------------------------------------------------------
int ADIOS1DataAdaptor::UpdateTimeStep()
{
  timer::MarkEvent mark("ADIOS1DataAdaptor::UpdateTimeStep");

  // update data object time and time step
  unsigned long timeStep = 0;
  double time = 0.0;

  if (this->Internals->Schema.ReadTimeStep(this->GetCommunicator(),
    this->Internals->Stream, timeStep, time))
    {
    SENSEI_ERROR("Failed to update time step")
    return -1;
    }

  this->SetDataTimeStep(timeStep);
  this->SetDataTime(time);

  // read metadata
  if (this->Internals->Schema.ReadMeshMetadata(this->GetCommunicator(),
    this->Internals->Stream))
    {
    SENSEI_ERROR("Failed to read metadata")
    return -1;
    }

  // set up the name-object map
  unsigned int nMeshes = 0;
  if (this->Internals->Schema.GetNumberOfObjects(nMeshes))
    {
    SENSEI_ERROR("Failed to get the number of meshes")
    return -1;
    }

  return 0;
}

//----------------------------------------------------------------------------
int ADIOS1DataAdaptor::GetNumberOfMeshes(unsigned int &numMeshes)
{
  numMeshes = 0;
  if (this->Internals->Schema.GetNumberOfObjects(numMeshes))
    return -1;
  return  0;
}

//----------------------------------------------------------------------------
int ADIOS1DataAdaptor::GetMeshMetadata(unsigned int id, MeshMetadataPtr &metadata)
{
  if (this->Internals->Schema.GetMeshMetadata(id, metadata))
    {
    SENSEI_ERROR("Failed to get metadata for object " << id)
    return -1;
    }

  return 0;
}

//----------------------------------------------------------------------------
int ADIOS1DataAdaptor::GetMesh(const std::string &meshName,
   bool structureOnly, vtkDataObject *&mesh)
{
  timer::MarkEvent mark("ADIOS1DataAdaptor::GetMesh");

  mesh = nullptr;

  // other wise we need to read the mesh at the current time step
  if (this->Internals->Schema.ReadObject(this->GetCommunicator(),
    this->Internals->Stream, meshName, mesh, structureOnly))
    {
    SENSEI_ERROR("Failed to read mesh \"" << meshName << "\"")
    return -1;
    }

  return 0;
}

//----------------------------------------------------------------------------
int ADIOS1DataAdaptor::AddGhostNodesArray(vtkDataObject *mesh,
  const std::string &meshName)
{
  return AddArray(mesh, meshName, vtkDataObject::POINT, "vtkGhostType");
}

//----------------------------------------------------------------------------
int ADIOS1DataAdaptor::AddGhostCellsArray(vtkDataObject *mesh,
  const std::string &meshName)
{
  return AddArray(mesh, meshName, vtkDataObject::CELL, "vtkGhostType");
}

//----------------------------------------------------------------------------
int ADIOS1DataAdaptor::AddArray(vtkDataObject* mesh,
  const std::string &meshName, int association, const std::string& arrayName)
{
  timer::MarkEvent mark("ADIOS1DataAdaptor::AddArray");

  // the mesh should never be null. there must have been an error
  // upstream.
  if (!mesh)
    {
    SENSEI_ERROR("Invalid mesh object")
    return -1;
    }

  if (this->Internals->Schema.ReadArray(this->GetCommunicator(),
    this->Internals->Stream, meshName, association, arrayName, mesh))
    {
    SENSEI_ERROR("Failed to read " << VTKUtils::GetAttributesName(association)
      << " data array \"" << arrayName << "\" from mesh \"" << meshName << "\"")
    return -1;
    }

  return 0;
}

//----------------------------------------------------------------------------
int ADIOS1DataAdaptor::ReleaseData()
{
  timer::MarkEvent mark("ADIOS1DataAdaptor::ReleaseData");
  return 0;
}

//----------------------------------------------------------------------------
void ADIOS1DataAdaptor::PrintSelf(ostream& os, vtkIndent indent)
{
  this->sensei::DataAdaptor::PrintSelf(os, indent);
}

}
