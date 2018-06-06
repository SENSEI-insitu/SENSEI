#include "ADIOSDataAdaptor.h"

#include "Error.h"
#include "Timer.h"
#include "ADIOSSchema.h"
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
// associate the mesh name to a data object and mesh metadata
using vtkDataObjectPtr = vtkSmartPointer<vtkDataObject>;
using ObjectType = std::pair<vtkDataObjectPtr, MeshMetadata>;
using ObjectMapType = std::map<std::string, ObjectType>;
using ObjectMapIterType = ObjectMapType::iterator;

// helpers to keep things legible accessing pairs of pairs
static void setObject(ObjectType &ot, vtkDataObject *dobj){ ot.first.TakeReference(dobj); }
static vtkDataObject *getObject(ObjectType &ot) { return ot.first.GetPointer(); }
static vtkDataObjectPtr &getObjectPtr(ObjectType &ot) { return ot.first; }
static MeshMetadata &getMetadata(ObjectType &ot){ return ot.second; }

static const std::string &getObjectName(ObjectMapIterType &it) { return it->first; }
static void setObject(ObjectMapIterType &it, vtkDataObject *dobj){ setObject(it->second, dobj); }
static vtkDataObject *getObject(ObjectMapIterType &it) { return getObject(it->second); }
static vtkDataObjectPtr &getObjectPtr(ObjectMapIterType &it) { return getObjectPtr(it->second); }
static MeshMetadata &getMetadata(ObjectMapIterType &it){ return getMetadata(it->second); }

static ObjectMapIterType find(ObjectMapType &objMap, const std::string &meshName)
{
  return objMap.find(meshName);
}

static bool good(ObjectMapType &objMap, const ObjectMapIterType &it)
{
  return objMap.end() != it;
}


struct ADIOSDataAdaptor::InternalsType
{
  InternalsType() : Stream() {}

  senseiADIOS::InputStream Stream;
  senseiADIOS::DataObjectCollectionSchema Schema;
  ObjectMapType ObjectMap;
};


//----------------------------------------------------------------------------
senseiNewMacro(ADIOSDataAdaptor);

//----------------------------------------------------------------------------
ADIOSDataAdaptor::ADIOSDataAdaptor() : Internals(nullptr)
{
  this->Internals = new InternalsType;
}

//----------------------------------------------------------------------------
ADIOSDataAdaptor::~ADIOSDataAdaptor()
{
  delete this->Internals;
}

//----------------------------------------------------------------------------
void ADIOSDataAdaptor::EnableDynamicMesh(const std::string &meshName, int val)
{
// TODO
   SENSEI_ERROR("TODO -- update to new metadata")
//  getMetadata(this->Internals->ObjectMap[meshName]).StaticMesh = val;
}

//----------------------------------------------------------------------------
int ADIOSDataAdaptor::Open(const std::string &method,
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
int ADIOSDataAdaptor::Open(ADIOS_READ_METHOD method, const std::string& fileName)
{
  timer::MarkEvent mark("ADIOSDataAdaptor::Open");

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
int ADIOSDataAdaptor::Close()
{
  timer::MarkEvent mark("ADIOSDataAdaptor::Close");

  this->Internals->ObjectMap.clear();
  this->Internals->Stream.Close();

  return 0;
}

//----------------------------------------------------------------------------
int ADIOSDataAdaptor::Advance()
{
  timer::MarkEvent mark("ADIOSDataAdaptor::Advance");

  if (this->Internals->Stream.AdvanceTimeStep())
    return -1;

  if (this->UpdateTimeStep())
    return -1;

  return 0;
}

//----------------------------------------------------------------------------
int ADIOSDataAdaptor::UpdateTimeStep()
{
  timer::MarkEvent mark("ADIOSDataAdaptor::UpdateTimeStep");

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

  // update the available meshes
  std::vector<std::string> names;
  if (this->Internals->Schema.ReadObjectNames(this->GetCommunicator(),
    this->Internals->Stream, names))
    {
    SENSEI_ERROR("Failed to update object names")
    return -1;
    }

   SENSEI_ERROR("TODO -- update to new metadata")
// TODO
/*
  // try to add new object, if one exists nothing changes, preserving metadata
  // this is necessary since static flag can be set before mesh names are read
  unsigned int nNames = names.size();
  for (unsigned int i = 0; i < nNames; ++i)
    {
    this->Internals->ObjectMap.insert(std::make_pair(names[i],
      ObjectType(vtkDataObjectPtr(), MeshMetadata(names[i]))));
    }
*/

  return 0;
}

//----------------------------------------------------------------------------
int ADIOSDataAdaptor::GetNumberOfMeshes(unsigned int &numMeshes)
{
  numMeshes = this->Internals->ObjectMap.size();
  return  0;
}

//----------------------------------------------------------------------------
int ADIOSDataAdaptor::GetMeshMetadata(unsigned int id, MeshMetadataPtr &metadata)
{
   SENSEI_ERROR("TODO -- update to new metadata")
// TODO
/*
  meshName = "";

  if (id >= this->Internals->ObjectMap.size())
    {
    SENSEI_ERROR("Mesh name " << id << " out of bounds. "
      << " only " << this->Internals->ObjectMap.size()
      << " mesh names available")
    return -1;
    }

  ObjectMapIterType it = this->Internals->ObjectMap.begin();

  for (unsigned int i = 0; i < id; ++i)
    ++it;

  meshName = getObjectName(it);
*/

  return 0;
}

//----------------------------------------------------------------------------
int ADIOSDataAdaptor::GetMesh(const std::string &meshName,
   bool structureOnly, vtkDataObject *&mesh)
{
  timer::MarkEvent mark("ADIOSDataAdaptor::GetMesh");

  mesh = nullptr;

  ObjectMapIterType it = find(this->Internals->ObjectMap, meshName);
  if (!good(this->Internals->ObjectMap, it))
    {
    SENSEI_ERROR("No mesh named \"" << meshName << "\"")
    return -1;
    }

  // if we already read, and we have a static mesh, and requested structure matches
  // what we have we can return cached mesh.
  vtkDataObjectPtr &pmesh = getObjectPtr(it);
  MeshMetadata &metadata = getMetadata(it);

// TODO
   SENSEI_ERROR("TODO -- update to new metadata")
/*
  if (pmesh && metadata.StaticMesh && (structureOnly == metadata.StructureOnly))
    {
    mesh = pmesh.GetPointer();
    return 0;
    }
*/

  // other wise we need to read the mesh at the current time step
  if (this->Internals->Schema.ReadObject(this->GetCommunicator(),
    this->Internals->Stream, meshName, mesh, structureOnly))
    {
    SENSEI_ERROR("Failed to read mesh \"" << meshName << "\"")
    return -1;
    }

  // cache the mesh
  pmesh = mesh;

  // get the ghost layer metadata
  int nGhostCellLayers = 0;
  int nGhostNodeLayers = 0;
  if (VTKUtils::GetGhostLayerMetadata(mesh, nGhostCellLayers, nGhostNodeLayers))
    {
    SENSEI_ERROR("missing ghost layer meta data")
    return -1;
    }

  // discover the available array names
  // cell data arrays
  std::set<std::string> cellArrays;

  if (this->Internals->Schema.ReadArrayNames(this->GetCommunicator(),
    this->Internals->Stream, meshName, mesh, vtkDataObject::CELL, cellArrays))
    {
    SENSEI_ERROR("Failed to read cell associated array names")
    return -1;
    }

  // point data arrays
  std::set<std::string> pointArrays;
  if (this->Internals->Schema.ReadArrayNames(this->GetCommunicator(),
    this->Internals->Stream, meshName, mesh, vtkDataObject::POINT, pointArrays))
    {
    SENSEI_ERROR("Failed to read point associated array names")
    return -1;
    }

// TODO
   SENSEI_ERROR("TODO -- update to new metadata")
/*
  // update the metadata
  metadata.MeshName = meshName;
  metadata.StructureOnly = structureOnly;
  metadata.NumberOfGhostCellLayers = nGhostCellLayers;
  metadata.NumberOfGhostNodeLayers = nGhostNodeLayers;
  metadata.SetArrayNames(vtkDataObject::CELL, cellArrays);
  metadata.SetArrayNames(vtkDataObject::POINT, pointArrays);
*/
  return 0;
}
// TODO
/*
//----------------------------------------------------------------------------
int ADIOSDataAdaptor::GetMeshHasGhostNodes(const std::string &meshName,
    int &nGhostNodeLayers)
{
  // look in cache, at the very least mesh names should exist
  ObjectMapIterType it = find(this->Internals->ObjectMap, meshName);
  if (!good(this->Internals->ObjectMap, it))
    {
    SENSEI_ERROR("No mesh named \"" << meshName << "\"")
    return -1;
    }

  // check if mesh and hence metadata exisst
  vtkDataObjectPtr &pmesh = getObjectPtr(it);
  if (!pmesh)
    {
    // initialize mesh and metadata
    vtkDataObject *dobj = nullptr;
    if (this->GetMesh(meshName, 0, dobj))
      {
      SENSEI_ERROR("Failed to get mesh \"" << meshName << "\"")
      return -1;
      }
    }

  MeshMetadata &metadata = getMetadata(it);
  nGhostNodeLayers = metadata.NumberOfGhostNodeLayers;

  return 0;
}

//----------------------------------------------------------------------------
int ADIOSDataAdaptor::GetMeshHasGhostCells(const std::string &meshName,
    int &nGhostCellLayers)
{
  // look in cache, at the very least mesh names should exist
  ObjectMapIterType it = find(this->Internals->ObjectMap, meshName);
  if (!good(this->Internals->ObjectMap, it))
    {
    SENSEI_ERROR("No mesh named \"" << meshName << "\"")
    return -1;
    }

  // check if mesh and hence metadata exist
  vtkDataObjectPtr &pmesh = getObjectPtr(it);
  if (!pmesh)
    {
    // initialize mesh and metadata
    vtkDataObject *dobj = nullptr;
    if (this->GetMesh(meshName, 0, dobj))
      {
      SENSEI_ERROR("Failed to get mesh \"" << meshName << "\"")
      return -1;
      }
    }

  MeshMetadata &metadata = getMetadata(it);
  nGhostCellLayers = metadata.NumberOfGhostCellLayers;

  return 0;
}
*/

//----------------------------------------------------------------------------
int ADIOSDataAdaptor::AddGhostNodesArray(vtkDataObject *mesh, const std::string &meshName)
{
  return AddArray(mesh, meshName, vtkDataObject::POINT, "vtkGhostType");
}

//----------------------------------------------------------------------------
int ADIOSDataAdaptor::AddGhostCellsArray(vtkDataObject *mesh, const std::string &meshName)
{
  return AddArray(mesh, meshName, vtkDataObject::CELL, "vtkGhostType");
}

//----------------------------------------------------------------------------
int ADIOSDataAdaptor::AddArray(vtkDataObject* mesh,
  const std::string &meshName, int association, const std::string& arrayName)
{
  timer::MarkEvent mark("ADIOSDataAdaptor::AddArray");

  // the mesh should never be null. there must have been an error
  // upstream.
  if (!mesh)
    {
    SENSEI_ERROR("Invalid mesh object")
    return -1;
    }

  if (this->Internals->Schema.ReadArray(this->GetCommunicator(),
    this->Internals->Stream, meshName, mesh, association, arrayName))
    {
    SENSEI_ERROR("Failed to read " << VTKUtils::GetAttributesName(association)
      << " data array \"" << arrayName << "\" from mesh \"" << meshName << "\"")
    return -1;
    }

  return 0;
}

// TODO
/*
//----------------------------------------------------------------------------
int ADIOSDataAdaptor::GetNumberOfArrays(const std::string &meshName,
  int association, unsigned int &numberOfArrays)
{
  numberOfArrays = 0;

  // look in cache, at the very least mesh names should exist
  ObjectMapIterType it = find(this->Internals->ObjectMap, meshName);
  if (!good(this->Internals->ObjectMap, it))
    {
    SENSEI_ERROR("No mesh named \"" << meshName << "\"")
    return -1;
    }

  // check if mesh and hence metadata exist
  vtkDataObjectPtr &pmesh = getObjectPtr(it);
  if (!pmesh)
    {
    // initialize mesh and metadata
    vtkDataObject *dobj = nullptr;
    if (this->GetMesh(meshName, 0, dobj))
      {
      SENSEI_ERROR("Failed to get mesh \"" << meshName << "\"")
      return -1;
      }
    }

  // check valid association
  if ((association != vtkDataObject::POINT) && (association != vtkDataObject::CELL))
    {
    SENSEI_ERROR("Invalid association " << association)
    return -1;
    }

  numberOfArrays = getMetadata(it).GetArrayNames(association).size();

  return 0;
}

//----------------------------------------------------------------------------
int ADIOSDataAdaptor::GetArrayName(const std::string &meshName,
  int association, unsigned int index, std::string &arrayName)
{
  // look in cache, at the very least mesh names should exist
  ObjectMapIterType it = find(this->Internals->ObjectMap, meshName);
  if (!good(this->Internals->ObjectMap, it))
    {
    SENSEI_ERROR("No mesh named \"" << meshName << "\"")
    return -1;
    }

  // check if mesh and hence metadata exist
  vtkDataObjectPtr &pmesh = getObjectPtr(it);
  if (!pmesh)
    {
    // initialize mesh and metadata
    vtkDataObject *dobj = nullptr;
    if (this->GetMesh(meshName, 0, dobj))
      {
      SENSEI_ERROR("Failed to get mesh \"" << meshName << "\"")
      return -1;
      }
    }

  MeshMetadata &metadata = getMetadata(it);

  if ((association != vtkDataObject::POINT) && (association != vtkDataObject::CELL))
    {
    SENSEI_ERROR("Invalid association " << association)
    return -1;
    }

  std::vector<std::string> &arrayNames = metadata.GetArrayNames(association);
  if (index >= arrayNames.size())
    {
    SENSEI_ERROR(<< VTKUtils::GetAttributesName(association)
      << " data array index " << index << " is out of bounds. "
      << arrayNames.size() << " available")
    return -1;
    }

  arrayName = arrayNames[index];

  return 0;
}
*/

//----------------------------------------------------------------------------
int ADIOSDataAdaptor::ReleaseData()
{
  timer::MarkEvent mark("ADIOSDataAdaptor::ReleaseData");

  ObjectMapType::iterator it = this->Internals->ObjectMap.begin();
  ObjectMapType::iterator end = this->Internals->ObjectMap.end();
  for (; it != end; ++it)
    {
    // keep the static meshes, release their data arrays
    MeshMetadata &md = getMetadata(it);
    if (md.StaticMesh)
      this->ReleaseAttributeData(getObject(it));
    else
      setObject(it, nullptr);
    }

  return 0;
}

//----------------------------------------------------------------------------
void ADIOSDataAdaptor::ReleaseAttributeData(vtkDataObject* dobj)
{
  if (vtkCompositeDataSet* cd = vtkCompositeDataSet::SafeDownCast(dobj))
    {
    vtkCompositeDataIterator *iter = cd->NewIterator();
    for (iter->InitTraversal(); !iter->IsDoneWithTraversal(); iter->GoToNextItem())
      {
      this->ReleaseAttributeData(iter->GetCurrentDataObject());
      }
    iter->Delete();
    }
  else if (vtkDataSet *ds = dynamic_cast<vtkDataSet*>(dobj))
    {
    ds->GetAttributes(vtkDataObject::CELL)->Initialize();
    ds->GetAttributes(vtkDataObject::POINT)->Initialize();
    }
}

//----------------------------------------------------------------------------
void ADIOSDataAdaptor::PrintSelf(ostream& os, vtkIndent indent)
{
  this->sensei::DataAdaptor::PrintSelf(os, indent);
}

}
