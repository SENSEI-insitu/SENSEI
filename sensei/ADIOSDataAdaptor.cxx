#include "ADIOSDataAdaptor.h"

#include "Error.h"
#include "Timer.h"
#include "ADIOSSchema.h"
#include "VTKUtils.h"

#include <vtkCompositeDataIterator.h>
#include <vtkDataSetAttributes.h>
#include <vtkInformation.h>
#include <vtkMultiBlockDataSet.h>
#include <vtkObjectFactory.h>
#include <vtkSmartPointer.h>
#include <vtkDataSet.h>

#include <sstream>

using vtkDataObjectPtr = vtkSmartPointer<vtkDataObject>;

// associates an object and flag indicating if it is static
// to a name
using ObjectMapType = std::map<std::string, std::pair<vtkDataObjectPtr, int>>;



namespace sensei
{

struct ADIOSDataAdaptor::InternalsType
{
  InternalsType() : Comm(MPI_COMM_WORLD), Stream() {}


  ObjectMapType::iterator FindObject(const std::string &name)
  {
    return this->ObjectMap.find(name);
  }

  vtkDataObject *GetObject(ObjectMapType::iterator &it)
  {
    return it->second.first.GetPointer();
  }

  const vtkDataObject *GetObject(ObjectMapType::const_iterator &it)
  {
    return it->second.first.GetPointer();
  }

  int GetStatic(ObjectMapType::iterator &it)
  {
    return it->second.second;
  }

  int GetStatic(ObjectMapType::const_iterator &it)
  {
    return it->second.second;
  }

  MPI_Comm Comm;
  senseiADIOS::InputStream Stream;
  senseiADIOS::DataObjectCollectionSchema Schema;
  std::set<std::string> ObjectNames;
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
  this->Internals->ObjectMap[meshName].second = val;
}

//----------------------------------------------------------------------------
int ADIOSDataAdaptor::Open(MPI_Comm comm,
  const std::string &method, const std::string& filename)
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

  return this->Open(comm, it->second, filename);
}

//----------------------------------------------------------------------------
int ADIOSDataAdaptor::Open(MPI_Comm comm,
  ADIOS_READ_METHOD method, const std::string& fileName)
{
  timer::MarkEvent mark("ADIOSDataAdaptor::Open");

  this->Internals->Comm = comm;

  if (this->Internals->Stream.Open(comm, method, fileName))
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
  this->Internals->ObjectNames.clear();
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

  if (this->Internals->Schema.ReadTimeStep(this->Internals->Comm,
    this->Internals->Stream, timeStep, time))
    {
    SENSEI_ERROR("Failed to update time step")
    return -1;
    }

  this->SetDataTimeStep(timeStep);
  this->SetDataTime(time);

  // update the available meshes
  this->Internals->ObjectNames.clear();

  std::vector<std::string> names;
  if (this->Internals->Schema.ReadObjectNames(this->Internals->Comm,
    this->Internals->Stream, names))
    {
    SENSEI_ERROR("Failed to update object names")
    return -1;
    }

  unsigned int nNames = names.size();
  for (unsigned int i = 0; i < nNames; ++i)
    this->Internals->ObjectNames.insert(names[i]);

  return 0;
}

//----------------------------------------------------------------------------
int ADIOSDataAdaptor::GetNumberOfMeshes(unsigned int &numMeshes)
{
  numMeshes = this->Internals->ObjectNames.size();
  return  0;
}

//----------------------------------------------------------------------------
int ADIOSDataAdaptor::GetMeshName(unsigned int id, std::string &meshName)
{
  meshName = "";

  if (id >= this->Internals->ObjectNames.size())
    {
    SENSEI_ERROR("Mesh name " << id << " out of bounds. "
      << " only " << this->Internals->ObjectNames.size()
      << " mesh names available")
    return -1;
    }

  std::set<std::string>::iterator it = this->Internals->ObjectNames.begin();

  for (unsigned int i = 0; i < id; ++i)
    ++it;

  meshName = *it;

  return 0;
}

//----------------------------------------------------------------------------
int ADIOSDataAdaptor::GetMesh(const std::string &meshName,
   bool structure_only, vtkDataObject *&mesh)
{
  timer::MarkEvent mark("ADIOSDataAdaptor::GetMesh");

  mesh = nullptr;
  int staticMesh = 1;

  ObjectMapType::iterator it = this->Internals->FindObject(meshName);
  if (it != this->Internals->ObjectMap.end())
    {
    mesh = this->Internals->GetObject(it);
    staticMesh = this->Internals->GetStatic(it);

    // TODO -- handle structure_only changed
    // static mesh and we have already read it
    if (mesh && staticMesh)
      return 0;
    }

  // get the mesh at the current time step
  if (this->Internals->Schema.ReadObject(this->Internals->Comm,
    this->Internals->Stream, meshName, mesh, structure_only))
    {
    SENSEI_ERROR("Failed to read mesh \"" << meshName << "\"")
    return -1;
    }

  // cache the mesh, TODO - record structure_only
  vtkDataObjectPtr ptr;
  ptr.TakeReference(mesh);

  this->Internals->ObjectMap[meshName] = std::make_pair(ptr, staticMesh);

/* TODO -- is caching array names worth complexity it will add??
  // discover the available array names
  // point data arrays
  std::set<std::string> point_arrays;
  if (this->Internals->Schema.ReadArrayNames(this->Internals->Comm,
    this->Internals->Stream, mesh, vtkDataObject::POINT, point_arrays))
    {
    SENSEI_ERROR("Failed to read point associated array names")
    return -1;
    }
  this->Internals->ArrayNames[vtkDataObject::POINT].assign(
    point_arrays.begin(), point_arrays.end());

  // cell data arrays
  std::set<std::string> cell_arrays;
  if (this->Internals->Schema.ReadArrayNames(this->Internals->Comm,
    this->Internals->Stream, mesh, vtkDataObject::CELL, cell_arrays))
    {
    SENSEI_ERROR("Failed to read cell associated array names")
    return -1;
    }
  this->Internals->ArrayNames[vtkDataObject::CELL].assign(
    cell_arrays.begin(), cell_arrays.end());
*/

  return 0;
}

/*
//----------------------------------------------------------------------------
std::string ADIOSDataAdaptor::GetAvailableArrays(const std::string &meshName,
  int association)
{
  // TODO -- handle meshName

  // if the mesh has not been initialized then do so now
  vtkDataObject *mesh = nullptr;
  if (!this->Internals->Mesh && this->GetMesh(meshName, false, mesh))
    return "";

  std::ostringstream arrays;
  unsigned int nArrays = 0;
  if (!this->GetNumberOfArrays(meshName, association, nArrays))
    {
    std::string arrayName;
    this->GetArrayName(meshName, association, 0, arrayName);
    arrays << "\"" << arrayName << "\"";
    for (unsigned int i = 1; i < nArrays; ++i)
      {
      arrayName.clear();
      this->GetArrayName(meshName, association, i, arrayName);
      arrays << "\"" << arrayName << "\"";
      }
    }
  else
    {
    arrays << "{}";
    }

  return arrays.str();
}
*/

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

  if (this->Internals->Schema.ReadArray(this->Internals->Comm,
    this->Internals->Stream, meshName, mesh, association, arrayName))
    {
    SENSEI_ERROR("Failed to read " << VTKUtils::GetAttributesName(association)
      << " data array \"" << arrayName << "\" from mesh \"" << meshName << "\"")
    return -1;
    }

  return 0;
}

//----------------------------------------------------------------------------
int ADIOSDataAdaptor::GetNumberOfArrays(const std::string &meshName,
  int association, unsigned int &numberOfArrays)
{
  numberOfArrays = 0;

  vtkDataObject *mesh;
  ObjectMapType::iterator it = this->Internals->FindObject(meshName);
  if (it == this->Internals->ObjectMap.end())
    {
    // this mesh has not been initialized do it now
    if (this->GetMesh(meshName, false, mesh))
      {
      SENSEI_ERROR("Failed to get mesh \"" << meshName << "\"")
      return -1;
      }
    }
  else
    {
    mesh = this->Internals->GetObject(it);
    }

  std::set<std::string> arrayNames;
  if (this->Internals->Schema.ReadArrayNames(this->Internals->Comm,
    this->Internals->Stream, meshName, mesh, association, arrayNames))
    {
    SENSEI_ERROR("Failed to read array names on mesh \"" << meshName << "\"")
    return -1;
    }

  numberOfArrays = arrayNames.size();

  return 0;
}

//----------------------------------------------------------------------------
int ADIOSDataAdaptor::GetArrayName(const std::string &meshName,
  int association, unsigned int index, std::string &arrayName)
{
  vtkDataObject *mesh;
  ObjectMapType::iterator it = this->Internals->FindObject(meshName);
  if (it == this->Internals->ObjectMap.end())
    {
    // this mesh has not been initialized do it now
    if (this->GetMesh(meshName, false, mesh))
      {
      SENSEI_ERROR("Failed to get mesh \"" << meshName << "\"")
      return -1;
      }
    }
  else
    {
    mesh = this->Internals->GetObject(it);
    }

  std::set<std::string> arrayNames;
  if (this->Internals->Schema.ReadArrayNames(this->Internals->Comm,
    this->Internals->Stream, meshName, mesh, association, arrayNames))
    {
    SENSEI_ERROR("Failed to read array names on mesh \"" << meshName << "\"")
    return -1;
    }

  // bounds check
  if (index >= arrayNames.size())
    {
    SENSEI_ERROR(<< VTKUtils::GetAttributesName(association)
      << " data array index " << index << " is out of bounds. "
      << arrayNames.size() << " available")
    return -1;
    }

  std::set<std::string>::iterator nit = arrayNames.begin();
  for (unsigned int i =0; i < index; ++i)
    ++nit;

  arrayName = *nit;

  return 0;
}

//----------------------------------------------------------------------------
int ADIOSDataAdaptor::ReleaseData()
{
  timer::MarkEvent mark("ADIOSDataAdaptor::ReleaseData");

  ObjectMapType::iterator it = this->Internals->ObjectMap.begin();
  ObjectMapType::iterator end = this->Internals->ObjectMap.end();
  for (; it != end; ++it)
    {
    // keep the static meshes, release their data arrays
    if (this->Internals->GetStatic(it))
      this->ReleaseAttributeData(this->Internals->GetObject(it));
    else
      it->second.first = nullptr;
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
