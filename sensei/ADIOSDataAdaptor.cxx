#include "ADIOSDataAdaptor.h"

#include "Error.h"
#include "Timer.h"
#include "ADIOSSchema.h"

#include <vtkCompositeDataIterator.h>
#include <vtkDataSetAttributes.h>
#include <vtkInformation.h>
#include <vtkMultiBlockDataSet.h>
#include <vtkObjectFactory.h>
#include <vtkSmartPointer.h>
#include <vtkDataSet.h>

#include <sstream>

using vtkDataObjectPtr = vtkSmartPointer<vtkDataObject>;
using ArrayMapType = std::map<int,std::vector<std::string>>;

namespace sensei
{
struct ADIOSDataAdaptor::InternalsType
{
  InternalsType() : Comm(MPI_COMM_WORLD), File(nullptr),
    Mesh(nullptr), StaticMesh(0), StructureOnly(0) {}

  MPI_Comm Comm;
  ADIOS_FILE *File;
  senseiADIOS::DataObjectSchema Schema;
  vtkDataObjectPtr Mesh;
  ArrayMapType ArrayNames;
  int StaticMesh;
  int StructureOnly;
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
void ADIOSDataAdaptor::EnableDynamicMesh(int val)
{
  this->Internals->StaticMesh = val ? 0 : 1;
}

//----------------------------------------------------------------------------
int ADIOSDataAdaptor::Open(MPI_Comm comm,
  ADIOS_READ_METHOD method, const std::string& filename)
{
  timer::MarkEvent mark("ADIOSDataAdaptor::Open");

  // initialize adios
  adios_read_init_method(method, comm, "verbose=2");

  // open the file
  ADIOS_FILE *fp = adios_read_open(filename.c_str(),
    method, comm, ADIOS_LOCKMODE_ALL, -1);

  if (!fp)
    {
    SENSEI_ERROR("failed to open " << filename)
    return -1;
    }

  // verify that it is one of ours
  if (this->Internals->Schema.CanRead(fp))
    {
    adios_read_close(fp);
    SENSEI_ERROR("Failed to open \"" << filename << "\". Stream "
      "was not written in the SENSEI ADIOS schema format")
    return -1;
    }

  //
  this->Internals->Comm = comm;
  this->Internals->File = fp;

  // initialize the time step
  if (this->UpdateTimeStep())
    return -1;

  return 0;
}

//----------------------------------------------------------------------------
int ADIOSDataAdaptor::Close()
{
  timer::MarkEvent mark("ADIOSDataAdaptor::Close");

  this->Internals->Mesh = 0;
  this->Internals->ArrayNames[vtkDataObject::POINT].clear();
  this->Internals->ArrayNames[vtkDataObject::CELL].clear();

  if (this->Internals->File)
    adios_read_close(this->Internals->File);

  this->Internals->File = nullptr;

  return 0;
}

//----------------------------------------------------------------------------
int ADIOSDataAdaptor::Advance()
{
  timer::MarkEvent mark("ADIOSDataAdaptor::Advance");

  adios_release_step(this->Internals->File);

  if (adios_advance_step(this->Internals->File, 0, /*timeout*/0.0))
    {
    /* TODO -- according to adios documentation adios_errno should be
    // err_end_of_stream when we have processed all time steps
    // however it is err_step_notready.
    if (adios_errno != err_end_of_stream)
      {
      SENSEI_ERROR("Failed to advance to next time step")
      return -1;
      }
    else
      return 1;*/
    return 1;
    }

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

  if (this->Internals->Schema.ReadTimeStep(this->Internals->File,
    timeStep, time))
    {
    SENSEI_ERROR("Failed to update time step")
    return -1;
    }

  this->SetDataTimeStep(timeStep);
  this->SetDataTime(time);

  return 0;
}

//----------------------------------------------------------------------------
vtkDataObject* ADIOSDataAdaptor::GetMesh(bool structure_only)
{
  timer::MarkEvent mark("ADIOSDataAdaptor::GetMesh");

  // if the mesh is static, we've already read it in, and it
  // has the desired structure  then we can return the cached
  // instance
  if (this->Internals->StaticMesh && this->Internals->Mesh.Get() &&
    (structure_only == this->Internals->StructureOnly))
    return this->Internals->Mesh.Get();

  // get the mesh at the current time step
  vtkDataObject *mesh = nullptr;
  if (this->Internals->Schema.ReadMesh(this->Internals->Comm,
    this->Internals->File, structure_only, mesh))
    {
    SENSEI_ERROR("Failed to read mesh")
    return nullptr;
    }

  // cache the mesh, making a note of the structure
  this->Internals->Mesh.TakeReference(mesh);
  this->Internals->StructureOnly = structure_only;

  // discover the available array names
  // point data arrays
  std::set<std::string> point_arrays;
  if (this->Internals->Schema.ReadArrayNames(this->Internals->Comm,
    this->Internals->File, mesh, vtkDataObject::POINT, point_arrays))
    {
    SENSEI_ERROR("Failed to read point associated array names")
    return nullptr;
    }
  this->Internals->ArrayNames[vtkDataObject::POINT].assign(
    point_arrays.begin(), point_arrays.end());

  // cell data arrays
  std::set<std::string> cell_arrays;
  if (this->Internals->Schema.ReadArrayNames(this->Internals->Comm,
    this->Internals->File, mesh, vtkDataObject::CELL, cell_arrays))
    {
    SENSEI_ERROR("Failed to read cell associated array names")
    return nullptr;
    }
  this->Internals->ArrayNames[vtkDataObject::CELL].assign(
    cell_arrays.begin(), cell_arrays.end());

  return mesh;
}

//----------------------------------------------------------------------------
std::string ADIOSDataAdaptor::GetAvailableArrays(int association)
{

  // if the mesh has not been initialized then do so now
  if (!this->Internals->Mesh && !this->GetMesh(false))
    return "";

  std::ostringstream arrays;
  int nArrays = this->GetNumberOfArrays(association);
  if (nArrays)
    {
    arrays << "\"" << this->GetArrayName(association, 0) << "\"";
    for (int i = 1; i < nArrays; ++i)
      arrays << ", \"" << this->GetArrayName(association, i) << "\"";
    }
  else
    arrays << "{}";
  return arrays.str();
}

//----------------------------------------------------------------------------
bool ADIOSDataAdaptor::AddArray(vtkDataObject* mesh, int association,
   const std::string& name)
{
  timer::MarkEvent mark("ADIOSDataAdaptor::AddArray");

  // the mesh should never be null. there must have been an error
  // upstream.
  if (!mesh)
    {
    SENSEI_ERROR("Invalid mesh object")
    return false;
    }

  if (this->Internals->Schema.ReadArray(this->Internals->Comm,
    this->Internals->File, mesh, association, name))
    {
    SENSEI_ERROR("Failed to read "
      << (association == vtkDataObject::POINT ? "point" : "cell")
      << " data array \"" << name << "\". Point data has: "
      << this->GetAvailableArrays(vtkDataObject::POINT) << ". Cell data has: "
      << this->GetAvailableArrays(vtkDataObject::CELL) << ".")
    return false;
    }

  return true;
}

//----------------------------------------------------------------------------
unsigned int ADIOSDataAdaptor::GetNumberOfArrays(int association)
{

  // if the mesh has not been initialized then do so now
  if (!this->Internals->Mesh && !this->GetMesh(false))
    return 0;

  return this->Internals->ArrayNames[association].size();
}

//----------------------------------------------------------------------------
std::string ADIOSDataAdaptor::GetArrayName(int association, unsigned int index)
{

  // if the mesh has not been initialized then do so now
  if (!this->Internals->Mesh && !this->GetMesh(false))
    return "";

  // bounds check
  size_t n_arrays = this->Internals->ArrayNames[association].size();
  if (index >= n_arrays)
    {
    const char *assocStr = (association == vtkDataObject::POINT ?
      "point" : "cell");

    SENSEI_ERROR(<< assocStr << "array index " << index
      << " is out of bounds [0 - " << n_arrays << ")")
    return "";
    }

  return this->Internals->ArrayNames[association][index];
}

//----------------------------------------------------------------------------
void ADIOSDataAdaptor::ReleaseData()
{
  timer::MarkEvent mark("ADIOSDataAdaptor::ReleaseData");
  if (this->Internals->StaticMesh)
    {
    // only release the data arrays, keep the mesh object around
    this->ReleaseAttributeData(this->Internals->Mesh);
    }
  else
    {
    // nuke it all
    this->Internals->Mesh = 0;
    this->Internals->ArrayNames[vtkDataObject::POINT].clear();
    this->Internals->ArrayNames[vtkDataObject::CELL].clear();
    }
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
  this->Superclass::PrintSelf(os, indent);
}

}
