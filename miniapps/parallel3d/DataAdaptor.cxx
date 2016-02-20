#include "DataAdaptor.h"

#include "vtkCellData.h"
#include "vtkDataObject.h"
#include "vtkDoubleArray.h"
#include "vtkImageData.h"
#include "vtkInformation.h"
#include "vtkObjectFactory.h"

namespace parallel3d
{
vtkStandardNewMacro(DataAdaptor);
//-----------------------------------------------------------------------------
DataAdaptor::DataAdaptor()
{
}

//-----------------------------------------------------------------------------
DataAdaptor::~DataAdaptor()
{
}

//-----------------------------------------------------------------------------
void DataAdaptor::Initialize(
  int g_x, int g_y, int g_z,
  int l_x, int l_y, int l_z,
  uint64_t start_extents_x, uint64_t start_extents_y, uint64_t start_extents_z,
  int tot_blocks_x, int tot_blocks_y, int tot_blocks_z,
  int block_id_x, int block_id_y, int block_id_z)
{
  // we only really need to save the local extents for our current example. So
  // we'll just save that.
  this->CellExtent[0] = start_extents_x;
  this->CellExtent[1] = start_extents_x + l_x - 1;
  this->CellExtent[2] = start_extents_y;
  this->CellExtent[3] = start_extents_y + l_y - 1;
  this->CellExtent[4] = start_extents_z;
  this->CellExtent[5] = start_extents_z + l_z - 1;

  // This is point-based.
  this->WholeExtent[0] = 0;
  this->WholeExtent[1] = g_x;
  this->WholeExtent[2] = 0;
  this->WholeExtent[3] = g_y;
  this->WholeExtent[4] = 0;
  this->WholeExtent[5] = g_z;
  this->GetInformation()->Set(vtkDataObject::DATA_EXTENT(),
    this->WholeExtent, 6);
}

//-----------------------------------------------------------------------------
void DataAdaptor::AddArray(const std::string& name, double* data)
{
  if (this->Variables[name] != data)
    {
    this->Variables[name] = data;
    this->Arrays.erase(name);
    }
}

//-----------------------------------------------------------------------------
void DataAdaptor::ClearArrays()
{
  this->Variables.clear();
  this->Arrays.clear();
}

//-----------------------------------------------------------------------------
vtkDataObject* DataAdaptor::GetMesh(bool vtkNotUsed(structure_only))
{
  if (!this->Mesh)
    {
    this->Mesh = vtkSmartPointer<vtkImageData>::New();
    this->Mesh->SetExtent(
      this->CellExtent[0], this->CellExtent[1] + 1,
      this->CellExtent[2], this->CellExtent[3] + 1,
      this->CellExtent[4], this->CellExtent[5] + 1);
    }
  return this->Mesh;
}

//-----------------------------------------------------------------------------
bool DataAdaptor::AddArray(vtkDataObject* mesh, int association, const char* name)
{
  if (association != vtkDataObject::FIELD_ASSOCIATION_CELLS || name == NULL)
    {
    return false;
    }

  VariablesType::iterator iterV = this->Variables.find(name);
  if (iterV == this->Variables.end())
    {
    return false;
    }

  vtkImageData* image = vtkImageData::SafeDownCast(mesh);
  assert(image != NULL);

  ArraysType::iterator iterA = this->Arrays.find(iterV->first);
  if (iterA == this->Arrays.end())
    {
    vtkSmartPointer<vtkDoubleArray>& vtkarray = this->Arrays[iterV->first];
    vtkarray = vtkSmartPointer<vtkDoubleArray>::New();
    vtkarray->SetName(name);
    const vtkIdType size = (this->CellExtent[1] - this->CellExtent[0] + 1) *
      (this->CellExtent[3] - this->CellExtent[2] + 1) *
      (this->CellExtent[5] - this->CellExtent[4] + 1);
    assert(size == image->GetNumberOfCells());
    vtkarray->SetArray(iterV->second, size, 1);
    vtkImageData::SafeDownCast(mesh)->GetCellData()->SetScalars(vtkarray);
    return true;
    }
  return true;
}

//-----------------------------------------------------------------------------
unsigned int DataAdaptor::GetNumberOfArrays(int association)
{
  return (association == vtkDataObject::FIELD_ASSOCIATION_CELLS)?
    static_cast<unsigned int>(this->Variables.size()): 0;
}

//-----------------------------------------------------------------------------
const char* DataAdaptor::GetArrayName(int association, unsigned int index)
{
  if (association != vtkDataObject::FIELD_ASSOCIATION_CELLS)
    {
    return NULL;
    }
  unsigned int count = 0;
  for (VariablesType::iterator iter=this->Variables.begin(), max=this->Variables.end();
    iter != max; ++iter, ++count)
    {
    if (count==index)
      {
      return iter->first.c_str();
      }
    }
  return NULL;
}

//-----------------------------------------------------------------------------
void DataAdaptor::ReleaseData()
{
  this->ClearArrays();
  this->Mesh = NULL;
}

}
