#include "DataAdaptor.h"
#include "Error.h"

#include <vtkCellData.h>
#include <vtkDataObject.h>
#include <vtkDoubleArray.h>
#include <vtkImageData.h>
#include <vtkInformation.h>
#include <vtkObjectFactory.h>

#include <limits>

namespace parallel3d
{
//-----------------------------------------------------------------------------
senseiNewMacro(DataAdaptor);

//-----------------------------------------------------------------------------
DataAdaptor::DataAdaptor()
{
  this->CellExtent[0] = std::numeric_limits<int>::max();
  this->CellExtent[1] = std::numeric_limits<int>::min();
  this->CellExtent[2] = std::numeric_limits<int>::max();
  this->CellExtent[3] = std::numeric_limits<int>::min();
  this->CellExtent[4] = std::numeric_limits<int>::max();
  this->CellExtent[5] = std::numeric_limits<int>::min();

  this->WholeExtent[0] = std::numeric_limits<int>::max();
  this->WholeExtent[1] = std::numeric_limits<int>::min();
  this->WholeExtent[2] = std::numeric_limits<int>::max();
  this->WholeExtent[3] = std::numeric_limits<int>::min();
  this->WholeExtent[4] = std::numeric_limits<int>::max();
  this->WholeExtent[5] = std::numeric_limits<int>::min();
}

//-----------------------------------------------------------------------------
DataAdaptor::~DataAdaptor()
{
}

//-----------------------------------------------------------------------------
void DataAdaptor::Initialize(int g_x, int g_y, int g_z, int l_x, int l_y, int l_z,
  uint64_t start_extents_x, uint64_t start_extents_y, uint64_t start_extents_z,
  int tot_blocks_x, int tot_blocks_y, int tot_blocks_z, int block_id_x,
  int block_id_y, int block_id_z)
{
  (void)tot_blocks_x;
  (void)tot_blocks_y;
  (void)tot_blocks_z;
  (void)block_id_x;
  (void)block_id_y;
  (void)block_id_z;

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
int DataAdaptor::GetNumberOfArrays(const std::string &meshName,
  int association, unsigned int &numberOfArrays)
{
  numberOfArrays = 0;

  if (meshName != "mesh")
    {
    SENSEI_ERROR("No mesh \"" << meshName << "\"")
    return -1;
    }

  if (association == vtkDataObject::FIELD_ASSOCIATION_CELLS)
    {
    numberOfArrays = this->Variables.size();
    }

  return 0;
}

//-----------------------------------------------------------------------------
int DataAdaptor::GetArrayName(const std::string &meshName, int association,
  unsigned int index, std::string &arrayName)
{
  arrayName = "";

  if (meshName != "mesh")
    {
    SENSEI_ERROR("No mesh \"" << meshName << "\"")
    return -1;
    }

  if (association != vtkDataObject::FIELD_ASSOCIATION_CELLS)
    {
    SENSEI_ERROR("No point data on mesh")
    return -1;
    }

  if (index >= this->Variables.size())
    {
    SENSEI_ERROR("Index out of bounds")
    return -1;
    }

  unsigned int count = 0;
  VariablesType::iterator end = this->Variables.end();
  VariablesType::iterator iter = this->Variables.begin();
  for (; iter != end; ++iter, ++count)
    {
    if (count == index)
      {
      arrayName = iter->first;
      return 0;
      }
    }

  return 0;
}

//-----------------------------------------------------------------------------
int DataAdaptor::GetNumberOfMeshes(unsigned int &numMeshes)
{
  numMeshes = 1;
  return 0;
}

//-----------------------------------------------------------------------------
int DataAdaptor::GetMeshName(unsigned int id, std::string &meshName)
{
  if (id != 0)
    {
    SENSEI_ERROR("Mesh id is out of range. 1 mesh available.")
    return -1;
    }
  meshName = "mesh";
  return 0;
}
//-----------------------------------------------------------------------------
int DataAdaptor::GetMesh(const std::string &meshName, bool structureOnly,
  vtkDataObject *&mesh)
{
  (void)structureOnly;
  if (meshName != "mesh")
    {
    SENSEI_ERROR("No mesh \"" << meshName << "\"")
    return -1;
    }

  if (!this->Mesh)
    {
    this->Mesh = vtkSmartPointer<vtkImageData>::New();

    this->Mesh->SetExtent(
      this->CellExtent[0], this->CellExtent[1] + 1,
      this->CellExtent[2], this->CellExtent[3] + 1,
      this->CellExtent[4], this->CellExtent[5] + 1);
    }

  mesh = this->Mesh;
  return 0;
}

//-----------------------------------------------------------------------------
int DataAdaptor::AddArray(vtkDataObject* mesh, const std::string &meshName,
  int association, const std::string& arrayName)
{
  if (meshName != "mesh")
    {
    SENSEI_ERROR("No mesh \"" << meshName << "\"")
    return -1;
    }

  if (association != vtkDataObject::FIELD_ASSOCIATION_CELLS)
    {
    SENSEI_ERROR("No point data on mesh")
    return -1;
    }

  VariablesType::iterator iterV = this->Variables.find(arrayName);

  if (iterV == this->Variables.end())
    {
    SENSEI_ERROR("no array named \"" << arrayName << "\"")
    return -1;
    }

  vtkDoubleArrayPtr vtkarray = vtkDoubleArrayPtr::New();
  vtkarray->SetName(arrayName.c_str());

  const vtkIdType size =
    (this->CellExtent[1] - this->CellExtent[0] + 1) *
    (this->CellExtent[3] - this->CellExtent[2] + 1) *
    (this->CellExtent[5] - this->CellExtent[4] + 1);

  vtkarray->SetArray(iterV->second, size, 1);

  vtkImageData* image = vtkImageData::SafeDownCast(mesh);
  image->GetCellData()->AddArray(vtkarray);

  assert(image);
  assert(size == image->GetNumberOfCells());

  return 0;
}

//-----------------------------------------------------------------------------
int DataAdaptor::ReleaseData()
{
  this->ClearArrays();
  this->Mesh = NULL;
  return 0;
}

}
