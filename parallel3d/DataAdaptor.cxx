#include "DataAdaptor.h"

#include "vtkObjectFactory.h"
#include "vtkDataObject.h"
#include "vtkDoubleArray.h"

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
  this->Extent[0] = start_extents_x;
  this->Extent[1] = start_extents_x + l_x - 1;
  this->Extent[2] = start_extents_y;
  this->Extent[3] = start_extents_y + l_y - 1;
  this->Extent[4] = start_extents_z;
  this->Extent[5] = start_extents_z + l_z - 1;
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
vtkDataObject* DataAdaptor::GetMesh()
{
  // our analysis doesn't need a mesh so we punt on it for now.
  // In theory, we'll create a new vtkImageData and return that.
  vtkGenericWarningMacro("TODO: Not implemented currently.");
  return NULL;
}

//-----------------------------------------------------------------------------
vtkAbstractArray* DataAdaptor::GetArray(int association, const char* name)
{
  if (association != vtkDataObject::FIELD_ASSOCIATION_POINTS || name == NULL)
    {
    return NULL;
    }

  VariablesType::iterator iterV = this->Variables.find(name);
  if (iterV == this->Variables.end())
    {
    return NULL;
    }

  ArraysType::iterator iterA = this->Arrays.find(iterV->first);
  if (iterA == this->Arrays.end())
    {
    vtkSmartPointer<vtkDoubleArray>& vtkarray = this->Arrays[iterV->first];
    vtkarray = vtkSmartPointer<vtkDoubleArray>::New();
    vtkarray->SetName(name);

    const vtkIdType size = (this->Extent[1] - this->Extent[0] + 1) *
      (this->Extent[3] - this->Extent[2] + 1) *
      (this->Extent[5] - this->Extent[4] + 1);
    vtkarray->SetArray(iterV->second, size, 1);
    return vtkarray;
    }
  else
    {
    return iterA->second;
    }
}

//-----------------------------------------------------------------------------
unsigned int DataAdaptor::GetNumberOfArrays(int association)
{
  return (association == vtkDataObject::FIELD_ASSOCIATION_POINTS)?
    static_cast<unsigned int>(this->Variables.size()): 0;
}

//-----------------------------------------------------------------------------
const char* DataAdaptor::GetArrayName(int association, unsigned int index)
{
  if (association != vtkDataObject::FIELD_ASSOCIATION_POINTS)
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
}

}
