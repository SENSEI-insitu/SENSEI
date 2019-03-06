#include "DataAdaptor.h"
#include "MeshMetadata.h"
#include "Error.h"

#include <vtkCellData.h>
#include <vtkDataObject.h>
#include <vtkDoubleArray.h>
#include <vtkImageData.h>
#include <vtkObjectFactory.h>

#include <limits>

namespace parallel3d
{
//-----------------------------------------------------------------------------
senseiNewMacro(DataAdaptor);

//-----------------------------------------------------------------------------
DataAdaptor::DataAdaptor()
{
  for (int i = 0; i < 3; ++i)
    this->Origin[i] = 0.0;

  for (int i = 0; i < 3; ++i)
    this->Spacing[i] = 0.0;

  for (int i = 0; i < 6; ++i)
    this->LocalExtent[i] = 0;

  for (int i = 0; i < 6; ++i)
    this->GlobalExtent[i] = 0;

  for (int i = 0; i < 3; ++i)
    this->Arrays[i] = nullptr;
}

//-----------------------------------------------------------------------------
DataAdaptor::~DataAdaptor()
{
}

//-----------------------------------------------------------------------------
void DataAdaptor::UpdateGeometry(double x_0, double y_0, double z_0,
  double dx, double dy, double dz, long g_nx, long g_ny, long g_nz,
  long offs_x, long offs_y, long offs_z, long l_nx, long l_ny, long l_nz)
{
  this->Origin[0] = x_0;
  this->Origin[1] = y_0;
  this->Origin[2] = z_0;

  this->Spacing[0] = dx;
  this->Spacing[1] = dy;
  this->Spacing[2] = dz;

  this->LocalExtent[0] = offs_x;
  this->LocalExtent[1] = offs_x + l_nx - 1;
  this->LocalExtent[2] = offs_y;
  this->LocalExtent[3] = offs_y + l_ny - 1;
  this->LocalExtent[4] = offs_z;
  this->LocalExtent[5] = offs_z + l_nz - 1;

  this->GlobalExtent[0] = 0;
  this->GlobalExtent[1] = g_nx - 1;
  this->GlobalExtent[2] = 0;
  this->GlobalExtent[3] = g_ny - 1;
  this->GlobalExtent[4] = 0;
  this->GlobalExtent[5] = g_nz - 1;
}

//-----------------------------------------------------------------------------
void DataAdaptor::UpdateArrays(double *pressure, double *temperature,
  double *density)
{
  this->Arrays[0] = pressure;
  this->Arrays[1] = temperature;
  this->Arrays[2] = density;
}

//-----------------------------------------------------------------------------
int DataAdaptor::GetNumberOfMeshes(unsigned int &numMeshes)
{
  numMeshes = 1;
  return 0;
}

//-----------------------------------------------------------------------------
int DataAdaptor::GetMeshMetadata(unsigned int id, sensei::MeshMetadataPtr &metadata)
{
  if (id !=0 )
    {
    SENSEI_ERROR("dataset index " << id << " is out of bounds")
    return -1;
    }

  int rank = 0;
  int nRanks = 1;

  MPI_Comm_rank(this->GetCommunicator(), &rank);
  MPI_Comm_size(this->GetCommunicator(), &nRanks);

  metadata->MeshName = "mesh";
  metadata->NumBlocks = nRanks;
  metadata->NumBlocksLocal = {1};
  metadata->MeshType = VTK_IMAGE_DATA;
  metadata->BlockType = VTK_IMAGE_DATA;
  metadata->NumArrays = 3;
  metadata->ArrayName = {"pressure", "temperature", "density"};
  metadata->ArrayCentering = {vtkDataObject::CELL, vtkDataObject::CELL, vtkDataObject::CELL};
  metadata->ArrayComponents = {1, 1, 1};

  if (metadata->Flags.BlockDecompSet())
    {
    metadata->BlockOwner = {rank};
    metadata->BlockIds = {rank};
    }

  if (metadata->Flags.BlockSizeSet())
    {
    metadata->BlockNumPoints = {(this->LocalExtent[1] - this->LocalExtent[0] + 2)*
      (this->LocalExtent[3] - this->LocalExtent[2] + 2)*
      (this->LocalExtent[5] - this->LocalExtent[4] + 2)};

    metadata->BlockNumCells = {(this->LocalExtent[1] - this->LocalExtent[0] + 1)*
      (this->LocalExtent[3] - this->LocalExtent[2] + 1)*
      (this->LocalExtent[5] - this->LocalExtent[4] + 1)};
    }

  if (metadata->Flags.BlockExtentsSet())
    {
    //metadata->Extent = std::vector<int>(this->GlobalExtent, this->GlobalExtent+6);
    metadata->BlockExtents.emplace_back(
      std::array<int,6>{{this->LocalExtent[0], this->LocalExtent[1],
        this->LocalExtent[2], this->LocalExtent[3],
        this->LocalExtent[4], this->LocalExtent[5]}});
    }

  if (metadata->Flags.BlockBoundsSet())
    {
    metadata->BlockBounds.emplace_back(std::move(std::array<double,6>{{
      this->Origin[0] + this->Spacing[0]*this->LocalExtent[0],
      this->Origin[0] + this->Spacing[0]*(this->LocalExtent[1] + 1),
      this->Origin[1] + this->Spacing[1]*this->LocalExtent[2],
      this->Origin[1] + this->Spacing[1]*(this->LocalExtent[3] + 1),
      this->Origin[2] + this->Spacing[2]*this->LocalExtent[4],
      this->Origin[2] + this->Spacing[2]*(this->LocalExtent[5] + 1)}}));
    }

  return 0;
}
//-----------------------------------------------------------------------------
int DataAdaptor::GetMesh(const std::string &meshName, bool structureOnly,
  vtkDataObject *&mesh)
{
  mesh = nullptr;

  if (meshName != "mesh")
    {
    SENSEI_ERROR("No mesh \"" << meshName << "\"")
    return -1;
    }

  // create the Cartesian mesh. this is the legacy approach to
  // parallel data used by ParaView. When each rank has only 1 block
  vtkImageData *id = vtkImageData::New();

  // structure only says the analysis doesn't need the mesh geometry.
  // it's more important with unstructured meshes, shown here for
  // illustrative purposes
  if (!structureOnly)
    {
    id->SetOrigin(this->Origin);
    id->SetSpacing(this->Spacing);

    // pass in the local extents thta describe the positioning of
    // this rank's data within the global domain.
    // +1 because we have cell based extent but vtk wants point based
    id->SetExtent(this->LocalExtent[0], this->LocalExtent[1] + 1,
      this->LocalExtent[2], this->LocalExtent[3] + 1, this->LocalExtent[4],
      this->LocalExtent[5] + 1);
    }

  // return the dataset, note that the analysis takes ownership
  mesh = id;

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

  if (association != vtkDataObject::CELL)
    {
    SENSEI_ERROR("No point data on mesh")
    return -1;
    }

  // figure out which array is being requested
  double *data = nullptr;
  if (!strcmp("pressure", arrayName.c_str()))
    {
    data = this->Arrays[0];
    }
  else if (!strcmp("temperature", arrayName.c_str()))
    {
    data = this->Arrays[1];
    }
  else if (!strcmp("density", arrayName.c_str()))
    {
    data = this->Arrays[2];
    }
  else
    {
    SENSEI_ERROR("No array named \"" << arrayName << "\"")
    return -1;
    }

  long nCells = (this->LocalExtent[1] - this->LocalExtent[0] + 1) *
    (this->LocalExtent[3] - this->LocalExtent[2] + 1) *
    (this->LocalExtent[5] - this->LocalExtent[4] + 1);

  // pass the array by zero copy into a vtk data array
  vtkDoubleArray *da = vtkDoubleArray::New();
  da->SetName(arrayName.c_str());
  da->SetArray(data, nCells, 1);

  // put the vtk data array into the mesh's cell data
  vtkImageData *id = dynamic_cast<vtkImageData*>(mesh);
  if (!id)
    {
    SENSEI_ERROR("invalid mesh "
      << (mesh ? mesh->GetClassName() : "nullptr"))
    return -1;
    }

  id->GetCellData()->AddArray(da);
  da->Delete();

  return 0;
}

//-----------------------------------------------------------------------------
int DataAdaptor::ReleaseData()
{
  return 0;
}

}
