#ifndef VTKUtils_h
#define VTKUtils_h

#include "MeshMetadata.h"

class vtkDataSet;
class vtkDataObject;
class vtkFieldData;
class vtkDataSetAttributes;
class vtkCompositeDataSet;

#include <vtkSmartPointer.h>
#include <functional>
#include <vector>
#include <mpi.h>

using vtkCompositeDataSetPtr = vtkSmartPointer<vtkCompositeDataSet>;

namespace sensei
{

/// A collection of generally useful funcitons implementing
/// common access patterns or operations on VTK data structures
namespace VTKUtils
{

/// returns the enum value given an association name. where name
/// can be one of: point, cell or, field
int GetAssociation(std::string assocStr, int &assoc);

/// returns the name of the association, point, cell or field
const char *GetAttributesName(int association);

/// returns the container for the associations: vtkPointData,
/// vtkCellData, or vtkFieldData
vtkFieldData *GetAttributes(vtkDataSet *dobj, int association);

/// callback that processes input and output datasets
/// return 0 for success, > zero to stop without error, < zero to stop with error
using BinaryDatasetFunction = std::function<int(vtkDataSet*, vtkDataSet*)>;

/// Applies the function to leaves of the structurally equivalent
/// input and output data objects.
int Apply(vtkDataObject *input, vtkDataObject *output,
  BinaryDatasetFunction &func);

/// callback that processes input and output datasets
/// return 0 for success, > zero to stop without error, < zero to stop with error
using DatasetFunction = std::function<int(vtkDataSet*)>;

/// Applies the function to the data object
/// The function is called once for each leaf dataset
int Apply(vtkDataObject *dobj, DatasetFunction &func);

/// Store ghost layer metadata in the mesh
int SetGhostLayerMetadata(vtkDataObject *mesh,
  int nGhostCellLayers, int nGhostNodeLayers);

/// Retreive ghost layer metadata from the mesh. returns non-zero if
/// no such metadata is found.
int GetGhostLayerMetadata(vtkDataObject *mesh,
  int &nGhostCellLayers, int &nGhostNodeLayers);

/// Get  metadata, note that data set variant is not meant to
/// be used on blocks of a multi-block
int GetMetadata(MPI_Comm comm, vtkDataSet *ds, MeshMetadataPtr);
int GetMetadata(MPI_Comm comm, vtkCompositeDataSet *cd, MeshMetadataPtr);

/// Given a data object ensure that it is a composite data set
/// If it already is, then the call is a no-op, if it is not
/// then it is converted to a multiblock. The flag take determines
/// if the smart pointer takes ownership or adds a reference.
vtkCompositeDataSetPtr AsCompositeData(MPI_Comm comm,
  vtkDataObject *dobj, bool take = true);

/// Return true if the mesh or block type is AMR
inline bool Amr(const MeshMetadataPtr &md)
{
  return (md->MeshType == VTK_OVERLAPPING_AMR);
}

/// Return true if the mesh or block type is logically Cartesian
inline bool Structured(const MeshMetadataPtr &md)
{
  return (md->BlockType == VTK_STRUCTURED_GRID) ||
    (md->MeshType == VTK_STRUCTURED_GRID);
}

/// Return true if the mesh or block type is polydata
inline bool Polydata(const MeshMetadataPtr &md)
{
  return (md->BlockType == VTK_POLY_DATA) || (md->MeshType == VTK_POLY_DATA);
}

/// Return true if the mesh or block type is unstructured
inline bool Unstructured(const MeshMetadataPtr &md)
{
  return (md->BlockType == VTK_UNSTRUCTURED_GRID) ||
    (md->MeshType == VTK_UNSTRUCTURED_GRID);
}

/// Return true if the mesh or block type is stretched Cartesian
inline bool StretchedCartesian(const MeshMetadataPtr &md)
{
  return (md->BlockType == VTK_RECTILINEAR_GRID) ||
    (md->MeshType == VTK_RECTILINEAR_GRID);
}

/// Return true if the mesh or block type is uniform Cartesian
inline bool UniformCartesian(const MeshMetadataPtr &md)
{
  return (md->BlockType == VTK_IMAGE_DATA) || (md->MeshType == VTK_IMAGE_DATA)
    || (md->BlockType == VTK_UNIFORM_GRID) || (md->MeshType == VTK_UNIFORM_GRID);
}

/// Return true if the mesh or block type is logically Cartesian
inline bool LogicallyCartesian(const MeshMetadataPtr &md)
{
  return Structured(md) || UniformCartesian(md) || StretchedCartesian(md);
}

// rank 0 writes a dataset for visualizing the domain decomp
int WriteDomainDecomp(MPI_Comm comm, const sensei::MeshMetadataPtr &md,
  const std::string fileName);

}
}

#endif
