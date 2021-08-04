#ifndef VTKUtils_h
#define VTKUtils_h

#include "MeshMetadata.h"
#include "Error.h"

class vtkDataSet;
class vtkDataObject;
class vtkFieldData;
class vtkDataSetAttributes;
class vtkCompositeDataSet;

#include <vtkDataArray.h>
#include <vtkAOSDataArrayTemplate.h>
#include <vtkSOADataArrayTemplate.h>

#include <vtkSmartPointer.h>
#include <vtkAOSDataArrayTemplate.h>
#include <vtkCellArray.h>

#include <functional>
#include <vector>
#include <mpi.h>

#define vtkCellTemplateMacro(code)                                                                 \
  vtkTemplateMacroCase(VTK_LONG_LONG, long long, code);                                            \
  vtkTemplateMacroCase(VTK_UNSIGNED_LONG_LONG, unsigned long long, code);                          \
  vtkTemplateMacroCase(VTK_ID_TYPE, vtkIdType, code);                                              \
  vtkTemplateMacroCase(VTK_LONG, long, code);                                                      \
  vtkTemplateMacroCase(VTK_UNSIGNED_LONG, unsigned long, code);                                    \
  vtkTemplateMacroCase(VTK_INT, int, code);                                                        \
  vtkTemplateMacroCase(VTK_UNSIGNED_INT, unsigned int, code);

using vtkCompositeDataSetPtr = vtkSmartPointer<vtkCompositeDataSet>;

#define vtkTemplateMacroFP(call)                                                                   \
  vtkTemplateMacroCase(VTK_DOUBLE, double, call);                                                  \
  vtkTemplateMacroCase(VTK_FLOAT, float, call);                                                    \

namespace sensei
{

/// A collection of generally useful funcitons implementing
/// common access patterns or operations on VTK data structures
namespace VTKUtils
{
/** given a vtkDataArray get a pointer to underlying data
 * this handles access from VTK's AOS and SOA layouts. For
 * SOA layout only single component arrays should be passed.
 */
template <typename VTK_TT>
VTK_TT *GetPointer(vtkDataArray *da)
{
  using AOS_ARRAY_TT = vtkAOSDataArrayTemplate<VTK_TT>;
  using SOA_ARRAY_TT = vtkSOADataArrayTemplate<VTK_TT>;

  AOS_ARRAY_TT *aosDa = nullptr;
  SOA_ARRAY_TT *soaDa = nullptr;

  if ((aosDa = dynamic_cast<AOS_ARRAY_TT*>(da)))
    {
    return aosDa->GetPointer(0);
    }
  else if ((soaDa = dynamic_cast<SOA_ARRAY_TT*>(da)))
    {
    return soaDa->GetPointer(0);
    }

  SENSEI_ERROR("Invalid vtkDataArray "
     << (da ? da->GetClassName() : "nullptr"))
  return nullptr;
}

/// given a VTK type enum returns the sizeof that type
unsigned int Size(int vtkt);

/// given a VTK data object enum returns true if it a legacy object
int IsLegacyDataObject(int code);

/// givne a VTK data object enum constructs an instance
vtkDataObject *NewDataObject(int code);

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
inline bool AMR(const MeshMetadataPtr &md)
{
  return (md->MeshType == VTK_OVERLAPPING_AMR) ||
    (md->MeshType == VTK_NON_OVERLAPPING_AMR);
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

/** Packs data from a cell array into another cell array keeping track of
 * where to insert into the output array. Use it to serialze verys, lines, polys
 * strips form a polytdata into a single cell array for transport
 */
template <typename VTK_TT, typename ARRAY_TT = vtkAOSDataArrayTemplate<VTK_TT>>
void PackCells(ARRAY_TT *coIn, ARRAY_TT *ccIn, ARRAY_TT *coOut, ARRAY_TT *ccOut,
  size_t &coId, size_t &ccId)
{
  // copy offsets
  size_t nOffs = coIn->GetNumberOfTuples();
  const VTK_TT *pSrc = coIn->GetPointer(0);
  VTK_TT *pDest = coOut->GetPointer(0);

  for (size_t i = 0; i < nOffs; ++i)
    pDest[coId + i] = pSrc[i];

  // update cell id
  coId += nOffs;

  // copy connectivity
  size_t nConn = ccIn->GetNumberOfTuples();
  pSrc = ccIn->GetPointer(0);
  pDest = ccOut->GetPointer(0);

  for (size_t i = 0; i < nConn; ++i)
    pDest[ccId + i] = pSrc[i];

  // update conn id
  ccId += nConn;
}

/// deserializes a buffer made by VTKUtils::PackCells.
template <typename VTK_TT,  typename ARRAY_TT = vtkAOSDataArrayTemplate<VTK_TT>>
void UnpackCells(size_t nc, ARRAY_TT *coIn, ARRAY_TT *ccIn,
  vtkCellArray *caOut, size_t &coId, size_t &ccId)
{
  // offsets
  size_t nCo = nc + 1;

  VTK_TT *pCoIn = coIn->GetPointer(0);

  ARRAY_TT *coOut = ARRAY_TT::New();

  coOut->SetNumberOfTuples(nCo);
  VTK_TT *pCoOut = coOut->GetPointer(0);

  for (size_t i = 0; i < nCo; ++i)
    pCoOut[i] = pCoIn[coId + i];

  coId += nCo;

  // connectivity
  size_t nCc = nc ? pCoOut[nCo - 1] : 0;

  ARRAY_TT *ccOut = ARRAY_TT::New();
  if (nCc)
    {
    VTK_TT *pCcIn = ccIn->GetPointer(0);

    ccOut->SetNumberOfTuples(nCc);
    VTK_TT *pCcOut = ccOut->GetPointer(0);

    for (size_t i = 0; i < nCc; ++i)
      pCcOut[i] = pCcIn[ccId + i];

    ccId += nCc;
    }

  // package as cells
  caOut->SetData(coOut, ccOut);

  coOut->Delete();
  ccOut->Delete();
}

}
}

#endif
