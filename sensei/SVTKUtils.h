#ifndef SVTKUtils_h
#define SVTKUtils_h

#include "MeshMetadata.h"
#include "Error.h"

class svtkDataSet;
class svtkDataObject;
class svtkFieldData;
class svtkDataSetAttributes;
class svtkCompositeDataSet;

#include <svtkDataArray.h>
#include <svtkAOSDataArrayTemplate.h>
#include <svtkSOADataArrayTemplate.h>

#include <svtkSmartPointer.h>
#include <svtkAOSDataArrayTemplate.h>
#include <svtkCellArray.h>

#include <functional>
#include <vector>
#include <mpi.h>

#define svtkCellTemplateMacro(code)                                         \
  svtkTemplateMacroCase(SVTK_LONG_LONG, long long, code);                   \
  svtkTemplateMacroCase(SVTK_UNSIGNED_LONG_LONG, unsigned long long, code); \
  svtkTemplateMacroCase(SVTK_ID_TYPE, svtkIdType, code);                    \
  svtkTemplateMacroCase(SVTK_LONG, long, code);                             \
  svtkTemplateMacroCase(SVTK_UNSIGNED_LONG, unsigned long, code);           \
  svtkTemplateMacroCase(SVTK_INT, int, code);                               \
  svtkTemplateMacroCase(SVTK_UNSIGNED_INT, unsigned int, code);

using svtkCompositeDataSetPtr = svtkSmartPointer<svtkCompositeDataSet>;

#define svtkTemplateMacroFP(call)                   \
  svtkTemplateMacroCase(SVTK_DOUBLE, double, call); \
  svtkTemplateMacroCase(SVTK_FLOAT, float, call);   \

namespace sensei
{

/// A collection of generally useful funcitons implementing
/// common access patterns or operations on SVTK data structures
namespace SVTKUtils
{
/** given a svtkDataArray get a pointer to underlying data
 * this handles access from SVTK's AOS and SOA layouts. For
 * SOA layout only single component arrays should be passed.
 */
template <typename SVTK_TT>
SVTK_TT *GetPointer(svtkDataArray *da)
{
  using AOS_ARRAY_TT = svtkAOSDataArrayTemplate<SVTK_TT>;
  using SOA_ARRAY_TT = svtkSOADataArrayTemplate<SVTK_TT>;

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

  SENSEI_ERROR("Invalid svtkDataArray "
     << (da ? da->GetClassName() : "nullptr"))
  return nullptr;
}

/// given a SVTK type enum returns the sizeof that type
unsigned int Size(int svtkt);

/// given a SVTK data object enum returns true if it a legacy object
int IsLegacyDataObject(int code);

/// givne a SVTK data object enum constructs an instance
svtkDataObject *NewDataObject(int code);

/// returns the enum value given an association name. where name
/// can be one of: point, cell or, field
int GetAssociation(std::string assocStr, int &assoc);

/// returns the name of the association, point, cell or field
const char *GetAttributesName(int association);

/// returns the container for the associations: svtkPointData,
/// svtkCellData, or svtkFieldData
svtkFieldData *GetAttributes(svtkDataSet *dobj, int association);

/// callback that processes input and output datasets
/// return 0 for success, > zero to stop without error, < zero to stop with error
using BinaryDatasetFunction = std::function<int(svtkDataSet*, svtkDataSet*)>;

/// Applies the function to leaves of the structurally equivalent
/// input and output data objects.
int Apply(svtkDataObject *input, svtkDataObject *output,
  BinaryDatasetFunction &func);

/// callback that processes input and output datasets
/// return 0 for success, > zero to stop without error, < zero to stop with error
using DatasetFunction = std::function<int(svtkDataSet*)>;

/// Applies the function to the data object
/// The function is called once for each leaf dataset
int Apply(svtkDataObject *dobj, DatasetFunction &func);

/// Store ghost layer metadata in the mesh
int SetGhostLayerMetadata(svtkDataObject *mesh,
  int nGhostCellLayers, int nGhostNodeLayers);

/// Retreive ghost layer metadata from the mesh. returns non-zero if
/// no such metadata is found.
int GetGhostLayerMetadata(svtkDataObject *mesh,
  int &nGhostCellLayers, int &nGhostNodeLayers);

/// Get  metadata, note that data set variant is not meant to
/// be used on blocks of a multi-block
int GetMetadata(MPI_Comm comm, svtkDataSet *ds, MeshMetadataPtr);
int GetMetadata(MPI_Comm comm, svtkCompositeDataSet *cd, MeshMetadataPtr);

/// Given a data object ensure that it is a composite data set
/// If it already is, then the call is a no-op, if it is not
/// then it is converted to a multiblock. The flag take determines
/// if the smart pointer takes ownership or adds a reference.
svtkCompositeDataSetPtr AsCompositeData(MPI_Comm comm,
  svtkDataObject *dobj, bool take = true);

/// Return true if the mesh or block type is AMR
inline bool AMR(const MeshMetadataPtr &md)
{
  return (md->MeshType == SVTK_OVERLAPPING_AMR) ||
    (md->MeshType == SVTK_NON_OVERLAPPING_AMR);
}

/// Return true if the mesh or block type is logically Cartesian
inline bool Structured(const MeshMetadataPtr &md)
{
  return (md->BlockType == SVTK_STRUCTURED_GRID) ||
    (md->MeshType == SVTK_STRUCTURED_GRID);
}

/// Return true if the mesh or block type is polydata
inline bool Polydata(const MeshMetadataPtr &md)
{
  return (md->BlockType == SVTK_POLY_DATA) || (md->MeshType == SVTK_POLY_DATA);
}

/// Return true if the mesh or block type is unstructured
inline bool Unstructured(const MeshMetadataPtr &md)
{
  return (md->BlockType == SVTK_UNSTRUCTURED_GRID) ||
    (md->MeshType == SVTK_UNSTRUCTURED_GRID);
}

/// Return true if the mesh or block type is stretched Cartesian
inline bool StretchedCartesian(const MeshMetadataPtr &md)
{
  return (md->BlockType == SVTK_RECTILINEAR_GRID) ||
    (md->MeshType == SVTK_RECTILINEAR_GRID);
}

/// Return true if the mesh or block type is uniform Cartesian
inline bool UniformCartesian(const MeshMetadataPtr &md)
{
  return (md->BlockType == SVTK_IMAGE_DATA) || (md->MeshType == SVTK_IMAGE_DATA)
    || (md->BlockType == SVTK_UNIFORM_GRID) || (md->MeshType == SVTK_UNIFORM_GRID);
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
template <typename SVTK_TT, typename ARRAY_TT = svtkAOSDataArrayTemplate<SVTK_TT>>
void PackCells(ARRAY_TT *coIn, ARRAY_TT *ccIn, ARRAY_TT *coOut, ARRAY_TT *ccOut,
  size_t &coId, size_t &ccId)
{
  // copy offsets
  size_t nOffs = coIn->GetNumberOfTuples();
  const SVTK_TT *pSrc = coIn->GetPointer(0);
  SVTK_TT *pDest = coOut->GetPointer(0);

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

/// deserializes a buffer made by SVTKUtils::PackCells.
template <typename SVTK_TT,  typename ARRAY_TT = svtkAOSDataArrayTemplate<SVTK_TT>>
void UnpackCells(size_t nc, ARRAY_TT *coIn, ARRAY_TT *ccIn,
  svtkCellArray *caOut, size_t &coId, size_t &ccId)
{
  // offsets
  size_t nCo = nc + 1;

  SVTK_TT *pCoIn = coIn->GetPointer(0);

  ARRAY_TT *coOut = ARRAY_TT::New();

  coOut->SetNumberOfTuples(nCo);
  SVTK_TT *pCoOut = coOut->GetPointer(0);

  for (size_t i = 0; i < nCo; ++i)
    pCoOut[i] = pCoIn[coId + i];

  coId += nCo;

  // connectivity
  size_t nCc = nc ? pCoOut[nCo - 1] : 0;

  ARRAY_TT *ccOut = ARRAY_TT::New();
  if (nCc)
    {
    SVTK_TT *pCcIn = ccIn->GetPointer(0);

    ccOut->SetNumberOfTuples(nCc);
    SVTK_TT *pCcOut = ccOut->GetPointer(0);

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
