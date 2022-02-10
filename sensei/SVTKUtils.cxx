#include "senseiConfig.h"
#include "SVTKUtils.h"
#include "MPIUtils.h"
#include "MeshMetadata.h"
#include "Error.h"


#include <svtkCompositeDataIterator.h>
#include <svtkCompositeDataSet.h>
#include <svtkDataSetAttributes.h>
#include <svtkDataArray.h>
#include <svtkAbstractArray.h>
#include <svtkDoubleArray.h>
#include <svtkFloatArray.h>
#include <svtkIntArray.h>
#include <svtkUnsignedIntArray.h>
#include <svtkLongArray.h>
#include <svtkUnsignedLongArray.h>
#include <svtkLongLongArray.h>
#include <svtkUnsignedLongLongArray.h>
#include <svtkCharArray.h>
#include <svtkUnsignedCharArray.h>
#include <svtkIdTypeArray.h>
#include <svtkPoints.h>
#include <svtkPolyData.h>
#include <svtkStructuredPoints.h>
#include <svtkStructuredGrid.h>
#include <svtkRectilinearGrid.h>
#include <svtkUnstructuredGrid.h>
#include <svtkImageData.h>
#include <svtkUniformGrid.h>
#include <svtkTable.h>
#include <svtkMultiBlockDataSet.h>
#include <svtkHierarchicalBoxDataSet.h>
#include <svtkMultiPieceDataSet.h>
#include <svtkHyperTreeGrid.h>
#include <svtkOverlappingAMR.h>
#include <svtkNonOverlappingAMR.h>
#include <svtkUniformGridAMR.h>
#include <svtkObjectFactory.h>
#include <svtkPointData.h>
#include <svtkSmartPointer.h>
#include <svtkDataObject.h>
#include <svtkDataSet.h>
#include <svtkPointSet.h>
#include <svtkAMRBox.h>
#include <svtkDataSetAttributes.h>
#include <svtkFieldData.h>
#include <svtkCellData.h>
#include <svtkPointData.h>
#include <svtkObjectBase.h>
#include <svtkObject.h>
#include <svtkCellArray.h>
#include <svtkCellTypes.h>
#include <svtkSmartPointer.h>
#include <svtkIntArray.h>
#include <svtkVersionMacros.h>
#if ((SVTK_VERSION_MAJOR >= 8) && (SVTK_VERSION_MINOR >= 2))
#include <svtkAOSDataArrayTemplate.h>
#include <svtkSOADataArrayTemplate.h>
#endif
#if defined(ENABLE_VTK_IO)
#include <svtkXMLUnstructuredGridWriter.h>
#endif
#include <svtkPoints.h>
#include <svtkDoubleArray.h>
#include <svtkIdTypeArray.h>
#include <svtkUnsignedCharArray.h>

#include <sstream>
#include <functional>
#include <mpi.h>

using svtkDataObjectPtr = svtkSmartPointer<svtkDataObject>;
using svtkCompositeDataIteratorPtr = svtkSmartPointer<svtkCompositeDataIterator>;

namespace sensei
{
namespace SVTKUtils
{

// --------------------------------------------------------------------------
unsigned int Size(int svtkt)
{
  switch (svtkt)
    {
    case SVTK_FLOAT:
      return sizeof(float);
      break;
    case SVTK_DOUBLE:
      return sizeof(double);
      break;
    case SVTK_CHAR:
      return sizeof(char);
      break;
    case SVTK_UNSIGNED_CHAR:
      return sizeof(unsigned char);
      break;
    case SVTK_INT:
      return sizeof(int);
      break;
    case SVTK_UNSIGNED_INT:
      return sizeof(unsigned int);
      break;
    case SVTK_LONG:
      return sizeof(long);
      break;
    case SVTK_UNSIGNED_LONG:
      return sizeof(unsigned long);
      break;
    case SVTK_LONG_LONG:
      return sizeof(long long);
      break;
    case SVTK_UNSIGNED_LONG_LONG:
      return sizeof(unsigned long long);
      break;
    case SVTK_ID_TYPE:
      return sizeof(svtkIdType);
      break;
    default:
      {
      SENSEI_ERROR("the adios type for svtk type enumeration " << svtkt
        << " is currently not implemented")
      MPI_Abort(MPI_COMM_WORLD, -1);
      }
    }
  return 0;
}

// --------------------------------------------------------------------------
int IsLegacyDataObject(int code)
{
  // this function is used to determine data parallelization strategy.
  // SVTK has 2, namely the legacy one in which each process holds 1
  // legacy dataset, and the more modern approach where SVTK composite
  // dataset holds any number of datasets on any number of processes.
  int ret = 0;
  switch (code)
    {
    // legacy
    case SVTK_POLY_DATA:
    case SVTK_STRUCTURED_POINTS:
    case SVTK_STRUCTURED_GRID:
    case SVTK_RECTILINEAR_GRID:
    case SVTK_UNSTRUCTURED_GRID:
    case SVTK_IMAGE_DATA:
    case SVTK_UNIFORM_GRID:
    case SVTK_TABLE:
    // others
    case SVTK_GRAPH:
    case SVTK_TREE:
    case SVTK_SELECTION:
    case SVTK_DIRECTED_GRAPH:
    case SVTK_UNDIRECTED_GRAPH:
    case SVTK_DIRECTED_ACYCLIC_GRAPH:
    case SVTK_ARRAY_DATA:
    case SVTK_REEB_GRAPH:
    case SVTK_MOLECULE:
    case SVTK_PATH:
    case SVTK_PIECEWISE_FUNCTION:
      ret = 1;
      break;
    // composite data etc
    case SVTK_MULTIBLOCK_DATA_SET:
    case SVTK_HIERARCHICAL_BOX_DATA_SET:
    case SVTK_MULTIPIECE_DATA_SET:
    case SVTK_HYPER_OCTREE:
    case SVTK_HYPER_TREE_GRID:
    case SVTK_OVERLAPPING_AMR:
    case SVTK_NON_OVERLAPPING_AMR:
    case SVTK_UNIFORM_GRID_AMR:
      ret = 0;
      break;
    // base classes
    case SVTK_DATA_OBJECT:
    case SVTK_DATA_SET:
    case SVTK_POINT_SET:
    case SVTK_COMPOSITE_DATA_SET:
    case SVTK_GENERIC_DATA_SET:
#if !(SVTK_MAJOR_VERSION == 6 && SVTK_MINOR_VERSION == 1)
    case SVTK_UNSTRUCTURED_GRID_BASE:
    case SVTK_PISTON_DATA_OBJECT:
#endif
    // deprecated/removed
    case SVTK_HIERARCHICAL_DATA_SET:
    case SVTK_TEMPORAL_DATA_SET:
    case SVTK_MULTIGROUP_DATA_SET:
    // unknown code
    default:
      SENSEI_ERROR("Neither legacy nor composite " << code)
      ret = -1;
    }
  return ret;
}

// --------------------------------------------------------------------------
svtkDataObject *NewDataObject(int code)
{
  svtkDataObject *ret = nullptr;
  switch (code)
    {
    // simple
    case SVTK_POLY_DATA:
      ret = svtkPolyData::New();
      break;
    case SVTK_STRUCTURED_POINTS:
      ret = svtkStructuredPoints::New();
      break;
    case SVTK_STRUCTURED_GRID:
      ret = svtkStructuredGrid::New();
      break;
    case SVTK_RECTILINEAR_GRID:
      ret = svtkRectilinearGrid::New();
      break;
    case SVTK_UNSTRUCTURED_GRID:
      ret = svtkUnstructuredGrid::New();
      break;
    case SVTK_IMAGE_DATA:
      ret = svtkImageData::New();
      break;
    case SVTK_UNIFORM_GRID:
      ret = svtkUniformGrid::New();
      break;
    case SVTK_TABLE:
      ret = svtkTable::New();
      break;
    // composite data etc
    case SVTK_MULTIBLOCK_DATA_SET:
      ret = svtkMultiBlockDataSet::New();
      break;
    case SVTK_HIERARCHICAL_BOX_DATA_SET:
      ret = svtkHierarchicalBoxDataSet::New();
      break;
    case SVTK_MULTIPIECE_DATA_SET:
      ret = svtkMultiPieceDataSet::New();
      break;
    case SVTK_HYPER_TREE_GRID:
      ret = svtkHyperTreeGrid::New();
      break;
    case SVTK_OVERLAPPING_AMR:
      ret = svtkOverlappingAMR::New();
      break;
    case SVTK_NON_OVERLAPPING_AMR:
      ret = svtkNonOverlappingAMR::New();
      break;
    case SVTK_UNIFORM_GRID_AMR:
      ret = svtkUniformGridAMR::New();
      break;
    // TODO
    case SVTK_GRAPH:
    case SVTK_TREE:
    case SVTK_SELECTION:
    case SVTK_DIRECTED_GRAPH:
    case SVTK_UNDIRECTED_GRAPH:
    case SVTK_DIRECTED_ACYCLIC_GRAPH:
    case SVTK_ARRAY_DATA:
    case SVTK_REEB_GRAPH:
    case SVTK_MOLECULE:
    case SVTK_PATH:
    case SVTK_PIECEWISE_FUNCTION:
      SENSEI_WARNING("Factory for " << code << " not yet implemented")
      break;
    // base classes
    case SVTK_DATA_OBJECT:
    case SVTK_DATA_SET:
    case SVTK_POINT_SET:
    case SVTK_COMPOSITE_DATA_SET:
    case SVTK_GENERIC_DATA_SET:
#if !(SVTK_MAJOR_VERSION == 6 && SVTK_MINOR_VERSION == 1)
    case SVTK_UNSTRUCTURED_GRID_BASE:
    case SVTK_PISTON_DATA_OBJECT:
#endif
    // deprecated/removed
    case SVTK_HIERARCHICAL_DATA_SET:
    case SVTK_TEMPORAL_DATA_SET:
    case SVTK_MULTIGROUP_DATA_SET:
    // unknown code
    default:
      SENSEI_ERROR("data object for " << code << " could not be construtced")
    }
  return ret;
}

//----------------------------------------------------------------------------
int GetAssociation(std::string assocStr, int &assoc)
{
  unsigned int n = assocStr.size();
  for (unsigned int i = 0; i < n; ++i)
    assocStr[i] = tolower(assocStr[i]);

  if (assocStr == "point")
    {
    assoc = svtkDataObject::POINT;
    return 0;
    }
  else if (assocStr == "cell")
    {
    assoc = svtkDataObject::CELL;
    return 0;
    }
  else if (assocStr == "field")
    {
    assoc = svtkDataObject::FIELD;
    return 0;
    }

  SENSEI_ERROR("Invalid association \"" << assocStr << "\"")
  return -1;
}

//----------------------------------------------------------------------------
const char *GetAttributesName(int association)
{
  switch (association)
    {
    case svtkDataObject::POINT:
      return "point";
      break;
    case svtkDataObject::CELL:
      return "cell";
      break;
    case svtkDataObject::FIELD:
      return "field";
      break;
    }
  SENSEI_ERROR("Invalid data set attributes association")
  return "";
}

//----------------------------------------------------------------------------
svtkFieldData *GetAttributes(svtkDataSet *dobj, int association)
{
  switch (association)
    {
    case svtkDataObject::POINT:
      return static_cast<svtkFieldData*>(dobj->GetPointData());
      break;
    case svtkDataObject::CELL:
      return static_cast<svtkFieldData*>(dobj->GetCellData());
      break;
    case svtkDataObject::FIELD:
      return static_cast<svtkFieldData*>(dobj->GetFieldData());
      break;
    }
  SENSEI_ERROR("Invalid data set attributes association")
  return nullptr;
}

//----------------------------------------------------------------------------
int Apply(svtkCompositeDataSet *cd, svtkCompositeDataSet *cdo,
  BinaryDatasetFunction &func)
{
  svtkCompositeDataIteratorPtr cdit;
  cdit.TakeReference(cd->NewIterator());
  while (!cdit->IsDoneWithTraversal())
    {
    svtkDataObject *obj = cd->GetDataSet(cdit);
    svtkDataObject *objOut = cdo->GetDataSet(cdit);

    // recurse through nested composite datasets
    if (svtkCompositeDataSet *cdn = dynamic_cast<svtkCompositeDataSet*>(obj))
      {
      svtkCompositeDataSet*cdnOut = static_cast<svtkCompositeDataSet*>(objOut);
      int ret = Apply(cdn, cdnOut, func);
      if (ret < 0)
        {
        // stop with error
        SENSEI_ERROR("Failed to apply to composite data set index "
          << cdit->GetCurrentFlatIndex())
        return -1;
        }
      else if (ret > 0)
        {
        // stop without error
        return 1;
        }
      }
    // process data set leaves
    else if(svtkDataSet *ds = dynamic_cast<svtkDataSet*>(obj))
      {
      svtkDataSet *dsOut = static_cast<svtkDataSet*>(objOut);
      int ret = func(ds, dsOut);
      if (ret < 0)
        {
        // stop with error
        SENSEI_ERROR("Function failed in apply at data set index "
          << cdit->GetCurrentFlatIndex())
        return -1;
        }
      else if (ret)
        {
        // stop without error
        return 1;
        }
      }
    else if (obj)
      {
      SENSEI_ERROR("Can't apply to " << obj->GetClassName())
      return -1;
      }
    cdit->GoToNextItem();
    }
  return 0;
}

//----------------------------------------------------------------------------
int Apply(svtkDataObject *dobj, svtkDataObject *dobjo,
  BinaryDatasetFunction &func)
{
  if (svtkCompositeDataSet *cd = dynamic_cast<svtkCompositeDataSet*>(dobj))
    {
    svtkCompositeDataSet *cdo = static_cast<svtkCompositeDataSet*>(dobjo);
    if (Apply(cd, cdo, func) < 0)
      {
      return -1;
      }
    }
  else if (svtkDataSet *ds = dynamic_cast<svtkDataSet*>(dobj))
    {
    svtkDataSet *dso = static_cast<svtkDataSet*>(dobjo);
    if (func(ds, dso) < 0)
      {
      return -1;
      }
    }
  else
    {
    SENSEI_ERROR("Unsupoorted data object type " << dobj->GetClassName())
    return -1;
    }

  return 0;
}

//----------------------------------------------------------------------------
int Apply(svtkCompositeDataSet *cd, DatasetFunction &func)
{
  svtkCompositeDataIteratorPtr cdit;
  cdit.TakeReference(cd->NewIterator());
  while (!cdit->IsDoneWithTraversal())
    {
    svtkDataObject *obj = cd->GetDataSet(cdit);
    // recurse through nested composite datasets
    if (svtkCompositeDataSet *cdn = dynamic_cast<svtkCompositeDataSet*>(obj))
      {
      int ret = Apply(cdn, func);
      if (ret < 0)
        {
        // stop with error
        SENSEI_ERROR("Failed to apply to composite data set index "
          << cdit->GetCurrentFlatIndex())
        return -1;
        }
      else if (ret > 0)
        {
        // stop without error
        return 1;
        }
      }
    // process data set leaves
    else if(svtkDataSet *ds = dynamic_cast<svtkDataSet*>(obj))
      {
      int ret = func(ds);
      if (ret < 0)
        {
        // stop with error
        SENSEI_ERROR("Function failed to apply to composite data set index "
          << cdit->GetCurrentFlatIndex())
        return -1;
        }
      else if (ret)
        {
        // stop without error
        return 1;
        }
      }
    else if (obj)
      {
      SENSEI_ERROR("Can't apply to " << obj->GetClassName())
      return -1;
      }
    cdit->GoToNextItem();
    }
  return 0;
}

//----------------------------------------------------------------------------
int Apply(svtkDataObject *dobj, DatasetFunction &func)
{
  if (svtkCompositeDataSet *cd = dynamic_cast<svtkCompositeDataSet*>(dobj))
    {
    if (Apply(cd, func) < 0)
      {
      return -1;
      }
    }
  else if (svtkDataSet *ds = dynamic_cast<svtkDataSet*>(dobj))
    {
    if (func(ds) < 0)
      {
      return -1;
      }
    }
  else
    {
    SENSEI_ERROR("Unsupoorted data object type " << dobj->GetClassName())
    return -1;
    }
  return 0;
}

//----------------------------------------------------------------------------
int GetGhostLayerMetadata(svtkDataObject *mesh,
  int &nGhostCellLayers, int &nGhostNodeLayers)
{
  // get the ghost layer metadata
  svtkFieldData *fd = mesh->GetFieldData();

  svtkIntArray *glmd =
    dynamic_cast<svtkIntArray*>(fd->GetArray("senseiGhostLayers"));

  if (!glmd)
    return -1;

  nGhostCellLayers = glmd->GetValue(0);
  nGhostNodeLayers = glmd->GetValue(1);

  return 0;
}

//----------------------------------------------------------------------------
int SetGhostLayerMetadata(svtkDataObject *mesh,
  int nGhostCellLayers, int nGhostNodeLayers)
{
  // pass ghost layer metadata in field data.
  svtkIntArray *glmd = svtkIntArray::New();
  glmd->SetName("senseiGhostLayers");
  glmd->SetNumberOfTuples(2);
  glmd->SetValue(0, nGhostCellLayers);
  glmd->SetValue(1, nGhostNodeLayers);

  svtkFieldData *fd = mesh->GetFieldData();
  fd->AddArray(glmd);
  glmd->Delete();

  return 0;
}

// --------------------------------------------------------------------------
int GetArrayMetadata(svtkDataSetAttributes *dsa, int centering,
  std::vector<std::string> &arrayNames, std::vector<int> &arrayCen,
  std::vector<int> &arrayComps, std::vector<int> &arrayType,
  std::vector<std::array<double,2>> &arrayRange,
  int &hasGhostArray)
{
  int na = dsa->GetNumberOfArrays();
  for (int i = 0; i < na; ++i)
    {
    svtkDataArray *da = dsa->GetArray(i);

    const char *name = da->GetName();
    arrayNames.emplace_back((name ? name : "unkown"));

    arrayCen.emplace_back(centering);
    arrayComps.emplace_back(da->GetNumberOfComponents());
    arrayType.emplace_back(da->GetDataType());

    arrayRange.emplace_back(std::array<double,2>({std::numeric_limits<double>::max(),
      std::numeric_limits<double>::lowest()}));

    if (!hasGhostArray && name && !strcmp("svtkGhostType", name))
      hasGhostArray = 1;
    }
  return 0;
}

// --------------------------------------------------------------------------
int GetArrayMetadata(svtkDataSetAttributes *dsa,
  std::vector<std::array<double,2>> &arrayRange)
{
  int na = dsa->GetNumberOfArrays();
  for (int i = 0; i < na; ++i)
    {
    svtkDataArray *da = dsa->GetArray(i);

    double rng[2];
    da->GetRange(rng);

    arrayRange.emplace_back(std::array<double,2>({rng[0], rng[1]}));
    }
  return 0;
}

// --------------------------------------------------------------------------
int GetArrayMetadata(svtkDataSet *ds, MeshMetadataPtr &metadata)
{
  SVTKUtils::GetArrayMetadata(ds->GetPointData(), svtkDataObject::POINT,
    metadata->ArrayName, metadata->ArrayCentering, metadata->ArrayComponents,
    metadata->ArrayType, metadata->ArrayRange, metadata->NumGhostNodes);

  SVTKUtils::GetArrayMetadata(ds->GetCellData(), svtkDataObject::CELL,
    metadata->ArrayName, metadata->ArrayCentering, metadata->ArrayComponents,
    metadata->ArrayType, metadata->ArrayRange, metadata->NumGhostCells);

  metadata->NumArrays = metadata->ArrayName.size();

  return 0;
}

// --------------------------------------------------------------------------
int GetBlockMetadata(int rank, int id, svtkDataSet *ds,
  const MeshMetadataFlags &flags, std::vector<int> &blockOwner,
  std::vector<int> &blockIds, std::vector<long> &blockPoints,
  std::vector<long> &blockCells, std::vector<long> &blockCellArraySize,
  std::vector<std::array<int,6>> &blockExtents,
  std::vector<std::array<double,6>> &blockBounds,
  std::vector<std::vector<std::array<double,2>>> &blockArrayRange)
{
  if (!ds)
    return -1;

  if (flags.BlockDecompSet())
    {
    blockOwner.emplace_back(rank);
    blockIds.emplace_back(id);
    }

  if (flags.BlockSizeSet())
    {
    long nPts = ds->GetNumberOfPoints();
    blockPoints.emplace_back(nPts);

    long nCells = ds->GetNumberOfCells();
    blockCells.emplace_back(nCells);

    long cellArraySize = 0;
    if (svtkUnstructuredGrid *ug = dynamic_cast<svtkUnstructuredGrid*>(ds))
      {
      cellArraySize = ug->GetCells()->GetConnectivityArray()->GetNumberOfTuples();
      }
    else if (svtkPolyData *pd = dynamic_cast<svtkPolyData*>(ds))
      {
      cellArraySize =
        pd->GetVerts()->GetConnectivityArray()->GetNumberOfTuples() +
        pd->GetLines()->GetConnectivityArray()->GetNumberOfTuples() +
        pd->GetPolys()->GetConnectivityArray()->GetNumberOfTuples() +
        pd->GetStrips()->GetConnectivityArray()->GetNumberOfTuples();
      }

    blockCellArraySize.emplace_back(cellArraySize);
    }

  if (flags.BlockExtentsSet())
    {
    std::array<int,6> ext = {1, 0, 1, 0, 1, 0};
    if (svtkImageData *im = dynamic_cast<svtkImageData*>(ds))
      {
      im->GetExtent(ext.data());
      }
    else if (svtkRectilinearGrid *rg = dynamic_cast<svtkRectilinearGrid*>(ds))
      {
      rg->GetExtent(ext.data());
      }
    else if (svtkStructuredGrid *sg = dynamic_cast<svtkStructuredGrid*>(ds))
      {
      sg->GetExtent(ext.data());
      }
    blockExtents.emplace_back(std::move(ext));

    // TODO -- for AMR meshes extract blocvk level
    }

  if (flags.BlockBoundsSet())
    {
    std::array<double,6> bounds = {
        std::numeric_limits<double>::max(), std::numeric_limits<double>::lowest(),
        std::numeric_limits<double>::max(), std::numeric_limits<double>::lowest(),
        std::numeric_limits<double>::max(), std::numeric_limits<double>::lowest()};

    ds->GetBounds(bounds.data());
    blockBounds.emplace_back(std::move(bounds));
    }

  if (flags.BlockArrayRangeSet())
    {
    std::vector<std::array<double,2>> arrayRange;
    GetArrayMetadata(ds->GetPointData(), arrayRange);
    GetArrayMetadata(ds->GetCellData(), arrayRange);
    blockArrayRange.emplace_back(std::move(arrayRange));
    }

  return 0;
}

// --------------------------------------------------------------------------
int GetBlockMetadata(int rank, int id, svtkDataSet *ds, MeshMetadataPtr metadata)
{
    return GetBlockMetadata(rank, id, ds, metadata->Flags,
      metadata->BlockOwner, metadata->BlockIds, metadata->BlockNumPoints,
      metadata->BlockNumCells, metadata->BlockCellArraySize,
      metadata->BlockExtents, metadata->BlockBounds, metadata->BlockArrayRange);
}

// --------------------------------------------------------------------------
int GetMetadata(MPI_Comm comm, svtkCompositeDataSet *cd, MeshMetadataPtr metadata)
{
  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  svtkOverlappingAMR *amrds = dynamic_cast<svtkOverlappingAMR*>(cd);

  metadata->MeshType = amrds ? SVTK_OVERLAPPING_AMR : SVTK_MULTIBLOCK_DATA_SET;

  // get global metadata
  svtkCompositeDataIterator *cdit = cd->NewIterator();
  if (!cdit->IsDoneWithTraversal())
    {
    svtkDataObject *bobj = cd->GetDataSet(cdit);

    metadata->BlockType = bobj->GetDataObjectType();

    // get array metadata
    if (svtkDataSet *ds = dynamic_cast<svtkDataSet*>(bobj))
      SVTKUtils::GetArrayMetadata(ds, metadata);

    if (svtkPointSet *ps = dynamic_cast<svtkPointSet*>(bobj))
      metadata->CoordinateType = ps->GetPoints()->GetData()->GetDataType();
    }

  // get block metadata
  int numBlocks = 0;
  int numBlocksLocal = 0;
  cdit->SetSkipEmptyNodes(0);

  for (cdit->InitTraversal(); !cdit->IsDoneWithTraversal(); cdit->GoToNextItem())
    {
    numBlocks += 1;

    svtkDataObject *dobj = cd->GetDataSet(cdit);
    int bid = std::max(0, int(cdit->GetCurrentFlatIndex() - 1));

    if (svtkDataSet *ds = dynamic_cast<svtkDataSet*>(dobj))
      {
      numBlocksLocal += 1;

      if (SVTKUtils::GetBlockMetadata(rank, bid, ds, metadata))
        {
        SENSEI_ERROR("Failed to get block metadata for block "
         << cdit->GetCurrentFlatIndex())
        cdit->Delete();
        return -1;
        }
      }
    }

  // set block counts
  metadata->NumBlocks = numBlocks;
  metadata->NumBlocksLocal = {numBlocksLocal};
  cdit->Delete();

  // get global bounds and extents
  if (metadata->Flags.BlockBoundsSet())
    MPIUtils::GlobalBounds(comm, metadata->BlockBounds, metadata->Bounds);

  if (metadata->Flags.BlockExtentsSet())
    MPIUtils::GlobalBounds(comm, metadata->BlockExtents, metadata->Extent);

  if (amrds)
    {
    // global view of block owner is always required
    if (metadata->Flags.BlockDecompSet())
      MPIUtils::GlobalViewV(comm, metadata->BlockOwner);

    // these are all always global views
    metadata->NumLevels = amrds->GetNumberOfLevels();
    metadata->BlockLevel.resize(metadata->NumBlocks);
    metadata->BlockExtents.resize(metadata->NumBlocks);
    metadata->RefRatio.resize(metadata->NumLevels);
    metadata->BlocksPerLevel.resize(metadata->NumLevels);

    int q = 0;
    for (int i = 0; i < metadata->NumLevels; ++i)
      {
      int rr = amrds->GetRefinementRatio(i);
      metadata->RefRatio[i] = {rr, rr, rr};

      int nb = amrds->GetNumberOfDataSets(i);
      metadata->BlocksPerLevel[i] = nb;

      std::vector<double> bounds;
      for (int j = 0; j < nb; ++j, ++q)
        {
        metadata->BlockLevel[q] = i;

        int *pbaq = metadata->BlockExtents[q].data();

        const svtkAMRBox &box = amrds->GetAMRBox(i, j);
        box.GetDimensions(pbaq, pbaq+3);
        }
      }
    }

  return 0;
}

// --------------------------------------------------------------------------
// note: not intended for use on the blocks of a multiblock
int GetMetadata(MPI_Comm comm, svtkDataSet *ds, MeshMetadataPtr metadata)
{
  int rank = 0;
  int nRanks = 1;

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nRanks);

  metadata->MeshType = ds->GetDataObjectType();
  metadata->BlockType = ds->GetDataObjectType();

  SVTKUtils::GetArrayMetadata(ds, metadata);

  if (SVTKUtils::GetBlockMetadata(rank, 0, ds, metadata))
    {
    SENSEI_ERROR("Failed to get block metadata for block " << rank)
    return -1;
    }

  metadata->NumBlocks = nRanks;
  metadata->NumBlocksLocal = {1};

  // get global bounds and extents
  if (metadata->Flags.BlockBoundsSet())
    MPIUtils::GlobalBounds(comm, metadata->BlockBounds, metadata->Bounds);

  if (metadata->Flags.BlockExtentsSet())
    MPIUtils::GlobalBounds(comm, metadata->BlockExtents, metadata->Extent);

  return 0;
}

// --------------------------------------------------------------------------
svtkCompositeDataSetPtr AsCompositeData(MPI_Comm comm,
  svtkDataObject *dobj, bool take)
{
  // make sure we have composite dataset if not create one
  svtkCompositeDataSetPtr cd;
  svtkCompositeDataSet *tmp = nullptr;
  if ((tmp = dynamic_cast<svtkCompositeDataSet*>(dobj)))
    {
    if (take)
      cd.TakeReference(tmp);
    else
      cd = tmp;
    }
  else
    {
    int rank = 0;
    int nRanks = 1;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nRanks);

    svtkMultiBlockDataSet *mb = svtkMultiBlockDataSet::New();
    mb->SetNumberOfBlocks(nRanks);
    mb->SetBlock(rank, dobj);
    if (take)
      dobj->Delete();
    cd.TakeReference(mb);
    }

  return cd;
}

/*
int arrayCpy(void *&wptr, svtkDataArray *da)
{
  unsigned long nt = da->GetNumberOfTuples();
  unsigned int nc = da->GetNumberOfComponents();

  switch (da->GetDataType())
    {
    svtkTemplateMacro(
      if (svtkAOSDataArrayTemplate<SVTK_TT> *aosda =
        dynamic_cast<svtkAOSDataArrayTemplate<SVTK_TT>*>(da))
        {
        unsigned long long nb = nt*nc*sizeof(SVTK_TT);
        SVTK_TT *pda = aosda->GetPointer(0);
        memcpy(wptr, pda, nb);
        ((char*)wptr) += nb;
        }
      else if (svtkSOADataArrayTemplate<SVTK_TT> *soada =
        dynamic_cast<svtkSOADataArrayTemplate<SVTK_TT>*>(da))
        {
        unsigned long long nb = nt*sizeof(SVTK_TT);
        for (unsigned int j = 0; j < nc; ++j)
          {
          SVTK_TT *pda = soada->GetComponentArrayPointer(j);
          memcpy(wptr, pda, nb);
          ((char*)wptr) += nb;
          }
        }
      else
        {
        SENSEI_ERROR("Invalid data array type " << da->GetClassName())
        return -1;
        }
    )
    }

  return 0;
}


// --------------------------------------------------------------------------
int DataArraySerializer::operator()(svtkDataSet *ds)
{
  svtkDataArray *da = m_centering == svtkDataObject::POINT ?
    ds->GetPointData()->GetName(m_name) : ds->GetCellData()->GetName(m_name);

  if (!da || arrayCpy(m_write_ptr, da))
    {
    SENSEI_ERROR("Failed to serialize "
      << GetAttributesName(m_centering) << " data array \""
      << m_name << "\"")
    return -1;
    }

  return 0;
}

// --------------------------------------------------------------------------
int PointsSerializer::operator()(svtkDataSet *ds)
{
  svtkPointSet *ps = dynamic_cast<svtkPointSet*>(ps);
  if (!ps)
    {
    SENSEI_ERROR("Invalid dataset type " << ds->GetClassName())
    return -1;
    }

  svtkDataArray *da = ps->GetPoints()->GetData();
  if (!da || arrayCpy(m_write_ptr, da))
    {
    SENSEI_ERROR("Failed to serialize points")
    return -1;
    }

  return 0;
}

// --------------------------------------------------------------------------
int CellTypesSerializer::operator()(svtkDataSet *ds)
{
  svtkDataArray *da = nullptr;
  if (svtkUnstructuredGrid *ug = dynamic_cast<svtkUnstructuredGrid*>(ds))
    {
    da = ug->GetCellTypesArray();
    if (!da || arrayCpy(m_write_ptr, da))
      {
      SENSEI_ERROR("Failed to serialize cell types")
      return -1;
      }
    }
  else if (svtkPolyData *pd = dynamic_cast<svtkPolyData*>(ds))
    {
    svtkIdType nv = pd->GetNumberOfVerts();
    memset(m_write_ptr, nv, SVTK_VERTEX);
    m_write_ptr += nv;

    svtkIdType nl = pd->GetNumberOfLines();
    memset(m_write_ptr, nl, SVTK_LINE);
    m_write_ptr += nl;

    svtkIdType np = pd->GetNumberOfPolys();
    memset(m_write_ptr, np, SVTK_POLYGON);
    m_write_ptr += np;

    svtkIdType ns = pd->GetNumberOfStrips();
    memset(m_write_ptr, ns, SVTK_TRIANGLE_STRIP)
    m_write_ptr += ns;
    }
  else
    {
    SENSEI_ERROR("Invalid dataset type " << ds->GetClassName())
    return -1;
    }

  return 0;
}

// --------------------------------------------------------------------------
int CellArraySerializer::operator()(svtkDataSet *ds)
{
  svtkDataArray *da = nullptr;
  if (svtkUnstructuredGrid *ug = dynamic_cast<svtkUnstructuredGrid*>(ds))
    {
    da = ug->GetCells()->GetData();
    if (!da || arrayCpy(m_write_ptr, da))
      {
      SENSEI_ERROR("Failed to serialize cells")
      return -1;
      }
    }
  else if (svtkPolyData *pd = dynamic_cast<svtkPolyData*>(ds))
    {
    da = pd->GetVerts()->GetData();
    if (!da || arrayCpy(m_write_ptr, da))
      {
      SENSEI_ERROR("Failed to serialize verts")
      return -1;
      }

    da = pd->GetLines()->GetData();
    if (!da || arrayCpy(m_write_ptr, da))
      {
      SENSEI_ERROR("Failed to serialize lines")
      return -1;
      }

    da = pd->GetLines()->GetPolys();
    if (!da || arrayCpy(m_write_ptr, da))
      {
      SENSEI_ERROR("Failed to serialize polys")
      return -1;
      }

    da = pd->GetStrips()->GetData();
    if (!da || arrayCpy(m_write_ptr, da))
      {
      SENSEI_ERROR("Failed to serialize strips")
      return -1;
      }
    }
  else
    {
    SENSEI_ERROR("Invalid dataset type " << ds->GetClassName())
    return -1;
    }

  return 0;
}

// --------------------------------------------------------------------------
int GetSizesAndOffsets(MPI_Comm comm,
  const sensei::MeshMetadataPtr &md,
  unsigned long long &num_points_total,
  unsigned long long &num_points_local,
  unsigned long long &point_offset_local,
  unsigned long long &num_cells_total,
  unsigned long long &num_cells_local,
  unsigned long long &cell_offset_local,
  unsigned long long &cell_array_size_total,
  unsigned long long &cell_array_size_local,
  unsigned long long &cell_array_offset_local)
{
  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  num_points_total = 0;
  num_points_local = 0;
  point_offset_local = 0;
  num_cells_total = 0;
  num_cells_local = 0;
  cell_offset_local = 0;
  cell_array_size_total = 0;
  cell_array_size_local = 0;
  cell_array_offset_local = 0;

  // calculate the number of points and cells total
  // and the local offset to each type of data
  unsigned int num_blocks = md->NumBlocks;
  for (unsigned int i = 0; i < num_blocks; ++i)
    {
    num_points_total += md->BlockNumPoints[i];
    num_cells_total += md->BlockNumCells[i];

    if ((md->BlockType == SVTK_POLYDATA) || (md->MeshType == SVTK_POLYDATA) ||
     (md->BlockType == SVTK_UNSTRUCTURED_GRID) || (md->MeshType == SVTK_UNSTRUCTURED_GRID))
     {
     cell_array_size_total += md->BlockCellArraySize[i];
     }

    if (md->BlockOwner[i] < rank)
      {
      point_offset_local += md->BlockNumPoints[i];
      cell_offset_local += md->BlockNumCells[i];
      if ((md->BlockType == SVTK_POLYDATA) || (md->MeshType == SVTK_POLYDATA) ||
        (md->BlockType == SVTK_UNSTRUCTURED_GRID) || (md->MeshType == SVTK_UNSTRUCTURED_GRID))
        {
        cell_array_offset_local += md->BlockCellArraySize[i];
        }
      }
    else if (md->BlockOwner[i] == rank)
      {
      num_points_local += md->BlockNumPoints[i]
      num_cells_local += md->BlockNumCells[i]
      if ((md->BlockType == SVTK_POLYDATA) || (md->MeshType == SVTK_POLYDATA) ||
        (md->BlockType == SVTK_UNSTRUCTURED_GRID) || (md->MeshType == SVTK_UNSTRUCTURED_GRID))
        {
        cell_array_size_local += md->BlockCellArraySize[i];
        }
      }
    }

  return 0;
}

// --------------------------------------------------------------------------
int GetLocalGeometrySizes(MPI_Comm comm,
  const sensei::MeshMetadataPtr &md,
  unsigned long long &num_points_local,
  unsigned long long &num_cells_local,
  unsigned long long &cell_array_size_local)
{
  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  unsigned int num_blocks = md->NumBlocks;
  for (unsigned int i = 0; i < num_blocks; ++i)
    {
    if (md->BlockOwner[i] == rank)
      {
      num_points_local += md->BlockNumPoints[i]
      num_cells_local += md->BlockNumCells[i]
      if ((md->BlockType == SVTK_POLYDATA) || (md->MeshType == SVTK_POLYDATA) ||
        (md->BlockType == SVTK_UNSTRUCTURED_GRID) || (md->MeshType == SVTK_UNSTRUCTURED_GRID))
        {
        cell_array_size_local += md->BlockCellArraySize[i];
        }
      }
    }

  return 0;
}
*/

#if defined(ENABLE_VTK_IO)
// helper for creating hexahedron
static
void HexPoints(long cid, const std::array<double,6> &bds, double *pCoords)
{
    long ii = 8*3*cid;
    pCoords[ii     ] = bds[0];
    pCoords[ii + 1 ] = bds[2];
    pCoords[ii + 2 ] = bds[4];

    pCoords[ii + 3 ] = bds[1];
    pCoords[ii + 4 ] = bds[2];
    pCoords[ii + 5 ] = bds[4];

    pCoords[ii + 6 ] = bds[1];
    pCoords[ii + 7 ] = bds[3];
    pCoords[ii + 8 ] = bds[4];

    pCoords[ii + 9 ] = bds[0];
    pCoords[ii + 10] = bds[3];
    pCoords[ii + 11] = bds[4];

    pCoords[ii + 12] = bds[0];
    pCoords[ii + 13] = bds[2];
    pCoords[ii + 14] = bds[5];

    pCoords[ii + 15] = bds[1];
    pCoords[ii + 16] = bds[2];
    pCoords[ii + 17] = bds[5];

    pCoords[ii + 18] = bds[1];
    pCoords[ii + 19] = bds[3];
    pCoords[ii + 20] = bds[5];

    pCoords[ii + 21] = bds[0];
    pCoords[ii + 22] = bds[3];
    pCoords[ii + 23] = bds[5];
}

// helper to make hexahedron cell
static
void HexCell(long cid, unsigned char *pCta, svtkIdType *pClocs, svtkIdType *pCids)
{
    // cell types & location
    pCta[cid] = SVTK_HEXAHEDRON;
    pClocs[cid] = cid*9;

    // cells
    long ii = 8*cid;
    long jj = 9*cid;
    pCids[jj    ] = 8;
    pCids[jj + 1] = ii;
    pCids[jj + 2] = ii + 1;
    pCids[jj + 3] = ii + 2;
    pCids[jj + 4] = ii + 3;
    pCids[jj + 5] = ii + 4;
    pCids[jj + 6] = ii + 5;
    pCids[jj + 7] = ii + 6;
    pCids[jj + 8] = ii + 7;
}
#endif

// --------------------------------------------------------------------------
int WriteDomainDecomp(MPI_Comm comm, const sensei::MeshMetadataPtr &md,
  const std::string fileName)
{
#if !defined(ENABLE_VTK_IO)
    (void)comm;
    (void)md;
    (void)fileName;
    SENSEI_ERROR("SVTK XML I/O capabilites are required by WriteDomainDecomp"
      " but are not present in this build")
    return -1;
#else
  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  if (rank != 0)
    return 0;

  if (!md->GlobalView)
    {
    SENSEI_ERROR("A global view is required")
    return -1;
    }

  bool haveAMR = SVTKUtils::AMR(md);

  int numPoints = 8*(md->NumBlocks + 1);
  int numCells = md->NumBlocks + 1;

  svtkDoubleArray *coords = svtkDoubleArray::New();
  coords->SetNumberOfComponents(3);
  coords->SetNumberOfTuples(numPoints);
  double *pCoords = coords->GetPointer(0);

  svtkIdTypeArray *cids = svtkIdTypeArray::New();
  cids->SetNumberOfTuples(numPoints+numCells);
  svtkIdType *pCids = cids->GetPointer(0);

  svtkUnsignedCharArray *cta = svtkUnsignedCharArray::New();
  cta->SetNumberOfTuples(numCells);
  unsigned char *pCta = cta->GetPointer(0);

  svtkIdTypeArray *clocs = svtkIdTypeArray::New();
  clocs->SetNumberOfTuples(numCells);
  svtkIdType *pClocs = clocs->GetPointer(0);

  svtkDoubleArray *owner = svtkDoubleArray::New();
  owner->SetNumberOfTuples(numCells);
  owner->SetName("BlockOwner");
  double *pOwner = owner->GetPointer(0);

  svtkDoubleArray *ids = svtkDoubleArray::New();
  ids->SetNumberOfTuples(numCells);
  ids->SetName("BlockIds");
  double *pIds = ids->GetPointer(0);

  svtkIntArray *lev = nullptr;
  int *pLev = nullptr;
  if (SVTKUtils::AMR(md))
    {
    lev = svtkIntArray::New();
    lev->SetNumberOfTuples(numCells);
    lev->SetName("BlockLevel");
    pLev = lev->GetPointer(0);
    }

  // define a hex for every block
  for (int i = 0; i < md->NumBlocks; ++i)
    {
    HexPoints(i, md->BlockBounds[i], pCoords);
    HexCell(i, pCta, pClocs, pCids);
    pIds[i] = md->BlockIds[i];
    pOwner[i] = md->BlockOwner[i];
    if (haveAMR)
      pLev[i] = md->BlockLevel[i];
    }

  // and one for an enclosing box
  HexPoints(md->NumBlocks, md->Bounds, pCoords);
  HexCell(md->NumBlocks, pCta, pClocs, pCids);
  pIds[md->NumBlocks] = -2;
  pOwner[md->NumBlocks] = -2;
  if (haveAMR)
    pLev[md->NumBlocks] = -2;

  svtkPoints *pts = svtkPoints::New();
  pts->SetData(coords);
  coords->Delete();

  svtkCellArray *ca = svtkCellArray::New();
  ca->SetCells(numCells, cids);
  cids->Delete();

  svtkUnstructuredGrid *ug = svtkUnstructuredGrid::New();
  ug->SetPoints(pts);
  ug->SetCells(cta, clocs, ca);
  ug->GetCellData()->AddArray(ids);
  ug->GetCellData()->AddArray(owner);
  if (haveAMR)
    {
    ug->GetCellData()->AddArray(lev);
    lev->Delete();
    }

  ids->Delete();
  owner->Delete();
  pts->Delete();
  ca->Delete();

  svtkXMLUnstructuredGridWriter *w = svtkXMLUnstructuredGridWriter::New();
  w->SetInputData(ug);
  w->SetFileName(fileName.c_str());
  w->SetCompressorTypeToNone();
  w->EncodeAppendedDataOff();
  w->Write();

  w->Delete();
  ug->Delete();

  return 0;
#endif
}

}
}
