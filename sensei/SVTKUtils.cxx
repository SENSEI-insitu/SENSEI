#include "senseiConfig.h"
#include "SVTKUtils.h"
#include "MPIUtils.h"
#include "MeshMetadata.h"
#include "Error.h"

#include <hamr_buffer.h>
#include <svtkHAMRDataArray.h>

#include <svtkDataArray.h>
#include <svtkAbstractArray.h>
#include <svtkAOSDataArrayTemplate.h>
#include <svtkSOADataArrayTemplate.h>
#include <svtkIdTypeArray.h>
#include <svtkDoubleArray.h>
#include <svtkFloatArray.h>
#include <svtkCharArray.h>
#include <svtkShortArray.h>
#include <svtkIntArray.h>
#include <svtkLongArray.h>
#include <svtkLongLongArray.h>
#include <svtkUnsignedCharArray.h>
#include <svtkUnsignedShortArray.h>
#include <svtkUnsignedIntArray.h>
#include <svtkUnsignedLongArray.h>
#include <svtkUnsignedLongLongArray.h>
#include <svtkIdTypeArray.h>
#include <svtkAOSDataArrayTemplate.h>
#include <svtkSOADataArrayTemplate.h>
#include <svtkCompositeDataIterator.h>
#include <svtkCompositeDataSet.h>
#include <svtkDataSetAttributes.h>
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
#include <svtkPoints.h>
#include <svtkObjectBase.h>
#include <svtkObject.h>
#include <svtkCellArray.h>
#include <svtkCellTypes.h>
#include <svtkSmartPointer.h>
#include <svtkCallbackCommand.h>
#include <svtkVersionMacros.h>
#include <svtkType.h>
#if defined(SENSEI_ENABLE_VTK_IO)
#include <vtkXMLUnstructuredGridWriter.h>
#endif
#if defined(SENSEI_ENABLE_VTK_CORE)
#include <vtkCommand.h>
#include <vtkCallbackCommand.h>
#include <vtkUnstructuredGrid.h>
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkPointData.h>
#include <vtkCompositeDataIterator.h>
#include <vtkCompositeDataSet.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkStructuredGrid.h>
#include <vtkRectilinearGrid.h>
#include <vtkImageData.h>
#include <vtkUniformGrid.h>
#include <vtkMultiBlockDataSet.h>
#include <vtkOverlappingAMR.h>
#include <vtkNonOverlappingAMR.h>
#include <vtkUniformGridAMR.h>
#include <vtkDataObject.h>
#include <vtkDataSet.h>
#include <vtkPointSet.h>
#include <vtkDataSetAttributes.h>
#include <vtkFieldData.h>
#include <vtkCellData.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkAMRBox.h>
#include <vtkDataArray.h>
#include <vtkAbstractArray.h>
#include <vtkAOSDataArrayTemplate.h>
#include <vtkSOADataArrayTemplate.h>
#include <vtkIdTypeArray.h>
#include <vtkDoubleArray.h>
#include <vtkFloatArray.h>
#include <vtkCharArray.h>
#include <vtkShortArray.h>
#include <vtkIntArray.h>
#include <vtkLongArray.h>
#include <vtkLongLongArray.h>
#include <vtkUnsignedCharArray.h>
#include <vtkUnsignedShortArray.h>
#include <vtkUnsignedIntArray.h>
#include <vtkUnsignedLongArray.h>
#include <vtkUnsignedLongLongArray.h>
#include <vtkIdTypeArray.h>
#include <vtkType.h>
#endif

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
    case SVTK_DOUBLE:
      return sizeof(double);
    case SVTK_CHAR:
      return sizeof(char);
    case SVTK_UNSIGNED_CHAR:
      return sizeof(unsigned char);
    case SVTK_INT:
      return sizeof(int);
    case SVTK_UNSIGNED_INT:
      return sizeof(unsigned int);
    case SVTK_LONG:
      return sizeof(long);
    case SVTK_UNSIGNED_LONG:
      return sizeof(unsigned long);
    case SVTK_LONG_LONG:
      return sizeof(long long);
    case SVTK_UNSIGNED_LONG_LONG:
      return sizeof(unsigned long long);
    case SVTK_ID_TYPE:
      return sizeof(svtkIdType);
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
    case svtkDataObject::CELL:
      return "cell";
    case svtkDataObject::FIELD:
      return "field";
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
    case svtkDataObject::CELL:
      return static_cast<svtkFieldData*>(dobj->GetCellData());
    case svtkDataObject::FIELD:
      return static_cast<svtkFieldData*>(dobj->GetFieldData());
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

#if defined(SENSEI_ENABLE_VTK_IO)
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
#if !defined(SENSEI_ENABLE_VTK_IO)
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

  vtkDoubleArray *coords = vtkDoubleArray::New();
  coords->SetNumberOfComponents(3);
  coords->SetNumberOfTuples(numPoints);
  double *pCoords = coords->GetPointer(0);

  vtkIdTypeArray *cids = vtkIdTypeArray::New();
  cids->SetNumberOfTuples(numPoints+numCells);
  vtkIdType *pCids = cids->GetPointer(0);

  vtkUnsignedCharArray *cta = vtkUnsignedCharArray::New();
  cta->SetNumberOfTuples(numCells);
  unsigned char *pCta = cta->GetPointer(0);

  vtkIdTypeArray *clocs = vtkIdTypeArray::New();
  clocs->SetNumberOfTuples(numCells);
  vtkIdType *pClocs = clocs->GetPointer(0);

  vtkDoubleArray *owner = vtkDoubleArray::New();
  owner->SetNumberOfTuples(numCells);
  owner->SetName("BlockOwner");
  double *pOwner = owner->GetPointer(0);

  vtkDoubleArray *ids = vtkDoubleArray::New();
  ids->SetNumberOfTuples(numCells);
  ids->SetName("BlockIds");
  double *pIds = ids->GetPointer(0);

  vtkIntArray *lev = nullptr;
  int *pLev = nullptr;
  if (SVTKUtils::AMR(md))
    {
    lev = vtkIntArray::New();
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

  vtkPoints *pts = vtkPoints::New();
  pts->SetData(coords);
  coords->Delete();

  vtkCellArray *ca = vtkCellArray::New();
  ca->SetCells(numCells, cids);
  cids->Delete();

  vtkUnstructuredGrid *ug = vtkUnstructuredGrid::New();
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

  vtkXMLUnstructuredGridWriter *w = vtkXMLUnstructuredGridWriter::New();
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


#if defined(SENSEI_ENABLE_VTK_CORE)
/** this will be called when the vtkDataArray is deleted. we release the held
 * reference to the corrsponding svtkDataArray
 */
void svtkObjectDelete(vtkObject *, unsigned long, void *clientData, void *)
{
    svtkObject *heldRef = (svtkObject*)clientData;
    heldRef->Delete();
}

/// type traits for VTK AOS Data Arrays
template <typename T>
struct vtkAOSDataArrayTT
{
    using Type = vtkAOSDataArrayTemplate<T>;
};

#define declareVtkAOSDataArrayTT(_CPP_T, _VTK_T)    \
template<>                                          \
struct vtkAOSDataArrayTT<_CPP_T>                    \
{                                                   \
    using Type = _VTK_T;                            \
};

declareVtkAOSDataArrayTT(char, vtkCharArray)
declareVtkAOSDataArrayTT(short, vtkShortArray)
declareVtkAOSDataArrayTT(int, vtkIntArray)
declareVtkAOSDataArrayTT(long, vtkLongArray)
declareVtkAOSDataArrayTT(long long, vtkLongLongArray)
declareVtkAOSDataArrayTT(unsigned char, vtkUnsignedCharArray)
declareVtkAOSDataArrayTT(unsigned short, vtkUnsignedShortArray)
declareVtkAOSDataArrayTT(unsigned int, vtkUnsignedIntArray)
declareVtkAOSDataArrayTT(unsigned long, vtkUnsignedLongArray)
declareVtkAOSDataArrayTT(unsigned long long, vtkUnsignedLongLongArray)
declareVtkAOSDataArrayTT(float, vtkFloatArray)
declareVtkAOSDataArrayTT(double, vtkDoubleArray)
#endif

// --------------------------------------------------------------------------
vtkTypeInt64Array *VTKObjectFactory::New(svtkTypeInt64Array *daIn)
{
#if !defined(SENSEI_ENABLE_VTK_CORE)
  (void)daIn;
  SENSEI_ERROR("Conversion from SVTK to VTK is not available in this build")
  return nullptr;
#else
  if (!daIn)
  {
    SENSEI_ERROR("Can't create a vtkTypeInt64Array from nullptr")
    return nullptr;
  }

  size_t nTups = daIn->GetNumberOfTuples();
  size_t nComps = daIn->GetNumberOfComponents();

  vtkTypeInt64Array *daOut = vtkTypeInt64Array::New();
  daOut->SetNumberOfComponents(nComps);
  daOut->SetArray(daIn->GetPointer(0), nTups*nComps, 1);
  daOut->SetName(daIn->GetName());

  // hold a reference to the VTK array.
  daIn->Register(nullptr);

  // release the held reference when the SVTK array signals it is finished
  vtkCallbackCommand *cc = vtkCallbackCommand::New();
  cc->SetCallback(svtkObjectDelete);
  cc->SetClientData(daIn);

  daOut->AddObserver(vtkCommand::DeleteEvent, cc);
  cc->Delete();

  return daOut;
#endif
}

// --------------------------------------------------------------------------
vtkTypeInt32Array *VTKObjectFactory::New(svtkTypeInt32Array *daIn)
{
#if !defined(SENSEI_ENABLE_VTK_CORE)
  (void)daIn;
  SENSEI_ERROR("Conversion from SVTK to VTK is not available in this build")
  return nullptr;
#else
  if (!daIn)
  {
    SENSEI_ERROR("Can't create a vtkTypeInt64Array from nullptr")
    return nullptr;
  }

  size_t nTups = daIn->GetNumberOfTuples();
  size_t nComps = daIn->GetNumberOfComponents();

  vtkTypeInt32Array *daOut = vtkTypeInt32Array::New();
  daOut->SetNumberOfComponents(nComps);
  daOut->SetArray(daIn->GetPointer(0), nTups*nComps, 1);
  daOut->SetName(daIn->GetName());

  // hold a reference to the VTK array.
  daIn->Register(nullptr);

  // release the held reference when the SVTK array signals it is finished
  vtkCallbackCommand *cc = vtkCallbackCommand::New();
  cc->SetCallback(svtkObjectDelete);
  cc->SetClientData(daIn);

  daOut->AddObserver(vtkCommand::DeleteEvent, cc);
  cc->Delete();

  return daOut;
#endif
}

// --------------------------------------------------------------------------
vtkDataArray *VTKObjectFactory::New(svtkDataArray *daIn)
{
#if !defined(SENSEI_ENABLE_VTK_CORE)
  (void)daIn;
  SENSEI_ERROR("Conversion from SVTK to VTK is not available in this build")
  return nullptr;
#else
  if (!daIn)
  {
    SENSEI_ERROR("Can't create a vtkDataArray from nullptr")
    return nullptr;
  }

  vtkDataArray *daOut = nullptr;

  size_t nTups = daIn->GetNumberOfTuples();
  size_t nComps = daIn->GetNumberOfComponents();

  switch (daIn->GetDataType())
  {
    svtkTemplateMacro(
    svtkAOSDataArrayTemplate<SVTK_TT> *aosIn =
      dynamic_cast<svtkAOSDataArrayTemplate<SVTK_TT>*>(daIn);

    svtkSOADataArrayTemplate<SVTK_TT> *soaIn =
      dynamic_cast<svtkSOADataArrayTemplate<SVTK_TT>*>(daIn);

    if (aosIn)
    {
      // AOS
      vtkAOSDataArrayTT<SVTK_TT>::Type *aosOut = vtkAOSDataArrayTT<SVTK_TT>::Type::New();
      aosOut->SetNumberOfComponents(nComps);
      aosOut->SetArray(aosIn->GetPointer(0), nTups*nComps, 1);
      daOut = static_cast<vtkDataArray*>(aosOut);
    }
    else if (soaIn)
    {
      // SOA
      vtkSOADataArrayTemplate<SVTK_TT> *soaOut = vtkSOADataArrayTemplate<SVTK_TT>::New();
      soaOut->SetNumberOfComponents(nComps);
      for (size_t j = 0; j < nComps; ++j)
      {
        soaOut->SetArray(j, soaIn->GetComponentArrayPointer(j), nTups, true, true);
      }
      daOut = static_cast<vtkDataArray*>(soaOut);
    }
    );
  }

  if (!daOut)
  {
    SENSEI_ERROR("Failed to create a vtkDataArray from the given "
      << daIn->GetClassName() << " instance")
    return nullptr;
  }

  daOut->SetName(daIn->GetName());

  // hold a reference to the SVTK array.
  daIn->Register(nullptr);

  // release the held reference when the VTK array signals it is finished
  vtkCallbackCommand *cc = vtkCallbackCommand::New();
  cc->SetCallback(svtkObjectDelete);
  cc->SetClientData(daIn);

  daOut->AddObserver(vtkCommand::DeleteEvent, cc);
  cc->Delete();

  return daOut;
#endif
}

// --------------------------------------------------------------------------
vtkCellArray *VTKObjectFactory::New(svtkCellArray *caIn)
{
#if !defined(SENSEI_ENABLE_VTK_CORE)
  (void)caIn;
  SENSEI_ERROR("Conversion from SVTK to VTK is not available in this build")
  return nullptr;
#else
  if (!caIn)
  {
    SENSEI_ERROR("Can't create a vtkCellArray from nullptr")
    return nullptr;
  }

  vtkCellArray *caOut = vtkCellArray::New();

  // zero-copy only works if the array types exactly match, the svtkCellArray
  // has overloads for the common types that will silenlty make copies and not
  // hold a reference to the passed array
  if (caIn->IsStorage64Bit())
  {
    vtkTypeInt64Array *offs = VTKObjectFactory::New(caIn->GetOffsetsArray64());
    if (!offs)
    {
      SENSEI_ERROR("Failed to create the offsets array")
      return nullptr;
    }

    vtkTypeInt64Array *conn = VTKObjectFactory::New(caIn->GetConnectivityArray64());
    if (!conn)
    {
      SENSEI_ERROR("Failed to create the connectivity array")
      return nullptr;
    }

    caOut->SetData(offs, conn);

    offs->Delete();
    conn->Delete();
  }
  else
  {
    vtkTypeInt32Array *offs = VTKObjectFactory::New(caIn->GetOffsetsArray32());
    if (!offs)
    {
      SENSEI_ERROR("Failed to create the offsets array")
      return nullptr;
    }

    vtkTypeInt32Array *conn = VTKObjectFactory::New(caIn->GetConnectivityArray32());
    if (!conn)
    {
      SENSEI_ERROR("Failed to create the connectivity array")
      return nullptr;
    }

    caOut->SetData(offs, conn);

    offs->Delete();
    conn->Delete();
  }

  return caOut;
#endif
}

// --------------------------------------------------------------------------
vtkCellData *VTKObjectFactory::New(svtkCellData *cdIn)
{
#if !defined(SENSEI_ENABLE_VTK_CORE)
  (void)cdIn;
  SENSEI_ERROR("Conversion from SVTK to VTK is not available in this build")
  return nullptr;
#else
  return static_cast<vtkCellData*>
    (VTKObjectFactory::New(static_cast<svtkFieldData*>(cdIn)));
#endif
}

// --------------------------------------------------------------------------
vtkPointData *VTKObjectFactory::New(svtkPointData *pdIn)
{
#if !defined(SENSEI_ENABLE_VTK_CORE)
  (void)pdIn;
  SENSEI_ERROR("Conversion from SVTK to VTK is not available in this build")
  return nullptr;
#else
  return static_cast<vtkPointData*>
    (VTKObjectFactory::New(static_cast<svtkFieldData*>(pdIn)));
#endif
}

// --------------------------------------------------------------------------
vtkFieldData *VTKObjectFactory::New(svtkFieldData *fdIn)
{
#if !defined(SENSEI_ENABLE_VTK_CORE)
  (void)fdIn;
  SENSEI_ERROR("Conversion from SVTK to VTK is not available in this build")
  return nullptr;
#else
  if (!fdIn)
  {
    SENSEI_ERROR("Can't create a vtkFieldData from nullptr")
    return nullptr;
  }

  vtkDataSetAttributes *fdOut = nullptr;
  if (dynamic_cast<svtkCellData*>(fdIn))
  {
    fdOut = vtkCellData::New();
  }
  else if (dynamic_cast<svtkPointData*>(fdIn))
  {
    fdOut = vtkPointData::New();
  }
  else
  {
    SENSEI_ERROR("Failed to create a vtkFieldData from the give "
      << fdIn->GetClassName() << " instance")
    return nullptr;
  }

  int nArrays = fdIn->GetNumberOfArrays();
  for (int i = 0; i < nArrays; ++i)
  {
    vtkDataArray *ai = VTKObjectFactory::New(fdIn->GetArray(i));
    if (!ai)
    {
      SENSEI_ERROR("Array " << i << " was not transfered")
    }
    else
    {
      fdOut->AddArray(ai);
      ai->Delete();
    }
  }

  vtkDataArray *gc = fdOut->GetArray("svtkGhostType");
  if (gc)
  {
    gc->SetName("vtkGhostType");
  }

  return fdOut;
#endif
}

// --------------------------------------------------------------------------
vtkPoints *VTKObjectFactory::New(svtkPoints *ptsIn)
{
#if !defined(SENSEI_ENABLE_VTK_CORE)
  (void)ptsIn;
  SENSEI_ERROR("Conversion from SVTK to VTK is not available in this build")
  return nullptr;
#else
  if (!ptsIn)
  {
    SENSEI_ERROR("Can't create a vtkPoints from nullptr")
    return nullptr;
  }

  vtkDataArray *pts = VTKObjectFactory::New(ptsIn->GetData());
  if (!pts)
  {
    SENSEI_ERROR("Failed to create a vtkPoints from the give "
      << ptsIn->GetClassName() << " instance")
    return nullptr;
  }

  vtkPoints *ptsOut = vtkPoints::New();
  ptsOut->SetData(pts);
  pts->Delete();

  return ptsOut;
#endif
}

// --------------------------------------------------------------------------
vtkImageData *VTKObjectFactory::New(svtkImageData *idIn)
{
#if !defined(SENSEI_ENABLE_VTK_CORE)
  (void)idIn;
  SENSEI_ERROR("Conversion from SVTK to VTK is not available in this build")
  return nullptr;
#else
  if (!idIn)
  {
    SENSEI_ERROR("Can't create a vtkImageData from nullptr")
    return nullptr;
  }

  vtkImageData *idOut = vtkImageData::New();

  // metadata
  idOut->SetExtent(idIn->GetExtent());
  idOut->SetSpacing(idIn->GetSpacing());
  idOut->SetOrigin(idIn->GetOrigin());

  // point data arrays
  vtkPointData *pd = VTKObjectFactory::New(idIn->GetPointData());
  if (!pd)
  {
    SENSEI_ERROR("Failed to transfer vtkPointData")
    idOut->Delete();
    return nullptr;
  }
  idOut->GetPointData()->ShallowCopy(pd);
  pd->Delete();

  // cell data arrays
  vtkCellData *cd = VTKObjectFactory::New(idIn->GetCellData());
  if (!cd)
  {
    SENSEI_ERROR("Failed to transfer vtkCellData")
    idOut->Delete();
    return nullptr;
  }
  idOut->GetCellData()->ShallowCopy(cd);
  cd->Delete();

  return idOut;
#endif
}

// --------------------------------------------------------------------------
vtkUniformGrid *VTKObjectFactory::New(svtkUniformGrid *ugIn)
{
#if !defined(SENSEI_ENABLE_VTK_CORE)
  (void)ugIn;
  SENSEI_ERROR("Conversion from SVTK to VTK is not available in this build")
  return nullptr;
#else
  if (!ugIn)
  {
    SENSEI_ERROR("Can't create a vtkUniformGrid from nullptr")
    return nullptr;
  }

  vtkUniformGrid *ugOut = vtkUniformGrid::New();

  // metadata
  ugOut->SetExtent(ugIn->GetExtent());
  ugOut->SetSpacing(ugIn->GetSpacing());
  ugOut->SetOrigin(ugIn->GetOrigin());

  // point data arrays
  vtkPointData *pd = VTKObjectFactory::New(ugIn->GetPointData());
  if (!pd)
  {
    SENSEI_ERROR("Failed to transfer vtkPointData")
    ugOut->Delete();
    return nullptr;
  }
  ugOut->GetPointData()->ShallowCopy(pd);
  pd->Delete();

  // cell data arrays
  vtkCellData *cd = VTKObjectFactory::New(ugIn->GetCellData());
  if (!cd)
  {
    SENSEI_ERROR("Failed to transfer vtkCellData")
    ugOut->Delete();
    return nullptr;
  }
  ugOut->GetCellData()->ShallowCopy(cd);
  cd->Delete();

  return ugOut;
#endif
}

// --------------------------------------------------------------------------
vtkRectilinearGrid *VTKObjectFactory::New(svtkRectilinearGrid *rgIn)
{
#if !defined(SENSEI_ENABLE_VTK_CORE)
  (void)rgIn;
  SENSEI_ERROR("Conversion from SVTK to VTK is not available in this build")
  return nullptr;
#else
  if (!rgIn)
  {
    SENSEI_ERROR("Can't create a vtkRectilinearGrid from nullptr")
    return nullptr;
  }

  vtkRectilinearGrid *rgOut = vtkRectilinearGrid::New();

  // metadata
  rgOut->SetExtent(rgIn->GetExtent());

  // x coordinates
  vtkDataArray *x = VTKObjectFactory::New(rgIn->GetXCoordinates());
  if (!x)
  {
    SENSEI_ERROR("Failed to transfer x coordinates")
    rgOut->Delete();
    return nullptr;
  }
  rgOut->SetXCoordinates(x);
  x->Delete();

  // y coordinates
  vtkDataArray *y = VTKObjectFactory::New(rgIn->GetYCoordinates());
  if (!y)
  {
    SENSEI_ERROR("Failed to transfer y coordinates")
    rgOut->Delete();
    return nullptr;
  }
  rgOut->SetYCoordinates(y);
  y->Delete();

  // z coordinates
  vtkDataArray *z = VTKObjectFactory::New(rgIn->GetZCoordinates());
  if (!z)
  {
    SENSEI_ERROR("Failed to transfer z coordinates")
    rgOut->Delete();
    return nullptr;
  }
  rgOut->SetZCoordinates(z);
  z->Delete();

  // point data arrays
  vtkPointData *pd = VTKObjectFactory::New(rgIn->GetPointData());
  if (!pd)
  {
    SENSEI_ERROR("Failed to transfer vtkPointData")
    rgOut->Delete();
    return nullptr;
  }
  rgOut->GetPointData()->ShallowCopy(pd);
  pd->Delete();

  // cell data arrays
  vtkCellData *cd = VTKObjectFactory::New(rgIn->GetCellData());
  if (!cd)
  {
    SENSEI_ERROR("Failed to transfer vtkCellData")
    rgOut->Delete();
    return nullptr;
  }
  rgOut->GetCellData()->ShallowCopy(cd);
  cd->Delete();

  return rgOut;
#endif
}

// --------------------------------------------------------------------------
vtkStructuredGrid *VTKObjectFactory::New(svtkStructuredGrid *sgIn)
{
#if !defined(SENSEI_ENABLE_VTK_CORE)
  (void)sgIn;
  SENSEI_ERROR("Conversion from SVTK to VTK is not available in this build")
  return nullptr;
#else
  if (!sgIn)
  {
    SENSEI_ERROR("Can't create a vtkStructuredGrid from nullptr")
    return nullptr;
  }

  vtkStructuredGrid *sgOut = vtkStructuredGrid::New();

  // metadata
  sgOut->SetExtent(sgIn->GetExtent());

  // points
  vtkPoints *pts = VTKObjectFactory::New(sgIn->GetPoints());
  if (!pts)
  {
    SENSEI_ERROR("Failed to transfer points of the svtkStructuredGrid")
    sgOut->Delete();
    return nullptr;
  }

  // point data arrays
  vtkPointData *pd = VTKObjectFactory::New(sgIn->GetPointData());
  if (!pd)
  {
    SENSEI_ERROR("Failed to transfer vtkPointData")
    sgOut->Delete();
    return nullptr;
  }
  sgOut->GetPointData()->ShallowCopy(pd);
  pd->Delete();

  // cell data arrays
  vtkCellData *cd = VTKObjectFactory::New(sgIn->GetCellData());
  if (!cd)
  {
    SENSEI_ERROR("Failed to transfer vtkCellData")
    sgOut->Delete();
    return nullptr;
  }
  sgOut->GetCellData()->ShallowCopy(cd);
  cd->Delete();

  return sgOut;
#endif
}

// --------------------------------------------------------------------------
vtkPolyData *VTKObjectFactory::New(svtkPolyData *pdIn)
{
#if !defined(SENSEI_ENABLE_VTK_CORE)
  (void)pdIn;
  SENSEI_ERROR("Conversion from SVTK to VTK is not available in this build")
  return nullptr;
#else
  if (!pdIn)
  {
    SENSEI_ERROR("Can't create a vtkPolyData from nullptr")
    return nullptr;
  }

  vtkPolyData *pdOut = vtkPolyData::New();

  // points
  vtkPoints *pts = VTKObjectFactory::New(pdIn->GetPoints());
  if (!pts)
  {
    SENSEI_ERROR("Failed to transfer points of the svtkPolyData")
    pdOut->Delete();
    return nullptr;
  }
  pdOut->SetPoints(pts);
  pts->Delete();

  // vert cells
  vtkCellArray *verts = VTKObjectFactory::New(pdIn->GetVerts());
  if (!verts)
  {
    SENSEI_ERROR("Failed to transfer verts of the svtkPolyData")
    pdOut->Delete();
    return nullptr;
  }
  pdOut->SetVerts(verts);
  verts->Delete();

  // line cells
  vtkCellArray *lines = VTKObjectFactory::New(pdIn->GetLines());
  if (!lines)
  {
    SENSEI_ERROR("Failed to transfer lines of the svtkPolyData")
    pdOut->Delete();
    return nullptr;
  }
  pdOut->SetLines(lines);
  lines->Delete();

  // poly cells
  vtkCellArray *polys = VTKObjectFactory::New(pdIn->GetPolys());
  if (!polys)
  {
    SENSEI_ERROR("Failed to transfer polys of the svtkPolyData")
    pdOut->Delete();
    return nullptr;
  }
  pdOut->SetPolys(polys);
  polys->Delete();

  // strip cells
  vtkCellArray *strips = VTKObjectFactory::New(pdIn->GetStrips());
  if (!strips)
  {
    SENSEI_ERROR("Failed to transfer strips of the svtkPolyData")
    pdOut->Delete();
    return nullptr;
  }
  pdOut->SetStrips(strips);
  strips->Delete();

  // point data arrays
  vtkPointData *pd = VTKObjectFactory::New(pdIn->GetPointData());
  if (!pd)
  {
    SENSEI_ERROR("Failed to transfer vtkPointData")
    pdOut->Delete();
    return nullptr;
  }
  pdOut->GetPointData()->ShallowCopy(pd);
  pd->Delete();

  // cell data arrays
  vtkCellData *cd = VTKObjectFactory::New(pdIn->GetCellData());
  if (!cd)
  {
    SENSEI_ERROR("Failed to transfer vtkCellData")
    pdOut->Delete();
    return nullptr;
  }
  pdOut->GetCellData()->ShallowCopy(cd);
  cd->Delete();

  return pdOut;
#endif
}

// --------------------------------------------------------------------------
vtkUnstructuredGrid *VTKObjectFactory::New(svtkUnstructuredGrid *ugIn)
{
#if !defined(SENSEI_ENABLE_VTK_CORE)
  (void)ugIn;
  SENSEI_ERROR("Conversion from SVTK to VTK is not available in this build")
  return nullptr;
#else
  if (!ugIn)
  {
    SENSEI_ERROR("Can't create a vtkUnstructuredGrid from nullptr")
    return nullptr;
  }

  vtkUnstructuredGrid *ugOut = vtkUnstructuredGrid::New();

  // cell types
  vtkUnsignedCharArray *ct =
    dynamic_cast<vtkUnsignedCharArray*>(VTKObjectFactory::New(ugIn->GetCellTypesArray()));

  if (!ct)
  {
    SENSEI_ERROR("Failed to transfer cell types from svtkUnstructuredGrid")
    ugOut->Delete();
    return nullptr;
  }

  // cells
  vtkCellArray *cells = VTKObjectFactory::New(ugIn->GetCells());
  if (!cells)
  {
    SENSEI_ERROR("Failed to transfer cells from svtkUnstructuredGrid")
    ugOut->Delete();
    return nullptr;
  }

  ugOut->SetCells(ct, cells);
  ct->Delete();
  cells->Delete();

  // points
  vtkPoints *pts = VTKObjectFactory::New(ugIn->GetPoints());
  if (!pts)
  {
    SENSEI_ERROR("Failed to transfer points of the svtkPolyData")
    ugOut->Delete();
    return nullptr;
  }
  ugOut->SetPoints(pts);
  pts->Delete();

  // point data arrays
  vtkPointData *pd = VTKObjectFactory::New(ugIn->GetPointData());
  if (!pd)
  {
    SENSEI_ERROR("Failed to transfer vtkPointData")
    ugOut->Delete();
    return nullptr;
  }
  ugOut->GetPointData()->ShallowCopy(pd);
  pd->Delete();

  // cell data arrays
  vtkCellData *cd = VTKObjectFactory::New(ugIn->GetCellData());
  if (!cd)
  {
    SENSEI_ERROR("Failed to transfer vtkCellData")
    ugOut->Delete();
    return nullptr;
  }
  ugOut->GetCellData()->ShallowCopy(cd);
  cd->Delete();

  return ugOut;
#endif
}

// --------------------------------------------------------------------------
vtkMultiBlockDataSet *VTKObjectFactory::New(svtkMultiBlockDataSet *mbIn)
{
#if !defined(SENSEI_ENABLE_VTK_CORE)
  (void)mbIn;
  SENSEI_ERROR("Conversion from SVTK to VTK is not available in this build")
  return nullptr;
#else
  if (!mbIn)
  {
    SENSEI_ERROR("Can't create a vtkMultiBlockDataSet from nullptr")
    return nullptr;
  }

  vtkMultiBlockDataSet *mbOut = vtkMultiBlockDataSet::New();

  // metadata
  int nBlocks = mbIn->GetNumberOfBlocks();

  mbOut->SetNumberOfBlocks(nBlocks);

  for (int i = 0; i < nBlocks; ++i)
  {
    svtkDataSet *dsIn = dynamic_cast<svtkDataSet*>(mbIn->GetBlock(i));
    if (dsIn)
    {
      vtkDataSet *dsOut = VTKObjectFactory::New(dsIn);
      if (!dsOut)
      {
        SENSEI_ERROR("Failed to transfer block "
          << i << " of the svtkMultiBlockDataSet")
        mbOut->Delete();
        return nullptr;
      }
      mbOut->SetBlock(i, dsOut);
      dsOut->Delete();
    }
  }

  return mbOut;
#endif
}

// --------------------------------------------------------------------------
vtkOverlappingAMR *VTKObjectFactory::New(svtkOverlappingAMR *amrIn)
{
#if !defined(SENSEI_ENABLE_VTK_CORE)
  (void)amrIn;
  SENSEI_ERROR("Conversion from SVTK to VTK is not available in this build")
  return nullptr;
#else
  if (!amrIn)
  {
    SENSEI_ERROR("Can't create a vtkOverlappingAMR from nullptr")
    return nullptr;
  }

  vtkOverlappingAMR *amrOut = vtkOverlappingAMR::New();

  // num levels and block
  int nLevels = amrIn->GetNumberOfLevels();

  std::vector<int> nBlocks(nLevels);
  for (int i = 0; i < nLevels; ++i)
    nBlocks[i] = amrIn->GetNumberOfDataSets(i);

  amrOut->Initialize(nLevels, nBlocks.data());

  // origin
  amrOut->SetOrigin(amrIn->GetOrigin());

  // level data
  for (int i = 0; i < nLevels; ++i)
  {
    // level spacing
    double dx[3] = {0.0};
    amrIn->GetSpacing(i, dx);
    amrOut->SetSpacing(i, dx);

    // refinement
    amrOut->SetRefinementRatio(i, amrIn->GetRefinementRatio(i));

    for (int j = 0; j < nBlocks[i]; ++j)
    {
      // origin
      /*double x0[3] = {0.0};
      amrIn->GetOrigin(i, j, x0);
      amrOut->SetOrigin(i, j, x0);*/

      // box
      svtkAMRBox abIn = amrIn->GetAMRBox(i, j);
      vtkAMRBox abOut(abIn.GetLoCorner(), abIn.GetHiCorner());
      amrOut->SetAMRBox(i, j, abOut);

      // gid
      amrOut->SetAMRBlockSourceIndex(i, j, amrIn->GetAMRBlockSourceIndex(i, j));

      // data
      svtkUniformGrid *ugIn = amrIn->GetDataSet(i, j);
      if (ugIn)
      {
        vtkUniformGrid *ugOut = VTKObjectFactory::New(ugIn);
        if (!ugOut)
        {
          SENSEI_ERROR("Failed to convert AMR block at level "
            << i << " block " << j)
          amrOut->Delete();
          return nullptr;
        }
        amrOut->SetDataSet(i, j, ugOut);
        ugOut->Delete();
      }
    }
  }

  return amrOut;
#endif
}

// --------------------------------------------------------------------------
vtkDataObject *VTKObjectFactory::New(svtkDataObject *objIn)
{
#if !defined(SENSEI_ENABLE_VTK_CORE)
  (void)objIn;
  SENSEI_ERROR("Conversion from SVTK to VTK is not available in this build")
  return nullptr;
#else
  if (!objIn)
  {
    SENSEI_ERROR("Can't create a vtkDataObject from nullptr")
    return nullptr;
  }

  svtkDataSet *dsIn = nullptr;
  svtkMultiBlockDataSet *mbIn = nullptr;
  svtkOverlappingAMR *amrIn = nullptr;

  if ((dsIn = dynamic_cast<svtkDataSet*>(objIn)))
  {
    return static_cast<vtkDataObject*>(VTKObjectFactory::New(dsIn));
  }
  else if ((mbIn = dynamic_cast<svtkMultiBlockDataSet*>(objIn)))
  {
    return static_cast<vtkDataObject*>(VTKObjectFactory::New(mbIn));
  }
  else if ((amrIn = dynamic_cast<svtkOverlappingAMR*>(objIn)))
  {
    return static_cast<vtkDataObject*>(VTKObjectFactory::New(amrIn));
  }

  SENSEI_ERROR("Failed to construct a VTK object from the given "
    << objIn->GetClassName() << " instance. Conversion not yet implemented.")

  return nullptr;
#endif
}

// --------------------------------------------------------------------------
vtkDataSet *VTKObjectFactory::New(svtkDataSet *dsIn)
{
#if !defined(SENSEI_ENABLE_VTK_CORE)
  (void)dsIn;
  SENSEI_ERROR("Conversion from SVTK to VTK is not available in this build")
  return nullptr;
#else
  if (!dsIn)
  {
    SENSEI_ERROR("Can't create a vtkDataSet from nullptr")
    return nullptr;
  }

  svtkImageData *idIn = nullptr;
  svtkUniformGrid *ungIn = nullptr;
  svtkRectilinearGrid *rgIn = nullptr;
  svtkStructuredGrid *sgIn = nullptr;
  svtkPolyData *pdIn = nullptr;
  svtkUnstructuredGrid *ugIn = nullptr;

  if ((idIn = dynamic_cast<svtkImageData*>(dsIn)))
  {
    return VTKObjectFactory::New(idIn);
  }
  else if ((ungIn = dynamic_cast<svtkUniformGrid*>(dsIn)))
  {
    return VTKObjectFactory::New(ungIn);
  }
  else if ((rgIn = dynamic_cast<svtkRectilinearGrid*>(dsIn)))
  {
    return VTKObjectFactory::New(rgIn);
  }
  else if ((sgIn = dynamic_cast<svtkStructuredGrid*>(dsIn)))
  {
    return VTKObjectFactory::New(sgIn);
  }
  else if ((pdIn = dynamic_cast<svtkPolyData*>(dsIn)))
  {
    return VTKObjectFactory::New(pdIn);
  }
  else if ((ugIn = dynamic_cast<svtkUnstructuredGrid*>(dsIn)))
  {
    return VTKObjectFactory::New(ugIn);
  }

  SENSEI_ERROR("Failed to construct a VTK object from the given "
    << dsIn->GetClassName() << " instance. Conversion not yet implemented.")

  return nullptr;
#endif
}



#if defined(SENSEI_ENABLE_VTK_CORE)
/** this will be called when the svtkDataArray is deleted. we release the held
 * reference to the corrsponding vtkDataArray
 */
void vtkObjectDelete(svtkObject *, unsigned long, void *clientData, void *)
{
    vtkObject *heldRef = (vtkObject*)clientData;
    heldRef->Delete();
}

/// type traits for SVTK AOS Data Arrays
template <typename T>
struct svtkAOSDataArrayTT
{
    using Type = svtkAOSDataArrayTemplate<T>;
};

#define declareSvtkAOSDataArrayTT(_CPP_T, _SVTK_T)  \
template<>                                          \
struct svtkAOSDataArrayTT<_CPP_T>                   \
{                                                   \
    using Type = _SVTK_T;                           \
};

declareSvtkAOSDataArrayTT(char, svtkCharArray)
declareSvtkAOSDataArrayTT(short, svtkShortArray)
declareSvtkAOSDataArrayTT(int, svtkIntArray)
declareSvtkAOSDataArrayTT(long, svtkLongArray)
declareSvtkAOSDataArrayTT(long long, svtkLongLongArray)
declareSvtkAOSDataArrayTT(unsigned char, svtkUnsignedCharArray)
declareSvtkAOSDataArrayTT(unsigned short, svtkUnsignedShortArray)
declareSvtkAOSDataArrayTT(unsigned int, svtkUnsignedIntArray)
declareSvtkAOSDataArrayTT(unsigned long, svtkUnsignedLongArray)
declareSvtkAOSDataArrayTT(unsigned long long, svtkUnsignedLongLongArray)
declareSvtkAOSDataArrayTT(float, svtkFloatArray)
declareSvtkAOSDataArrayTT(double, svtkDoubleArray)
#endif

// --------------------------------------------------------------------------
svtkTypeInt64Array *SVTKObjectFactory::New(vtkTypeInt64Array *daIn)
{
#if !defined(SENSEI_ENABLE_VTK_CORE)
  (void)daIn;
  SENSEI_ERROR("Conversion from VTK to SVTK is not available in this build")
  return nullptr;
#else
  if (!daIn)
  {
    SENSEI_ERROR("Can't create a svtkTypeInt64Array from nullptr")
    return nullptr;
  }

  size_t nTups = daIn->GetNumberOfTuples();
  size_t nComps = daIn->GetNumberOfComponents();

  svtkTypeInt64Array *daOut = svtkTypeInt64Array::New();
  daOut->SetNumberOfComponents(nComps);
  daOut->SetArray(daIn->GetPointer(0), nTups*nComps, 1);
  daOut->SetName(daIn->GetName());

  // hold a reference to the VTK array.
  daIn->Register(nullptr);

  // release the held reference when the SVTK array signals it is finished
  svtkCallbackCommand *cc = svtkCallbackCommand::New();
  cc->SetCallback(vtkObjectDelete);
  cc->SetClientData(daIn);

  daOut->AddObserver(svtkCommand::DeleteEvent, cc);
  cc->Delete();

  return daOut;
#endif
}

// --------------------------------------------------------------------------
svtkTypeInt32Array *SVTKObjectFactory::New(vtkTypeInt32Array *daIn)
{
#if !defined(SENSEI_ENABLE_VTK_CORE)
  (void)daIn;
  SENSEI_ERROR("Conversion from VTK to SVTK is not available in this build")
  return nullptr;
#else
  if (!daIn)
  {
    SENSEI_ERROR("Can't create a svtkTypeInt64Array from nullptr")
    return nullptr;
  }

  size_t nTups = daIn->GetNumberOfTuples();
  size_t nComps = daIn->GetNumberOfComponents();

  svtkTypeInt32Array *daOut = svtkTypeInt32Array::New();
  daOut->SetNumberOfComponents(nComps);
  daOut->SetArray(daIn->GetPointer(0), nTups*nComps, 1);
  daOut->SetName(daIn->GetName());

  // hold a reference to the VTK array.
  daIn->Register(nullptr);

  // release the held reference when the SVTK array signals it is finished
  svtkCallbackCommand *cc = svtkCallbackCommand::New();
  cc->SetCallback(vtkObjectDelete);
  cc->SetClientData(daIn);

  daOut->AddObserver(svtkCommand::DeleteEvent, cc);
  cc->Delete();

  return daOut;
#endif
}

// --------------------------------------------------------------------------
svtkDataArray *SVTKObjectFactory::New(vtkDataArray *daIn)
{
#if !defined(SENSEI_ENABLE_VTK_CORE)
  (void)daIn;
  SENSEI_ERROR("Conversion from VTK to SVTK is not available in this build")
  return nullptr;
#else
  if (!daIn)
  {
    SENSEI_ERROR("Can't create a svtkDataArray from nullptr")
    return nullptr;
  }

  svtkDataArray *daOut = nullptr;

  size_t nTups = daIn->GetNumberOfTuples();
  size_t nComps = daIn->GetNumberOfComponents();

  switch (daIn->GetDataType())
  {
    vtkTemplateMacro(
    vtkAOSDataArrayTemplate<VTK_TT> *aosIn =
      dynamic_cast<vtkAOSDataArrayTemplate<VTK_TT>*>(daIn);

    vtkSOADataArrayTemplate<VTK_TT> *soaIn =
      dynamic_cast<vtkSOADataArrayTemplate<VTK_TT>*>(daIn);

    if (aosIn)
    {
      // AOS
      svtkAOSDataArrayTT<VTK_TT>::Type *aosOut = svtkAOSDataArrayTT<VTK_TT>::Type::New();
      aosOut->SetNumberOfComponents(nComps);
      aosOut->SetArray(aosIn->GetPointer(0), nTups*nComps, 1);
      daOut = static_cast<svtkDataArray*>(aosOut);
    }
    else if (soaIn)
    {
      // SOA
      svtkSOADataArrayTemplate<VTK_TT> *soaOut = svtkSOADataArrayTemplate<VTK_TT>::New();
      soaOut->SetNumberOfComponents(nComps);
      for (size_t j = 0; j < nComps; ++j)
      {
        soaOut->SetArray(j, soaIn->GetComponentArrayPointer(j), nTups, true, true);
      }
      daOut = static_cast<svtkDataArray*>(soaOut);
    }
    );
  }

  if (!daOut)
  {
    SENSEI_ERROR("Failed to create a svtkDataArray from the given "
      << daIn->GetClassName() << " instance")
    return nullptr;
  }

  daOut->SetName(daIn->GetName());

  // hold a reference to the VTK array.
  daIn->Register(nullptr);

  // release the held reference when the SVTK array signals it is finished
  svtkCallbackCommand *cc = svtkCallbackCommand::New();
  cc->SetCallback(vtkObjectDelete);
  cc->SetClientData(daIn);

  daOut->AddObserver(svtkCommand::DeleteEvent, cc);
  cc->Delete();

  return daOut;
#endif
}

// --------------------------------------------------------------------------
svtkCellArray *SVTKObjectFactory::New(vtkCellArray *caIn)
{
#if !defined(SENSEI_ENABLE_VTK_CORE)
  (void)caIn;
  SENSEI_ERROR("Conversion from VTK to SVTK is not available in this build")
  return nullptr;
#else
  if (!caIn)
  {
    SENSEI_ERROR("Can't create a svtkCellArray from nullptr")
    return nullptr;
  }

  svtkCellArray *caOut = svtkCellArray::New();

  // zero-copy only works if the array types exactly match, the vtkCellArray
  // has overloads for the common types that will silenlty make copies and not
  // hold a reference to the passed array
  if (caIn->IsStorage64Bit())
  {
    svtkTypeInt64Array *offs = SVTKObjectFactory::New(caIn->GetOffsetsArray64());
    if (!offs)
    {
      SENSEI_ERROR("Failed to create the offsets array")
      return nullptr;
    }

    svtkTypeInt64Array *conn = SVTKObjectFactory::New(caIn->GetConnectivityArray64());
    if (!conn)
    {
      SENSEI_ERROR("Failed to create the connectivity array")
      return nullptr;
    }

    caOut->SetData(offs, conn);

    offs->Delete();
    conn->Delete();
  }
  else
  {
    svtkTypeInt32Array *offs = SVTKObjectFactory::New(caIn->GetOffsetsArray32());
    if (!offs)
    {
      SENSEI_ERROR("Failed to create the offsets array")
      return nullptr;
    }

    svtkTypeInt32Array *conn = SVTKObjectFactory::New(caIn->GetConnectivityArray32());
    if (!conn)
    {
      SENSEI_ERROR("Failed to create the connectivity array")
      return nullptr;
    }

    caOut->SetData(offs, conn);

    offs->Delete();
    conn->Delete();
  }

  return caOut;
#endif
}

// --------------------------------------------------------------------------
svtkCellData *SVTKObjectFactory::New(vtkCellData *cdIn)
{
#if !defined(SENSEI_ENABLE_VTK_CORE)
  (void)cdIn;
  SENSEI_ERROR("Conversion from VTK to SVTK is not available in this build")
  return nullptr;
#else
  return static_cast<svtkCellData*>
    (SVTKObjectFactory::New(static_cast<vtkFieldData*>(cdIn)));
#endif
}

// --------------------------------------------------------------------------
svtkPointData *SVTKObjectFactory::New(vtkPointData *pdIn)
{
#if !defined(SENSEI_ENABLE_VTK_CORE)
  (void)pdIn;
  SENSEI_ERROR("Conversion from VTK to SVTK is not available in this build")
  return nullptr;
#else
  return static_cast<svtkPointData*>
    (SVTKObjectFactory::New(static_cast<vtkFieldData*>(pdIn)));
#endif
}

// --------------------------------------------------------------------------
svtkFieldData *SVTKObjectFactory::New(vtkFieldData *fdIn)
{
#if !defined(SENSEI_ENABLE_VTK_CORE)
  (void)fdIn;
  SENSEI_ERROR("Conversion from VTK to SVTK is not available in this build")
  return nullptr;
#else
  if (!fdIn)
  {
    SENSEI_ERROR("Can't create a svtkFieldData from nullptr")
    return nullptr;
  }

  svtkDataSetAttributes *fdOut = nullptr;
  if (dynamic_cast<vtkCellData*>(fdIn))
  {
    fdOut = svtkCellData::New();
  }
  else if (dynamic_cast<vtkPointData*>(fdIn))
  {
    fdOut = svtkPointData::New();
  }
  else
  {
    SENSEI_ERROR("Failed to create a svtkFieldData from the give "
      << fdIn->GetClassName() << " instance")
    return nullptr;
  }

  int nArrays = fdIn->GetNumberOfArrays();
  for (int i = 0; i < nArrays; ++i)
  {
    svtkDataArray *ai = SVTKObjectFactory::New(fdIn->GetArray(i));
    if (!ai)
    {
      SENSEI_ERROR("Array " << i << " was not transfered")
    }
    else
    {
      fdOut->AddArray(ai);
      ai->Delete();
    }
  }

  svtkDataArray *gc = fdOut->GetArray("vtkGhostType");
  if (gc)
  {
    gc->SetName("svtkGhostType");
  }

  return fdOut;
#endif
}

// --------------------------------------------------------------------------
svtkPoints *SVTKObjectFactory::New(vtkPoints *ptsIn)
{
#if !defined(SENSEI_ENABLE_VTK_CORE)
  (void)ptsIn;
  SENSEI_ERROR("Conversion from VTK to SVTK is not available in this build")
  return nullptr;
#else
  if (!ptsIn)
  {
    SENSEI_ERROR("Can't create a svtkPoints from nullptr")
    return nullptr;
  }

  svtkDataArray *pts = SVTKObjectFactory::New(ptsIn->GetData());
  if (!pts)
  {
    SENSEI_ERROR("Failed to create a svtkPoints from the give "
      << ptsIn->GetClassName() << " instance")
    return nullptr;
  }

  svtkPoints *ptsOut = svtkPoints::New();
  ptsOut->SetData(pts);
  pts->Delete();

  return ptsOut;
#endif
}

// --------------------------------------------------------------------------
svtkImageData *SVTKObjectFactory::New(vtkImageData *idIn)
{
#if !defined(SENSEI_ENABLE_VTK_CORE)
  (void)idIn;
  SENSEI_ERROR("Conversion from VTK to SVTK is not available in this build")
  return nullptr;
#else
  if (!idIn)
  {
    SENSEI_ERROR("Can't create a svtkImageData from nullptr")
    return nullptr;
  }

  svtkImageData *idOut = svtkImageData::New();

  // metadata
  idOut->SetExtent(idIn->GetExtent());
  idOut->SetSpacing(idIn->GetSpacing());
  idOut->SetOrigin(idIn->GetOrigin());

  // point data arrays
  svtkPointData *pd = SVTKObjectFactory::New(idIn->GetPointData());
  if (!pd)
  {
    SENSEI_ERROR("Failed to transfer svtkPointData")
    idOut->Delete();
    return nullptr;
  }
  idOut->GetPointData()->ShallowCopy(pd);
  pd->Delete();

  // cell data arrays
  svtkCellData *cd = SVTKObjectFactory::New(idIn->GetCellData());
  if (!cd)
  {
    SENSEI_ERROR("Failed to transfer svtkCellData")
    idOut->Delete();
    return nullptr;
  }
  idOut->GetCellData()->ShallowCopy(cd);
  cd->Delete();

  return idOut;
#endif
}

// --------------------------------------------------------------------------
svtkUniformGrid *SVTKObjectFactory::New(vtkUniformGrid *ugIn)
{
#if !defined(SENSEI_ENABLE_VTK_CORE)
  (void)ugIn;
  SENSEI_ERROR("Conversion from VTK to SVTK is not available in this build")
  return nullptr;
#else
  if (!ugIn)
  {
    SENSEI_ERROR("Can't create a svtkUniformGrid from nullptr")
    return nullptr;
  }

  svtkUniformGrid *ugOut = svtkUniformGrid::New();

  // metadata
  ugOut->SetExtent(ugIn->GetExtent());
  ugOut->SetSpacing(ugIn->GetSpacing());
  ugOut->SetOrigin(ugIn->GetOrigin());

  // point data arrays
  svtkPointData *pd = SVTKObjectFactory::New(ugIn->GetPointData());
  if (!pd)
  {
    SENSEI_ERROR("Failed to transfer svtkPointData")
    ugOut->Delete();
    return nullptr;
  }
  ugOut->GetPointData()->ShallowCopy(pd);
  pd->Delete();

  // cell data arrays
  svtkCellData *cd = SVTKObjectFactory::New(ugIn->GetCellData());
  if (!cd)
  {
    SENSEI_ERROR("Failed to transfer svtkCellData")
    ugOut->Delete();
    return nullptr;
  }
  ugOut->GetCellData()->ShallowCopy(cd);
  cd->Delete();

  return ugOut;
#endif
}

// --------------------------------------------------------------------------
svtkRectilinearGrid *SVTKObjectFactory::New(vtkRectilinearGrid *rgIn)
{
#if !defined(SENSEI_ENABLE_VTK_CORE)
  (void)rgIn;
  SENSEI_ERROR("Conversion from VTK to SVTK is not available in this build")
  return nullptr;
#else
  if (!rgIn)
  {
    SENSEI_ERROR("Can't create a svtkRectilinearGrid from nullptr")
    return nullptr;
  }

  svtkRectilinearGrid *rgOut = svtkRectilinearGrid::New();

  // metadata
  rgOut->SetExtent(rgIn->GetExtent());

  // x coordinates
  svtkDataArray *x = SVTKObjectFactory::New(rgIn->GetXCoordinates());
  if (!x)
  {
    SENSEI_ERROR("Failed to transfer x coordinates")
    rgOut->Delete();
    return nullptr;
  }
  rgOut->SetXCoordinates(x);
  x->Delete();

  // y coordinates
  svtkDataArray *y = SVTKObjectFactory::New(rgIn->GetYCoordinates());
  if (!y)
  {
    SENSEI_ERROR("Failed to transfer y coordinates")
    rgOut->Delete();
    return nullptr;
  }
  rgOut->SetYCoordinates(y);
  y->Delete();

  // z coordinates
  svtkDataArray *z = SVTKObjectFactory::New(rgIn->GetZCoordinates());
  if (!z)
  {
    SENSEI_ERROR("Failed to transfer z coordinates")
    rgOut->Delete();
    return nullptr;
  }
  rgOut->SetZCoordinates(z);
  z->Delete();

  // point data arrays
  svtkPointData *pd = SVTKObjectFactory::New(rgIn->GetPointData());
  if (!pd)
  {
    SENSEI_ERROR("Failed to transfer svtkPointData")
    rgOut->Delete();
    return nullptr;
  }
  rgOut->GetPointData()->ShallowCopy(pd);
  pd->Delete();

  // cell data arrays
  svtkCellData *cd = SVTKObjectFactory::New(rgIn->GetCellData());
  if (!cd)
  {
    SENSEI_ERROR("Failed to transfer svtkCellData")
    rgOut->Delete();
    return nullptr;
  }
  rgOut->GetCellData()->ShallowCopy(cd);
  cd->Delete();

  return rgOut;
#endif
}

// --------------------------------------------------------------------------
svtkStructuredGrid *SVTKObjectFactory::New(vtkStructuredGrid *sgIn)
{
#if !defined(SENSEI_ENABLE_VTK_CORE)
  (void)sgIn;
  SENSEI_ERROR("Conversion from VTK to SVTK is not available in this build")
  return nullptr;
#else
  if (!sgIn)
  {
    SENSEI_ERROR("Can't create a svtkStructuredGrid from nullptr")
    return nullptr;
  }

  svtkStructuredGrid *sgOut = svtkStructuredGrid::New();

  // metadata
  sgOut->SetExtent(sgIn->GetExtent());

  // points
  svtkPoints *pts = SVTKObjectFactory::New(sgIn->GetPoints());
  if (!pts)
  {
    SENSEI_ERROR("Failed to transfer points of the vtkStructuredGrid")
    sgOut->Delete();
    return nullptr;
  }

  // point data arrays
  svtkPointData *pd = SVTKObjectFactory::New(sgIn->GetPointData());
  if (!pd)
  {
    SENSEI_ERROR("Failed to transfer svtkPointData")
    sgOut->Delete();
    return nullptr;
  }
  sgOut->GetPointData()->ShallowCopy(pd);
  pd->Delete();

  // cell data arrays
  svtkCellData *cd = SVTKObjectFactory::New(sgIn->GetCellData());
  if (!cd)
  {
    SENSEI_ERROR("Failed to transfer svtkCellData")
    sgOut->Delete();
    return nullptr;
  }
  sgOut->GetCellData()->ShallowCopy(cd);
  cd->Delete();

  return sgOut;
#endif
}

// --------------------------------------------------------------------------
svtkPolyData *SVTKObjectFactory::New(vtkPolyData *pdIn)
{
#if !defined(SENSEI_ENABLE_VTK_CORE)
  (void)pdIn;
  SENSEI_ERROR("Conversion from VTK to SVTK is not available in this build")
  return nullptr;
#else
  if (!pdIn)
  {
    SENSEI_ERROR("Can't create a svtkPolyData from nullptr")
    return nullptr;
  }

  svtkPolyData *pdOut = svtkPolyData::New();

  // points
  svtkPoints *pts = SVTKObjectFactory::New(pdIn->GetPoints());
  if (!pts)
  {
    SENSEI_ERROR("Failed to transfer points of the vtkPolyData")
    pdOut->Delete();
    return nullptr;
  }
  pdOut->SetPoints(pts);
  pts->Delete();

  // vert cells
  svtkCellArray *verts = SVTKObjectFactory::New(pdIn->GetVerts());
  if (!verts)
  {
    SENSEI_ERROR("Failed to transfer verts of the vtkPolyData")
    pdOut->Delete();
    return nullptr;
  }
  pdOut->SetVerts(verts);
  verts->Delete();

  // line cells
  svtkCellArray *lines = SVTKObjectFactory::New(pdIn->GetLines());
  if (!lines)
  {
    SENSEI_ERROR("Failed to transfer lines of the vtkPolyData")
    pdOut->Delete();
    return nullptr;
  }
  pdOut->SetLines(lines);
  lines->Delete();

  // poly cells
  svtkCellArray *polys = SVTKObjectFactory::New(pdIn->GetPolys());
  if (!polys)
  {
    SENSEI_ERROR("Failed to transfer polys of the vtkPolyData")
    pdOut->Delete();
    return nullptr;
  }
  pdOut->SetPolys(polys);
  polys->Delete();

  // strip cells
  svtkCellArray *strips = SVTKObjectFactory::New(pdIn->GetStrips());
  if (!strips)
  {
    SENSEI_ERROR("Failed to transfer strips of the vtkPolyData")
    pdOut->Delete();
    return nullptr;
  }
  pdOut->SetStrips(strips);
  strips->Delete();

  // point data arrays
  svtkPointData *pd = SVTKObjectFactory::New(pdIn->GetPointData());
  if (!pd)
  {
    SENSEI_ERROR("Failed to transfer svtkPointData")
    pdOut->Delete();
    return nullptr;
  }
  pdOut->GetPointData()->ShallowCopy(pd);
  pd->Delete();

  // cell data arrays
  svtkCellData *cd = SVTKObjectFactory::New(pdIn->GetCellData());
  if (!cd)
  {
    SENSEI_ERROR("Failed to transfer svtkCellData")
    pdOut->Delete();
    return nullptr;
  }
  pdOut->GetCellData()->ShallowCopy(cd);
  cd->Delete();

  return pdOut;
#endif
}

// --------------------------------------------------------------------------
svtkUnstructuredGrid *SVTKObjectFactory::New(vtkUnstructuredGrid *ugIn)
{
#if !defined(SENSEI_ENABLE_VTK_CORE)
  (void)ugIn;
  SENSEI_ERROR("Conversion from VTK to SVTK is not available in this build")
  return nullptr;
#else
  if (!ugIn)
  {
    SENSEI_ERROR("Can't create a svtkUnstructuredGrid from nullptr")
    return nullptr;
  }

  svtkUnstructuredGrid *ugOut = svtkUnstructuredGrid::New();

  // cell types
  svtkUnsignedCharArray *ct =
    dynamic_cast<svtkUnsignedCharArray*>(SVTKObjectFactory::New(ugIn->GetCellTypesArray()));

  if (!ct)
  {
    SENSEI_ERROR("Failed to transfer cell types from vtkUnstructuredGrid")
    ugOut->Delete();
    return nullptr;
  }

  // cells
  svtkCellArray *cells = SVTKObjectFactory::New(ugIn->GetCells());
  if (!cells)
  {
    SENSEI_ERROR("Failed to transfer cells from vtkUnstructuredGrid")
    ugOut->Delete();
    return nullptr;
  }

  ugOut->SetCells(ct, cells);
  ct->Delete();
  cells->Delete();

  // points
  svtkPoints *pts = SVTKObjectFactory::New(ugIn->GetPoints());
  if (!pts)
  {
    SENSEI_ERROR("Failed to transfer points of the vtkPolyData")
    ugOut->Delete();
    return nullptr;
  }
  ugOut->SetPoints(pts);
  pts->Delete();

  // point data arrays
  svtkPointData *pd = SVTKObjectFactory::New(ugIn->GetPointData());
  if (!pd)
  {
    SENSEI_ERROR("Failed to transfer svtkPointData")
    ugOut->Delete();
    return nullptr;
  }
  ugOut->GetPointData()->ShallowCopy(pd);
  pd->Delete();

  // cell data arrays
  svtkCellData *cd = SVTKObjectFactory::New(ugIn->GetCellData());
  if (!cd)
  {
    SENSEI_ERROR("Failed to transfer svtkCellData")
    ugOut->Delete();
    return nullptr;
  }
  ugOut->GetCellData()->ShallowCopy(cd);
  cd->Delete();

  return ugOut;
#endif
}

// --------------------------------------------------------------------------
svtkMultiBlockDataSet *SVTKObjectFactory::New(vtkMultiBlockDataSet *mbIn)
{
#if !defined(SENSEI_ENABLE_VTK_CORE)
  (void)mbIn;
  SENSEI_ERROR("Conversion from VTK to SVTK is not available in this build")
  return nullptr;
#else
  if (!mbIn)
  {
    SENSEI_ERROR("Can't create a svtkMultiBlockDataSet from nullptr")
    return nullptr;
  }

  svtkMultiBlockDataSet *mbOut = svtkMultiBlockDataSet::New();

  // metadata
  int nBlocks = mbIn->GetNumberOfBlocks();

  mbOut->SetNumberOfBlocks(nBlocks);

  for (int i = 0; i < nBlocks; ++i)
  {
    vtkDataSet *dsIn = dynamic_cast<vtkDataSet*>(mbIn->GetBlock(i));
    if (dsIn)
    {
      svtkDataSet *dsOut = SVTKObjectFactory::New(dsIn);
      if (!dsOut)
      {
        SENSEI_ERROR("Failed to transfer block "
          << i << " of the vtkMultiBlockDataSet")
        mbOut->Delete();
        return nullptr;
      }
      mbOut->SetBlock(i, dsOut);
      dsOut->Delete();
    }
  }

  return mbOut;
#endif
}

// --------------------------------------------------------------------------
svtkOverlappingAMR *SVTKObjectFactory::New(vtkOverlappingAMR *amrIn)
{
#if !defined(SENSEI_ENABLE_VTK_CORE)
  (void)amrIn;
  SENSEI_ERROR("Conversion from VTK to SVTK is not available in this build")
  return nullptr;
#else
  if (!amrIn)
  {
    SENSEI_ERROR("Can't create a svtkOverlappingAMR from nullptr")
    return nullptr;
  }

  svtkOverlappingAMR *amrOut = svtkOverlappingAMR::New();

  // num levels and block
  int nLevels = amrIn->GetNumberOfLevels();

  std::vector<int> nBlocks(nLevels);
  for (int i = 0; i < nLevels; ++i)
    nBlocks[i] = amrIn->GetNumberOfDataSets(i);

  amrOut->Initialize(nLevels, nBlocks.data());

  // origin
  amrOut->SetOrigin(amrIn->GetOrigin());

  // level data
  for (int i = 0; i < nLevels; ++i)
  {
    // level origin
    /*double x0[3] = {0.0};
    amrIn->GetOrigin(i, x0);
    amrOut->SetOrigin(i, x0);*/

    // level spacing
    double dx[3] = {0.0};
    amrIn->GetSpacing(i, dx);
    amrOut->SetSpacing(i, dx);

    // refinement
    amrOut->SetRefinementRatio(i, amrIn->GetRefinementRatio(i));

    for (int j = 0; j < nBlocks[i]; ++j)
    {
      // box
      vtkAMRBox abIn = amrIn->GetAMRBox(i, j);
      svtkAMRBox abOut(abIn.GetLoCorner(), abIn.GetHiCorner());
      amrOut->SetAMRBox(i, j, abOut);

      // gid
      amrOut->SetAMRBlockSourceIndex(i, j, amrIn->GetAMRBlockSourceIndex(i, j));

      // data
      vtkUniformGrid *ugIn = amrIn->GetDataSet(i, j);
      if (ugIn)
      {
        svtkUniformGrid *ugOut = SVTKObjectFactory::New(ugIn);
        if (!ugOut)
        {
          SENSEI_ERROR("Failed to convert AMR block at level "
            << i << " block " << j)
          amrOut->Delete();
          return nullptr;
        }
        amrOut->SetDataSet(i, j, ugOut);
        ugOut->Delete();
      }
    }
  }

  return amrOut;
#endif
}

// --------------------------------------------------------------------------
svtkDataObject *SVTKObjectFactory::New(vtkDataObject *objIn)
{
#if !defined(SENSEI_ENABLE_VTK_CORE)
  (void)objIn;
  SENSEI_ERROR("Conversion from VTK to SVTK is not available in this build")
  return nullptr;
#else
  if (!objIn)
  {
    SENSEI_ERROR("Can't create a svtkDataObject from nullptr")
    return nullptr;
  }

  vtkDataSet *dsIn = nullptr;
  vtkMultiBlockDataSet *mbIn = nullptr;
  vtkOverlappingAMR *amrIn = nullptr;

  if ((dsIn = dynamic_cast<vtkDataSet*>(objIn)))
  {
    return SVTKObjectFactory::New(dsIn);
  }
  else if ((mbIn = dynamic_cast<vtkMultiBlockDataSet*>(objIn)))
  {
    return SVTKObjectFactory::New(mbIn);
  }
  else if ((amrIn = dynamic_cast<vtkOverlappingAMR*>(objIn)))
  {
    return SVTKObjectFactory::New(amrIn);
  }

  SENSEI_ERROR("Failed to construct a SVTK object from the given "
    << objIn->GetClassName() << " instance. Conversion not yet implemented.")

  return nullptr;
#endif
}

// --------------------------------------------------------------------------
svtkDataSet *SVTKObjectFactory::New(vtkDataSet *dsIn)
{
#if !defined(SENSEI_ENABLE_VTK_CORE)
  (void)dsIn;
  SENSEI_ERROR("Conversion from VTK to SVTK is not available in this build")
  return nullptr;
#else
  if (!dsIn)
  {
    SENSEI_ERROR("Can't create a svtkDataSet from nullptr")
    return nullptr;
  }

  vtkImageData *idIn = nullptr;
  vtkUniformGrid *ungIn = nullptr;
  vtkRectilinearGrid *rgIn = nullptr;
  vtkStructuredGrid *sgIn = nullptr;
  vtkPolyData *pdIn = nullptr;
  vtkUnstructuredGrid *ugIn = nullptr;

  if ((idIn = dynamic_cast<vtkImageData*>(dsIn)))
  {
    return SVTKObjectFactory::New(idIn);
  }
  else if ((ungIn = dynamic_cast<vtkUniformGrid*>(dsIn)))
  {
    return SVTKObjectFactory::New(ungIn);
  }
  else if ((rgIn = dynamic_cast<vtkRectilinearGrid*>(dsIn)))
  {
    return SVTKObjectFactory::New(rgIn);
  }
  else if ((sgIn = dynamic_cast<vtkStructuredGrid*>(dsIn)))
  {
    return SVTKObjectFactory::New(sgIn);
  }
  else if ((pdIn = dynamic_cast<vtkPolyData*>(dsIn)))
  {
    return SVTKObjectFactory::New(pdIn);
  }
  else if ((ugIn = dynamic_cast<vtkUnstructuredGrid*>(dsIn)))
  {
    return SVTKObjectFactory::New(ugIn);
  }

  SENSEI_ERROR("Failed to construct a SVTK object from the given "
    << dsIn->GetClassName() << " instance. Conversion not yet implemented.")

  return nullptr;
#endif
}

// **************************************************************************
template <typename SVTK_TT>
int write(FILE *fh, svtkDataArray *da)
{
  // first put the data into a buffer on the host. this will be a deep
  // copy because the VTK file format is big endian and we need to
  // rearange bytes
  long nt = da->GetNumberOfTuples();
  long nc = da->GetNumberOfComponents();
  long nct =  nc*nt;
  long nb = nct*sizeof(SVTK_TT);

  hamr::buffer<SVTK_TT> tmp(hamr::get_host_allocator(), nct);

  if (svtkHAMRDataArray<SVTK_TT> *hamrda =
    dynamic_cast<svtkHAMRDataArray<SVTK_TT>*>(da))
  {
    auto spda = hamrda->GetHostAccessible();
    hamrda->Synchronize();

    memcpy(tmp.data(), spda.get(), nb);
  }
  else if (svtkAOSDataArrayTemplate<SVTK_TT> *aosda =
    dynamic_cast<svtkAOSDataArrayTemplate<SVTK_TT>*>(da))
  {
    SVTK_TT *pda = aosda->GetPointer(0);
    memcpy(tmp.data(), pda, nb);
  }
  else
  {
    SENSEI_ERROR("Invalid data array type " << da->GetClassName())
    fclose(fh);
    return -1;
  }

  // rearange bytes to big endian in place as required by the VTK file format
  if (sizeof(SVTK_TT) == 8)
  {
    uint64_t *ptmp = (uint64_t*)tmp.data();
    for (size_t i = 0; i < tmp.size(); ++i)
      ptmp[i] = __builtin_bswap64(ptmp[i]);
  }
  else if (sizeof(SVTK_TT) == 4)
  {
    uint32_t *ptmp = (uint32_t*)tmp.data();
    for (size_t i = 0; i < tmp.size(); ++i)
      ptmp[i] = __builtin_bswap32(ptmp[i]);
  }
  else if (sizeof(SVTK_TT) == 2)
  {
    uint16_t *ptmp = (uint16_t*)tmp.data();
    for (size_t i = 0; i < tmp.size(); ++i)
      ptmp[i] = __builtin_bswap16(ptmp[i]);
  }
  else if (sizeof(SVTK_TT) != 1)
  {
    SENSEI_ERROR("Invalid element size " << sizeof(SVTK_TT) << " bytes")
    fclose(fh);
    return -1;
  }

  // replace ' ' with '_' in unsigned types
  char typeName[128];
  strncpy(typeName, da->GetDataTypeAsString(), 127);
  typeName[127] = '\0';

  while (char *ptr = strchr(typeName, ' '))
    *ptr = '_';

  // write the array
  fprintf(fh, "SCALARS %s %s %ld\n"
          "LOOKUP_TABLE default\n", da->GetName(), typeName, nc);

  fwrite(tmp.data(), sizeof(SVTK_TT), nct, fh);

  return 0;
}

// **************************************************************************
int WriteVTK(const char *fn, long npx, long npy, long npz,
  double x0, double y0, double z0, double dx, double dy, double dz,
  const std::vector<svtkDataArray*> &cellData,
  const std::vector<svtkDataArray*> &pointData)
{
  // write the file in vtk format
  FILE *fh = fopen(fn, "w");
  if (!fh)
  {
      SENSEI_ERROR("Failed to open \"" << fn << "\"")
      return -1;
  }

  // write the file in vtk format
  fprintf(fh, "# vtk DataFile Version 2.0\n"
              "sensei in-situ output\n"
              "BINARY\n"
              "DATASET STRUCTURED_POINTS\n"
              "DIMENSIONS %ld %ld %ld\n"
              "ORIGIN %g %g %g\n"
              "SPACING %g %g %g\n",
              npx, npy, npz, x0, y0, z0, dx, dy, dz);

  // write the point data arrays
  unsigned int npa = pointData.size();
  if (npa)
  {
    long np = npx*npy*npz;
    fprintf(fh, "POINT_DATA %ld\n", np);
    for (unsigned int i = 0; i < npa; ++i)
    {
      svtkDataArray *da = pointData[i];
      assert(np == da->GetNumberOfTuples());
      switch (da->GetDataType())
      {
      svtkTemplateMacro(
        write<SVTK_TT>(fh, da);
      );}
    }
  }

  // write the cell data arrays
  unsigned int nca = cellData.size();
  if (nca)
  {
    long nc = std::max(npx - 1l, 1l)*std::max(npy - 1l, 1l)*std::max(npz - 1l, 1l);
    fprintf(fh, "CELL_DATA %ld\n", nc);
    for (unsigned int i = 0; i < nca; ++i)
    {
      svtkDataArray *da = cellData[i];
      assert(nc == da->GetNumberOfTuples());
      switch (da->GetDataType())
      {
      svtkTemplateMacro(
        write<SVTK_TT>(fh, da);
      );}
    }
  }

  fclose(fh);
  return 0;
}

}
}
