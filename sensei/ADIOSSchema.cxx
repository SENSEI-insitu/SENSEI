#include "ADIOSSchema.h"
#include "VTKUtils.h"
#include "Error.h"

#include <vtkCellTypes.h>
#include <vtkCellData.h>
#include <vtkCompositeDataIterator.h>
#include <vtkCompositeDataSet.h>
#include <vtkDataSetAttributes.h>
#include <vtkDoubleArray.h>
#include <vtkFloatArray.h>
#include <vtkIntArray.h>
#include <vtkUnsignedIntArray.h>
#include <vtkLongArray.h>
#include <vtkUnsignedLongArray.h>
#include <vtkLongLongArray.h>
#include <vtkUnsignedLongLongArray.h>
#include <vtkCharArray.h>
#include <vtkUnsignedCharArray.h>
#include <vtkIdTypeArray.h>
#include <vtkCellArray.h>
#include <vtkInformation.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkStructuredPoints.h>
#include <vtkStructuredGrid.h>
#include <vtkRectilinearGrid.h>
#include <vtkUnstructuredGrid.h>
#include <vtkImageData.h>
#include <vtkUniformGrid.h>
#include <vtkTable.h>
#include <vtkMultiBlockDataSet.h>
#include <vtkHierarchicalBoxDataSet.h>
#include <vtkMultiPieceDataSet.h>
#include <vtkHyperOctree.h>
#include <vtkHyperTreeGrid.h>
#include <vtkOverlappingAMR.h>
#include <vtkNonOverlappingAMR.h>
#include <vtkUniformGridAMR.h>
#include <vtkObjectFactory.h>
#include <vtkPointData.h>
#include <vtkSmartPointer.h>

#include <mpi.h>

#include <adios.h>
#include <adios_read.h>

#include <vector>
#include <map>
#include <set>
#include <string>
#include <functional>
#include <sstream>

namespace senseiADIOS
{

// --------------------------------------------------------------------------
ADIOS_DATATYPES adiosIdType()
{
  if (sizeof(vtkIdType) == sizeof(int64_t))
    {
    return adios_long; // 64 bits
    }
  else if(sizeof(vtkIdType) == sizeof(int32_t))
    {
    return adios_integer; // 32 bits
    }
  else
    {
    SENSEI_ERROR("No conversion from vtkIdType to ADIOS_DATATYPES")
    MPI_Abort(MPI_COMM_WORLD, -1);
    }
  return adios_unknown;
}

// --------------------------------------------------------------------------
ADIOS_DATATYPES adiosType(vtkDataArray* da)
{
  if (dynamic_cast<vtkFloatArray*>(da))
    {
    return adios_real;
    }
  else if (dynamic_cast<vtkDoubleArray*>(da))
    {
    return adios_double;
    }
  else if (dynamic_cast<vtkCharArray*>(da))
    {
    return adios_byte;
    }
  else if (dynamic_cast<vtkIntArray*>(da))
    {
    return adios_integer;
    }
  else if (dynamic_cast<vtkLongArray*>(da))
    {
    if (sizeof(long) == 4)
      return adios_integer; // 32 bits
    return adios_long; // 64 bits
    }
  else if (dynamic_cast<vtkLongLongArray*>(da))
    {
    return adios_long; // 64 bits
    }
  else if (dynamic_cast<vtkUnsignedCharArray*>(da))
    {
    return adios_unsigned_byte;
    }
  else if (dynamic_cast<vtkUnsignedIntArray*>(da))
    {
    return adios_unsigned_integer;
    }
  else if (dynamic_cast<vtkUnsignedLongArray*>(da))
    {
    if (sizeof(unsigned long) == 4)
      return adios_unsigned_integer; // 32 bits
    return adios_unsigned_long; // 64 bits
    }
  else if (dynamic_cast<vtkUnsignedLongLongArray*>(da))
    {
    return adios_unsigned_long; // 64 bits
    }
  else if (dynamic_cast<vtkIdTypeArray*>(da))
    {
    return adiosIdType();
    }
  else
    {
    SENSEI_ERROR("the adios type for data array \"" << da->GetClassName()
      << "\" is currently not implemented")
    MPI_Abort(MPI_COMM_WORLD, -1);
    }
  return adios_unknown;
}

// --------------------------------------------------------------------------
int isLegacyDataObject(int code)
{
  // this function is used to determine data parallelization strategy.
  // VTK has 2, namely the legacy one in which each process holds 1
  // legacy dataset, and the more modern approach where VTK composite
  // dataset holds any number of datasets on any number of processes.
  int ret = 0;
  switch (code)
    {
    // legacy
    case VTK_POLY_DATA:
    case VTK_STRUCTURED_POINTS:
    case VTK_STRUCTURED_GRID:
    case VTK_RECTILINEAR_GRID:
    case VTK_UNSTRUCTURED_GRID:
    case VTK_IMAGE_DATA:
    case VTK_UNIFORM_GRID:
    case VTK_TABLE:
    // others
    case VTK_GRAPH:
    case VTK_TREE:
    case VTK_SELECTION:
    case VTK_DIRECTED_GRAPH:
    case VTK_UNDIRECTED_GRAPH:
    case VTK_DIRECTED_ACYCLIC_GRAPH:
    case VTK_ARRAY_DATA:
    case VTK_REEB_GRAPH:
    case VTK_MOLECULE:
    case VTK_PATH:
    case VTK_PIECEWISE_FUNCTION:
      ret = 1;
      break;
    // composite data etc
    case VTK_MULTIBLOCK_DATA_SET:
    case VTK_HIERARCHICAL_BOX_DATA_SET:
    case VTK_MULTIPIECE_DATA_SET:
    case VTK_HYPER_OCTREE:
    case VTK_HYPER_TREE_GRID:
    case VTK_OVERLAPPING_AMR:
    case VTK_NON_OVERLAPPING_AMR:
    case VTK_UNIFORM_GRID_AMR:
      ret = 0;
      break;
    // base classes
    case VTK_DATA_OBJECT:
    case VTK_DATA_SET:
    case VTK_POINT_SET:
    case VTK_COMPOSITE_DATA_SET:
    case VTK_GENERIC_DATA_SET:
#if !(VTK_MAJOR_VERSION == 6 && VTK_MINOR_VERSION == 1)
    case VTK_UNSTRUCTURED_GRID_BASE:
    case VTK_PISTON_DATA_OBJECT:
#endif
    // deprecated/removed
    case VTK_HIERARCHICAL_DATA_SET:
    case VTK_TEMPORAL_DATA_SET:
    case VTK_MULTIGROUP_DATA_SET:
    // unknown code
    default:
      SENSEI_ERROR("Neither legacy nor composite " << code)
      ret = -1;
    }
  return ret;
}

// --------------------------------------------------------------------------
vtkDataObject *newDataObject(int code)
{
  vtkDataObject *ret = nullptr;
  switch (code)
    {
    // simple
    case VTK_POLY_DATA:
      ret = vtkPolyData::New();
      break;
    case VTK_STRUCTURED_POINTS:
      ret = vtkStructuredPoints::New();
      break;
    case VTK_STRUCTURED_GRID:
      ret = vtkStructuredGrid::New();
      break;
    case VTK_RECTILINEAR_GRID:
      ret = vtkRectilinearGrid::New();
      break;
    case VTK_UNSTRUCTURED_GRID:
      ret = vtkUnstructuredGrid::New();
      break;
    case VTK_IMAGE_DATA:
      ret = vtkImageData::New();
      break;
    case VTK_UNIFORM_GRID:
      ret = vtkUniformGrid::New();
      break;
    case VTK_TABLE:
      ret = vtkTable::New();
      break;
    // composite data etc
    case VTK_MULTIBLOCK_DATA_SET:
      ret = vtkMultiBlockDataSet::New();
      break;
    case VTK_HIERARCHICAL_BOX_DATA_SET:
      ret = vtkHierarchicalBoxDataSet::New();
      break;
    case VTK_MULTIPIECE_DATA_SET:
      ret = vtkMultiPieceDataSet::New();
      break;
    case VTK_HYPER_OCTREE:
      ret = vtkHyperOctree::New();
      break;
    case VTK_HYPER_TREE_GRID:
      ret = vtkHyperTreeGrid::New();
      break;
    case VTK_OVERLAPPING_AMR:
      ret = vtkOverlappingAMR::New();
      break;
    case VTK_NON_OVERLAPPING_AMR:
      ret = vtkNonOverlappingAMR::New();
      break;
    case VTK_UNIFORM_GRID_AMR:
      ret = vtkUniformGridAMR::New();
      break;
    // TODO
    case VTK_GRAPH:
    case VTK_TREE:
    case VTK_SELECTION:
    case VTK_DIRECTED_GRAPH:
    case VTK_UNDIRECTED_GRAPH:
    case VTK_DIRECTED_ACYCLIC_GRAPH:
    case VTK_ARRAY_DATA:
    case VTK_REEB_GRAPH:
    case VTK_MOLECULE:
    case VTK_PATH:
    case VTK_PIECEWISE_FUNCTION:
      SENSEI_WARNING("Factory for " << code << " not yet implemented")
      break;
    // base classes
    case VTK_DATA_OBJECT:
    case VTK_DATA_SET:
    case VTK_POINT_SET:
    case VTK_COMPOSITE_DATA_SET:
    case VTK_GENERIC_DATA_SET:
#if !(VTK_MAJOR_VERSION == 6 && VTK_MINOR_VERSION == 1)
    case VTK_UNSTRUCTURED_GRID_BASE:
    case VTK_PISTON_DATA_OBJECT:
#endif
    // deprecated/removed
    case VTK_HIERARCHICAL_DATA_SET:
    case VTK_TEMPORAL_DATA_SET:
    case VTK_MULTIGROUP_DATA_SET:
    // unknown code
    default:
      SENSEI_ERROR("data object for " << code << " could not be construtced")
    }
  return ret;
}

// --------------------------------------------------------------------------
bool streamIsFileBased(ADIOS_READ_METHOD method)
{
  switch(method)
    {
    case ADIOS_READ_METHOD_BP:
    case ADIOS_READ_METHOD_BP_AGGREGATE:
      return true;
    case ADIOS_READ_METHOD_DATASPACES:
    case ADIOS_READ_METHOD_DIMES:
    case ADIOS_READ_METHOD_FLEXPATH:
    case ADIOS_READ_METHOD_ICEE:
      return false;
    }
  SENSEI_ERROR("Unknown read method " << method)
  return false;
}

// --------------------------------------------------------------------------
template <typename val_t>
int adiosInq(InputStream &iStream, const std::string &path, val_t &val)
{
  ADIOS_VARINFO *vinfo = adios_inq_var(iStream.File, path.c_str());
  if (!vinfo)
    {
    SENSEI_ERROR("ADIOS stream is missing \"" << path << "\"")
    return -1;
    }
  val = *static_cast<val_t*>(vinfo->value);
  adios_free_varinfo(vinfo);
  return 0;
}

// dataset_function takes a vtkDataSet*, it might be nullptr,
// does some processing, and returns and integer code.
//
// return codes:
//  1   : successfully processed the dataset, end the traversal
//  0   : successfully processed the dataset, continue traversal
//  < 0 : an error occured, report it and end traversal
using dataset_function =
  std::function<int(unsigned int,unsigned int,vtkDataSet*)>;

// --------------------------------------------------------------------------
// apply given function to each leaf in the data object.
int apply(unsigned int doid, unsigned int dsid, vtkDataObject* dobj,
  dataset_function &func, int skip_empty=1)
{
  if (vtkCompositeDataSet* cd = dynamic_cast<vtkCompositeDataSet*>(dobj))
    {
    vtkSmartPointer<vtkCompositeDataIterator> iter;
    iter.TakeReference(cd->NewIterator());
    iter->SetSkipEmptyNodes(skip_empty);
    for (iter->InitTraversal(); !iter->IsDoneWithTraversal(); iter->GoToNextItem())
      {
      // recurse
      int ierr = 0;
      if ((ierr = apply(doid, iter->GetCurrentFlatIndex(),
        iter->GetCurrentDataObject(), func, skip_empty)))
        return ierr;
      }
    }
  else if (vtkDataSet *ds = dynamic_cast<vtkDataSet*>(dobj))
    {
    int ierr = func(doid, dsid, ds);
#ifdef ADIOSSchemaDEBUG
    if (ierr < 0)
      {
      SENSEI_ERROR("Apply failed, functor returned error code " << ierr)
      }
#endif
    return ierr;
    }
  return 0;
}

// --------------------------------------------------------------------------
// returns the number of datasets(on this process) with the given type
template<typename dataset_t>
unsigned int getNumberOfDatasets(vtkDataObject *dobj)
{
  unsigned int number_of_datasets = 0;
  if (vtkCompositeDataSet *cd = dynamic_cast<vtkCompositeDataSet*>(dobj))
    {
    // function to count number of datasets of the given type
    dataset_function func = [&number_of_datasets](unsigned int, unsigned int, vtkDataSet *ds) -> int
    {
      if (dynamic_cast<dataset_t*>(ds))
        ++number_of_datasets;
      return 0;
    };

    if (apply(0, 0, dobj, func))
      return -1;
    }
  else if (dynamic_cast<dataset_t*>(dobj))
    {
    ++number_of_datasets;
    }

  return number_of_datasets;
}

// --------------------------------------------------------------------------
unsigned int getNumberOfDatasets(MPI_Comm comm, vtkDataObject *dobj,
  int local_only)
{
  unsigned int number_of_datasets = 0;
  // function that counts number of datasets of any type
  dataset_function func = [&number_of_datasets](unsigned int, unsigned int, vtkDataSet*) -> int
  {
    ++number_of_datasets;
    return 0;
  };

  if (apply(0, 0, dobj, func) < 0)
    return -1;

  if (!local_only)
    MPI_Allreduce(MPI_IN_PLACE, &number_of_datasets, 1,
      MPI_UNSIGNED, MPI_SUM, comm);

  return number_of_datasets;
}



// --------------------------------------------------------------------------
int Schema::DefineVariables(MPI_Comm comm, int64_t gh, unsigned int doid,
  vtkDataObject *dobj)
{
  // lambda that defines variables for a dataset
  dataset_function func =
    [this,gh](unsigned int doid, unsigned int dsid, vtkDataSet *ds) -> int
  {
    int ierr = this->DefineVariables(gh, doid, dsid, ds);
    if (ierr < 0)
      SENSEI_ERROR("Failed to define variables on data object "
        << doid << " dataset " << dsid);
    return ierr;
  };

  // if the given object is a simple dataset then id is the MPI rank.
  // shift by 1 to be consistent with composite datasets whose flat
  // index starts at 1
  int dsid = 0;
  if (dynamic_cast<vtkDataSet*>(dobj))
    {
    MPI_Comm_rank(comm, &dsid);
    dsid += 1;
    }

  // apply to leaf datasets
  if (apply(doid, dsid, dobj, func) < 0)
    return -1;

  return 0;
}

// --------------------------------------------------------------------------
int Schema::DefineVariables(int64_t, unsigned int, unsigned int, vtkDataSet *)
{
  return 0;
}

// --------------------------------------------------------------------------
uint64_t Schema::GetSize(MPI_Comm comm, vtkDataObject *dobj)
{
  (void)comm;

  uint64_t size = 0;

  // function that accumulates size of a dataset
  dataset_function func =
    [this,&size](unsigned int, unsigned int, vtkDataSet *ds) -> int
  {
    size += this->GetSize(ds);
    return 0;
  };

  // apply to leaf datasets
  if (apply(0, 0, dobj, func) < 0)
    return 0;

  return size;
}

// --------------------------------------------------------------------------
uint64_t Schema::GetSize(vtkDataSet *)
{
  return 0;
}

// --------------------------------------------------------------------------
int Schema::Write(MPI_Comm comm, int64_t fh, unsigned int doid,
  vtkDataObject *dobj)
{
  (void)comm;

  // lambda that writes a dataset
  dataset_function func =
    [this,fh](unsigned int doid, unsigned int dsid, vtkDataSet *ds) -> int
  {
    int ierr = this->Write(fh, doid, dsid, ds);
    if (ierr < 0)
      SENSEI_ERROR("Failed to write data object "
        << doid << " dataset " << dsid);
    return ierr;
  };

  // if the given object is a simple dataset then id is the MPI rank.
  // shift by 1 to be consistent with composite datasets whose flat
  // index starts at 1
  int dsid = 0;
  if (dynamic_cast<vtkDataSet*>(dobj))
    {
    MPI_Comm_rank(comm, &dsid);
    dsid += 1;
    }

  // apply to leaf datasets
  if (apply(doid, dsid, dobj, func) < 0)
    return -1;

  return 0;
}

// --------------------------------------------------------------------------
int Schema::Write(int64_t, unsigned int, unsigned int, vtkDataSet *)
{
  return 0;
}

// --------------------------------------------------------------------------
int Schema::Read(MPI_Comm comm, InputStream &iStream, unsigned int doid,
  vtkDataObject *&dobj)
{
  (void)comm;

  // function that reads a dataset
  dataset_function func =
    [&](unsigned int doid, unsigned int dsid, vtkDataSet *ds) -> int
  {
    int ierr = this->Read(comm, iStream, doid, dsid, ds);
    if (ierr < 0)
      SENSEI_ERROR("Failed to read data object " << doid << " dataset " << dsid);
    return ierr;
  };

  // if the given object is a simple dataset then id is the MPI rank.
  // shift by 1 to be consistent with composite datasets whose flat
  // index starts at 1
  int dsid = 0;
  if (dynamic_cast<vtkDataSet*>(dobj))
    {
    MPI_Comm_rank(comm, &dsid);
    dsid += 1;
    }

  // apply to leaf datasets
  if (apply(doid, dsid, dobj, func) < 0)
    return -1;

  return 0;
}

// --------------------------------------------------------------------------
int Schema::Read(MPI_Comm, InputStream &, unsigned int, unsigned int,
  vtkDataSet *&)
{
  return 0;
}




// helper for writing strings. different ADIOS transports handle
// strings differently and do not follow the documentation which
// complicates things to the point that we factor the code into
// its own class. the string is written into a byte array and
// will be null terminated.
class StringSchema
{
public:
  static int DefineVariables(int64_t gh, const std::string &path);

  static uint64_t GetSize(const std::string &str);

  static int Write(uint64_t fh, const std::string &path,
    const std::string &str);

  static int Read(InputStream &iStream, ADIOS_SELECTION *sel,
    const std::string &path, std::string &str);
};

// --------------------------------------------------------------------------
uint64_t StringSchema::GetSize(const std::string &str)
{
  return str.size() + 1 + sizeof(int);
}

// --------------------------------------------------------------------------
int StringSchema::DefineVariables(int64_t gh, const std::string &path)
{
  // a second variable holding the local, global, length is required
  // for writing. according to trhe docs you could write a constant
  // string literal, but that only works with BP and not FLEXPATH
  std::string len = path + "_len";
  adios_define_var(gh, len.c_str(), "", adios_integer,
    "", "", "0");

  // define the string
  adios_define_var(gh, path.c_str(), "", adios_byte,
    len.c_str(), len.c_str(), "0");

  return 0;
}

// --------------------------------------------------------------------------
int StringSchema::Read(InputStream &iStream, ADIOS_SELECTION *sel,
  const std::string &path, std::string &result)
{
  // get metadata
  ADIOS_VARINFO *vinfo = adios_inq_var(iStream.File, path.c_str());
  if (!vinfo)
    {
    SENSEI_ERROR("ADIOS stream is missing \"" << path << "\"")
    return -1;
    }

  // allocate a buffer
  char *tmp = static_cast<char*>(malloc(vinfo->dims[0]));

  // read it
  adios_schedule_read(iStream.File, sel, path.c_str(), 0, 1, tmp);
  if (adios_perform_reads(iStream.File, 1))
    {
    SENSEI_ERROR("Failed to read string at \"" << path << "\"")
    return -1;
    }

  // clean up and pass string back
  adios_free_varinfo(vinfo);
  result = tmp;
  free(tmp);

  return 0;
}

// --------------------------------------------------------------------------
int StringSchema::Write(uint64_t fh, const std::string &path,
  const std::string &str)
{
  std::string len = path + "_len";
  int n = str.size() + 1;
  if (adios_write(fh, len.c_str(), &n) ||
    adios_write(fh, path.c_str(), str.c_str()))
    {
    SENSEI_ERROR("Failed to write string at \"" << path << "\"")
    return -1;
    }
  return 0;
}



// --------------------------------------------------------------------------
int Extent3DSchema::DefineVariables(int64_t gh, unsigned int doid,
  unsigned int dsid, vtkDataSet *ds)
{
  if (dynamic_cast<vtkImageData*>(ds))
    {
    std::ostringstream oss;
    oss << "data_object_" << doid << "/dataset_" << dsid;

    std::string dataset_id = oss.str();

    std::string path_len = dataset_id + "/extent_len";
    adios_define_var(gh, path_len.c_str(), "", adios_integer, "", "", "");

    // /data_object_<id>/dataset_<id>/extent
    std::string path = dataset_id + "/extent";
    adios_define_var(gh, path.c_str(), "", adios_integer,
      path_len.c_str(), path_len.c_str(), "0");

    path_len = dataset_id + "/origin_len";
    adios_define_var(gh, path_len.c_str(), "", adios_integer, "", "", "");

    // /data_object_<id>/dataset_<id>/origin
    path = dataset_id + "/origin";
    adios_define_var(gh, path.c_str(), "", adios_double,
      path_len.c_str(), path_len.c_str(), "0");

    path_len = dataset_id + "/spacing_len";
    adios_define_var(gh, path_len.c_str(), "", adios_integer, "", "", "");

    // /data_object_<id>/dataset_<id>/spacing
    path = dataset_id + "/spacing";
    adios_define_var(gh, path.c_str(), "", adios_double,
      path_len.c_str(), path_len.c_str(), "0");
    }
  return 0;
}

// --------------------------------------------------------------------------
uint64_t Extent3DSchema::GetSize(vtkDataSet *ds)
{
  if (dynamic_cast<vtkImageData*>(ds))
    return 3*sizeof(unsigned long) + 6*sizeof(int) +
      6*sizeof(double) + 3*sizeof(int);

  return 0;
}

// --------------------------------------------------------------------------
int Extent3DSchema::Write(int64_t fh, unsigned int doid, unsigned int dsid,
  vtkDataSet *ds)
{
  if (vtkImageData *img = dynamic_cast<vtkImageData*>(ds))
    {
    std::ostringstream oss;
    oss << "data_object_" << doid << "/dataset_" << dsid << "/";
    std::string dataset_id = oss.str();

    // extent
    int extent_len = 6;
    std::string path = dataset_id + "extent_len";
    adios_write(fh, path.c_str(), &extent_len);

    path = dataset_id + "extent";
    int extent[6] = {0};
    img->GetExtent(extent);
    adios_write(fh, path.c_str(), extent);

    // origin
    int origin_len = 3;
    path = dataset_id + "origin_len";
    adios_write(fh, path.c_str(), &origin_len);

    path = dataset_id + "origin";
    double origin[3] = {0.0};
    img->GetOrigin(origin);
    adios_write(fh, path.c_str(), origin);

    // spacing
    int spacing_len = 3;
    path = dataset_id + "spacing_len";
    adios_write(fh, path.c_str(), &spacing_len);

    path = dataset_id + "spacing";
    double spacing[3] = {0.0};
    img->GetSpacing(spacing);
    adios_write(fh, path.c_str(), spacing);
    }
  return 0;
}

// --------------------------------------------------------------------------
int Extent3DSchema::Read(MPI_Comm comm, InputStream &iStream,
  unsigned int doid, unsigned int dsid, vtkDataSet *&ds)
{
  if (vtkImageData *img = dynamic_cast<vtkImageData*>(ds))
    {
    std::ostringstream oss;
    oss << "data_object_" << doid << "/dataset_" << dsid << "/";
    std::string dataset_id = oss.str();

    ADIOS_SELECTION *sel = nullptr;
    if (!streamIsFileBased(iStream.ReadMethod))
      {
      int rank = 0;
      MPI_Comm_rank(comm, &rank);
      sel = adios_selection_writeblock(rank);
      if (!sel)
        {
        SENSEI_ERROR("Failed to make the selction")
        return -1;
        }
      }

    // extent
    std::string path = dataset_id + "extent";
    int extent[6] = {0};
    adios_schedule_read(iStream.File, sel, path.c_str(), 0, 1, extent);

    // origin
    path = dataset_id + "origin";
    double origin[3] = {0.0};
    adios_schedule_read(iStream.File, sel, path.c_str(), 0, 1, origin);

    // spacing
    path = dataset_id + "spacing";
    double spacing[3] = {0.0};
    adios_schedule_read(iStream.File, sel, path.c_str(), 0, 1, spacing);

    if (adios_perform_reads(iStream.File, 1))
      {
      SENSEI_ERROR("Failed to read extents")
      return -1;
      }

    if (sel)
      adios_selection_delete(sel);

    img->SetExtent(extent);
    img->SetOrigin(origin);
    img->SetSpacing(spacing);
    }
  return 0;
}



// compile time helper class to convert attribute type
// enum into a string
template<int att_t> struct datasetAttributeString;

template<> struct datasetAttributeString<vtkDataObject::POINT>
{
  static std::string str(){ return "point";  }
};

template<> struct datasetAttributeString<vtkDataObject::CELL>
{
  static std::string str(){ return "cell";  }
};

template<int att_t>
struct DatasetAttributesSchema<att_t>::InternalsType
{
  using IdMapType = std::map<unsigned int, int>;
  using NameIdMapType = std::map<std::string, IdMapType>;
  using MeshMapType = std::map<unsigned int, NameIdMapType>;
  MeshMapType NameIdMap;
};

// --------------------------------------------------------------------------
template<int att_t>
DatasetAttributesSchema<att_t>::DatasetAttributesSchema()
{
  this->Internals = new InternalsType;
}

// --------------------------------------------------------------------------
template<int att_t>
DatasetAttributesSchema<att_t>::~DatasetAttributesSchema()
{
  delete this->Internals;
}

// --------------------------------------------------------------------------
template<int att_t>
int DatasetAttributesSchema<att_t>::DefineVariables(int64_t gh,
  unsigned int doid, unsigned int dsid, vtkDataSet* ds)
{
  if (ds)
    {
    std::ostringstream oss;
    oss << "data_object_" << doid << "/dataset_" << dsid << "/";
    std::string dataset_id = oss.str();

    std::string att_type = datasetAttributeString<att_t>::str() + "_data/";

    std::string att_path = dataset_id + att_type;

    // /data_object_<id>/dataset_<id>/<att_type>/number_of_arrays
    std::string path = att_path + "number_of_arrays";
    adios_define_var(gh, path.c_str(), "", adios_integer, "", "", "");

    vtkDataSetAttributes* dsa = ds->GetAttributes(att_t);
    for (int i = 0, max = dsa->GetNumberOfArrays(); i < max; ++i)
      {
      vtkDataArray* array = dsa->GetArray(i);
      oss.str("");
      oss << "array_" << i;

      std::string array_id = oss.str();
      std::string array_path = att_path + array_id;

      // /data_object_<id>/dataset_<id>/<att_type>/array_<i>/name
      StringSchema::DefineVariables(gh, array_path + "/name");

      // /data_object_<id>/dataset_<id>/<att_type>/array_<i>/number_of_elements
      std::string elem_path = array_path + "/number_of_elements";
      adios_define_var(gh, elem_path.c_str(), "", adiosIdType(), "", "", "");

      // /data_object_<id>/dataset_<id>/<att_type>/array_<i>/number_of_components
      path = array_path + "/number_of_components";
      adios_define_var(gh, path.c_str(), "", adios_integer, "", "", "");

      // /data_object_<id>/dataset_<id>/<att_type>/array_<i>/element_type
      path = array_path + "/element_type";
      adios_define_var(gh, path.c_str(), "", adios_integer, "", "", "");

      // /data_object_<id>/dataset_<id>/<att_type>/array_<i>/data
      path = array_path + "/data";
      adios_define_var(gh, path.c_str(), "", adiosType(array),
        elem_path.c_str(), elem_path.c_str(), "0");
      }
    }

  return 0;
}

// --------------------------------------------------------------------------
template<int att_t>
uint64_t DatasetAttributesSchema<att_t>::GetSize(vtkDataSet *ds)
{
  int64_t size = 0;
  if (ds)
    {
    vtkDataSetAttributes* dsa = ds->GetAttributes(att_t);

    // per array offset and length
    int number_of_arrays = dsa->GetNumberOfArrays();
    size = number_of_arrays*(2*sizeof(int) + sizeof(vtkIdType));

    // size of the arrays themselves
    for (int i = 0; i < number_of_arrays; ++i)
      {
      vtkDataArray *da = dsa->GetArray(i);

      size += StringSchema::GetSize(da->GetName());

      size += da->GetNumberOfTuples()*
        da->GetNumberOfComponents()*da->GetElementComponentSize();
      }
    }

  return size;
}

// -------------------------------------------------------------------------
template<int att_t>
int DatasetAttributesSchema<att_t>::Write(int64_t fh,
  unsigned int doid, unsigned int dsid, vtkDataSet *ds)
{
  if (ds)
    {
    vtkDataSetAttributes* dsa = ds->GetAttributes(att_t);

    std::ostringstream oss;
    oss << "data_object_" << doid << "/dataset_" << dsid << "/";
    std::string dataset_id = oss.str();

    std::string att_type = datasetAttributeString<att_t>::str() + "_data/";
    std::string att_path = dataset_id + att_type;

    // /data_object_<id>/dataset_<id>/<att_type>/number_of_arrays
    int n_arrays = dsa->GetNumberOfArrays();
    std::string path = att_path + "number_of_arrays";
    adios_write(fh, path.c_str(), &n_arrays);

    for (int i = 0; i < n_arrays; ++i)
      {
      vtkDataArray* array = dsa->GetArray(i);
      oss.str("");
      oss << "array_" << i;

      std::string array_id = oss.str();
      std::string array_path = att_path + array_id;

      // /data_object_<id>/dataset_<id>/<att_type>/array_<i>/name
      StringSchema::Write(fh, array_path + "/name", array->GetName());

      // /data_objevct_<id>/dataset_<id>/<att_type>/array_<i>/number_of_elements
      vtkIdType n_elem = array->GetNumberOfTuples()*array->GetNumberOfComponents();
      std::string elem_path = array_path + "/number_of_elements";
      adios_write(fh, elem_path.c_str(), &n_elem);

      // /data_object_<id>/dataset_<id>/<att_type>/array_<i>/number_of_components
      path = array_path + "/number_of_components";
      int n_comp = array->GetNumberOfComponents();
      adios_write(fh, path.c_str(), &n_comp);

      // /data_object_<id>/dataset_<id>/<att_type>/array_<i>/element_type
      path = array_path + "/element_type";
      int elem_type = array->GetDataType();
      adios_write(fh, path.c_str(), &elem_type);

      // /data_object_<id>/dataset_<id>/<att_type>/array_<i>
      path = array_path + "/data";
      adios_write(fh, path.c_str(), array->GetVoidPointer(0));
      }
    }

  return 0;
}

// --------------------------------------------------------------------------
template<int att_t>
int DatasetAttributesSchema<att_t>::Read(MPI_Comm comm, InputStream &iStream,
  unsigned int doid, unsigned int dsid, vtkDataSet *&ds)
{
  if (ds)
    {
    vtkDataSetAttributes* dsa = ds->GetAttributes(att_t);

    std::ostringstream oss;
    oss << "data_object_" << doid << "/dataset_" << dsid << "/";
    std::string dataset_id = oss.str();

    std::string att_type = datasetAttributeString<att_t>::str() + "_data/";
    std::string att_path = dataset_id + att_type;

    // /data_object_<id>/dataset_<id>/<att_type>/number_of_arrays
    int n_arrays = 0;
    std::string path = att_path + "number_of_arrays";
    if (adiosInq(iStream, att_path, n_arrays))
      return -1;

    ADIOS_SELECTION *sel = nullptr;
    if (!streamIsFileBased(iStream.ReadMethod))
      {
      int rank = 0;
      MPI_Comm_rank(comm, &rank);
      sel = adios_selection_writeblock(rank);
      if (!sel)
        {
        SENSEI_ERROR("Failed to make the selction")
        return -1;
        }
      }

    for (int i = 0; i < n_arrays; ++i)
      {
      oss.str("");
      oss << "array_" << i;

      std::string array_id = oss.str();
      std::string array_path = att_path + array_id;

      // /data_object_<id>/dataset_<id>/<att_type>/array_<i>/name
      std::string name;
      path = array_path + "/name";
      if (StringSchema::Read(iStream, sel, path, name))
        return -1;

      // /data_object_<id>/dataset_<id>/<att_type>/array_<i>/number_of_elements
      vtkIdType n_elem = 0;
      path = array_path + "/number_of_elements";
      if (adiosInq(iStream, path, n_elem))
        return -1;

      // /data_object_<id>/dataset_<id>/<att_type>/array_<i>/number_of_components
      int n_comp = 0;
      path = array_path + "/number_of_components";
      if (adiosInq(iStream, path, n_comp))
        return -1;

      // /dataset_<id>/<att_type>/array_<i>/element_type
      int elem_type = 0;
      path = array_path + "/element_type";
      if (adiosInq(iStream, path, elem_type))
        return -1;

      // /data_object_<id>//dataset_<id>/<att_type>/array_<i>/data
      path = array_path + "/data";
      vtkDataArray *array = vtkDataArray::CreateDataArray(elem_type);
      array->SetNumberOfComponents(n_comp);
      array->SetNumberOfTuples(n_elem/n_comp);
      array->SetName(name.c_str());
      adios_schedule_read(iStream.File, sel, path.c_str(),
        0, 1, array->GetVoidPointer(0));
      dsa->AddArray(array);
      array->Delete();

      if (adios_perform_reads(iStream.File, 1))
        {
        SENSEI_ERROR("Failed to read data for "
          << datasetAttributeString<att_t>::str() << " data array \""
          << array->GetName() << "\"")
        return -1;
        }

      }
    if (sel)
      adios_selection_delete(sel);
    }
  return 0;
}

// --------------------------------------------------------------------------
template<int att_t>
int DatasetAttributesSchema<att_t>::ReadArrayNames(MPI_Comm comm,
  InputStream &iStream, unsigned int doid, vtkDataObject *dobj,
  std::set<std::string> &array_names)
{
  // adaptor function
  dataset_function func =
    [&](unsigned int doid, unsigned int dsid, vtkDataSet* ds) -> int
  {
    int ierr = this->ReadArrayNames(comm, iStream, doid, dsid, ds, array_names);
    if (ierr < 0)
      SENSEI_ERROR("Failed to get array names in dataset "
        << "data object " << doid << " dataset " << dsid)
    return ierr;
  };

  this->Internals->NameIdMap[doid].clear();

  // if the given object is a simple dataset then id is the MPI rank.
  // shift by 1 to be consistent with composite datasets whose flat
  // index starts at 1
  int dsid = 0;
  if (dynamic_cast<vtkDataSet*>(dobj))
    {
    MPI_Comm_rank(comm, &dsid);
    dsid += 1;
    }

  if (apply(doid, dsid, dobj, func, 0) < 0)
    return -1;

  return 0;
}

// --------------------------------------------------------------------------
template<int att_t>
int DatasetAttributesSchema<att_t>::ReadArrayNames(MPI_Comm comm,
  InputStream &iStream, unsigned int doid, unsigned int dsid, vtkDataSet *ds,
  std::set<std::string> &array_names)
{
  std::ostringstream oss;
  oss << "data_object_" << doid << "/dataset_" << dsid << "/";
  std::string dataset_id = oss.str();

  std::string att_type = datasetAttributeString<att_t>::str() + "_data/";
  std::string att_path = dataset_id + att_type;

  // /data_object_<id>/dataset_<id>/<att_type>/number_of_arrays
  int n_arrays = 0;
  std::string path = att_path + "number_of_arrays";
  if (adiosInq(iStream, path, n_arrays))
    return -1;

  ADIOS_SELECTION *sel = nullptr;
  if (!streamIsFileBased(iStream.ReadMethod))
    {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    sel = adios_selection_writeblock(rank);
    if (!sel)
      {
      SENSEI_ERROR("Failed to make the selction")
      return -1;
      }
    }

  for (int i = 0; i < n_arrays; ++i)
    {
    oss.str("");
    oss << "array_" << i;

    std::string array_id = oss.str();
    std::string array_path = att_path + array_id;

    // /data_object_<id>/dataset_<id>/<att_type>/array_<i>/name
    std::string name;
    path = array_path + "/name";
    if (StringSchema::Read(iStream, sel, path, name))
      return -1;

    // add to list of arrays
    array_names.insert(name);

    // cacahe the array id by block and name
    if (ds)
      this->Internals->NameIdMap[doid][name][dsid] = i;
    }

  if (sel)
    adios_selection_delete(sel);

  return 0;
}

// --------------------------------------------------------------------------
template<int att_t>
int DatasetAttributesSchema<att_t>::ReadArray(MPI_Comm comm,
  InputStream &iStream, const std::string &array_name, unsigned int doid,
  vtkDataObject *dobj)
{
  // adaptor function
  dataset_function func =
    [&](unsigned int doid, unsigned int dsid, vtkDataSet* ds) -> int
  {
    int ierr = this->ReadArray(comm, iStream, array_name, doid, dsid, ds);
    if (ierr < 0)
      SENSEI_ERROR("Failed to read array \"" << array_name
        << "\" from dataset " << "data object " << doid << " dataset " << dsid)
    return ierr;
  };


  // may need to build the name map, if the user never called
  // ReadArrayNames
  std::set<std::string> tmp;
  if (((this->Internals->NameIdMap.find(doid) == this->Internals->NameIdMap.end())
    || this->Internals->NameIdMap[doid].empty()) &&
    this->ReadArrayNames(comm, iStream, doid, dobj, tmp))
    return -1;

  // if the given object is a simple dataset then id is the MPI rank.
  // shift by 1 to be consistent with composite datasets whose flat
  // index starts at 1
  int dsid = 0;
  if (dynamic_cast<vtkDataSet*>(dobj))
    {
    MPI_Comm_rank(comm, &dsid);
    dsid += 1;
    }

  if (apply(doid, dsid, dobj, func) < 0)
    return -1;

  return 0;
}

// --------------------------------------------------------------------------
template<int att_t>
int DatasetAttributesSchema<att_t>::ReadArray(MPI_Comm comm,
  InputStream &iStream, const std::string &array_name, unsigned int doid,
  unsigned int dsid, vtkDataSet *ds)
{
  if (ds)
    {
    vtkDataSetAttributes* dsa = ds->GetAttributes(att_t);

    // use mesh name, array name, and block id to get the array id
    typename InternalsType::MeshMapType::iterator mesh_map_it =
      this->Internals->NameIdMap.find(doid);
    if (mesh_map_it == this->Internals->NameIdMap.end())
      {
      SENSEI_ERROR("No object id " << doid)
      return -1;
      }

    typename InternalsType::NameIdMapType::iterator name_map_it =
      mesh_map_it->second.find(array_name);
    if (name_map_it == mesh_map_it->second.end())
      {
      SENSEI_ERROR("No array named \"" << array_name << "\" in "
        << datasetAttributeString<att_t>::str() << " data of object "
        << doid)
      return -1;
      }

    typename InternalsType::IdMapType::iterator id_map_it =
      name_map_it->second.find(dsid);
    if (id_map_it == name_map_it->second.end())
      {
      SENSEI_ERROR("No array named \"" << array_name << "\" in "
        << datasetAttributeString<att_t>::str() << " data of object "
        << doid << " block " << dsid)
      return -1;
      }

    int array_id = id_map_it->second;

    std::ostringstream oss;
    oss << "data_object_" << doid << "/dataset_" << dsid << "/"
      << datasetAttributeString<att_t>::str() << "_data/"
      << "array_" << array_id;

    std::string array_path = oss.str();

    // /data_object_<id>/dataset_<id>/<att_type>/array_<i>/number_of_elements
    vtkIdType n_elem = 0;
    std::string path = array_path + "/number_of_elements";
    if (adiosInq(iStream, path, n_elem))
      return -1;

    // /data_object_<id>/dataset_<id>/<att_type>/array_<i>/number_of_components
    int n_comp = 0;
    path = array_path + "/number_of_components";
    if (adiosInq(iStream, path, n_comp))
      return -1;

    // /data_object_<id>/dataset_<id>/<att_type>/array_<i>/element_type
    int elem_type = 0;
    path = array_path + "/element_type";
    if (adiosInq(iStream, path, elem_type))
      return -1;

    ADIOS_SELECTION *sel = nullptr;
    if (!streamIsFileBased(iStream.ReadMethod))
      {
      int rank = 0;
      MPI_Comm_rank(comm, &rank);
      sel = adios_selection_writeblock(rank);
      if (!sel)
        {
        SENSEI_ERROR("Failed to make the selction")
        return -1;
        }
      }

    // /data_object_<id>/dataset_<id>/<att_type>/array_<i>/data
    path = array_path + "/data";
    vtkDataArray *array = vtkDataArray::CreateDataArray(elem_type);
    array->SetNumberOfComponents(n_comp);
    array->SetNumberOfTuples(n_elem/n_comp);
    array->SetName(array_name.c_str());
    adios_schedule_read(iStream.File, sel, path.c_str(),
      0, 1, array->GetVoidPointer(0));
    dsa->AddArray(array);
    array->Delete();

    if (adios_perform_reads(iStream.File, 1))
      {
      SENSEI_ERROR("Failed to read " << datasetAttributeString<att_t>::str()
        << " data array \"" << array_name << "\"")
      return -1;
      }

    if (sel)
      adios_selection_delete(sel);
    }

  return 0;
}



// --------------------------------------------------------------------------
int PointsSchema::DefineVariables(int64_t gh, unsigned int doid,
  unsigned int dsid, vtkDataSet *ds)
{
  if (vtkPointSet *ps = dynamic_cast<vtkPointSet*>(ds))
    {
    std::ostringstream oss;
    oss << "data_object_" << doid << "/dataset_" << dsid << "/points/";
    std::string dataset_id = oss.str();

    // /data_object_<id>/dataset_<id>/points/number_of_elements
    std::string path_len = dataset_id + "number_of_elements";
    adios_define_var(gh, path_len.c_str(), "", adios_unsigned_long, "", "", "");

    // /data_object_<id>/dataset_<id>/points/elem_type
    std::string path = dataset_id + "elem_type";
    adios_define_var(gh, path.c_str(), "", adios_integer, "", "", "");

    // /data_object_<id>/dataset_<id>/points/data
    vtkDataArray *pts = ps->GetPoints()->GetData();

    path = dataset_id + "data";
    adios_define_var(gh, path.c_str(), "", adiosType(pts),
      path_len.c_str(), path_len.c_str(), "0");
    }
  return 0;
}

// --------------------------------------------------------------------------
uint64_t PointsSchema::GetSize(vtkDataSet *ds)
{
  uint64_t size = 0;
  if (dynamic_cast<vtkPointSet*>(ds))
    size += sizeof(unsigned long)     // number of points
      + this->GetPointsSize(ds);      // points array
  return size;
}

// -------------------------------------------------------------------------
int PointsSchema::Write(int64_t fh, unsigned int doid, unsigned int dsid,
  vtkDataSet *ds)
{
  if (vtkPointSet *ps = dynamic_cast<vtkPointSet*>(ds))
    {
    std::ostringstream oss;
    oss << "data_object_" << doid << "/dataset_" << dsid << "/points/";
    std::string dataset_id = oss.str();

    // /data_object_<id>/dataset_<id>/points/number_of_elements
    unsigned long number_of_elements = 3*ds->GetNumberOfPoints();
    std::string path = dataset_id + "number_of_elements";
    adios_write(fh, path.c_str(), &number_of_elements);

    vtkDataArray *pts = ps->GetPoints()->GetData();

    // /data_object_<id>/dataset_<id>/points/type
    path = dataset_id + "elem_type";
    int points_type = pts->GetDataType();
    adios_write(fh, path.c_str(), &points_type);

    // /data_object_<id>/dataset_<id>/points/data
    path = dataset_id + "data";
    adios_write(fh, path.c_str(), pts->GetVoidPointer(0));
    }
  return 0;
}

// --------------------------------------------------------------------------
int PointsSchema::Read(MPI_Comm comm, InputStream &iStream, unsigned int doid,
  unsigned int dsid, vtkDataSet *&ds)
{
  if (vtkPointSet *ps = dynamic_cast<vtkPointSet*>(ds))
    {
    std::ostringstream oss;
    oss << "data_object_" << doid << "/dataset_" << dsid << "/points/";
    std::string dataset_id = oss.str();

    // /data_object_<id>/dataset_<id>/points/number_of_elements
    unsigned long number_of_elements = 0;
    std::string path = dataset_id + "number_of_elements";
    if (adiosInq(iStream, path, number_of_elements))
      return -1;

    // /data_object_<id>/dataset_<id>/points/type
    int elem_type = 0;
    path = dataset_id + "elem_type";
    if (adiosInq(iStream, path, elem_type))
      return -1;

    // /data_object_<id>/dataset_<id>/points/data
    vtkDataArray *pts = vtkDataArray::CreateDataArray(elem_type);
    pts->SetNumberOfComponents(3);
    pts->SetNumberOfTuples(number_of_elements/3);
    pts->SetName("points");

    ADIOS_SELECTION *sel = nullptr;
    if (!streamIsFileBased(iStream.ReadMethod))
      {
      int rank = 0;
      MPI_Comm_rank(comm, &rank);
      sel = adios_selection_writeblock(rank);
      if (!sel)
        {
        SENSEI_ERROR("Failed to make the selction")
        return -1;
        }
      }

    path = dataset_id + "data";
    adios_schedule_read(iStream.File, sel, path.c_str(),
      0, 1, pts->GetVoidPointer(0));

    if (adios_perform_reads(iStream.File, 1))
      {
      SENSEI_ERROR("Failed to read points")
      return -1;
      }

    if (sel)
      adios_selection_delete(sel);

    vtkPoints *points = vtkPoints::New();
    points->SetData(pts);
    pts->Delete();

    ps->SetPoints(points);
    points->Delete();
    }
  return 0;
}

// --------------------------------------------------------------------------
uint64_t PointsSchema::GetPointsSize(vtkDataSet *ds)
{
  int64_t size = 0;
  if (vtkPointSet *ps = dynamic_cast<vtkPointSet*>(ds))
    {
    vtkDataArray *pts = ps->GetPoints()->GetData();

    size += pts->GetNumberOfTuples()*
      pts->GetNumberOfComponents()*pts->GetElementComponentSize();
    }
  return size;
}

// --------------------------------------------------------------------------
unsigned long PointsSchema::GetPointsLength(vtkDataSet *ds)
{
  unsigned long length = 0;
  if (vtkPointSet *ps = dynamic_cast<vtkPointSet*>(ds))
    {
    vtkDataArray *pts = ps->GetPoints()->GetData();
    length += pts->GetNumberOfTuples()*pts->GetNumberOfComponents();
    }
  return length;
}

// --------------------------------------------------------------------------
unsigned long PointsSchema::GetNumberOfPoints(vtkDataSet *ds)
{
  return ds ? ds->GetNumberOfPoints() : 0;
}



// --------------------------------------------------------------------------
int CellsSchema::DefineVariables(int64_t gh, unsigned int doid,
  unsigned int dsid, vtkDataSet* ds)
{
  if (dynamic_cast<vtkPolyData*>(ds) || dynamic_cast<vtkUnstructuredGrid*>(ds))
    {
    std::ostringstream oss;
    oss << "data_object_" << doid << "/dataset_" << dsid << "/cells/";
    std::string dataset_id = oss.str();

    // /data_object_<id>/dataset_<id>/cells/number_of_cells
    std::string path_len = dataset_id + "number_of_cells";
    adios_define_var(gh, path_len.c_str(), "", adios_unsigned_long, "", "", "");

    // /data_object_<id>/dataset_<id>/cells/cell_types
    std::string path_array = dataset_id + "cell_types";
    adios_define_var(gh, path_array.c_str(), "", adios_unsigned_byte,
      path_len.c_str(), path_len.c_str(), "0");

    // /data_object_<id>/dataset_<id>/cells/number_of_cells
    path_len = dataset_id + "number_of_elements";
    adios_define_var(gh, path_len.c_str(), "", adios_unsigned_long, "", "", "");

    // /data_object_<id>/dataset_<id>/cells/data
    path_array = dataset_id + "data";
    adios_define_var(gh, path_array.c_str(), "", adiosIdType(),
      path_len.c_str(), path_len.c_str(), "0");
    }

  return 0;
}

// --------------------------------------------------------------------------
uint64_t CellsSchema::GetSize(vtkDataSet *ds)
{
  uint64_t size = 0;
  if (dynamic_cast<vtkPolyData*>(ds) || dynamic_cast<vtkUnstructuredGrid*>(ds))
    {
    size += 2*sizeof(unsigned long) +                    // number_of_cells, number_of_elementss
      this->GetNumberOfCells(ds)*sizeof(unsigned char) + // cell types array
      this->GetCellsLength(ds)*sizeof(vtkIdType);        // cells array
    }
  return size;
}

// -------------------------------------------------------------------------
int CellsSchema::Write(int64_t fh, unsigned int doid, unsigned int dsid,
  vtkDataSet *ds)
{
  vtkPolyData *pd = dynamic_cast<vtkPolyData*>(ds);
  vtkUnstructuredGrid *ug = dynamic_cast<vtkUnstructuredGrid*>(ds);

  if (pd || ug)
    {
    std::ostringstream oss;
    oss << "data_object_" << doid << "/dataset_" << dsid << "/cells/";
    std::string dataset_id = oss.str();

    // /data_object_<id>/dataset_<id>/cells/number_of_cells
    std::string path = dataset_id + "number_of_cells";
    unsigned long number_of_cells = ds->GetNumberOfCells();
    adios_write(fh, path.c_str(), &number_of_cells);

    if (ug)
      {
      // /data_object_<id>/dataset_<id>/cells/cell_types
      path = dataset_id + "cell_types";
      adios_write(fh, path.c_str(), ug->GetCellTypesArray()->GetPointer(0));

      // /data_object_<id>/dataset_<id>/cells/number_of_elements
      std::string path = dataset_id + "number_of_elements";
      unsigned long number_of_elements = ug->GetCells()->GetData()->GetNumberOfTuples();
      adios_write(fh, path.c_str(), &number_of_elements);

      // /data_object_<id>/dataset_<id>/cells/data
      path = dataset_id + "data";
      adios_write(fh, path.c_str(), ug->GetCells()->GetData()->GetPointer(0));
      }
    else if (pd)
      {
      // first move the polydata's various cell arrays into a single
      // contiguous array. and build a cell types array. doing it this
      // way simplifies the file format as we don't need to keep track
      // of all 4 cells arrays.
      std::vector<char> types;
      std::vector<vtkIdType> cells;


      vtkIdType nv = pd->GetNumberOfVerts();
      if (nv)
        {
        types.insert(types.end(), nv, VTK_VERTEX);
        vtkIdType *pv = pd->GetVerts()->GetData()->GetPointer(0);
        cells.insert(cells.end(), pv, pv + pd->GetVerts()->GetData()->GetNumberOfTuples());
        }

      vtkIdType nl = pd->GetNumberOfLines();
      if (nl)
        {
        types.insert(types.end(), nl, VTK_LINE);
        vtkIdType *pl = pd->GetLines()->GetData()->GetPointer(0);
        cells.insert(cells.end(), pl, pl + pd->GetLines()->GetData()->GetNumberOfTuples());
        }

      vtkIdType np = pd->GetNumberOfPolys();
      if (np)
        {
        types.insert(types.end(), np, VTK_POLYGON);
        vtkIdType *pp = pd->GetPolys()->GetData()->GetPointer(0);
        cells.insert(cells.end(), pp, pp + pd->GetPolys()->GetData()->GetNumberOfTuples());
        }

      vtkIdType ns = pd->GetNumberOfStrips();
      if (ns)
        {
        types.insert(types.end(), ns, VTK_TRIANGLE_STRIP);
        vtkIdType *ps = pd->GetStrips()->GetData()->GetPointer(0);
        cells.insert(cells.end(), ps, ps + pd->GetStrips()->GetData()->GetNumberOfTuples());
        }

      // /data_object_<id>/dataset_<id>/cells/cell_types
      path = dataset_id + "cell_types";
      adios_write(fh, path.c_str(), types.data());

      // /data_object_<id>/dataset_<id>/cells/number_of_elements
      std::string path = dataset_id + "number_of_elements";
      unsigned long number_of_elements = cells.size();
      adios_write(fh, path.c_str(), &number_of_elements);

      // /data_object_<id>/dataset_<id>/cells/data
      path = dataset_id + "data";
      adios_write(fh, path.c_str(), cells.data());
      }
    }

  return 0;
}

// --------------------------------------------------------------------------
int CellsSchema::Read(MPI_Comm comm, InputStream &iStream, unsigned int doid,
  unsigned int dsid, vtkDataSet *&ds)
{
  vtkPolyData *pd = dynamic_cast<vtkPolyData*>(ds);
  vtkUnstructuredGrid *ug = dynamic_cast<vtkUnstructuredGrid*>(ds);

  if (pd || ug)
    {
    std::ostringstream oss;
    oss << "data_object_" << doid << "/dataset_" << dsid << "/cells/";
    std::string dataset_id = oss.str();

    // /data_object_<id>/dataset_<id>/cells/number_of_cells
    unsigned long number_of_cells = 0;
    std::string path = dataset_id + "number_of_cells";
    if (adiosInq(iStream, path, number_of_cells))
      return -1;

    // /data_object_<id>/dataset_<id>/cells/cell_types
    path = dataset_id + "cell_types";

    vtkUnsignedCharArray *types = vtkUnsignedCharArray::New();
    types->SetNumberOfTuples(number_of_cells);

    ADIOS_SELECTION *sel = nullptr;
    if (!streamIsFileBased(iStream.ReadMethod))
      {
      int rank = 0;
      MPI_Comm_rank(comm, &rank);
      sel = adios_selection_writeblock(rank);
      if (!sel)
        {
        SENSEI_ERROR("Failed to make the selction")
        return -1;
        }
      }

    adios_schedule_read(iStream.File, sel, path.c_str(),
      0, 1, types->GetVoidPointer(0));

    // /data_object_<id>/dataset_<id>/cells/number_of_elements
    unsigned long number_of_elements = 0;
    path = dataset_id + "number_of_elements";
    if (adiosInq(iStream, path, number_of_elements))
      return -1;

    // /data_object_<id>/dataset_<id>/cells/data
    path = dataset_id + "data";
    vtkIdTypeArray *cells = vtkIdTypeArray::New();
    cells->SetNumberOfTuples(number_of_elements);

    adios_schedule_read(iStream.File, sel, path.c_str(),
      0, 1, cells->GetVoidPointer(0));

    if (adios_perform_reads(iStream.File, 1))
      {
      SENSEI_ERROR("Failed to read cells")
      return -1;
      }

    if (sel)
      adios_selection_delete(sel);

    if (ug)
      {
      // build locations
      vtkIdTypeArray *locs = vtkIdTypeArray::New();
      locs->SetNumberOfTuples(number_of_cells);
      vtkIdType *p_locs = locs->GetPointer(0);
      vtkIdType *p_cells = cells->GetPointer(0);
      p_locs[0] = 0;
      for (unsigned long i = 1; i < number_of_cells; ++i)
        p_locs[i] = p_locs[i-1] + p_cells[p_locs[i-1]];

      // pass types, locs, and cells
      vtkCellArray *ca = vtkCellArray::New();
      ca->SetCells(number_of_cells, cells);
      cells->Delete();

      ug->SetCells(types, locs, ca);

      types->Delete();
      locs->Delete();
      ca->Delete();
      }
    else if (pd)
      {
      unsigned char *p_types = types->GetPointer(0);
      vtkIdType *p_cells = cells->GetPointer(0);

      // assumptions made here:
      // data is serialized in the order verts, lines, polys, strips

      // find first and last vert and number of verts
      unsigned long i = 0;
      unsigned long n_verts = 0;
      vtkIdType *vert_begin = p_cells;
      while ((i < number_of_cells) && (p_types[i] == VTK_VERTEX))
        {
        p_cells += p_cells[0] + 1;
        ++n_verts;
        ++i;
        }
      vtkIdType *vert_end = p_cells;

      // find first and last line and number of lines
      unsigned long n_lines = 0;
      vtkIdType *line_begin = p_cells;
      while ((i < number_of_cells) && (p_types[i] == VTK_LINE))
        {
        p_cells += p_cells[0] + 1;
        ++n_lines;
        ++i;
        }
      vtkIdType *line_end = p_cells;

      // find first and last poly and number of polys
      unsigned long n_polys = 0;
      vtkIdType *poly_begin = p_cells;
      while ((i < number_of_cells) && (p_types[i] == VTK_VERTEX))
        {
        p_cells += p_cells[0] + 1;
        ++n_polys;
        ++i;
        }
      vtkIdType *poly_end = p_cells;

      // find first and last strip and number of strips
      unsigned long n_strips = 0;
      vtkIdType *strip_begin = p_cells;
      while ((i < number_of_cells) && (p_types[i] == VTK_VERTEX))
        {
        p_cells += p_cells[0] + 1;
        ++n_strips;
        ++i;
        }
      vtkIdType *strip_end = p_cells;

      // pass verts
      unsigned long n_tups = vert_end - vert_begin;
      vtkIdTypeArray *verts = vtkIdTypeArray::New();
      verts->SetNumberOfTuples(n_tups);
      vtkIdType *p_verts = verts->GetPointer(0);

      for (unsigned long j = 0; j < n_tups; ++j)
        p_verts[j] = vert_begin[j];

      vtkCellArray *ca = vtkCellArray::New();
      ca->SetCells(n_verts, verts);
      verts->Delete();

      pd->SetVerts(ca);
      ca->Delete();

      // pass lines
      n_tups = line_end - line_begin;
      vtkIdTypeArray *lines = vtkIdTypeArray::New();
      lines->SetNumberOfTuples(n_tups);
      vtkIdType *p_lines = lines->GetPointer(0);

      for (unsigned long j = 0; j < n_tups; ++j)
        p_lines[j] = line_begin[j];

      ca = vtkCellArray::New();
      ca->SetCells(n_lines, lines);
      lines->Delete();

      pd->SetLines(ca);
      ca->Delete();

      // pass polys
      n_tups = poly_end - poly_begin;
      vtkIdTypeArray *polys = vtkIdTypeArray::New();
      polys->SetNumberOfTuples(n_tups);
      vtkIdType *p_polys = polys->GetPointer(0);

      for (unsigned long j = 0; j < n_tups; ++j)
        p_polys[j] = poly_begin[j];

      ca = vtkCellArray::New();
      ca->SetCells(n_polys, polys);
      polys->Delete();

      pd->SetPolys(ca);
      ca->Delete();

      // pass strips
      n_tups = strip_end - strip_begin;
      vtkIdTypeArray *strips = vtkIdTypeArray::New();
      strips->SetNumberOfTuples(n_tups);
      vtkIdType *p_strips = strips->GetPointer(0);

      for (unsigned long j = 0; j < n_tups; ++j)
        p_strips[j] = strip_begin[j];

      ca = vtkCellArray::New();
      ca->SetCells(n_strips, strips);
      strips->Delete();

      pd->SetStrips(ca);
      ca->Delete();

      pd->BuildCells();

      cells->Delete();
      types->Delete();
      }
    }

  return 0;
}

// --------------------------------------------------------------------------
uint64_t CellsSchema::GetCellsSize(vtkDataSet *ds)
{
  return CellsSchema::GetCellsLength(ds)*sizeof(vtkIdType);
}

// --------------------------------------------------------------------------
unsigned long CellsSchema::GetCellsLength(vtkDataSet *ds)
{
  unsigned long length = 0;
  if (ds)
    {
    switch(ds->GetDataObjectType())
      {
      case VTK_POLY_DATA:
        {
        vtkPolyData *pd = static_cast<vtkPolyData*>(ds);
        length += (pd->GetVerts()->GetData()->GetNumberOfTuples() +
           pd->GetLines()->GetData()->GetNumberOfTuples() +
           pd->GetStrips()->GetData()->GetNumberOfTuples() +
           pd->GetPolys()->GetData()->GetNumberOfTuples());
        }
        break;
      case VTK_UNSTRUCTURED_GRID:
        {
        vtkUnstructuredGrid *ug = static_cast<vtkUnstructuredGrid*>(ds);
        length += ug->GetCells()->GetData()->GetNumberOfTuples();
        }
        break;
      }
    }
  return length;
}

// --------------------------------------------------------------------------
unsigned long CellsSchema::GetNumberOfCells(vtkDataSet *ds)
{
  return ds ? ds->GetNumberOfCells() : 0;
}



struct DatasetSchema::InternalsType
{
  std::set<unsigned int> Decomp;
  Extent3DSchema Extent;
  CellsSchema Cells;
  PointsSchema Points;
  PointDataSchema PointData;
  CellDataSchema CellData;
};

// --------------------------------------------------------------------------
DatasetSchema::DatasetSchema()
{
  this->Internals = new InternalsType;
}

// --------------------------------------------------------------------------
DatasetSchema::~DatasetSchema()
{
  delete this->Internals;
}

// --------------------------------------------------------------------------
void DatasetSchema::ClearDecomp()
{
  this->Internals->Decomp.clear();
}

// --------------------------------------------------------------------------
void DatasetSchema::SetDecomp(unsigned int id0, unsigned int id1)
{
  for (unsigned int i = id0; i < id1; ++i)
    this->Internals->Decomp.insert(i);
}

// --------------------------------------------------------------------------
int DatasetSchema::DefineVariables(MPI_Comm comm, int64_t gh,
  unsigned int doid, vtkDataObject *dobj)
{
  if (this->Schema::DefineVariables(comm, gh, doid, dobj) ||
    this->Internals->Extent.DefineVariables(comm, gh, doid, dobj) ||
    this->Internals->Cells.DefineVariables(comm, gh, doid, dobj) ||
    this->Internals->Points.DefineVariables(comm, gh, doid, dobj) ||
    this->Internals->PointData.DefineVariables(comm, gh, doid, dobj) ||
    this->Internals->CellData.DefineVariables(comm, gh, doid, dobj))
    {
    SENSEI_ERROR("Failed to define variables")
    return -1;
    }
  return 0;
}

// --------------------------------------------------------------------------
int DatasetSchema::DefineVariables(int64_t gh, unsigned doid,
  unsigned int dsid, vtkDataSet *ds)
{
  if (ds)
    {
    std::ostringstream oss;
    oss << "data_object_" << doid << "/dataset_" << dsid << "/";
    std::string dataset_id = oss.str();

    // /data_object_<id>/dataset_<id>/data_object_type
    std::string path = dataset_id + "data_object_type";
    adios_define_var(gh, path.c_str(), "", adios_integer, "", "", "");
    }
  return 0;
}

// --------------------------------------------------------------------------
uint64_t DatasetSchema::GetSize(MPI_Comm comm, vtkDataObject *dobj)
{
  return this->Internals->Extent.GetSize(comm, dobj) +
    this->Internals->Cells.GetSize(comm, dobj) +
    this->Internals->Points.GetSize(comm, dobj) +
    this->Internals->CellData.GetSize(comm, dobj) +
    this->Internals->PointData.GetSize(comm, dobj) +
    this->Schema::GetSize(comm, dobj);
}

// --------------------------------------------------------------------------
uint64_t DatasetSchema::GetSize(vtkDataSet *ds)
{
  if (ds)
    return sizeof(int);
  return 0;
}

// -------------------------------------------------------------------------
int DatasetSchema::Write(MPI_Comm comm, int64_t fh, unsigned int doid,
 vtkDataObject *dobj)
{
  if (this->Schema::Write(comm, fh, doid, dobj) ||
    this->Internals->Extent.Write(comm, fh, doid, dobj) ||
    this->Internals->Cells.Write(comm, fh, doid, dobj) ||
    this->Internals->Points.Write(comm, fh, doid, dobj) ||
    this->Internals->PointData.Write(comm, fh, doid, dobj) ||
    this->Internals->CellData.Write(comm, fh, doid, dobj))
    {
    SENSEI_ERROR("Failed to write")
    return -1;
    }

  return 0;
}

// --------------------------------------------------------------------------
int DatasetSchema::Write(int64_t fh, unsigned int doid, unsigned int dsid,
  vtkDataSet *ds)
{
  if (ds)
    {
    std::ostringstream oss;
    oss << "data_object_" << doid << "/dataset_" << dsid << "/";
    std::string dataset_id = oss.str();

    // /data_object_<id>/dataset_<id>/data_object_type
    std::string path = dataset_id + "data_object_type";
    int dobj_type = ds->GetDataObjectType();
    adios_write(fh, path.c_str(), &dobj_type);
    }
  return 0;
}

// --------------------------------------------------------------------------
int DatasetSchema::Read(MPI_Comm comm, InputStream &iStream,
  unsigned int doid, vtkDataObject *&dobj)
{
  if (this->Schema::Read(comm, iStream, doid, dobj) ||
    this->Internals->Extent.Read(comm, iStream, doid, dobj) ||
    this->Internals->Cells.Read(comm, iStream, doid, dobj) ||
    this->Internals->Points.Read(comm, iStream, doid, dobj) ||
    this->Internals->PointData.Read(comm, iStream, doid, dobj) ||
    this->Internals->CellData.Read(comm, iStream, doid, dobj))
    {
    SENSEI_ERROR("Failed to read")
    return -1;
    }

  return 0;
}

// --------------------------------------------------------------------------
int DatasetSchema::Read(MPI_Comm comm, InputStream &iStream,
  unsigned int doid, unsigned int dsid, vtkDataSet *&ds)
{
  (void)comm;

  ds = nullptr;

  // consult the domain decomp, if it is ours then
  // construct an instance
  if (this->Internals->Decomp.find(dsid) != this->Internals->Decomp.end())
    {
    std::ostringstream oss;
    oss << "data_object_" << doid << "/dataset_" << dsid << "/";
    std::string dataset_id = oss.str();

    // /data_object_<id>/dataset_<id>/data_object_type
    int dobj_type = 0;
    std::string path = dataset_id + "data_object_type";
    if (adiosInq(iStream, path, dobj_type))
      return -1;

    ds = dynamic_cast<vtkDataSet*>(newDataObject(dobj_type));
    }

  return 0;
}

// --------------------------------------------------------------------------
int DatasetSchema::ReadMesh(MPI_Comm comm, InputStream &iStream,
  unsigned int doid, vtkDataObject *&dobj, bool structure_only)
{
  // for composite datasets we need to initialize the local
  // datasets
  vtkCompositeDataSet *cd = nullptr;
  if ((cd = dynamic_cast<vtkCompositeDataSet*>(dobj)))
    {
    vtkCompositeDataIterator *it = cd->NewIterator();
    it->SkipEmptyNodesOff();

    for (it->InitTraversal(); !it->IsDoneWithTraversal(); it->GoToNextItem())
      {
      int dsid = it->GetCurrentFlatIndex();

      vtkDataSet *ds = nullptr;
      if (this->Read(comm, iStream, doid, dsid, ds))
        {
        it->Delete();
        SENSEI_ERROR("Failed to read data object " << doid << " dataset " << dsid);
        return -1;
        }

      cd->SetDataSet(it, ds);

      if (ds)
        ds->Delete();
      }
    it->Delete();
    }

  // structure only, means no topolgy and geometry
  if (structure_only)
    return 0;

  // read topology (cells) and geometry (points/extents)
  if (this->Internals->Extent.Read(comm, iStream, doid, dobj) ||
    this->Internals->Cells.Read(comm, iStream, doid, dobj) ||
    this->Internals->Points.Read(comm, iStream, doid, dobj))
    return -1;

  return 0;
}

// --------------------------------------------------------------------------
int DatasetSchema::ReadArrayNames(MPI_Comm comm, InputStream &iStream,
  unsigned int doid, vtkDataObject *dobj, int association,
  std::set<std::string> &array_names)
{
  switch (association)
    {
    case vtkDataObject::POINT:
      return this->Internals->PointData.ReadArrayNames(comm,
        iStream, doid, dobj, array_names);
      break;

    case vtkDataObject::CELL:
      return this->Internals->CellData.ReadArrayNames(comm,
        iStream, doid, dobj, array_names);
      break;
    }
  SENSEI_ERROR("Invalid array association " << association)
  return -1;
}

// --------------------------------------------------------------------------
int DatasetSchema::ReadArray(MPI_Comm comm, InputStream &iStream,
  unsigned int doid, vtkDataObject *dobj, int association,
  const std::string &name)
{
  (void)comm;

  switch (association)
    {
    case vtkDataObject::POINT:
      return this->Internals->PointData.ReadArray(comm,
        iStream, name, doid, dobj);
      break;
    case vtkDataObject::CELL:
      return this->Internals->CellData.ReadArray(comm,
        iStream, name, doid, dobj);
      break;
    }
  SENSEI_ERROR("Invalid array association " << association)
  return -1;
}



class VersionSchema
{
public:
  VersionSchema() : Revision(2), LowestCompatibleRevision(2) {}

  uint64_t GetSize(){ return sizeof(unsigned int); }

  int DefineVariables(int64_t gh);

  int Write(int64_t fh);

  int Read(InputStream &iStream);

private:
  unsigned int Revision;
  unsigned int LowestCompatibleRevision;
};

// --------------------------------------------------------------------------
int VersionSchema::DefineVariables(int64_t gh)
{
  // all ranks need to write this info for FLEXPATH method
  // but not the MPI method.
  // /SENSEIDataObjectSchema
  adios_define_var(gh, "DataObjectSchema", "", adios_unsigned_integer,
    "", "", "");
  return 0;
}

// --------------------------------------------------------------------------
int VersionSchema::Write(int64_t fh)
{
  // all ranks need to write this info for FLEXPATH method
  // but not the MPI method.
  adios_write(fh, "DataObjectSchema", &this->Revision);
  return 0;
}

// --------------------------------------------------------------------------
int VersionSchema::Read(InputStream &iStream)
{
  // check for the tag. if it is not present, this connot
  // be one of our files
  unsigned int revision = 0;
  if (adiosInq(iStream, "DataObjectSchema", revision))
    return -1;

  // test for version backward compatibility.
  if (revision < this->LowestCompatibleRevision)
    {
    SENSEI_ERROR("Schema revision " << this->LowestCompatibleRevision
      << " is incompatible with with revision " << revision
      << " found in the current stream")
    return -2;
    }

  return 0;
}



struct DataObjectSchema::InternalsType
{
  DatasetSchema Dataset;
};

// --------------------------------------------------------------------------
DataObjectSchema::DataObjectSchema()
{
  this->Internals = new InternalsType;
}

// --------------------------------------------------------------------------
DataObjectSchema::~DataObjectSchema()
{
  delete this->Internals;
}

// --------------------------------------------------------------------------
int DataObjectSchema::DefineVariables(MPI_Comm comm, int64_t gh,
  unsigned int doid, vtkDataObject* dobj)
{
  std::ostringstream oss;
  oss << "data_object_" << doid << "/";

  // /data_object_<id>/number_of_datasets
  std::string path = oss.str() + "number_of_datasets";
  adios_define_var(gh, path.c_str(), "", adios_unsigned_integer,
    "", "", "");

  // /data_object_<id>/data_object_type
  path = oss.str() + "data_object_type";
  adios_define_var(gh, path.c_str(), "", adios_integer, "", "", "");

  if (this->Internals->Dataset.DefineVariables(comm, gh, doid, dobj))
    {
    SENSEI_ERROR("Failed to define variables")
    return -1;
    }

  return 0;
}

// --------------------------------------------------------------------------
uint64_t DataObjectSchema::GetSize(MPI_Comm comm, vtkDataObject *dobj)
{
  return 2*sizeof(unsigned int) + sizeof(int) +
    this->Internals->Dataset.GetSize(comm, dobj);
}

// --------------------------------------------------------------------------
int DataObjectSchema::Write(MPI_Comm comm, int64_t fh, unsigned int doid,
  vtkDataObject *dobj)
{
  (void)comm;

  unsigned int n_datasets = getNumberOfDatasets(comm, dobj, 0);
  int dobj_type = dobj->GetDataObjectType();

  std::ostringstream oss;
  oss << "data_object_" << doid << "/";

  std::string path = oss.str() + "number_of_datasets";
  adios_write(fh, path.c_str(), &n_datasets);

  path = oss.str() + "data_object_type";
  adios_write(fh, path.c_str(), &dobj_type);

  if (this->Internals->Dataset.Write(comm, fh, doid, dobj))
    {
    SENSEI_ERROR("Failed to write datasets")
    return -1;
    }

  return 0;
}

// --------------------------------------------------------------------------
int DataObjectSchema::InitializeDataObject(MPI_Comm comm, InputStream &iStream,
  unsigned int doid, vtkDataObject *&dobj)
{
  int rank = 0;
  int n_ranks = 1;

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &n_ranks);

  // read the number of datasets stored in the stream
  // and the type of the root object
  std::ostringstream oss;
  oss << "data_object_" << doid << "/";

  std::string path = oss.str() + "number_of_datasets";
  unsigned int n_datasets = 0;
  if (adiosInq(iStream, path, n_datasets))
    return -1;

  int dobj_type = 0;
  path = oss.str() + "data_object_type";
  if (adiosInq(iStream, path, dobj_type))
    return -1;

  // determine if we have the old style decompostion, namely 1 legacy dataset
  // with one dataset per node, or the more modern and flexible one based on
  // VTK composite datasets.
  int legacyRootObject = isLegacyDataObject(dobj_type);

  // in VTK the legacy datasets are used in parallel only
  // when there are 1 dataset per MPI rank. Be sure that is the
  // case here. If it is not then we must use a composite
  // dataset to store multiple legacy datasets per MPI rank
  if (legacyRootObject && (static_cast<unsigned>(n_ranks) != n_datasets))
    dobj_type = VTK_MULTIBLOCK_DATA_SET;

  // construct the root object
  dobj = newDataObject(dobj_type);
  if (!dobj)
    {
    SENSEI_ERROR("Failed to create top level data object")
    return -1;
    }

  // inbitialize the root object
  if (vtkMultiBlockDataSet *ds = dynamic_cast<vtkMultiBlockDataSet*>(dobj))
    {
    ds->SetNumberOfBlocks(n_datasets);
    }
  else if (vtkMultiPieceDataSet *ds = dynamic_cast<vtkMultiPieceDataSet*>(dobj))
    {
    ds->SetNumberOfPieces(n_datasets);
    }
  // handle other cases here as needed
  // nothing to do for vtk dataset
  else if (!dynamic_cast<vtkDataSet*>(dobj))
    {
    SENSEI_ERROR("Failed to initialize " << dobj->GetClassName())
    return -1;
    }

  // compute the domain decomposition
  int n_local = n_datasets / n_ranks;
  int n_large = n_datasets % n_ranks;

  int id0 = 1 + n_local*rank + (rank < n_large ? rank : n_large);
  int id1 = id0 + n_local + (rank < n_large ? 1 : 0);

  this->Internals->Dataset.ClearDecomp();
  this->Internals->Dataset.SetDecomp(id0, id1);

  return 0;
}

// --------------------------------------------------------------------------
int DataObjectSchema::Read(MPI_Comm comm, InputStream &iStream,
  unsigned int doid, vtkDataObject *&dobj)
{
  // create the data object
  if (this->InitializeDataObject(comm, iStream, doid, dobj))
    {
    SENSEI_ERROR("Failed to initialize data object " << doid)
    return -1;
    }

  // process datasets
  if (this->Internals->Dataset.Read(comm, iStream, doid, dobj))
    {
    SENSEI_ERROR("Failed to read datasets")
    return -1;
    }

  return 0;
}

// --------------------------------------------------------------------------
int DataObjectSchema::ReadMesh(MPI_Comm comm, InputStream &iStream,
  unsigned int doid, vtkDataObject *&dobj, bool structure_only)
{
  // create the data object
  if (this->InitializeDataObject(comm, iStream, doid, dobj))
    {
    SENSEI_ERROR("Failed to initialize data object")
    return -1;
    }

  // process datasets
  if (this->Internals->Dataset.ReadMesh(comm, iStream,
    doid, dobj, structure_only))
    {
    SENSEI_ERROR("Failed to read datasets")
    return -1;
    }

  return 0;
}

// --------------------------------------------------------------------------
int DataObjectSchema::ReadArrayNames(MPI_Comm comm, InputStream &iStream,
  unsigned int doid, vtkDataObject *dobj, int association,
  std::set<std::string> &array_names)
{
  return this->Internals->Dataset.ReadArrayNames(comm, iStream, doid, dobj,
    association, array_names);
}

// --------------------------------------------------------------------------
int DataObjectSchema::ReadArray(MPI_Comm comm, InputStream &iStream,
  unsigned int doid, vtkDataObject *dobj, int association,
  const std::string &name)
{
  return this->Internals->Dataset.ReadArray(comm, iStream, doid,
    dobj, association, name);
}



using ObjectNameIdMapType = std::map<std::string, unsigned int>;

struct DataObjectCollectionSchema::InternalsType
{
  VersionSchema Version;
  DataObjectSchema DataObject;
  ObjectNameIdMapType ObjectNameIdMap;
};

// --------------------------------------------------------------------------
DataObjectCollectionSchema::DataObjectCollectionSchema()
{
  this->Internals = new DataObjectCollectionSchema::InternalsType;
}

// --------------------------------------------------------------------------
DataObjectCollectionSchema::~DataObjectCollectionSchema()
{
  delete this->Internals;
}

// --------------------------------------------------------------------------
int DataObjectCollectionSchema::DefineVariables(MPI_Comm comm, int64_t gh,
  const std::vector<std::string> &object_names,
  const std::vector<vtkDataObject*> &objects)
{
  // mark the file as ours and declare version it is written with
  this->Internals->Version.DefineVariables(gh);

  unsigned int n_objects = objects.size();

  if (objects.size() != object_names.size())
    {
    SENSEI_ERROR("Objects and names are not 1 to 1. "
      << n_objects << " objects and " << object_names.size() << " names")
    return -1;
    }

  // /time
  // /time_step
  adios_define_var(gh, "time_step" ,"", adios_unsigned_long, "", "", "");
  adios_define_var(gh, "time" ,"", adios_double, "", "", "");

  // /number_of_data_objects
  adios_define_var(gh, "number_of_data_objects", "", adios_integer,
    "", "", "");

  for (unsigned int i = 0; i < n_objects; ++i)
    {
    std::ostringstream oss;
    oss << "data_object_" << i << "/";
    std::string object_id = oss.str();

    // /data_object_<id>/name
    StringSchema::DefineVariables(gh, object_id + "name");

    if (this->Internals->DataObject.DefineVariables(comm, gh, i, objects[i]))
      {
      SENSEI_ERROR("Failed to define variables for object "
        << i << " " << object_names[i])
      return -1;
      }
    }

  return 0;
}

// --------------------------------------------------------------------------
uint64_t DataObjectCollectionSchema::GetSize(MPI_Comm comm,
  const std::vector<std::string> &object_names,
  const std::vector<vtkDataObject*> &objects)
{
  unsigned int n_objects = objects.size();

  if (objects.size() != object_names.size())
    {
    SENSEI_ERROR("Objects and names are not the same length. "
      << n_objects << " objects and " << object_names.size()
      << " names")
    return -1;
    }

  uint64_t size_of_names = 0;
  uint64_t size_of_objects = 0;
  for (unsigned int i = 0; i < n_objects; ++i)
    {
    size_of_names += StringSchema::GetSize(object_names[i]);
    size_of_objects += this->Internals->DataObject.GetSize(comm, objects[i]);
    }

  return sizeof(int) + size_of_names +
    size_of_objects + this->Internals->Version.GetSize();
}

// --------------------------------------------------------------------------
int DataObjectCollectionSchema::Write(MPI_Comm comm, int64_t fh,
  unsigned long time_step, double time,
  const std::vector<std::string> &object_names,
  const std::vector<vtkDataObject*> &objects)
{
  unsigned int n_objects = objects.size();
  if (objects.size() != object_names.size())
    {
    SENSEI_ERROR("Objects and names are not 1 to 1. "
      << n_objects << " objects and " << object_names.size()
      << " names")
    return -1;
    }

  // all ranks need to write this info for FLEXPATH method
  // but not the MPI method.
  adios_write(fh, "time_step", &time_step);
  adios_write(fh, "time", &time);

  // /number_of_data_objects
  std::string path = "number_of_data_objects";
  adios_write(fh, path.c_str(), &n_objects);

  for (unsigned int i = 0; i < n_objects; ++i)
    {
    std::ostringstream oss;
    oss << "data_object_" << i << "/";
    std::string object_id = oss.str();

    // /data_object_<id>/name
    path = object_id + "name";
    StringSchema::Write(fh, path, object_names[i]);

    if (this->Internals->DataObject.Write(comm, fh, i, objects[i]))
      {
      SENSEI_ERROR("Failed to write object " << i << " " << object_names[i])
      return -1;
      }
    }

  return 0;
}

// --------------------------------------------------------------------------
int DataObjectCollectionSchema::ReadObject(MPI_Comm comm,
  InputStream &iStream, const std::string &object_name,
  vtkDataObject *&object, bool structure_only)
{
  object = nullptr;

  unsigned int doid = 0;
  if (this->GetObjectId(comm, iStream, object_name, doid))
    {
    SENSEI_ERROR("Failed to get object id for \"" << object_name << "\"")
    return -1;
    }

  if (this->Internals->DataObject.ReadMesh(comm, iStream,
    doid, object, structure_only))
    {
    SENSEI_ERROR("Failed to read object " << doid << " \""
      << object_name << "\"")
    return -1;
    }

  return 0;
}



// --------------------------------------------------------------------------
int DataObjectCollectionSchema::ReadObjectNames(MPI_Comm comm,
  InputStream &iStream, std::vector<std::string> &object_names)
{
  (void)comm;

  if (this->Internals->ObjectNameIdMap.empty())
    {
    // /number_of_data_objects
    unsigned int n_objects = 0;
    if (adiosInq(iStream, "number_of_data_objects", n_objects))
      return -1;

    ADIOS_SELECTION *sel = nullptr;
    if (!streamIsFileBased(iStream.ReadMethod))
      {
      int rank = 0;
      MPI_Comm_rank(comm, &rank);
      sel = adios_selection_writeblock(rank);
      if (!sel)
        {
        SENSEI_ERROR("Failed to make the selction")
        return -1;
        }
      }

    for (unsigned int i = 0; i < n_objects; ++i)
      {
      std::ostringstream oss;
      oss << "data_object_" << i << "/";
      std::string data_object_id = oss.str();

      // /data_object_<id>/name
      std::string name;
      std::string path = data_object_id + "name";
      if (StringSchema::Read(iStream, sel, path, name))
        return -1;

      // store name to object id conversion
      this->Internals->ObjectNameIdMap[name] = i;
      }

    if (sel)
      adios_selection_delete(sel);
    }

  // copy object names
  ObjectNameIdMapType::iterator it =
    this->Internals->ObjectNameIdMap.begin();

  ObjectNameIdMapType::iterator end =
    this->Internals->ObjectNameIdMap.end();

  for (; it != end; ++it)
    object_names.push_back(it->first);

  return 0;
}

// --------------------------------------------------------------------------
int DataObjectCollectionSchema::GetObjectId(MPI_Comm comm,
 InputStream &iStream, const std::string &object_name,
  unsigned int &doid)
{
  doid = 0;

  std::vector<std::string> tmp;
  if (this->Internals->ObjectNameIdMap.empty() &&
    this->ReadObjectNames(comm, iStream, tmp))
    {
    SENSEI_ERROR("Failed to read object names")
    return -1;
    }

  ObjectNameIdMapType::iterator it =
    this->Internals->ObjectNameIdMap.find(object_name);

  if (it == this->Internals->ObjectNameIdMap.end())
    {
    SENSEI_ERROR("No object named \"" << object_name << "\"")
    return -1;
    }

  doid = it->second;

  return 0;
}

// --------------------------------------------------------------------------
int DataObjectCollectionSchema::ReadArrayNames(MPI_Comm comm,
  InputStream &iStream, const std::string &object_name,
  vtkDataObject *object, int association,
  std::set<std::string> &array_names)
{
  unsigned int doid = 0;
  if (this->GetObjectId(comm, iStream, object_name, doid))
    {
    SENSEI_ERROR("Failed to get object id for \"" << object_name << "\"")
    return -1;
    }

  if (this->Internals->DataObject.ReadArrayNames(comm, iStream, doid,
    object, association, array_names))
    {
    SENSEI_ERROR("Failed to read array names for object \""
      << object_name << "\"")
    return -1;
    }

  return 0;
}

// --------------------------------------------------------------------------
int DataObjectCollectionSchema::ReadArray(MPI_Comm comm, InputStream &iStream,
  const std::string &object_name, vtkDataObject *dobj, int association,
  const std::string &array_name)
{

  unsigned int doid = 0;
  if (this->GetObjectId(comm, iStream, object_name, doid))
    {
    SENSEI_ERROR("Failed to get object id for \"" << object_name << "\"")
    return -1;
    }

  if (this->Internals->DataObject.ReadArray(comm, iStream, doid, dobj,
    association, array_name))
    {
    SENSEI_ERROR("Failed to read "
      << sensei::VTKUtils::GetAttributesName(association)
      << " data array \"" << array_name << "\" from object \"" << object_name
      << "\"")
    return -1;
    }

  return 0;
}

// --------------------------------------------------------------------------
int DataObjectCollectionSchema::ReadTimeStep(MPI_Comm comm,
  InputStream &iStream, unsigned long &time_step, double &time)
{
  (void)comm;

  // read time and step values
  if (adiosInq(iStream, "time", time))
      return -1;

  if (adiosInq(iStream, "time_step", time_step))
      return -1;

  return 0;
}


/*
// --------------------------------------------------------------------------
int OutputStream::Open(MPI_Comm comm, const std::string fileName, char mode)
{
  if (adios_open(&handle, "SENSEI", fileName.c_str(), mode, comm))
    {
    SENSEI_ERROR("Failed to open \"" << fileName
      << "\" for " << mode << " mode")
    return -1;
    }

  return 0;
}

// --------------------------------------------------------------------------
int OutputStream::SetGroupSize(uint64_t size)
{
  if (adios_group_size(handle, size + sizeof(int), &group_size))
    {
    SENSEI_ERROR("Failed to set group size to " << size << " bytes")
    return -1;
    }

  return 0;
}
*/

// --------------------------------------------------------------------------
int InputStream::Open(MPI_Comm comm, ADIOS_READ_METHOD method,
  const std::string &fileName)
{
  this->Close();

  // initialize adios
  adios_read_init_method(method, comm, "verbose=1");

  // open the file
  ADIOS_FILE *file = adios_read_open(fileName.c_str(), method, comm,
    streamIsFileBased(method) ? ADIOS_LOCKMODE_ALL :
    ADIOS_LOCKMODE_CURRENT, -1.0f);

  if (!file)
    {
    SENSEI_ERROR("Failed to open \"" << fileName << "\" for reading")
    return -1;
    }

  this->File = file;
  this->ReadMethod = method;

/*
  // verify that it is one of ours
  DataObjectSchema schema;
  if (schema.CanRead(comm, *this))
    {
    SENSEI_ERROR("Failed to open \"" << fileName << "\". Stream "
      "was not written in the SENSEI ADIOS schema format")
    this->Close();
    return -1;
    }
*/
  return 0;
}

// --------------------------------------------------------------------------
int InputStream::AdvanceTimeStep()
{
  adios_release_step(this->File);

  if (adios_advance_step(this->File, 0,
    streamIsFileBased(this->ReadMethod) ? 0.0f : -1.0f))
    {
    //SENSEI_ERROR("Failed to advance to the next time step")
    return -1;
    }

  return 0;
}

// --------------------------------------------------------------------------
int InputStream::Close()
{
  if (this->File)
    {
    adios_read_close(this->File);
    adios_read_finalize_method(this->ReadMethod);
    this->File = nullptr;
    this->ReadMethod = static_cast<ADIOS_READ_METHOD>(-1);
    }

  return 0;
}

}
