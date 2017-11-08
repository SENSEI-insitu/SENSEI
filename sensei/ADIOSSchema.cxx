#include "ADIOSSchema.h"
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
using dataset_function = std::function<int(unsigned int,vtkDataSet*)>;

// --------------------------------------------------------------------------
// apply given function to each leaf in the data object.
int apply(unsigned int id, vtkDataObject* dobj, dataset_function &func, int skip_empty=1)
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
      if ((ierr = apply(iter->GetCurrentFlatIndex(),
        iter->GetCurrentDataObject(), func, skip_empty)))
        return ierr;
      }
    }
  else if (vtkDataSet *ds = dynamic_cast<vtkDataSet*>(dobj))
    {
    int ierr = func(id, ds);
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
    dataset_function func = [&number_of_datasets](unsigned int, vtkDataSet *ds) -> int
    {
      if (dynamic_cast<dataset_t*>(ds))
        ++number_of_datasets;
      return 0;
    };

    if (apply(0, dobj, func))
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
  dataset_function func = [&number_of_datasets](unsigned int, vtkDataSet*) -> int
  {
    ++number_of_datasets;
    return 0;
  };

  if (apply(0, dobj, func) < 0)
    return -1;

  if (!local_only)
    MPI_Allreduce(MPI_IN_PLACE, &number_of_datasets, 1,
      MPI_UNSIGNED, MPI_SUM, comm);

  return number_of_datasets;
}



// --------------------------------------------------------------------------
int Schema::DefineVariables(MPI_Comm comm, int64_t gh, vtkDataObject *dobj)
{
  // lambda that defines variables for a dataset
  dataset_function func = [this,gh](unsigned int id, vtkDataSet *ds) -> int
  {
    int ierr = this->DefineVariables(gh, id, ds);
    if (ierr < 0)
      SENSEI_ERROR("Failed to define variables on dataset " << id);
    return ierr;
  };

  // if the given object is a simple dataset then id is the MPI rank.
  // shift by 1 to be consistent with composite datasets whose flat
  // index starts at 1
  int id = 0;
  if (dynamic_cast<vtkDataSet*>(dobj))
    {
    MPI_Comm_rank(comm, &id);
    id += 1;
    }

  // apply to leaf datasets
  if (apply(id, dobj, func) < 0)
    return -1;

  return 0;
}

// --------------------------------------------------------------------------
int Schema::DefineVariables(int64_t, unsigned int, vtkDataSet *)
{
  return 0;
}

// --------------------------------------------------------------------------
uint64_t Schema::GetSize(MPI_Comm comm, vtkDataObject *dobj)
{
  (void)comm;

  uint64_t size = 0;

  // function that accumulates size of a dataset
  dataset_function func = [this,&size](unsigned int, vtkDataSet *ds) -> int
  {
    size += this->GetSize(ds);
    return 0;
  };

  // apply to leaf datasets
  if (apply(0, dobj, func) < 0)
    return 0;

  return size;
}

// --------------------------------------------------------------------------
uint64_t Schema::GetSize(vtkDataSet *)
{
  return 0;
}

// --------------------------------------------------------------------------
int Schema::Write(MPI_Comm comm, int64_t fh, vtkDataObject *dobj)
{
  (void)comm;

  // lambda that writes a dataset
  dataset_function func = [this,fh](unsigned int id, vtkDataSet *ds) -> int
  {
    int ierr = this->Write(fh, id, ds);
    if (ierr < 0)
      SENSEI_ERROR("Failed to write dataset " << id);
    return ierr;
  };

  // if the given object is a simple dataset then id is the MPI rank.
  // shift by 1 to be consistent with composite datasets whose flat
  // index starts at 1
  int id = 0;
  if (dynamic_cast<vtkDataSet*>(dobj))
    {
    MPI_Comm_rank(comm, &id);
    id += 1;
    }

  // apply to leaf datasets
  if (apply(id, dobj, func) < 0)
    return -1;

  return 0;
}

// --------------------------------------------------------------------------
int Schema::Write(int64_t, unsigned int, vtkDataSet *)
{
  return 0;
}

// --------------------------------------------------------------------------
int Schema::Read(MPI_Comm comm, InputStream &iStream, vtkDataObject *&dobj)
{
  (void)comm;

  // function that reads a dataset
  dataset_function func = [&](unsigned int id, vtkDataSet *ds) -> int
  {
    int ierr = this->Read(comm, iStream, id, ds);
    if (ierr < 0)
      SENSEI_ERROR("Failed to read dataset " << id);
    return ierr;
  };

  // if the given object is a simple dataset then id is the MPI rank.
  // shift by 1 to be consistent with composite datasets whose flat
  // index starts at 1
  int id = 0;
  if (dynamic_cast<vtkDataSet*>(dobj))
    {
    MPI_Comm_rank(comm, &id);
    id += 1;
    }

  // apply to leaf datasets
  if (apply(id, dobj, func) < 0)
    return -1;

  return 0;
}

// --------------------------------------------------------------------------
int Schema::Read(MPI_Comm, InputStream &, unsigned int, vtkDataSet *&)
{
  return 0;
}



// --------------------------------------------------------------------------
int Extent3DSchema::DefineVariables(int64_t gh, unsigned int id, vtkDataSet *ds)
{
  if (dynamic_cast<vtkImageData*>(ds))
    {
    std::ostringstream oss;
    oss << "dataset_" << id;

    std::string dataset_id = oss.str();

    std::string path_len = dataset_id + "/extent_len";
    adios_define_var(gh, path_len.c_str(), "", adios_integer, "", "", "");

    std::string path = dataset_id + "/extent";
    adios_define_var(gh, path.c_str(), "", adios_integer,
      path_len.c_str(), path_len.c_str(), "0");

    path_len = dataset_id + "/origin_len";
    adios_define_var(gh, path_len.c_str(), "", adios_integer, "", "", "");

    path = dataset_id + "/origin";
    adios_define_var(gh, path.c_str(), "", adios_double,
      path_len.c_str(), path_len.c_str(), "0");

    path_len = dataset_id + "/spacing_len";
    adios_define_var(gh, path_len.c_str(), "", adios_integer, "", "", "");

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
int Extent3DSchema::Write(int64_t fh, unsigned int id, vtkDataSet *ds)
{
  if (vtkImageData *img = dynamic_cast<vtkImageData*>(ds))
    {
    std::ostringstream oss;
    oss << "dataset_" << id << "/";
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
int Extent3DSchema::Read(MPI_Comm comm, InputStream &iStream, unsigned int id,
  vtkDataSet *&ds)
{
  if (vtkImageData *img = dynamic_cast<vtkImageData*>(ds))
    {
    std::ostringstream oss;
    oss << "dataset_" << id << "/";
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
  NameIdMapType NameIdMap;
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
  unsigned int id, vtkDataSet* ds)
{
  if (ds)
    {
    std::ostringstream oss;
    oss << "dataset_" << id << "/";
    std::string dataset_id = oss.str();

    std::string att_type = datasetAttributeString<att_t>::str() + "_data/";

    std::string att_path = dataset_id + att_type;

    // /dataset_<id>/<att_type>/number_of_arrays
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

      // how do you best write a C string in ADIOS? An adios_string type
      // scalar didn't work. writing it as a byte array works. unfortunately
      // bpls displays it as an array of integer values.

      // /dataset_<id>/<att_type>/array_<i>/name_len
      std::string len = array_path + "/name_len";
      adios_define_var(gh, len.c_str(), "", adios_integer,
        "", "", "0");

      // /dataset_<id>/<att_type>/array_<i>/name
      path = array_path + "/name";
      adios_define_var(gh, path.c_str(), "", adios_byte,
        len.c_str(), len.c_str(), "0");

      // /dataset_<id>/<att_type>/array_<i>/number_of_elements
      std::string elem_path = array_path + "/number_of_elements";
      adios_define_var(gh, elem_path.c_str(), "", adiosIdType(), "", "", "");

      // /dataset_<id>/<att_type>/array_<i>/number_of_components
      path = array_path + "/number_of_components";
      adios_define_var(gh, path.c_str(), "", adios_integer, "", "", "");

      // /dataset_<id>/<att_type>/array_<i>/element_type
      path = array_path + "/element_type";
      adios_define_var(gh, path.c_str(), "", adios_integer, "", "", "");

      // /dataset_<id>/<att_type>/array_<i>/data
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

      size += strlen(da->GetName()) + 1 + sizeof(int);

      size += da->GetNumberOfTuples()*
        da->GetNumberOfComponents()*da->GetElementComponentSize();
      }
    }

  return size;
}

// -------------------------------------------------------------------------
template<int att_t>
int DatasetAttributesSchema<att_t>::Write(int64_t fh,
  unsigned int id, vtkDataSet *ds)
{
  if (ds)
    {
    vtkDataSetAttributes* dsa = ds->GetAttributes(att_t);

    std::ostringstream oss;
    oss << "dataset_" << id << "/";
    std::string dataset_id = oss.str();

    std::string att_type = datasetAttributeString<att_t>::str() + "_data/";
    std::string att_path = dataset_id + att_type;

    // /dataset_<id>/<att_type>/number_of_arrays
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

      // /dataset_<id>/<att_type>/array_<i>/name_len
      //const char *name = array->GetName();
      path = array_path + "/name_len";
      int len = strlen(array->GetName()) + 1;
      adios_write(fh, path.c_str(), &len);

      // /dataset_<id>/<att_type>/array_<i>/name
      path = array_path + "/name";
      adios_write(fh, path.c_str(), array->GetName());

      // /dataset_<id>/<att_type>/array_<i>/number_of_elements
      vtkIdType n_elem = array->GetNumberOfTuples()*array->GetNumberOfComponents();
      std::string elem_path = array_path + "/number_of_elements";
      adios_write(fh, elem_path.c_str(), &n_elem);

      // /dataset_<id>/<att_type>/array_<i>/number_of_components
      path = array_path + "/number_of_components";
      int n_comp = array->GetNumberOfComponents();
      adios_write(fh, path.c_str(), &n_comp);

      // /dataset_<id>/<att_type>/array_<i>/element_type
      path = array_path + "/element_type";
      int elem_type = array->GetDataType();
      adios_write(fh, path.c_str(), &elem_type);

      // /dataset_<id>/<att_type>/array_<i>
      path = array_path + "/data";
      adios_write(fh, path.c_str(), array->GetVoidPointer(0));
      }
    }

  return 0;
}

// --------------------------------------------------------------------------
template<int att_t>
int DatasetAttributesSchema<att_t>::Read(MPI_Comm comm, InputStream &iStream,
  unsigned int id, vtkDataSet *&ds)
{
  if (ds)
    {
    vtkDataSetAttributes* dsa = ds->GetAttributes(att_t);

    std::ostringstream oss;
    oss << "dataset_" << id << "/";
    std::string dataset_id = oss.str();

    std::string att_type = datasetAttributeString<att_t>::str() + "_data/";
    std::string att_path = dataset_id + att_type;

    // /dataset_<id>/<att_type>/number_of_arrays
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

      // /dataset_<id>/<att_type>/array_<i>/name
      path = array_path + "/name";
      ADIOS_VARINFO *vinfo = adios_inq_var(iStream.File, path.c_str());
      if (!vinfo)
        {
        SENSEI_ERROR("ADIOS stream is missing \"" << path << "\"")
        return -1;
        }
      char *name = static_cast<char*>(malloc(vinfo->dims[0]));
      adios_schedule_read(iStream.File, sel, path.c_str(), 0, 1, name);
      if (adios_perform_reads(iStream.File, 1))
        {
        SENSEI_ERROR("Failed to read " << datasetAttributeString<att_t>::str()
          << "data array name")
        return -1;
        }
      adios_free_varinfo(vinfo);

      // /dataset_<id>/<att_type>/array_<i>/number_of_elements
      vtkIdType n_elem = 0;
      path = array_path + "/number_of_elements";
      if (adiosInq(iStream, path, n_elem))
        return -1;

      // /dataset_<id>/<att_type>/array_<i>/number_of_components
      int n_comp = 0;
      path = array_path + "/number_of_components";
      if (adiosInq(iStream, path, n_comp))
        return -1;

      // /dataset_<id>/<att_type>/array_<i>/element_type
      int elem_type = 0;
      path = array_path + "/element_type";
      if (adiosInq(iStream, path, elem_type))
        return -1;

      // /dataset_<id>/<att_type>/array_<i>/data
      path = array_path + "/data";
      vtkDataArray *array = vtkDataArray::CreateDataArray(elem_type);
      array->SetNumberOfComponents(n_comp);
      array->SetNumberOfTuples(n_elem/n_comp);
      array->SetName(name);
      adios_schedule_read(iStream.File, sel, path.c_str(),
        0, 1, array->GetVoidPointer(0));
      dsa->AddArray(array);
      array->Delete();
      free(name);

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
  InputStream &iStream, vtkDataObject *dobj, std::set<std::string> &array_names)
{
  // adaptor function
  dataset_function func = [&](unsigned int id, vtkDataSet* ds) -> int
  {
    int ierr = this->ReadArrayNames(comm, iStream, id, ds, array_names);
    if (ierr < 0)
      SENSEI_ERROR("Failed to get array names in dataset " << id)
    return ierr;
  };

  this->Internals->NameIdMap.clear();

  // if the given object is a simple dataset then id is the MPI rank.
  // shift by 1 to be consistent with composite datasets whose flat
  // index starts at 1
  int id = 0;
  if (dynamic_cast<vtkDataSet*>(dobj))
    {
    MPI_Comm_rank(comm, &id);
    id += 1;
    }

  if (apply(id, dobj, func, 0) < 0)
    return -1;

  return 0;
}

// --------------------------------------------------------------------------
template<int att_t>
int DatasetAttributesSchema<att_t>::ReadArrayNames(MPI_Comm comm,
  InputStream &iStream, unsigned int id, vtkDataSet *ds,
  std::set<std::string> &array_names)
{
  std::ostringstream oss;
  oss << "dataset_" << id << "/";
  std::string dataset_id = oss.str();

  std::string att_type = datasetAttributeString<att_t>::str() + "_data/";
  std::string att_path = dataset_id + att_type;

  // /dataset_<id>/<att_type>/number_of_arrays
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

    // /dataset_<id>/<att_type>/array_<i>/name
    path = array_path + "/name";
    ADIOS_VARINFO *vinfo = adios_inq_var(iStream.File, path.c_str());
    if (!vinfo)
      {
      SENSEI_ERROR("ADIOS stream is missing \"" << path << "\"")
      return -1;
      }
    char *tmp = static_cast<char*>(malloc(vinfo->dims[0]));
    adios_schedule_read(iStream.File, sel, path.c_str(), 0, 1, tmp);
    if (adios_perform_reads(iStream.File, 1))
      {
      SENSEI_ERROR("Failed to read "
        << datasetAttributeString<att_t>::str() << " data array names")
      return -1;
      }
    adios_free_varinfo(vinfo);

    std::string name(tmp);
    free(tmp);

    // add to list of arrays
    array_names.insert(name);

    // cacahe the array id by block and name
    if (ds)
      this->Internals->NameIdMap[name][id] = i;
    }

  if (sel)
    adios_selection_delete(sel);

  return 0;
}

// --------------------------------------------------------------------------
template<int att_t>
int DatasetAttributesSchema<att_t>::ReadArray(MPI_Comm comm,
  InputStream &iStream, const std::string &array_name, vtkDataObject *dobj)
{
  // adaptor function
  dataset_function func = [&](unsigned int id, vtkDataSet* ds) -> int
  {
    int ierr = this->ReadArray(comm, iStream, array_name, id, ds);
    if (ierr < 0)
      SENSEI_ERROR("Failed to read array \"" << array_name
        << "\" from dataset " << id)
    return ierr;
  };

  // may need to build the name map, if the user never called
  // ReadArrayNames
  std::set<std::string> tmp;
  if (this->Internals->NameIdMap.empty() &&
    this->ReadArrayNames(comm, iStream, dobj, tmp))
    return -1;

  // if the given object is a simple dataset then id is the MPI rank.
  // shift by 1 to be consistent with composite datasets whose flat
  // index starts at 1
  int id = 0;
  if (dynamic_cast<vtkDataSet*>(dobj))
    {
    MPI_Comm_rank(comm, &id);
    id += 1;
    }

  if (apply(id, dobj, func) < 0)
    return -1;

  return 0;
}

// --------------------------------------------------------------------------
template<int att_t>
int DatasetAttributesSchema<att_t>::ReadArray(MPI_Comm comm, InputStream &iStream,
  const std::string &array_name, unsigned int id, vtkDataSet *ds)
{
  if (ds)
    {
    vtkDataSetAttributes* dsa = ds->GetAttributes(att_t);

    // use name and block id to get the array id
    typename InternalsType::NameIdMapType::iterator name_map_it;
    typename InternalsType::IdMapType::iterator id_map_it;
    if (((name_map_it = this->Internals->NameIdMap.find(array_name)) ==
      this->Internals->NameIdMap.end())
      || ((id_map_it = name_map_it->second.find(id)) == name_map_it->second.end()))
      {
      SENSEI_ERROR("No array \"" << array_name << "\" in "
        << datasetAttributeString<att_t>::str() << " data")
      return -1;
      }
    int array_id = id_map_it->second;

    std::ostringstream oss;
    oss << "dataset_" << id << "/"
      << datasetAttributeString<att_t>::str() << "_data/"
      << "array_" << array_id;

    std::string array_path = oss.str();

    // /dataset_<id>/<att_type>/array_<i>/number_of_elements
    vtkIdType n_elem = 0;
    std::string path = array_path + "/number_of_elements";
    if (adiosInq(iStream, path, n_elem))
      return -1;

    // /dataset_<id>/<att_type>/array_<i>/number_of_components
    int n_comp = 0;
    path = array_path + "/number_of_components";
    if (adiosInq(iStream, path, n_comp))
      return -1;

    // /dataset_<id>/<att_type>/array_<i>/element_type
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

    // /dataset_<id>/<att_type>/array_<i>/data
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
int PointsSchema::DefineVariables(int64_t gh, unsigned int id, vtkDataSet *ds)
{
  if (vtkPointSet *ps = dynamic_cast<vtkPointSet*>(ds))
    {
    std::ostringstream oss;
    oss << "dataset_" << id << "/points/";
    std::string dataset_id = oss.str();

    // dataset_<id>/n_elem
    std::string path_len = dataset_id + "n_elem";
    adios_define_var(gh, path_len.c_str(), "", adios_unsigned_long, "", "", "");

    // dataset_<id>/elem_type
    std::string path = dataset_id + "elem_type";
    adios_define_var(gh, path.c_str(), "", adios_integer, "", "", "");

    // dataset_<id>/points
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
int PointsSchema::Write(int64_t fh, unsigned int id, vtkDataSet *ds)
{
  if (vtkPointSet *ps = dynamic_cast<vtkPointSet*>(ds))
    {
    std::ostringstream oss;
    oss << "dataset_" << id << "/points/";
    std::string dataset_id = oss.str();

    // dataset_<id>/points/n_elem
    unsigned long n_elem = 3*ds->GetNumberOfPoints();
    std::string path = dataset_id + "n_elem";
    adios_write(fh, path.c_str(), &n_elem);

    vtkDataArray *pts = ps->GetPoints()->GetData();

    // dataset_<id>/points/type
    path = dataset_id + "elem_type";
    int points_type = pts->GetDataType();
    adios_write(fh, path.c_str(), &points_type);

    // dataset_<id>/points/data
    path = dataset_id + "data";
    adios_write(fh, path.c_str(), pts->GetVoidPointer(0));
    }
  return 0;
}

// --------------------------------------------------------------------------
int PointsSchema::Read(MPI_Comm comm, InputStream &iStream, unsigned int id,
  vtkDataSet *&ds)
{
  if (vtkPointSet *ps = dynamic_cast<vtkPointSet*>(ds))
    {
    std::ostringstream oss;
    oss << "dataset_" << id << "/points/";
    std::string dataset_id = oss.str();

    // dataset_<id>/points/n_elem
    unsigned long n_elem = 0;
    std::string path = dataset_id + "n_elem";
    if (adiosInq(iStream, path, n_elem))
      return -1;

    // dataset_<id>/points/type
    int elem_type = 0;
    path = dataset_id + "elem_type";
    if (adiosInq(iStream, path, elem_type))
      return -1;

    // dataset_<id>/points/data
    vtkDataArray *pts = vtkDataArray::CreateDataArray(elem_type);
    pts->SetNumberOfComponents(3);
    pts->SetNumberOfTuples(n_elem/3);
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
int CellsSchema::DefineVariables(int64_t gh, unsigned int id, vtkDataSet* ds)
{
  if (dynamic_cast<vtkPolyData*>(ds) || dynamic_cast<vtkUnstructuredGrid*>(ds))
    {
    std::ostringstream oss;
    oss << "dataset_" << id << "/cells/";
    std::string dataset_id = oss.str();

    // dataset_<id>/cells/number_of_cells
    std::string path_len = dataset_id + "n_cells";
    adios_define_var(gh, path_len.c_str(), "", adios_unsigned_long, "", "", "");

    // /dataset_<id>/cells/cell_types
    std::string path_array = dataset_id + "cell_types";
    adios_define_var(gh, path_array.c_str(), "", adios_unsigned_byte,
      path_len.c_str(), path_len.c_str(), "0");

    // dataset_<id>/cells/number_of_cells
    path_len = dataset_id + "n_elem";
    adios_define_var(gh, path_len.c_str(), "", adios_unsigned_long, "", "", "");

    // dataset_<id>/cells/data
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
    size += 2*sizeof(unsigned long) +                    // n_cells, n_elems
      this->GetNumberOfCells(ds)*sizeof(unsigned char) + // cell types array
      this->GetCellsLength(ds)*sizeof(vtkIdType);        // cells array
    }
  return size;
}

// -------------------------------------------------------------------------
int CellsSchema::Write(int64_t fh, unsigned int id, vtkDataSet *ds)
{
  vtkPolyData *pd = dynamic_cast<vtkPolyData*>(ds);
  vtkUnstructuredGrid *ug = dynamic_cast<vtkUnstructuredGrid*>(ds);

  if (pd || ug)
    {
    std::ostringstream oss;
    oss << "dataset_" << id << "/cells/";
    std::string dataset_id = oss.str();

    // dataset_<id>/cells/n_cells
    std::string path = dataset_id + "n_cells";
    unsigned long n_cells = ds->GetNumberOfCells();
    adios_write(fh, path.c_str(), &n_cells);

    if (ug)
      {
      // dataset_<id>/cells/cell_types
      path = dataset_id + "cell_types";
      adios_write(fh, path.c_str(), ug->GetCellTypesArray()->GetPointer(0));

      // dataset_<id>/cells/n_elem
      std::string path = dataset_id + "n_elem";
      unsigned long n_elem = ug->GetCells()->GetData()->GetNumberOfTuples();
      adios_write(fh, path.c_str(), &n_elem);

      // dataset_<id>/cells/data
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

      // dataset_<id>/cells/cell_types
      path = dataset_id + "cell_types";
      adios_write(fh, path.c_str(), types.data());

      // dataset_<id>/cells/n_elem
      std::string path = dataset_id + "n_elem";
      unsigned long n_elem = cells.size();
      adios_write(fh, path.c_str(), &n_elem);

      // dataset_<id>/cells/data
      path = dataset_id + "data";
      adios_write(fh, path.c_str(), cells.data());
      }
    }

  return 0;
}

// --------------------------------------------------------------------------
int CellsSchema::Read(MPI_Comm comm, InputStream &iStream, unsigned int id,
  vtkDataSet *&ds)
{
  vtkPolyData *pd = dynamic_cast<vtkPolyData*>(ds);
  vtkUnstructuredGrid *ug = dynamic_cast<vtkUnstructuredGrid*>(ds);

  if (pd || ug)
    {
    std::ostringstream oss;
    oss << "dataset_" << id << "/cells/";
    std::string dataset_id = oss.str();

    // dataset_<id>/cells/n_cells
    unsigned long n_cells = 0;
    std::string path = dataset_id + "n_cells";
    if (adiosInq(iStream, path, n_cells))
      return -1;

    // dataset_<id>/cells/cell_types
    path = dataset_id + "cell_types";

    vtkUnsignedCharArray *types = vtkUnsignedCharArray::New();
    types->SetNumberOfTuples(n_cells);

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

    // dataset_<id>/cells/n_elem
    unsigned long n_elem = 0;
    path = dataset_id + "n_elem";
    if (adiosInq(iStream, path, n_elem))
      return -1;

    // dataset_<id>/cells/data
    path = dataset_id + "data";
    vtkIdTypeArray *cells = vtkIdTypeArray::New();
    cells->SetNumberOfTuples(n_elem);

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
      locs->SetNumberOfTuples(n_cells);
      vtkIdType *p_locs = locs->GetPointer(0);
      vtkIdType *p_cells = cells->GetPointer(0);
      p_locs[0] = 0;
      for (unsigned long i = 1; i < n_cells; ++i)
        p_locs[i] = p_locs[i-1] + p_cells[p_locs[i-1]];

      // pass types, locs, and cells
      vtkCellArray *ca = vtkCellArray::New();
      ca->SetCells(n_cells, cells);
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
      // data is serialized in the order verts, lines, polys strips

      // find first and last vert and number of verts
      unsigned long i = 0;
      unsigned long n_verts = 0;
      vtkIdType *vert_begin = p_cells;
      while ((i < n_cells) && (p_types[i] == VTK_VERTEX))
        {
        p_cells += p_cells[0] + 1;
        ++n_verts;
        ++i;
        }
      vtkIdType *vert_end = p_cells;

      // find first and last line and number of lines
      unsigned long n_lines = 0;
      vtkIdType *line_begin = p_cells;
      while ((i < n_cells) && (p_types[i] == VTK_LINE))
        {
        p_cells += p_cells[0] + 1;
        ++n_lines;
        ++i;
        }
      vtkIdType *line_end = p_cells;

      // find first and last poly and number of polys
      unsigned long n_polys = 0;
      vtkIdType *poly_begin = p_cells;
      while ((i < n_cells) && (p_types[i] == VTK_VERTEX))
        {
        p_cells += p_cells[0] + 1;
        ++n_polys;
        ++i;
        }
      vtkIdType *poly_end = p_cells;

      // find first and last strip and number of strips
      unsigned long n_strips = 0;
      vtkIdType *strip_begin = p_cells;
      while ((i < n_cells) && (p_types[i] == VTK_VERTEX))
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
void DatasetSchema::SetDecomp( unsigned int id0, unsigned int id1)
{
  for (unsigned int i = id0; i < id1; ++i)
    this->Internals->Decomp.insert(i);
}

// --------------------------------------------------------------------------
int DatasetSchema::DefineVariables(MPI_Comm comm, int64_t gh, vtkDataObject *dobj)
{
  if (this->Schema::DefineVariables(comm, gh, dobj) ||
    this->Internals->Extent.DefineVariables(comm, gh, dobj) ||
    this->Internals->Cells.DefineVariables(comm, gh, dobj) ||
    this->Internals->Points.DefineVariables(comm, gh, dobj) ||
    this->Internals->PointData.DefineVariables(comm, gh, dobj) ||
    this->Internals->CellData.DefineVariables(comm, gh, dobj))
    {
    SENSEI_ERROR("Failed to define variables")
    return -1;
    }
  return 0;
}

// --------------------------------------------------------------------------
int DatasetSchema::DefineVariables(int64_t gh, unsigned int id, vtkDataSet *ds)
{
  if (ds)
    {
    std::ostringstream oss;
    oss << "dataset_" << id << "/";
    std::string dataset_id = oss.str();

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
int DatasetSchema::Write(MPI_Comm comm, int64_t fh, vtkDataObject *dobj)
{
  if (this->Schema::Write(comm, fh, dobj) ||
    this->Internals->Extent.Write(comm, fh, dobj) ||
    this->Internals->Cells.Write(comm, fh, dobj) ||
    this->Internals->Points.Write(comm, fh, dobj) ||
    this->Internals->PointData.Write(comm, fh, dobj) ||
    this->Internals->CellData.Write(comm, fh, dobj))
    {
    SENSEI_ERROR("Failed to write")
    return -1;
    }

  return 0;
}

// --------------------------------------------------------------------------
int DatasetSchema::Write(int64_t fh, unsigned int id, vtkDataSet *ds)
{
  if (ds)
    {
    std::ostringstream oss;
    oss << "dataset_" << id << "/";
    std::string dataset_id = oss.str();

    std::string path = dataset_id + "data_object_type";
    int dobj_type = ds->GetDataObjectType();
    adios_write(fh, path.c_str(), &dobj_type);
    }
  return 0;
}

// --------------------------------------------------------------------------
int DatasetSchema::Read(MPI_Comm comm, InputStream &iStream, vtkDataObject *&dobj)
{
  if (this->Schema::Read(comm, iStream, dobj) ||
    this->Internals->Extent.Read(comm, iStream, dobj) ||
    this->Internals->Cells.Read(comm, iStream, dobj) ||
    this->Internals->Points.Read(comm, iStream, dobj) ||
    this->Internals->PointData.Read(comm, iStream, dobj) ||
    this->Internals->CellData.Read(comm, iStream, dobj))
    {
    SENSEI_ERROR("Failed to read")
    return -1;
    }

  return 0;
}

// --------------------------------------------------------------------------
int DatasetSchema::Read(MPI_Comm comm, InputStream &iStream, unsigned int id,
  vtkDataSet *&ds)
{
  (void)comm;

  ds = nullptr;

  // consult the domain decomp, if it is ours then
  // construct an instance
  if (this->Internals->Decomp.find(id) != this->Internals->Decomp.end())
    {
    std::ostringstream oss;
    oss << "dataset_" << id << "/";
    std::string dataset_id = oss.str();

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
  bool structure_only, vtkDataObject *&dobj)
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
      int id = it->GetCurrentFlatIndex();

      vtkDataSet *ds = nullptr;
      if (this->Read(comm, iStream, id, ds))
        {
        it->Delete();
        SENSEI_ERROR("Failed to read dataset " << id);
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
  if (this->Internals->Extent.Read(comm, iStream, dobj) ||
    this->Internals->Cells.Read(comm, iStream, dobj) ||
    this->Internals->Points.Read(comm, iStream, dobj))
    return -1;

  return 0;
}

// --------------------------------------------------------------------------
int DatasetSchema::ReadArrayNames(MPI_Comm comm, InputStream &iStream,
  vtkDataObject *dobj, int association, std::set<std::string> &array_names)
{
  switch (association)
    {
    case vtkDataObject::POINT:
      return this->Internals->PointData.ReadArrayNames(comm,
        iStream, dobj, array_names);
      break;

    case vtkDataObject::CELL:
      return this->Internals->CellData.ReadArrayNames(comm,
        iStream, dobj, array_names);
      break;
    }
  SENSEI_ERROR("Invalid array association " << association)
  return -1;
}

// --------------------------------------------------------------------------
int DatasetSchema::ReadArray(MPI_Comm comm, InputStream &iStream,
  vtkDataObject *dobj, int association, const std::string &name)
{
  (void)comm;

  switch (association)
    {
    case vtkDataObject::POINT:
      return this->Internals->PointData.ReadArray(comm, iStream, name, dobj);
      break;
    case vtkDataObject::CELL:
      return this->Internals->CellData.ReadArray(comm, iStream, name, dobj);
      break;
    }
  SENSEI_ERROR("Invalid array association " << association)
  return -1;
}



struct DataObjectSchema::InternalsType
{
  InternalsType() : SchemaRevision(1), LowestCompatibleRevision(1) {}

  DatasetSchema dataset;

  unsigned int SchemaRevision;
  unsigned int LowestCompatibleRevision;
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
  vtkDataObject* dobj)
{
  // all ranks need to write this info for FLEXPATH method
  // but not the MPI method.
  adios_define_var(gh, "SENSEIDataObjectSchema", "", adios_unsigned_integer,
    "", "", "");

  adios_define_var(gh, "number_of_datasets", "", adios_unsigned_integer,
    "", "", "");

  adios_define_var(gh, "data_object_type" ,"", adios_integer, "", "", "");

  adios_define_var(gh, "time_step" ,"", adios_unsigned_long, "", "", "");
  adios_define_var(gh, "time" ,"", adios_double, "", "", "");

  if (this->Internals->dataset.DefineVariables(comm, gh, dobj))
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
    sizeof(unsigned long) + sizeof(double) +
    this->Internals->dataset.GetSize(comm, dobj);
}

// --------------------------------------------------------------------------
int DataObjectSchema::WriteTimeStep(MPI_Comm comm, int64_t fh,
  unsigned long time_step, double time)
{
  (void)comm;

  // all ranks need to write this info for FLEXPATH method
  // but not the MPI method.
  adios_write(fh, "time_step", &time_step);
  adios_write(fh, "time", &time);

  return 0;
}

// --------------------------------------------------------------------------
int DataObjectSchema::Write(MPI_Comm comm, int64_t fh, vtkDataObject *dobj)
{
  (void)comm;

  unsigned int n_datasets = getNumberOfDatasets(comm, dobj, 0);
  int dobj_type = dobj->GetDataObjectType();

  // all ranks need to write this info for FLEXPATH method
  // but not the MPI method.
  adios_write(fh, "SENSEIDataObjectSchema",
    &this->Internals->SchemaRevision);

  adios_write(fh, "number_of_datasets", &n_datasets);
  adios_write(fh, "data_object_type", &dobj_type);

  if (this->Internals->dataset.Write(comm, fh, dobj))
    {
    SENSEI_ERROR("Failed to write datasets")
    return -1;
    }

  return 0;
}

// --------------------------------------------------------------------------
int DataObjectSchema::CanRead(MPI_Comm comm, InputStream &iStream)
{
  (void)comm;

  // check for the tag. if it is not present, this connot
  // be one of our files
  unsigned int revision = 0;
  if (adiosInq(iStream, "SENSEIDataObjectSchema", revision))
    return -1;

  // test for version backward compatibility.
  if (revision < this->Internals->LowestCompatibleRevision)
    {
    SENSEI_ERROR("Schema revision " << this->Internals->LowestCompatibleRevision
      << " is incompatible with with revision " << revision
      << " found in the current stream")
    return -2;
    }

  return 0;
}

// --------------------------------------------------------------------------
int DataObjectSchema::InitializeDataObject(MPI_Comm comm, InputStream &iStream,
  vtkDataObject *&dobj)
{
  int rank = 0;
  int n_ranks = 1;

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &n_ranks);

  // read the number of datasets stored in the stream
  // and the type of the root object
  unsigned int n_datasets = 0;
  if (adiosInq(iStream, "number_of_datasets", n_datasets))
    return -1;

  int dobj_type = 0;
  if (adiosInq(iStream, "data_object_type", dobj_type))
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

  this->Internals->dataset.ClearDecomp();
  this->Internals->dataset.SetDecomp(id0, id1);

  return 0;
}

// --------------------------------------------------------------------------
int DataObjectSchema::Read(MPI_Comm comm, InputStream &iStream, vtkDataObject *&dobj)
{
  // create the data object
  if (this->InitializeDataObject(comm, iStream, dobj))
    {
    SENSEI_ERROR("Failed to initialize data object")
    return -1;
    }

  // process datasets
  if (this->Internals->dataset.Read(comm, iStream, dobj))
    {
    SENSEI_ERROR("Failed to read datasets")
    return -1;
    }

  return 0;
}

// --------------------------------------------------------------------------
int DataObjectSchema::ReadMesh(MPI_Comm comm, InputStream &iStream,
  bool structure_only, vtkDataObject *&dobj)
{
  // create the data object
  if (this->InitializeDataObject(comm, iStream, dobj))
    {
    SENSEI_ERROR("Failed to initialize data object")
    return -1;
    }

  // process datasets
  if (this->Internals->dataset.ReadMesh(comm, iStream, structure_only, dobj))
    {
    SENSEI_ERROR("Failed to read datasets")
    return -1;
    }

  return 0;
}

// --------------------------------------------------------------------------
int DataObjectSchema::ReadArrayNames(MPI_Comm comm, InputStream &iStream,
  vtkDataObject *dobj, int association, std::set<std::string> &array_names)
{
  return this->Internals->dataset.ReadArrayNames(comm, iStream, dobj,
    association, array_names);
}

// --------------------------------------------------------------------------
int DataObjectSchema::ReadArray(MPI_Comm comm, InputStream &iStream,
  vtkDataObject *dobj, int association, const std::string &name)
{
  return this->Internals->dataset.ReadArray(comm, iStream, dobj, association, name);
}

// --------------------------------------------------------------------------
int DataObjectSchema::ReadTimeStep(MPI_Comm comm, InputStream &iStream,
  unsigned long &time_step, double &time)
{
  (void)comm;

  // read time and step values
  if (adiosInq(iStream, "time", time))
      return -1;

  if (adiosInq(iStream, "time_step", time_step))
      return -1;

  return 0;
}



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

  // verify that it is one of ours
  DataObjectSchema schema;
  if (schema.CanRead(comm, *this))
    {
    SENSEI_ERROR("Failed to open \"" << fileName << "\". Stream "
      "was not written in the SENSEI ADIOS schema format")
    this->Close();
    return -1;
    }

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
