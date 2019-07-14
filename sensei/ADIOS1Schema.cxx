#include "ADIOS1Schema.h"
#include "MeshMetadataMap.h"
#include "BinaryStream.h"
#include "Partitioner.h"
#include "VTKUtils.h"
#include "MPIUtils.h"
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

namespace senseiADIOS1
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
ADIOS_DATATYPES adiosType(int vtkt)
{
  switch (vtkt)
    {
    case VTK_FLOAT:
      return adios_real;
      break;
    case VTK_DOUBLE:
      return adios_double;
      break;
    case VTK_CHAR:
      return adios_byte;
      break;
    case VTK_UNSIGNED_CHAR:
      return adios_byte;
      break;
    case VTK_INT:
      return adios_integer;
      break;
    case VTK_UNSIGNED_INT:
      return adios_unsigned_integer;
      break;
    case VTK_LONG:
      if (sizeof(long) == 4)
        return adios_integer; // 32 bits
      return adios_long; // 64 bits
      break;
    case VTK_UNSIGNED_LONG:
      if (sizeof(long) == 4)
        return adios_unsigned_integer; // 32 bits
      return adios_unsigned_long; // 64 bits
      break;
    case VTK_LONG_LONG:
      return adios_long;
      break;
    case VTK_UNSIGNED_LONG_LONG:
      return adios_unsigned_long; // 64 bits
      break;
    case VTK_ID_TYPE:
      return adiosIdType();
      break;
    default:
      {
      SENSEI_ERROR("the adios type for vtk type enumeration " << vtkt
        << " is currently not implemented")
      MPI_Abort(MPI_COMM_WORLD, -1);
      }
    }
  return adios_unknown;
}

// --------------------------------------------------------------------------
unsigned int size(int vtkt)
{
  switch (vtkt)
    {
    case VTK_FLOAT:
      return sizeof(float);
      break;
    case VTK_DOUBLE:
      return sizeof(double);
      break;
    case VTK_CHAR:
      return sizeof(char);
      break;
    case VTK_UNSIGNED_CHAR:
      return sizeof(unsigned char);
      break;
    case VTK_INT:
      return sizeof(int);
      break;
    case VTK_UNSIGNED_INT:
      return sizeof(unsigned int);
      break;
    case VTK_LONG:
      return sizeof(long);
      break;
    case VTK_UNSIGNED_LONG:
      return sizeof(unsigned long);
      break;
    case VTK_LONG_LONG:
      return sizeof(long long);
      break;
    case VTK_UNSIGNED_LONG_LONG:
      return sizeof(unsigned long long);
      break;
    case VTK_ID_TYPE:
      return sizeof(vtkIdType);
      break;
    default:
      {
      SENSEI_ERROR("the adios type for vtk type enumeration " << vtkt
        << " is currently not implemented")
      MPI_Abort(MPI_COMM_WORLD, -1);
      }
    }
  return 0;
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




// helper for writing binary streams of data. binary stream is a sequence
// of bytes that has externally defined meaning.
class BinaryStreamSchema
{
public:
  static int DefineVariables(int64_t gh, const std::string &path);

  static int Write(uint64_t fh, const std::string &path,
    const sensei::BinaryStream &md);

  static int Read(InputStream &iStream, ADIOS_SELECTION *sel,
    const std::string &path, sensei::BinaryStream &md);
};

// --------------------------------------------------------------------------
int BinaryStreamSchema::DefineVariables(int64_t gh, const std::string &path)
{
  // a second variable holding the local, global, length is required
  // for writing. according to trhe docs you could write a constant
  // string literal, but that only works with BP and not FLEXPATH
  std::string len = path + "_len";
  adios_define_var(gh, len.c_str(), "", adios_integer,
    "", "", "0");

  // define the stream
  adios_define_var(gh, path.c_str(), "", adios_byte,
    len.c_str(), len.c_str(), "0");

  return 0;
}

// --------------------------------------------------------------------------
int BinaryStreamSchema::Read(InputStream &iStream, ADIOS_SELECTION *sel,
  const std::string &path, sensei::BinaryStream &str)
{
  // get metadata
  ADIOS_VARINFO *vinfo = adios_inq_var(iStream.File, path.c_str());
  if (!vinfo)
    {
    SENSEI_ERROR("ADIOS stream is missing \"" << path << "\"")
    return -1;
    }

  // allocate a buffer
  int nbytes = vinfo->dims[0];
  str.Resize(nbytes);
  str.SetReadPos(0);
  str.SetWritePos(nbytes);

  // read it
  adios_schedule_read(iStream.File, sel, path.c_str(), 0, 1, str.GetData());
  if (adios_perform_reads(iStream.File, 1))
    {
    SENSEI_ERROR("Failed to read BinaryStream at \"" << path << "\"")
    return -1;
    }

  // clean up and pass string back
  adios_free_varinfo(vinfo);

  return 0;
}

// --------------------------------------------------------------------------
int BinaryStreamSchema::Write(uint64_t fh, const std::string &path,
  const sensei::BinaryStream &str)
{
  std::string len = path + "_len";
  int n = str.Size();
  if (adios_write(fh, len.c_str(), &n) ||
    adios_write(fh, path.c_str(), str.GetData()))
    {
    SENSEI_ERROR("Failed to write BinaryStream at \"" << path << "\"")
    return -1;
    }
  return 0;
}



class VersionSchema
{
public:
  VersionSchema() : Revision(3), LowestCompatibleRevision(3) {}

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


// --------------------------------------------------------------------------
int InputStream::SetReadMethod(const std::string &method)
{
  size_t n = method.size();
  std::string lcase_method(n, ' ');
  for (size_t i = 0; i < n; ++i)
    lcase_method[i] = tolower(method[i]);

  std::map<std::string, ADIOS_READ_METHOD> methods;
  methods["bp"] = ADIOS_READ_METHOD_BP;
  methods["bp_aggregate"] = ADIOS_READ_METHOD_BP_AGGREGATE;
  methods["dataspaces"] = ADIOS_READ_METHOD_DATASPACES;
  methods["dimes"] = ADIOS_READ_METHOD_DIMES;
  methods["flexpath"] = ADIOS_READ_METHOD_FLEXPATH;

  std::map<std::string, ADIOS_READ_METHOD>::iterator it =
    methods.find(lcase_method);

  if (it == methods.end())
    {
    SENSEI_ERROR("Unsupported read method requested \"" << method << "\"")
    return -1;
    }

  this->ReadMethod = it->second;

  return 0;
}

// --------------------------------------------------------------------------
int InputStream::Open(MPI_Comm comm)
{
  return this->Open(comm, this->ReadMethod, this->FileName);
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
  this->FileName = fileName;

  return 0;
}

// --------------------------------------------------------------------------
int InputStream::AdvanceTimeStep()
{
  adios_release_step(this->File);

  if (adios_advance_step(this->File, 0,
    streamIsFileBased(this->ReadMethod) ? 0.0f : -1.0f))
    {
    this->Close();
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



struct ArraySchema
{
  int DefineVariables(MPI_Comm comm, int64_t gh,
    const std::string &ons, const sensei::MeshMetadataPtr &md);

  int DefineVariable(MPI_Comm comm, int64_t gh, const std::string &ons,
    int i, int array_type, int num_components, int array_cen,
    unsigned long long num_points_total, unsigned long long num_cells_total,
    unsigned int num_blocks, const std::vector<long> &block_num_points,
    const std::vector<long> &block_num_cells,
    const std::vector<int> &block_owner, std::vector<int64_t> &write_ids);

  int Write(MPI_Comm comm, int64_t fh,
    const sensei::MeshMetadataPtr &md, vtkCompositeDataSet *dobj);

  int Write(MPI_Comm comm, int64_t fh, unsigned int i,
    const std::string &array_name, int array_cen, vtkCompositeDataSet *dobj,
    unsigned int num_blocks, const std::vector<int> &block_owner,
    const std::vector<int64_t> &writeIds);

  int Read(MPI_Comm comm, ADIOS_FILE *fh, const std::string &ons,
    const std::string &array_name, int centering,
    const sensei::MeshMetadataPtr &md, vtkCompositeDataSet *dobj);

  int Read(MPI_Comm comm, ADIOS_FILE *fh, const std::string &ons,
    unsigned int i, const std::string &array_name, int array_type,
    unsigned long long num_components, int array_cen, unsigned int num_blocks,
    const std::vector<long> &block_num_points,
    const std::vector<long> &block_num_cells, const std::vector<int> &block_owner,
    vtkCompositeDataSet *dobj);

  std::map<std::string,std::vector<int64_t>> WriteIds;
};


// --------------------------------------------------------------------------
int ArraySchema::DefineVariable(MPI_Comm comm, int64_t gh,
  const std::string &ons, int i, int array_type, int num_components,
  int array_cen, unsigned long long num_points_total,
  unsigned long long num_cells_total, unsigned int num_blocks,
  const std::vector<long> &block_num_points,
  const std::vector<long> &block_num_cells,
  const std::vector<int> &block_owner, std::vector<int64_t> &write_ids)
{
  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  // validate centering
  if ((array_cen != vtkDataObject::POINT) && (array_cen != vtkDataObject::CELL))
    {
    SENSEI_ERROR("Invalid array centering at array " << i)
    return -1;
    }

  // put each data array in its own namespace
  std::ostringstream ans;
  ans << ons << "data_array_" << i << "/";

  // select global size either point or cell data
  unsigned long long num_elem_total = (array_cen == vtkDataObject::POINT ?
    num_points_total : num_cells_total)*num_components;

  // global size as a string
  std::ostringstream gdims;
  gdims << num_elem_total;

  // adios type of the array
  ADIOS_DATATYPES elem_type = adiosType(array_type);

  // define the variable once for each block
  unsigned long long block_offset = 0;
  for (unsigned int j = 0; j < num_blocks; ++j)
    {
    // get the block size
    unsigned long long num_elem_local = (array_cen == vtkDataObject::POINT ?
      block_num_points[j] : block_num_cells[j])*num_components;

    // define the variable for a local block
    if (block_owner[j] ==  rank)
      {
      // local size as a string
      std::ostringstream ldims;
      ldims << num_elem_local;

      // offset as a string
      std::ostringstream boffs;
      boffs << block_offset;

      // /data_object_<id>/data_array_<id>/data
      std::string path = ans.str() + "data";
      int64_t write_id = adios_define_var(gh, path.c_str(), "", elem_type,
         ldims.str().c_str(), gdims.str().c_str(), boffs.str().c_str());

      // save the write id to tell adios which block we are writing later
      write_ids[i*num_blocks + j] = write_id;
      }

    // update the block offset
    block_offset += num_elem_local;
    }

  return 0;
}

// --------------------------------------------------------------------------
int ArraySchema::DefineVariables(MPI_Comm comm, int64_t gh,
  const std::string &ons, const sensei::MeshMetadataPtr &md)
{
  std::vector<int64_t> &writeIds = this->WriteIds[md->MeshName];

  // allocate write ids
  unsigned int num_blocks = md->NumBlocks;
  unsigned int num_arrays = md->NumArrays;

  unsigned int num_ghost_arrays =
    md->NumGhostCells ? 1 : 0 + md->NumGhostNodes ? 1 : 0;

  writeIds.resize(num_blocks*(num_arrays + num_ghost_arrays));

  // compute global sizes
  unsigned long long num_points_total = 0;
  unsigned long long num_cells_total = 0;
  for (unsigned int j = 0; j < num_blocks; ++j)
    {
    num_points_total += md->BlockNumPoints[j];
    num_cells_total += md->BlockNumCells[j];
    }

  // define data arrays
  for (unsigned int i = 0; i < num_arrays; ++i)
    {
    if (this->DefineVariable(comm, gh, ons, i, md->ArrayType[i],
      md->ArrayComponents[i], md->ArrayCentering[i], num_points_total,
      num_cells_total, num_blocks, md->BlockNumPoints, md->BlockNumCells,
      md->BlockOwner, writeIds))
      return -1;
    }

  // define ghost arrays
  if (md->NumGhostCells)
    {
    if (this->DefineVariable(comm, gh, ons, num_arrays, VTK_UNSIGNED_CHAR,
      1, vtkDataObject::CELL, num_points_total, num_cells_total, num_blocks,
      md->BlockNumPoints, md->BlockNumCells, md->BlockOwner, writeIds))
      return -1;
    num_arrays += 1;
    }

  if (md->NumGhostNodes && this->DefineVariable(comm, gh, ons, num_arrays,
      VTK_UNSIGNED_CHAR, 1, vtkDataObject::POINT, num_points_total,
      num_cells_total, num_blocks, md->BlockNumPoints, md->BlockNumCells,
      md->BlockOwner, writeIds))
      return -1;

  return 0;
}


// --------------------------------------------------------------------------
int ArraySchema::Write(MPI_Comm comm, int64_t fh, unsigned int i,
  const std::string &array_name, int array_cen, vtkCompositeDataSet *dobj,
  unsigned int num_blocks, const std::vector<int> &block_owner,
  const std::vector<int64_t> &writeIds)
{
  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  vtkCompositeDataIterator *it = dobj->NewIterator();
  it->SetSkipEmptyNodes(0);
  it->InitTraversal();

  for (unsigned int j = 0; j < num_blocks; ++j)
    {
    if (block_owner[j] == rank)
      {
      vtkDataSet *ds = dynamic_cast<vtkDataSet*>(it->GetCurrentDataObject());
      if (!ds)
        {
        SENSEI_ERROR("Failed to get block " << j)
        return -1;
        }

      vtkDataSetAttributes *dsa = array_cen == vtkDataObject::POINT ?
        dynamic_cast<vtkDataSetAttributes*>(ds->GetPointData()) :
        dynamic_cast<vtkDataSetAttributes*>(ds->GetCellData());

      vtkDataArray *da = dsa->GetArray(array_name.c_str());
      if (!da)
        {
        SENSEI_ERROR("Failed to get array \"" << array_name << "\"")
        return -1;
        }

      adios_write_byid(fh, writeIds[i*num_blocks + j], da->GetVoidPointer(0));
      }

    it->GoToNextItem();
    }

  it->Delete();

  return 0;
}

// --------------------------------------------------------------------------
int ArraySchema::Write(MPI_Comm comm, int64_t fh,
  const sensei::MeshMetadataPtr &md, vtkCompositeDataSet *dobj)
{
  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  std::vector<int64_t> &writeIds = this->WriteIds[md->MeshName];

  // write data arrays
  unsigned int num_arrays = md->NumArrays;
  for (unsigned int i = 0; i < num_arrays; ++i)
    {
    if (this->Write(comm, fh, i, md->ArrayName[i], md->ArrayCentering[i],
      dobj, md->NumBlocks, md->BlockOwner, writeIds))
      return -1;
    }

  // write ghost arrays
  if (md->NumGhostCells)
    {
    if (this->Write(comm, fh, num_arrays, "vtkGhostType", vtkDataObject::CELL,
      dobj, md->NumBlocks, md->BlockOwner, writeIds))
      return -1;
    num_arrays += 1;
    }

  if (md->NumGhostNodes && this->Write(comm, fh, num_arrays,
    "vtkGhostType", vtkDataObject::POINT, dobj, md->NumBlocks,
     md->BlockOwner, writeIds))
    return -1;

  return 0;
}

// --------------------------------------------------------------------------
int ArraySchema::Read(MPI_Comm comm, ADIOS_FILE *fh, const std::string &ons,
  unsigned int i, const std::string &array_name, int array_type,
  unsigned long long num_components, int array_cen, unsigned int num_blocks,
  const std::vector<long> &block_num_points,
  const std::vector<long> &block_num_cells, const std::vector<int> &block_owner,
  vtkCompositeDataSet *dobj)
{
  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  // put each data array in its own namespace
  std::ostringstream ans;
  ans << ons << "data_array_" << i << "/";

  vtkCompositeDataIterator *it = dobj->NewIterator();
  it->SetSkipEmptyNodes(0);
  it->InitTraversal();

  // read each block
  unsigned long long block_offset = 0;
  for (unsigned int j = 0; j < num_blocks; ++j)
    {
    // get the block size
    unsigned long long num_elem_local = (array_cen == vtkDataObject::POINT ?
      block_num_points[j] : block_num_cells[j])*num_components;

    // define the variable for a local block
    if (block_owner[j] ==  rank)
      {
      uint64_t start = block_offset;
      uint64_t count = num_elem_local;;
      ADIOS_SELECTION *sel = adios_selection_boundingbox(1, &start, &count);

      vtkDataArray *array = vtkDataArray::CreateDataArray(array_type);
      array->SetNumberOfComponents(num_components);
      array->SetNumberOfTuples(num_elem_local);
      array->SetName(array_name.c_str());

      // /data_object_<id>/data_array_<id>/data
      std::string path = ans.str() + "data";
      adios_schedule_read(fh, sel, path.c_str(),
        0, 1, array->GetVoidPointer(0));

      if (adios_perform_reads(fh, 1))
        {
        SENSEI_ERROR("Failed to read points")
        return -1;
        }

      adios_selection_delete(sel);

      // pass to vtk
      vtkDataSet *ds = dynamic_cast<vtkDataSet*>(it->GetCurrentDataObject());
      if (!ds)
        {
        SENSEI_ERROR("Failed to get block " << j)
        return -1;
        }

      vtkDataSetAttributes *dsa = array_cen == vtkDataObject::POINT ?
        dynamic_cast<vtkDataSetAttributes*>(ds->GetPointData()) :
        dynamic_cast<vtkDataSetAttributes*>(ds->GetCellData());

      dsa->AddArray(array);
      array->Delete();
      }

    // update the block offset
    block_offset += num_elem_local;

    // next block
    it->GoToNextItem();
    }

  it->Delete();
  return 0;
}

// --------------------------------------------------------------------------
int ArraySchema::Read(MPI_Comm comm, ADIOS_FILE *fh, const std::string &ons,
  const std::string &name, int centering, const sensei::MeshMetadataPtr &md,
  vtkCompositeDataSet *dobj)
{
  unsigned int num_blocks = md->NumBlocks;
  unsigned int num_arrays = md->NumArrays;

  // read ghost arrays
  if (name == "vtkGhostType")
    {
    unsigned int i = centering == vtkDataObject::CELL ?
      num_arrays : num_arrays + 1;

    return this->Read(comm, fh, ons, i, "vtkGhostType", VTK_UNSIGNED_CHAR,
      1, centering, num_blocks, md->BlockNumPoints, md->BlockNumCells,
      md->BlockOwner, dobj);
    }

  // read data arrays
  for (unsigned int i = 0; i < num_arrays; ++i)
    {
    const std::string &array_name = md->ArrayName[i];
    int array_cen = md->ArrayCentering[i];

    // skip all but the requested array
    if ((centering != array_cen) || (name != array_name))
      continue;

    return this->Read(comm, fh, ons, i, array_name, md->ArrayType[i],
      md->ArrayComponents[i], array_cen, num_blocks, md->BlockNumPoints,
      md->BlockNumCells, md->BlockOwner, dobj);
    }

  return 0;
}


struct PointSchema
{
  int DefineVariables(MPI_Comm comm, int64_t gh,
    const std::string &ons, const sensei::MeshMetadataPtr &md);

  int Write(MPI_Comm comm, int64_t fh,
    const sensei::MeshMetadataPtr &md, vtkCompositeDataSet *dobj);

  int Read(MPI_Comm comm, ADIOS_FILE *fh, const std::string &ons,
    const sensei::MeshMetadataPtr &md, vtkCompositeDataSet *dobj);

  std::map<std::string, std::vector<int64_t>> WriteIds;
};

// --------------------------------------------------------------------------
int PointSchema::DefineVariables(MPI_Comm comm, int64_t gh,
  const std::string &ons, const sensei::MeshMetadataPtr &md)
{
  if (sensei::VTKUtils::Unstructured(md) || sensei::VTKUtils::Structured(md)
    || sensei::VTKUtils::Polydata(md))
    {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    // allocate write ids
    std::vector<int64_t> &writeIds = this->WriteIds[md->MeshName];
    unsigned int num_blocks = md->NumBlocks;
    writeIds.resize(num_blocks);

    // calc global size
    unsigned long long num_total = 0;
    for (unsigned int j = 0; j < num_blocks; ++j)
      {
      num_total += md->BlockNumPoints[j];
      }

    // data type for points
    ADIOS_DATATYPES type = adiosType(md->CoordinateType);

    // global sizes as a strings
    std::ostringstream gdims;
    gdims << 3*num_total;

    // define the variable once for each block
    unsigned long long block_offset = 0;

    for (unsigned int j = 0; j < num_blocks; ++j)
      {
      // get the block size
      unsigned long long num_local = md->BlockNumPoints[j];

      // define the variable for a local block
      if (md->BlockOwner[j] ==  rank)
        {
        // local size as a string
        std::ostringstream ldims;
        ldims << 3*num_local;

        // offset as a string
        std::ostringstream boffs;
        boffs << 3*block_offset;

        // /data_object_<id>/data_array_<id>/points
        std::string path_pts = ons + "points";
        int64_t write_id = adios_define_var(gh, path_pts.c_str(), "",
           type, ldims.str().c_str(), gdims.str().c_str(), boffs.str().c_str());

        // save the id for subsequent write
        writeIds[j] = write_id;
        }

      // update the block offset
      block_offset += num_local;
      }
    }

  return 0;
}

// --------------------------------------------------------------------------
int PointSchema::Write(MPI_Comm comm, int64_t fh,
  const sensei::MeshMetadataPtr &md, vtkCompositeDataSet *dobj)
{
  if (sensei::VTKUtils::Unstructured(md) || sensei::VTKUtils::Structured(md)
    || sensei::VTKUtils::Polydata(md))
    {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    std::vector<int64_t> &writeIds = this->WriteIds[md->MeshName];

    vtkCompositeDataIterator *it = dobj->NewIterator();
    it->SetSkipEmptyNodes(0);
    it->InitTraversal();

    unsigned int num_blocks = md->NumBlocks;
    for (unsigned int j = 0; j < num_blocks; ++j)
      {
      if (md->BlockOwner[j] == rank)
        {
        vtkPointSet *ds = dynamic_cast<vtkPointSet*>(it->GetCurrentDataObject());
        if (!ds)
          {
          SENSEI_ERROR("Failed to get block " << j)
          return -1;
          }

        adios_write_byid(fh, writeIds[j], ds->GetPoints()->GetData()->GetVoidPointer(0));
        }

      it->GoToNextItem();
      }
    it->Delete();
    }

  return 0;
}

// --------------------------------------------------------------------------
int PointSchema::Read(MPI_Comm comm, ADIOS_FILE *fh, const std::string &ons,
  const sensei::MeshMetadataPtr &md, vtkCompositeDataSet *dobj)
{
  if (sensei::VTKUtils::Unstructured(md) || sensei::VTKUtils::Structured(md)
    || sensei::VTKUtils::Polydata(md))
    {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    vtkCompositeDataIterator *it = dobj->NewIterator();
    it->SetSkipEmptyNodes(0);
    it->InitTraversal();

    // read local blocks
    unsigned long long block_offset = 0;
    unsigned int num_blocks = md->NumBlocks;
    for (unsigned int j = 0; j < num_blocks; ++j)
      {
      // get the block size
      unsigned long long num_local = md->BlockNumPoints[j];

      // read local block
      if (md->BlockOwner[j] ==  rank)
        {
        uint64_t start = 3*block_offset;
        uint64_t count = 3*num_local;;
        ADIOS_SELECTION *sel = adios_selection_boundingbox(1, &start, &count);

        vtkDataArray *points = vtkDataArray::CreateDataArray(md->CoordinateType);
        points->SetNumberOfComponents(3);
        points->SetNumberOfTuples(num_local);
        points->SetName("points");

        std::string path = ons + "points";
        adios_schedule_read(fh, sel, path.c_str(),
          0, 1, points->GetVoidPointer(0));

        if (adios_perform_reads(fh, 1))
          {
          SENSEI_ERROR("Failed to read points")
          return -1;
          }

        adios_selection_delete(sel);

        // pass into vtk
        vtkPoints *pts = vtkPoints::New();
        pts->SetData(points);
        points->Delete();

        vtkPointSet *ds = dynamic_cast<vtkPointSet*>(it->GetCurrentDataObject());
        if (!ds)
          {
          SENSEI_ERROR("Failed to get block " << j)
          return -1;
          }

        ds->SetPoints(pts);
        pts->Delete();
        }

      // update the block offset
      block_offset += num_local;

      // next block
      it->GoToNextItem();
      }

    it->Delete();
    }
  return 0;
}



struct UnstructuredCellSchema
{
  int DefineVariables(MPI_Comm comm, int64_t gh,
    const std::string &ons, const sensei::MeshMetadataPtr &md);

  int Write(MPI_Comm comm, int64_t fh,
    const sensei::MeshMetadataPtr &md, vtkCompositeDataSet *dobj);

  int Read(MPI_Comm comm, ADIOS_FILE *fh, const std::string &ons,
    const sensei::MeshMetadataPtr &md, vtkCompositeDataSet *dobj);

  std::map<std::string, std::vector<int64_t>> TypeWriteIds;
  std::map<std::string, std::vector<int64_t>> ArrayWriteIds;
};

// --------------------------------------------------------------------------
int UnstructuredCellSchema::DefineVariables(MPI_Comm comm, int64_t gh,
  const std::string &ons, const sensei::MeshMetadataPtr &md)
{
  if (sensei::VTKUtils::Unstructured(md))
    {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    // allocate write ids
    unsigned int num_blocks = md->NumBlocks;

    std::vector<int64_t> &typeWriteIds = this->TypeWriteIds[md->MeshName];
    typeWriteIds.resize(num_blocks);

    std::vector<int64_t> &arrayWriteIds = this->ArrayWriteIds[md->MeshName];
    arrayWriteIds.resize(num_blocks);

    // calculate global size
    unsigned long long num_cells_total = 0;
    unsigned long long cell_array_size_total = 0;

    for (unsigned int j = 0; j < num_blocks; ++j)
      {
      num_cells_total += md->BlockNumCells[j];
      cell_array_size_total += md->BlockCellArraySize[j];
      }

    // data type for cells
    ADIOS_DATATYPES cell_array_type = adiosIdType();

    // global sizes as a strings
    std::ostringstream cell_types_gdims;
    cell_types_gdims << num_cells_total;

    std::ostringstream cell_array_gdims;
    cell_array_gdims << cell_array_size_total;

    // define the variable once for each block
    unsigned long long cell_types_block_offset = 0;
    unsigned long long cell_array_block_offset = 0;

    for (unsigned int j = 0; j < num_blocks; ++j)
      {
      // get the block size
      unsigned long long num_cells_local = md->BlockNumCells[j];
      unsigned long long cell_array_size_local = md->BlockCellArraySize[j];

      // define the variable for a local block
      if (md->BlockOwner[j] ==  rank)
        {
        // local size as a string
        std::ostringstream cell_array_ldims;
        cell_array_ldims << cell_array_size_local;

        // offset as a string
        std::ostringstream cell_array_boffs;
        cell_array_boffs << cell_array_block_offset;

        // /data_object_<id>/cell_array
        std::string path_ca = ons + "cell_array";
        int64_t cell_array_write_id = adios_define_var(gh, path_ca.c_str(), "",
           cell_array_type, cell_array_ldims.str().c_str(), cell_array_gdims.str().c_str(),
           cell_array_boffs.str().c_str());

        // save the id for subsequent write
        arrayWriteIds[j] = cell_array_write_id;

        // local size as a string
        std::ostringstream cell_types_ldims;
        cell_types_ldims << num_cells_local;

        // offset as a string
        std::ostringstream cell_types_boffs;
        cell_types_boffs << cell_types_block_offset;

        // /data_object_<id>/cell_types
        std::string path_ct = ons + "cell_types";
        int64_t cell_type_write_id = adios_define_var(gh, path_ct.c_str(), "",
           adios_byte, cell_types_ldims.str().c_str(), cell_types_gdims.str().c_str(),
           cell_types_boffs.str().c_str());

        // save the write id to tell adios which block we are writing later
        typeWriteIds[j] = cell_type_write_id;
        }

      // update the block offset
      cell_types_block_offset += num_cells_local;
      cell_array_block_offset += cell_array_size_local;
      }
    }

  return 0;
}

// --------------------------------------------------------------------------
int UnstructuredCellSchema::Write(MPI_Comm comm, int64_t fh,
  const sensei::MeshMetadataPtr &md, vtkCompositeDataSet *dobj)
{
  if (sensei::VTKUtils::Unstructured(md))
    {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    std::vector<int64_t> &arrayWriteIds = this->ArrayWriteIds[md->MeshName];
    std::vector<int64_t> &typeWriteIds = this->TypeWriteIds[md->MeshName];

    vtkCompositeDataIterator *it = dobj->NewIterator();
    it->SetSkipEmptyNodes(0);
    it->InitTraversal();

    unsigned int num_blocks = md->NumBlocks;
    for (unsigned int j = 0; j < num_blocks; ++j)
      {
      // read local block
      if (md->BlockOwner[j] ==  rank)
        {
        vtkUnstructuredGrid *ds =
          dynamic_cast<vtkUnstructuredGrid*>(it->GetCurrentDataObject());
        if (!ds)
          {
          SENSEI_ERROR("Failed to get block " << j)
          return -1;
          }

        adios_write_byid(fh, typeWriteIds[j], ds->GetCellTypesArray()->GetVoidPointer(0));
        adios_write_byid(fh, arrayWriteIds[j], ds->GetCells()->GetData()->GetVoidPointer(0));
        }
      it->GoToNextItem();
      }
    it->Delete();
    }

  return 0;
}

// --------------------------------------------------------------------------
int UnstructuredCellSchema::Read(MPI_Comm comm, ADIOS_FILE *fh,
  const std::string &ons, const sensei::MeshMetadataPtr &md, vtkCompositeDataSet *dobj)
{
  if (sensei::VTKUtils::Unstructured(md))
    {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    vtkCompositeDataIterator *it = dobj->NewIterator();
    it->SetSkipEmptyNodes(0);
    it->InitTraversal();

    // calc block offsets 
    unsigned long long cell_types_block_offset = 0;
    unsigned long long cell_array_block_offset = 0;

    unsigned int num_blocks = md->NumBlocks;
    for (unsigned int j = 0; j < num_blocks; ++j)
      {
      // get the block size
      unsigned long long num_cells_local = md->BlockNumCells[j];
      unsigned long long cell_array_size_local = md->BlockCellArraySize[j];

      // define the variable for a local block
      if (md->BlockOwner[j] ==  rank)
        {
        // /data_object_<id>/cell_types
        uint64_t ct_start = cell_types_block_offset;
        uint64_t ct_count = num_cells_local;;
        ADIOS_SELECTION *ct_sel = adios_selection_boundingbox(1, &ct_start, &ct_count);

        vtkUnsignedCharArray *cell_types = vtkUnsignedCharArray::New();
        cell_types->SetNumberOfComponents(1);
        cell_types->SetNumberOfTuples(num_cells_local);
        cell_types->SetName("cell_types");

        std::string ct_path = ons + "cell_types";
        adios_schedule_read(fh, ct_sel, ct_path.c_str(),
          0, 1, cell_types->GetVoidPointer(0));

        if (adios_perform_reads(fh, 1))
          {
          SENSEI_ERROR("Failed to read cell types")
          return -1;
          }

        adios_selection_delete(ct_sel);

        // /data_object_<id>/cell_array
        uint64_t ca_start = cell_array_block_offset;
        uint64_t ca_count = cell_array_size_local;;
        ADIOS_SELECTION *ca_sel = adios_selection_boundingbox(1, &ca_start, &ca_count);

        vtkIdTypeArray *cell_array = vtkIdTypeArray::New();
        cell_array->SetNumberOfComponents(1);
        cell_array->SetNumberOfTuples(cell_array_size_local);
        cell_array->SetName("cell_array");

        std::string ca_path = ons + "cell_array";
        adios_schedule_read(fh, ca_sel, ca_path.c_str(),
          0, 1, cell_array->GetVoidPointer(0));

        if (adios_perform_reads(fh, 1))
          {
          SENSEI_ERROR("Failed to read cell_types")
          return -1;
          }

        adios_selection_delete(ca_sel);

        vtkUnstructuredGrid *ds =
          dynamic_cast<vtkUnstructuredGrid*>(it->GetCurrentDataObject());
        if (!ds)
          {
          SENSEI_ERROR("Failed to get block " << j)
          return -1;
          }
        // build locations
        vtkIdTypeArray *cell_locs = vtkIdTypeArray::New();
        cell_locs->SetNumberOfTuples(num_cells_local);
        vtkIdType *p_locs = cell_locs->GetPointer(0);
        vtkIdType *p_cells = cell_array->GetPointer(0);
        p_locs[0] = 0;
        for (unsigned long i = 1; i < num_cells_local; ++i)
          p_locs[i] = p_locs[i-1] + p_cells[p_locs[i-1]] + 1;

        // pass types, cell_locs, and cells
        vtkCellArray *ca = vtkCellArray::New();
        ca->SetCells(num_cells_local, cell_array);
        cell_array->Delete();

        ds->SetCells(cell_types, cell_locs, ca);

        cell_locs->Delete();
        cell_array->Delete();
        cell_types->Delete();
        }

      // update the block offset
      cell_types_block_offset += num_cells_local;
      cell_array_block_offset += cell_array_size_local;
      }
    }

  return 0;
}



struct PolydataCellSchema
{
  int DefineVariables(MPI_Comm comm, int64_t gh,
    const std::string &ons, const sensei::MeshMetadataPtr &md);

  int Write(MPI_Comm comm, int64_t fh,
    const sensei::MeshMetadataPtr &md, vtkCompositeDataSet *dobj);

  int Read(MPI_Comm comm, ADIOS_FILE *fh,
    const std::string &ons, const sensei::MeshMetadataPtr &md,
    vtkCompositeDataSet *dobj);

  std::map<std::string, std::vector<int64_t>> TypeWriteIds;
  std::map<std::string, std::vector<int64_t>> ArrayWriteIds;
};

// --------------------------------------------------------------------------
int PolydataCellSchema::DefineVariables(MPI_Comm comm, int64_t gh,
  const std::string &ons, const sensei::MeshMetadataPtr &md)
{
  if (sensei::VTKUtils::Polydata(md))
    {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    // allocate write ids
    unsigned int num_blocks = md->NumBlocks;

    std::vector<int64_t> &typeWriteIds = this->TypeWriteIds[md->MeshName];
    typeWriteIds.resize(num_blocks);

    std::vector<int64_t> &arrayWriteIds = this->ArrayWriteIds[md->MeshName];
    arrayWriteIds.resize(num_blocks);

    // calculate global size
    unsigned long long num_cells_total = 0;
    unsigned long long cell_array_size_total = 0;

    for (unsigned int j = 0; j < num_blocks; ++j)
      {
      num_cells_total += md->BlockNumCells[j];
      cell_array_size_total += md->BlockCellArraySize[j];
      }

    // data type for cells
    ADIOS_DATATYPES cell_array_type = adiosIdType();

    // global sizes as a strings
    std::ostringstream cell_types_gdims;
    cell_types_gdims << num_cells_total;

    std::ostringstream cell_array_gdims;
    cell_array_gdims << cell_array_size_total;

    // define the variable once for each block
    unsigned long long cell_type_block_offset = 0;
    unsigned long long cell_array_block_offset = 0;

    for (unsigned int j = 0; j < num_blocks; ++j)
      {
      // get the block size
      unsigned long long num_cells_local = md->BlockNumCells[j];
      unsigned long long cell_array_size_local = md->BlockCellArraySize[j];

      // define the variable for a local block
      if (md->BlockOwner[j] ==  rank)
        {
        // local size as a string
        std::ostringstream cell_array_ldims;
        cell_array_ldims << cell_array_size_local;

        // local size as a string
        std::ostringstream cell_types_ldims;
        cell_types_ldims << num_cells_local;

        // offset as a string
        std::ostringstream cell_types_boffs;
        cell_types_boffs << cell_type_block_offset;

        // /data_object_<id>/cell_types
        std::string path_ct = ons + "cell_types";
        int64_t cell_type_write_id = adios_define_var(gh, path_ct.c_str(), "",
           adios_byte, cell_types_ldims.str().c_str(), cell_types_gdims.str().c_str(),
           cell_types_boffs.str().c_str());

        // save the write id to tell adios which block we are writing later
        typeWriteIds[j] = cell_type_write_id;

        // offset as a string
        std::ostringstream cell_array_boffs;
        cell_array_boffs << cell_array_block_offset;

        // /data_object_<id>/cell_array
        std::string path_ca = ons + "cell_array";
        int64_t cell_array_write_id = adios_define_var(gh, path_ca.c_str(), "",
           cell_array_type, cell_array_ldims.str().c_str(), cell_array_gdims.str().c_str(),
           cell_array_boffs.str().c_str());

        // save the id for subsequent write
        arrayWriteIds[j] = cell_array_write_id;
        }

      // update the block offset
      cell_type_block_offset += num_cells_local;
      cell_array_block_offset += cell_array_size_local;
      }
    }

  return 0;
}

// --------------------------------------------------------------------------
int PolydataCellSchema::Write(MPI_Comm comm, int64_t fh,
  const sensei::MeshMetadataPtr &md, vtkCompositeDataSet *dobj)
{
  if (sensei::VTKUtils::Polydata(md))
    {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    std::vector<int64_t> &typeWriteIds = this->TypeWriteIds[md->MeshName];
    std::vector<int64_t> &arrayWriteIds = this->ArrayWriteIds[md->MeshName];

    vtkCompositeDataIterator *it = dobj->NewIterator();
    it->SetSkipEmptyNodes(0);
    it->InitTraversal();

    unsigned int num_blocks = md->NumBlocks;
    for (unsigned int j = 0; j < num_blocks; ++j)
      {
      if (md->BlockOwner[j] == rank)
        {
        vtkPolyData *pd = dynamic_cast<vtkPolyData*>(it->GetCurrentDataObject());
        if (!pd)
          {
          SENSEI_ERROR("Failed to get block " << j << " not polydata")
          return -1;
          }

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

        adios_write_byid(fh, typeWriteIds[j], types.data());
        adios_write_byid(fh, arrayWriteIds[j], cells.data());
        }

      // go to the next block
      it->GoToNextItem();
      }
    it->Delete();
    }

  return 0;
}

// --------------------------------------------------------------------------
int PolydataCellSchema::Read(MPI_Comm comm, ADIOS_FILE *fh,
  const std::string &ons, const sensei::MeshMetadataPtr &md,
  vtkCompositeDataSet *dobj)
{
  if (sensei::VTKUtils::Polydata(md))
    {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    vtkCompositeDataIterator *it = dobj->NewIterator();
    it->SetSkipEmptyNodes(0);
    it->InitTraversal();

    unsigned long long cell_block_offset = 0;
    unsigned long long cell_array_block_offset = 0;

    // read local blocks
    unsigned int num_blocks = md->NumBlocks;
    for (unsigned int j = 0; j < num_blocks; ++j)
      {
      // get the block size
      unsigned long long num_cells_local = md->BlockNumCells[j];
      unsigned long long cell_array_size_local = md->BlockCellArraySize[j];

      if (md->BlockOwner[j] == rank)
        {
        std::vector<vtkIdType> cell_array(cell_array_size_local);
        std::vector<unsigned char> cell_types(num_cells_local);

        uint64_t ct_start = cell_block_offset;
        uint64_t ct_count = num_cells_local;;
        ADIOS_SELECTION *ct_sel = adios_selection_boundingbox(1, &ct_start, &ct_count);

        // /data_object_<id>/cell_types
        std::string ct_path = ons + "cell_types";
        adios_schedule_read(fh, ct_sel, ct_path.c_str(),
          0, 1, cell_types.data());

        if (adios_perform_reads(fh, 1))
          {
          SENSEI_ERROR("Failed to read cell types")
          return -1;
          }

        adios_selection_delete(ct_sel);

        uint64_t ca_start = cell_array_block_offset;
        uint64_t ca_count = cell_array_size_local;;
        ADIOS_SELECTION *ca_sel = adios_selection_boundingbox(1, &ca_start, &ca_count);

        // /data_object_<id>/cell_array
        std::string ca_path = ons + "cell_array";
        adios_schedule_read(fh, ca_sel, ca_path.c_str(),
          0, 1, cell_array.data());

        if (adios_perform_reads(fh, 1))
          {
          SENSEI_ERROR("Failed to read cell_types")
          return -1;
          }

        adios_selection_delete(ca_sel);

        unsigned char *p_types = cell_types.data();
        vtkIdType *p_cells = cell_array.data();

        // assumptions made here:
        // data is serialized in the order verts, lines, polys, strips

        // find first and last vert and number of verts
        unsigned long i = 0;
        unsigned long n_verts = 0;
        vtkIdType *vert_begin = p_cells;
        while ((i < num_cells_local) && (p_types[i] == VTK_VERTEX))
          {
          p_cells += p_cells[0] + 1;
          ++n_verts;
          ++i;
          }
        vtkIdType *vert_end = p_cells;

        // find first and last line and number of lines
        unsigned long n_lines = 0;
        vtkIdType *line_begin = p_cells;
        while ((i < num_cells_local) && (p_types[i] == VTK_LINE))
          {
          p_cells += p_cells[0] + 1;
          ++n_lines;
          ++i;
          }
        vtkIdType *line_end = p_cells;

        // find first and last poly and number of polys
        unsigned long n_polys = 0;
        vtkIdType *poly_begin = p_cells;
        while ((i < num_cells_local) && (p_types[i] == VTK_VERTEX))
          {
          p_cells += p_cells[0] + 1;
          ++n_polys;
          ++i;
          }
        vtkIdType *poly_end = p_cells;

        // find first and last strip and number of strips
        unsigned long n_strips = 0;
        vtkIdType *strip_begin = p_cells;
        while ((i < num_cells_local) && (p_types[i] == VTK_VERTEX))
          {
          p_cells += p_cells[0] + 1;
          ++n_strips;
          ++i;
          }
        vtkIdType *strip_end = p_cells;

        // pass into vtk
        vtkPolyData *pd = dynamic_cast<vtkPolyData*>(it->GetCurrentDataObject());
        if (!pd)
          {
          SENSEI_ERROR("Failed to get block " << j)
          return -1;
          }

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
        }
      // go to the next block
      it->GoToNextItem();

      // update the block offset
      cell_block_offset += num_cells_local;
      cell_array_block_offset += cell_array_size_local;
      }

    it->Delete();
    }

  return 0;
}



struct LogicallyCartesianSchema
{
  int DefineVariables(MPI_Comm comm, int64_t gh, const std::string &ons,
    const sensei::MeshMetadataPtr &md);

  int Write(MPI_Comm comm, int64_t fh,
    const sensei::MeshMetadataPtr &md, vtkCompositeDataSet *dobj);

  int Read(MPI_Comm comm, ADIOS_FILE *fh,
    const std::string &ons, const sensei::MeshMetadataPtr &md,
    vtkCompositeDataSet *dobj);

  std::map<std::string, std::vector<int64_t>> WriteIds;
};

// --------------------------------------------------------------------------
int LogicallyCartesianSchema::DefineVariables(MPI_Comm comm, int64_t gh,
  const std::string &ons, const sensei::MeshMetadataPtr &md)
{
  if (sensei::VTKUtils::LogicallyCartesian(md))
    {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    // allocate write ids
    unsigned int num_blocks = md->NumBlocks;

    std::vector<int64_t> &writeIds = this->WriteIds[md->MeshName];
    writeIds.resize(num_blocks);

    // global sizes as a strings
    std::ostringstream hexplet_gdims;
    hexplet_gdims << 6*num_blocks;

    // define for each block
    for (unsigned int j = 0; j < num_blocks; ++j)
      {
      // define the variable for a local block
      if (md->BlockOwner[j] ==  rank)
        {
        // local size as a string
        std::ostringstream hexplet_ldims;
        hexplet_ldims << 6;

        // offset as a string
        std::ostringstream hexplet_boffs;
        hexplet_boffs << 6*j;

        // /data_object_<id>/data_array_<id>/extent
        std::string path_extent = ons + "extent";
        int64_t extent_write_id = adios_define_var(gh, path_extent.c_str(), "",
           adios_integer, hexplet_ldims.str().c_str(), hexplet_gdims.str().c_str(),
           hexplet_boffs.str().c_str());

        // save the id for subsequent write
        writeIds[j] = extent_write_id;
        }
      }
    }

  return 0;
}

// --------------------------------------------------------------------------
int LogicallyCartesianSchema::Write(MPI_Comm comm, int64_t fh,
  const sensei::MeshMetadataPtr &md, vtkCompositeDataSet *dobj)
{
  if (sensei::VTKUtils::LogicallyCartesian(md))
    {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    std::vector<int64_t> &writeIds = this->WriteIds[md->MeshName];

    vtkCompositeDataIterator *it = dobj->NewIterator();
    it->SetSkipEmptyNodes(0);
    it->InitTraversal();

    unsigned int num_blocks = md->NumBlocks;
    for (unsigned int j = 0; j < num_blocks; ++j)
      {
      // define the variable for a local block
      if (md->BlockOwner[j] ==  rank)
        {
        vtkDataObject *dobj = it->GetCurrentDataObject();
        if (!dobj)
          {
          SENSEI_ERROR("Failed to get block " << j)
          return -1;
          }
        switch (md->BlockType)
          {
          case VTK_RECTILINEAR_GRID:
            adios_write_byid(fh, writeIds[j],
              dynamic_cast<vtkRectilinearGrid*>(dobj)->GetExtent());
            break;
          case VTK_IMAGE_DATA:
            adios_write_byid(fh, writeIds[j],
              dynamic_cast<vtkImageData*>(dobj)->GetExtent());
            break;
          case VTK_STRUCTURED_GRID:
            adios_write_byid(fh, writeIds[j],
              dynamic_cast<vtkStructuredGrid*>(dobj)->GetExtent());
            break;
          }
        }
      it->GoToNextItem();
      }
    it->Delete();
    }

  return 0;
}

// --------------------------------------------------------------------------
int LogicallyCartesianSchema::Read(MPI_Comm comm, ADIOS_FILE *fh,
  const std::string &ons, const sensei::MeshMetadataPtr &md,
  vtkCompositeDataSet *dobj)
{
  if (sensei::VTKUtils::LogicallyCartesian(md))
    {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    vtkCompositeDataIterator *it = dobj->NewIterator();
    it->SetSkipEmptyNodes(0);
    it->InitTraversal();

    // read each block
    unsigned int num_blocks = md->NumBlocks;
    for (unsigned int j = 0; j < num_blocks; ++j)
      {
      // read the variable for a local block
      if (md->BlockOwner[j] ==  rank)
        {
        // /data_object_<id>/data_array_<id>/extent
        uint64_t hexplet_start = 6*j;
        uint64_t hexplet_count = 6;
        ADIOS_SELECTION *hexplet_sel =
          adios_selection_boundingbox(1, &hexplet_start, &hexplet_count);

        int ext[6] = {0};
        std::string extent_path = ons + "extent";
        adios_schedule_read(fh, hexplet_sel, extent_path.c_str(), 0, 1, ext);

        if (adios_perform_reads(fh, 1))
          {
          SENSEI_ERROR("Failed to read cell_types")
          return -1;
          }

        adios_selection_delete(hexplet_sel);

        // update the vtk object
        vtkDataObject *dobj = it->GetCurrentDataObject();
        if (!dobj)
          {
          SENSEI_ERROR("Failed to get block " << j)
          return -1;
          }
        switch (md->BlockType)
          {
          case VTK_RECTILINEAR_GRID:
            dynamic_cast<vtkRectilinearGrid*>(dobj)->SetExtent(ext);
            break;
          case VTK_IMAGE_DATA:
              dynamic_cast<vtkImageData*>(dobj)->SetExtent(ext);
            break;
          case VTK_STRUCTURED_GRID:
              dynamic_cast<vtkStructuredGrid*>(dobj)->SetExtent(ext);
            break;
          }
        }
      // next block
      it->GoToNextItem();
      }
    it->Delete();
    }

  return 0;
}



struct UniformCartesianSchema
{
  int DefineVariables(MPI_Comm comm, int64_t gh, const std::string &ons,
    const sensei::MeshMetadataPtr &md);

  int Write(MPI_Comm comm, int64_t fh,
    const sensei::MeshMetadataPtr &md, vtkCompositeDataSet *dobj);

  int Read(MPI_Comm comm, ADIOS_FILE *fh, const std::string &ons,
    const sensei::MeshMetadataPtr &md, vtkCompositeDataSet *dobj);

  std::map<std::string, std::vector<int64_t>> OriginWriteIds;
  std::map<std::string, std::vector<int64_t>> SpacingWriteIds;
};

// --------------------------------------------------------------------------
int UniformCartesianSchema::DefineVariables(MPI_Comm comm, int64_t gh,
  const std::string &ons, const sensei::MeshMetadataPtr &md)
{
  if (sensei::VTKUtils::UniformCartesian(md))
    {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    // allocate write ids
    unsigned int num_blocks = md->NumBlocks;

    std::vector<int64_t> &originWriteIds = this->OriginWriteIds[md->MeshName];
    originWriteIds.resize(num_blocks);

    std::vector<int64_t> &spacingWriteIds = this->SpacingWriteIds[md->MeshName];
    spacingWriteIds.resize(num_blocks);

    // global sizes as a strings
    std::ostringstream triplet_gdims;
    triplet_gdims << 3*num_blocks;

    // define for each block
    for (unsigned int j = 0; j < num_blocks; ++j)
      {
      // define the variable for a local block
      if (md->BlockOwner[j] ==  rank)
        {
        // local size as a string
        std::ostringstream triplet_ldims;
        triplet_ldims << 3;

        // offset as a string
        std::ostringstream triplet_boffs;
        triplet_boffs << 3*j;

        // /data_object_<id>/data_array_<id>/origin
        std::string path_origin = ons + "origin";
        int64_t origin_write_id = adios_define_var(gh, path_origin.c_str(), "",
           adios_double, triplet_ldims.str().c_str(), triplet_gdims.str().c_str(),
           triplet_boffs.str().c_str());

        // save the id for subsequent write
        originWriteIds[j] = origin_write_id;

        // /data_object_<id>/data_array_<id>/spacing
        std::string path_spacing = ons + "spacing";
        int64_t spacing_write_id = adios_define_var(gh, path_spacing.c_str(), "",
           adios_double, triplet_ldims.str().c_str(), triplet_gdims.str().c_str(),
           triplet_boffs.str().c_str());

        // save the id for subsequent write
        spacingWriteIds[j] = spacing_write_id;
        }
      }
    }

  return 0;
}

// --------------------------------------------------------------------------
int UniformCartesianSchema::Write(MPI_Comm comm, int64_t fh,
  const sensei::MeshMetadataPtr &md, vtkCompositeDataSet *dobj)
{
  if (sensei::VTKUtils::UniformCartesian(md))
    {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    std::vector<int64_t> &originWriteIds = this->OriginWriteIds[md->MeshName];
    std::vector<int64_t> &spacingWriteIds = this->SpacingWriteIds[md->MeshName];

    vtkCompositeDataIterator *it = dobj->NewIterator();
    it->SetSkipEmptyNodes(0);
    it->InitTraversal();

    unsigned int num_blocks = md->NumBlocks;
    for (unsigned int j = 0; j < num_blocks; ++j)
      {
      // define the variable for a local block
      if (md->BlockOwner[j] ==  rank)
        {
        vtkImageData *ds = dynamic_cast<vtkImageData*>(it->GetCurrentDataObject());
        if (!ds)
          {
          SENSEI_ERROR("Failed to get block " << j)
          return -1;
          }

        adios_write_byid(fh, originWriteIds[j], ds->GetOrigin());
        adios_write_byid(fh, spacingWriteIds[j], ds->GetSpacing());
        }

      it->GoToNextItem();
      }
    it->Delete();
    }

  return 0;
}

// --------------------------------------------------------------------------
int UniformCartesianSchema::Read(MPI_Comm comm, ADIOS_FILE *fh,
  const std::string &ons, const sensei::MeshMetadataPtr &md,
  vtkCompositeDataSet *dobj)
{
  if (sensei::VTKUtils::UniformCartesian(md))
    {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    vtkCompositeDataIterator *it = dobj->NewIterator();
    it->SetSkipEmptyNodes(0);
    it->InitTraversal();

    // define for each block
    unsigned int num_blocks = md->NumBlocks;
    for (unsigned int j = 0; j < num_blocks; ++j)
      {
      // define the variable for a local block
      if (md->BlockOwner[j] ==  rank)
        {
        uint64_t triplet_start = 3*j;
        uint64_t triplet_count = 3;
        ADIOS_SELECTION *triplet_sel =
          adios_selection_boundingbox(1, &triplet_start, &triplet_count);

        // /data_object_<id>/data_array_<id>/origin
        double x0[3] = {0.0};
        std::string origin_path = ons + "origin";
        adios_schedule_read(fh, triplet_sel, origin_path.c_str(), 0, 1, x0);

        // /data_object_<id>/data_array_<id>/spacing
        double dx[3] = {0.0};
        std::string spacing_path = ons + "spacing";
        adios_schedule_read(fh, triplet_sel, spacing_path.c_str(), 0, 1, dx);

        if (adios_perform_reads(fh, 1))
          {
          SENSEI_ERROR("Failed to read cell_types")
          return -1;
          }

        adios_selection_delete(triplet_sel);

        // update the vtk object
        vtkImageData *ds = dynamic_cast<vtkImageData*>(it->GetCurrentDataObject());
        if (!ds)
          {
          SENSEI_ERROR("Failed to get block " << j << " not image data")
          return -1;
          }

        ds->SetOrigin(x0);
        ds->SetSpacing(dx);
        }
      // next block
      it->GoToNextItem();
      }
    it->Delete();
    }

  return 0;
}



struct StretchedCartesianSchema
{
  int DefineVariables(MPI_Comm comm, int64_t gh, const std::string &ons,
    const sensei::MeshMetadataPtr &md);

  int Write(MPI_Comm comm, int64_t fh,
    const sensei::MeshMetadataPtr &md, vtkCompositeDataSet *dobj);

  int Read(MPI_Comm comm, ADIOS_FILE *fh, const std::string &ons,
    const sensei::MeshMetadataPtr &md, vtkCompositeDataSet *dobj);

  std::map<std::string, std::vector<int64_t>> XCoordWriteIds;
  std::map<std::string, std::vector<int64_t>> YCoordWriteIds;
  std::map<std::string, std::vector<int64_t>> ZCoordWriteIds;
};

// --------------------------------------------------------------------------
int StretchedCartesianSchema::DefineVariables(MPI_Comm comm, int64_t gh,
  const std::string &ons, const sensei::MeshMetadataPtr &md)
{
  if (sensei::VTKUtils::StretchedCartesian(md))
    {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    unsigned int num_blocks = md->NumBlocks;

    // allocate write ids
    std::vector<int64_t> &xCoordWriteIds = this->XCoordWriteIds[md->MeshName];
    xCoordWriteIds.resize(num_blocks);

    std::vector<int64_t> &yCoordWriteIds = this->YCoordWriteIds[md->MeshName];
    yCoordWriteIds.resize(num_blocks);

    std::vector<int64_t> &zCoordWriteIds = this->ZCoordWriteIds[md->MeshName];
    zCoordWriteIds.resize(num_blocks);

    // calc global size
    unsigned long long nx_total = 0;
    unsigned long long ny_total = 0;
    unsigned long long nz_total = 0;
    for (unsigned int j = 0; j < num_blocks; ++j)
      {
      int *ext = md->BlockExtents[j].data();
      nx_total += ext[1] - ext[0] + 2;
      ny_total += ext[3] - ext[2] + 2;
      nz_total += ext[5] - ext[4] + 2;
      }

    // data type for points
    ADIOS_DATATYPES point_type = adiosType(md->CoordinateType);

    // global sizes as a strings
    std::ostringstream x_gdims;
    x_gdims << nx_total;

    std::ostringstream y_gdims;
    y_gdims << ny_total;

    std::ostringstream z_gdims;
    z_gdims << nz_total;

    // define the variable once for each block
    unsigned long long x_block_offset = 0;
    unsigned long long y_block_offset = 0;
    unsigned long long z_block_offset = 0;

    for (unsigned int j = 0; j < num_blocks; ++j)
      {
      // get the block size
      int *ext = md->BlockExtents[j].data();
      unsigned long long nx_local = ext[1] - ext[0] + 2;
      unsigned long long ny_local = ext[3] - ext[2] + 2;
      unsigned long long nz_local = ext[5] - ext[4] + 2;

      // define the variable for a local block
      if (md->BlockOwner[j] ==  rank)
        {
        // local size as a string
        std::ostringstream x_ldims;
        x_ldims << nx_local;

        std::ostringstream y_ldims;
        y_ldims << ny_local;

        std::ostringstream z_ldims;
        z_ldims << nz_local;

        // offset as a string
        std::ostringstream x_boffs;
        x_boffs << x_block_offset;

        std::ostringstream y_boffs;
        y_boffs << y_block_offset;

        std::ostringstream z_boffs;
        z_boffs << z_block_offset;

        // /data_object_<id>/data_array_<id>/x_coords
        std::string path_xc = ons + "x_coords";
        int64_t xc_write_id = adios_define_var(gh, path_xc.c_str(), "",
           point_type, x_ldims.str().c_str(), x_gdims.str().c_str(),
           x_boffs.str().c_str());

        // save the id for subsequent write
        xCoordWriteIds[j] = xc_write_id;

        // /data_object_<id>/data_array_<id>/y_coords
        std::string path_yc = ons + "y_coords";
        int64_t yc_write_id = adios_define_var(gh, path_yc.c_str(), "",
           point_type, y_ldims.str().c_str(), y_gdims.str().c_str(),
           y_boffs.str().c_str());

        // save the id for subsequent write
        yCoordWriteIds[j] = yc_write_id;

        // /data_object_<id>/data_array_<id>/z_coords
        std::string path_zc = ons + "z_coords";
        int64_t zc_write_id = adios_define_var(gh, path_zc.c_str(), "",
           point_type, z_ldims.str().c_str(), z_gdims.str().c_str(),
           z_boffs.str().c_str());

        // save the id for subsequent write
        zCoordWriteIds[j] = zc_write_id;
        }

      // update the block offset
      x_block_offset += nx_local;
      y_block_offset += ny_local;
      z_block_offset += nz_local;
      }
    }

  return 0;
}

// --------------------------------------------------------------------------
int StretchedCartesianSchema::Write(MPI_Comm comm, int64_t fh,
  const sensei::MeshMetadataPtr &md, vtkCompositeDataSet *dobj)
{
  if (sensei::VTKUtils::StretchedCartesian(md))
    {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    std::vector<int64_t> &xCoordWriteIds = this->XCoordWriteIds[md->MeshName];
    std::vector<int64_t> &yCoordWriteIds = this->YCoordWriteIds[md->MeshName];
    std::vector<int64_t> &zCoordWriteIds = this->ZCoordWriteIds[md->MeshName];

    vtkCompositeDataIterator *it = dobj->NewIterator();
    it->SetSkipEmptyNodes(0);
    it->InitTraversal();

    unsigned int num_blocks = md->NumBlocks;
    for (unsigned int j = 0; j < num_blocks; ++j)
      {
      if (md->BlockOwner[j] ==  rank)
        {
        vtkRectilinearGrid *ds = dynamic_cast<vtkRectilinearGrid*>(it->GetCurrentDataObject());
        if (!ds)
          {
          SENSEI_ERROR("Failed to get block " << j << " not unstructured")
          return -1;
          }

        adios_write_byid(fh, xCoordWriteIds[j], ds->GetXCoordinates()->GetVoidPointer(0));
        adios_write_byid(fh, yCoordWriteIds[j], ds->GetXCoordinates()->GetVoidPointer(0));
        adios_write_byid(fh, zCoordWriteIds[j], ds->GetXCoordinates()->GetVoidPointer(0));
        }
      it->GoToNextItem();
      }
    it->Delete();
    }

  return 0;
}

// --------------------------------------------------------------------------
int StretchedCartesianSchema::Read(MPI_Comm comm, ADIOS_FILE *fh,
  const std::string &ons, const sensei::MeshMetadataPtr &md,
  vtkCompositeDataSet *dobj)
{
  if (sensei::VTKUtils::StretchedCartesian(md))
    {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    vtkCompositeDataIterator *it = dobj->NewIterator();
    it->SetSkipEmptyNodes(0);
    it->InitTraversal();

    unsigned long long xc_offset = 0;
    unsigned long long yc_offset = 0;
    unsigned long long zc_offset = 0;

    unsigned int num_blocks = md->NumBlocks;
    for (unsigned int j = 0; j < num_blocks; ++j)
      {
      // get the block size
      int *ext = md->BlockExtents[j].data();
      unsigned long long nx_local = ext[1] - ext[0] + 2;
      unsigned long long ny_local = ext[3] - ext[2] + 2;
      unsigned long long nz_local = ext[5] - ext[4] + 2;

      // define the variable for a local block
      if (md->BlockOwner[j] ==  rank)
        {
        // /data_object_<id>/data_array_<id>/x_coords
        uint64_t x_start = xc_offset;
        uint64_t x_count = nx_local;;
        ADIOS_SELECTION *xc_sel = adios_selection_boundingbox(1, &x_start, &x_count);

        vtkDataArray *x_coords = vtkDataArray::CreateDataArray(md->CoordinateType);
        x_coords->SetNumberOfComponents(1);
        x_coords->SetNumberOfTuples(nx_local);
        x_coords->SetName("x_coords");

        std::string xc_path = ons + "x_coords";
        adios_schedule_read(fh, xc_sel, xc_path.c_str(),
          0, 1, x_coords->GetVoidPointer(0));

        // /data_object_<id>/data_array_<id>/y_coords
        uint64_t y_start = yc_offset;
        uint64_t y_count = ny_local;;
        ADIOS_SELECTION *yc_sel = adios_selection_boundingbox(1, &y_start, &y_count);

        vtkDataArray *y_coords = vtkDataArray::CreateDataArray(md->CoordinateType);
        y_coords->SetNumberOfComponents(1);
        y_coords->SetNumberOfTuples(ny_local);
        y_coords->SetName("y_coords");

        std::string yc_path = ons + "y_coords";
        adios_schedule_read(fh, yc_sel, yc_path.c_str(),
          0, 1, y_coords->GetVoidPointer(0));

        // /data_object_<id>/data_array_<id>/z_coords
        uint64_t z_start = zc_offset;
        uint64_t z_count = nz_local;;
        ADIOS_SELECTION *zc_sel = adios_selection_boundingbox(1, &z_start, &z_count);

        vtkDataArray *z_coords = vtkDataArray::CreateDataArray(md->CoordinateType);
        z_coords->SetNumberOfComponents(1);
        z_coords->SetNumberOfTuples(nz_local);
        z_coords->SetName("z_coords");

        std::string zc_path = ons + "z_coords";
        adios_schedule_read(fh, zc_sel, zc_path.c_str(),
          0, 1, z_coords->GetVoidPointer(0));

        if (adios_perform_reads(fh, 1))
          {
          SENSEI_ERROR("Failed to read stretched Cartesian block " << j)
          return -1;
          }

        adios_selection_delete(xc_sel);
        adios_selection_delete(yc_sel);
        adios_selection_delete(zc_sel);

        // update the vtk object
        vtkRectilinearGrid *ds = dynamic_cast<vtkRectilinearGrid*>(it->GetCurrentDataObject());
        if (!ds)
          {
          SENSEI_ERROR("Failed to get block " << j)
          return -1;
          }

        ds->SetXCoordinates(x_coords);
        ds->SetYCoordinates(y_coords);
        ds->SetZCoordinates(z_coords);

        x_coords->Delete();
        y_coords->Delete();
        z_coords->Delete();
        }

      // next block
      it->GoToNextItem();

      // update the block offset
      xc_offset += nx_local;
      yc_offset += ny_local;
      zc_offset += nz_local;
      }

    it->Delete();
    }

  return 0;
}



struct DataObjectSchema
{
  int DefineVariables(MPI_Comm comm, int64_t gh,
    unsigned int doid,  const sensei::MeshMetadataPtr &md);

  int Write(MPI_Comm comm, int64_t fh, unsigned int doid,
    const sensei::MeshMetadataPtr &md, vtkCompositeDataSet *dobj);

  int ReadMesh(MPI_Comm comm, ADIOS_FILE *fh,
    unsigned int doid, const sensei::MeshMetadataPtr &md,
    vtkCompositeDataSet *&dobj, bool structure_only);

  int ReadArray(MPI_Comm comm, ADIOS_FILE *fh,
    unsigned int doid, const std::string &name, int association,
    const sensei::MeshMetadataPtr &md, vtkCompositeDataSet *dobj);

  int InitializeDataObject(MPI_Comm comm,
    const sensei::MeshMetadataPtr &md, vtkCompositeDataSet *&dobj);

  ArraySchema DataArrays;
  PointSchema Points;
  UnstructuredCellSchema UnstructuredCells;
  PolydataCellSchema PolydataCells;
  UniformCartesianSchema UniformCartesian;
  StretchedCartesianSchema StretchedCartesian;
  LogicallyCartesianSchema LogicallyCartesian;
};

// --------------------------------------------------------------------------
int DataObjectSchema::DefineVariables(MPI_Comm comm, int64_t gh,
  unsigned int doid, const sensei::MeshMetadataPtr &md)
{
  // put each data object in its own namespace
  std::ostringstream ons;
  ons << "data_object_" << doid << "/";

  if (this->DataArrays.DefineVariables(comm, gh, ons.str(), md) ||
    this->Points.DefineVariables(comm, gh, ons.str(), md) ||
    this->UnstructuredCells.DefineVariables(comm, gh, ons.str(), md) ||
    this->PolydataCells.DefineVariables(comm, gh, ons.str(), md) ||
    this->UniformCartesian.DefineVariables(comm, gh, ons.str(), md) ||
    this->StretchedCartesian.DefineVariables(comm, gh, ons.str(), md) ||
    this->LogicallyCartesian.DefineVariables(comm, gh, ons.str(), md))
    {
    SENSEI_ERROR("Failed to define variables for object "
      << doid << " \"" << md->MeshName << "\"")
    return -1;
    }

  return 0;
}

// --------------------------------------------------------------------------
int DataObjectSchema::Write(MPI_Comm comm, int64_t fh, unsigned int doid,
  const sensei::MeshMetadataPtr &md, vtkCompositeDataSet *dobj)
{
  if (this->DataArrays.Write(comm, fh, md, dobj) ||
    this->Points.Write(comm, fh, md, dobj) ||
    this->UnstructuredCells.Write(comm, fh, md, dobj) ||
    this->PolydataCells.Write(comm, fh, md, dobj) ||
    this->UniformCartesian.Write(comm, fh, md, dobj) ||
    this->StretchedCartesian.Write(comm, fh, md, dobj) ||
    this->LogicallyCartesian.Write(comm, fh, md, dobj))
    {
    SENSEI_ERROR("Failed to write for object "
      << doid << " \"" << md->MeshName << "\"")
    return -1;
    }

  return 0;
}

// --------------------------------------------------------------------------
int DataObjectSchema::ReadMesh(MPI_Comm comm, ADIOS_FILE *fh,
  unsigned int doid, const sensei::MeshMetadataPtr &md,
  vtkCompositeDataSet *&dobj, bool structure_only)
{
  // create the data object
  dobj = nullptr;
  if (this->InitializeDataObject(comm, md, dobj))
    {
    SENSEI_ERROR("Failed to initialize data object")
    return -1;
    }

  std::ostringstream ons;
  ons << "data_object_" << doid << "/";

  if ((!structure_only &&
    (this->Points.Read(comm, fh, ons.str(), md, dobj) ||
    this->UnstructuredCells.Read(comm, fh, ons.str(), md, dobj) ||
    this->PolydataCells.Read(comm, fh, ons.str(), md, dobj))) ||
    this->UniformCartesian.Read(comm, fh, ons.str(), md, dobj) ||
    this->StretchedCartesian.Read(comm, fh, ons.str(), md, dobj) ||
    this->LogicallyCartesian.Read(comm, fh, ons.str(), md, dobj))
    {
    SENSEI_ERROR("Failed to define variables for object "
      << doid << " \"" << md->MeshName << "\"")
    return -1;
    }

  return 0;
}

// --------------------------------------------------------------------------
int DataObjectSchema::ReadArray(MPI_Comm comm, ADIOS_FILE *fh,
  unsigned int doid, const std::string &name, int association,
  const sensei::MeshMetadataPtr &md, vtkCompositeDataSet *dobj)
{
  std::ostringstream ons;
  ons << "data_object_" << doid << "/";

  if (this->DataArrays.Read(comm, fh, ons.str(), name, association, md, dobj))
    {
    SENSEI_ERROR("Failed to define variables for object "
      << doid << " \"" << md->MeshName << "\"")
    return -1;
    }

  return 0;
}

// --------------------------------------------------------------------------
int DataObjectSchema::InitializeDataObject(MPI_Comm comm,
  const sensei::MeshMetadataPtr &md, vtkCompositeDataSet *&dobj)
{
  dobj = nullptr;

  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  // allocate the local dataset
  vtkMultiBlockDataSet *mbds = vtkMultiBlockDataSet::New();
  mbds->SetNumberOfBlocks(md->NumBlocks);
  for (int i = 0; i < md->NumBlocks; ++i)
    {
    if (md->BlockOwner[i] == rank)
      {
      vtkDataObject *ds = newDataObject(md->BlockType);
      mbds->SetBlock(md->BlockIds[i], ds);
      }
    }

  dobj = mbds;

  return 0;
}



struct DataObjectCollectionSchema::InternalsType
{
  VersionSchema Version;
  DataObjectSchema DataObject;
  sensei::MeshMetadataMap SenderMdMap;
  sensei::MeshMetadataMap ReceiverMdMap;
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
int DataObjectCollectionSchema::ReadMeshMetadata(MPI_Comm comm, InputStream &iStream)
{
  (void)comm;

  this->Internals->SenderMdMap.Clear();
  this->Internals->ReceiverMdMap.Clear();

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

  // read the sender mesh metadta
  for (unsigned int i = 0; i < n_objects; ++i)
    {
    std::ostringstream oss;
    oss << "data_object_" << i << "/";
    std::string data_object_id = oss.str();

    // /data_object_<id>/metadata
    sensei::BinaryStream bs;
    std::string path = data_object_id + "metadata";
    if (BinaryStreamSchema::Read(iStream, sel, path, bs))
      return -1;

    sensei::MeshMetadataPtr md = sensei::MeshMetadata::New();
    md->FromStream(bs);

    // Don't add internally generated arrays, as these
    // interfere with ghost cell/node arrays which are
    // also special cases.
    /*md->ArrayName.push_back("SenderBlockOwner");
    md->ArrayCentering.push_back(vtkDataObject::CELL);
    md->ArrayComponents.push_back(1);
    md->ArrayType.push_back(VTK_INT);

    md->ArrayName.push_back("ReceiverBlockOwner");
    md->ArrayCentering.push_back(vtkDataObject::CELL);
    md->ArrayComponents.push_back(1);
    md->ArrayType.push_back(VTK_INT);

    md->NumArrays += 2;*/

    this->Internals->SenderMdMap.PushBack(md);
    }

  // resize the receiver mesh metatadata, this will be set
  // later by the whomever is controling how the data lands
  this->Internals->ReceiverMdMap.Resize(n_objects);

  if (sel)
    adios_selection_delete(sel);

  return 0;
}

// --------------------------------------------------------------------------
int DataObjectCollectionSchema::GetSenderMeshMetadata(unsigned int id,
  sensei::MeshMetadataPtr &md)
{
  if (this->Internals->SenderMdMap.GetMeshMetadata(id, md))
    {
    SENSEI_ERROR("Failed to get mesh metadata for object " << id)
    return -1;
    }

  return 0;
}

// --------------------------------------------------------------------------
int DataObjectCollectionSchema::SetReceiverMeshMetadata(unsigned int id,
  sensei::MeshMetadataPtr &md)
{
  return this->Internals->ReceiverMdMap.SetMeshMetadata(id, md);
}


// --------------------------------------------------------------------------
int DataObjectCollectionSchema::GetReceiverMeshMetadata(unsigned int id,
  sensei::MeshMetadataPtr &md)
{
  if (this->Internals->ReceiverMdMap.GetMeshMetadata(id, md))
    {
    SENSEI_ERROR("Failed to get mesh metadata for object " << id)
    return -1;
    }

  return 0;
}

// --------------------------------------------------------------------------
int DataObjectCollectionSchema::GetNumberOfObjects(unsigned int &num)
{
  num = this->Internals->SenderMdMap.Size();
  return 0;
}

// --------------------------------------------------------------------------
int DataObjectCollectionSchema::GetObjectId(MPI_Comm comm,
  const std::string &object_name, unsigned int &doid)
{
  (void)comm;

  doid = 0;

  if (this->Internals->SenderMdMap.GetMeshId(object_name, doid))
    {
    SENSEI_ERROR("Failed to get the id of \"" << object_name << "\"")
    return -1;
    }

  return 0;
}

// --------------------------------------------------------------------------
int DataObjectCollectionSchema::DefineVariables(MPI_Comm comm, int64_t gh,
  const std::vector<sensei::MeshMetadataPtr> &metadata)
{
  // mark the file as ours and declare version it is written with
  this->Internals->Version.DefineVariables(gh);

  // /time
  // /time_step
  adios_define_var(gh, "time_step" ,"", adios_unsigned_long, "", "", "");
  adios_define_var(gh, "time" ,"", adios_double, "", "", "");

  // /number_of_data_objects
  unsigned int n_objects = metadata.size();
  adios_define_var(gh, "number_of_data_objects", "", adios_integer,
    "", "", "");

  for (unsigned int i = 0; i < n_objects; ++i)
    {
    std::ostringstream oss;
    oss << "data_object_" << i << "/";
    std::string object_id = oss.str();

    // what follows depends on a global view of the metadata
    if (!metadata[i]->GlobalView)
      {
      SENSEI_ERROR("A global view of metadata is required")
      return -1;
      }

    // /data_object_<id>/metadata
    BinaryStreamSchema::DefineVariables(gh, object_id + "metadata");

    if (this->Internals->DataObject.DefineVariables(comm, gh, i, metadata[i]))
      {
      SENSEI_ERROR("Failed to define variables for object "
        << i << " " << metadata[i]->MeshName)
      return -1;
      }
    }

  return 0;
}

// --------------------------------------------------------------------------
int DataObjectCollectionSchema::Write(MPI_Comm comm, int64_t fh,
  unsigned long time_step, double time,
  const std::vector<sensei::MeshMetadataPtr> &metadata,
  const std::vector<vtkCompositeDataSet*> &objects)
{
  unsigned int n_objects = objects.size();
  if (n_objects != metadata.size())
    {
    SENSEI_ERROR("Missing metadata for some objects. "
      << n_objects << " data objects and " << metadata.size()
      << " metadata")
    return -1;
    }

  // write the schema version
  this->Internals->Version.Write(fh);

  // all ranks need to write this info for FLEXPATH method
  // but not the MPI method.
  adios_write(fh, "time_step", &time_step);
  adios_write(fh, "time", &time);

  // /number_of_data_objects
  std::string path = "number_of_data_objects";
  adios_write(fh, path.c_str(), &n_objects);

  for (unsigned int i = 0; i < n_objects; ++i)
    {
    sensei::BinaryStream bs;
    metadata[i]->ToStream(bs);

    std::ostringstream oss;
    oss << "data_object_" << i << "/";
    std::string object_id = oss.str();

    // /data_object_<id>/metadata
    path = object_id + "metadata";
    BinaryStreamSchema::Write(fh, path, bs);

    if (this->Internals->DataObject.Write(comm, fh, i, metadata[i], objects[i]))
      {
      SENSEI_ERROR("Failed to write object " << i << " \""
        << metadata[i]->MeshName << "\"")
      return -1;
      }
    }

  return 0;
}

// --------------------------------------------------------------------------
bool DataObjectCollectionSchema::CanRead(InputStream &iStream)
{
  return this->Internals->Version.Read(iStream) == 0;
}

// --------------------------------------------------------------------------
int DataObjectCollectionSchema::ReadObject(MPI_Comm comm,
  InputStream &iStream, const std::string &object_name,
  vtkDataObject *&dobj, bool structure_only)
{
  dobj = nullptr;

  unsigned int doid = 0;
  if (this->GetObjectId(comm, object_name, doid))
    {
    SENSEI_ERROR("Failed to get object id for \"" << object_name << "\"")
    return -1;
    }

  sensei::MeshMetadataPtr md;
  if (this->Internals->ReceiverMdMap.GetMeshMetadata(doid, md))
    {
    SENSEI_ERROR("Failed to get metadata for  \"" << object_name << "\"")
    return -1;
    }

  vtkCompositeDataSet *cd = dynamic_cast<vtkCompositeDataSet*>(dobj);
  if (this->Internals->DataObject.ReadMesh(comm,
    iStream.File, doid, md, cd, structure_only))
    {
    SENSEI_ERROR("Failed to read object " << doid << " \""
      << object_name << "\"")
    return -1;
    }
  dobj = cd;

  return 0;
}

// --------------------------------------------------------------------------
int DataObjectCollectionSchema::ReadArray(MPI_Comm comm,
  InputStream &iStream, const std::string &object_name, int association,
  const std::string &array_name, vtkDataObject *dobj)
{
  // convert the mesh name into its id
  unsigned int doid = 0;
  if (this->GetObjectId(comm, object_name, doid))
    {
    SENSEI_ERROR("Failed to get object id for \"" << object_name << "\"")
    return -1;
    }

  // our factory will create vtkMultiBlock even if the sender has a legacy
  // dataset type. this enables block based re-partitioning.
  vtkCompositeDataSet *cds = dynamic_cast<vtkCompositeDataSet*>(dobj);
  if (!cds)
    {
    SENSEI_ERROR("Composite data required")
    return -1;
    }

  // get the receiver metadata. this tells how the data should land on the
  // receiver side.
  sensei::MeshMetadataPtr md;
  if (this->Internals->ReceiverMdMap.GetMeshMetadata(doid, md))
    {
    SENSEI_ERROR("Failed to get receiver metadata for  \"" << object_name << "\"")
    return -1;
    }

  // handle a special case to let us visualize block owner for debugging
  if (array_name.rfind("BlockOwner") != std::string::npos)
    {
    // if not generating owner for the receiver, get the sender metadata
    sensei::MeshMetadataPtr omd = md;
    if (array_name.find("Sender") == 0)
      {
      if (this->Internals->SenderMdMap.GetMeshMetadata(doid, omd))
        {
        SENSEI_ERROR("Failed to get sender metadata for  \"" << object_name << "\"")
        return -1;
        }
      }

    // add an array filled with BlockOwner, from either sender or receiver
    // metadata
    if (this->AddBlockOwnerArray(comm, array_name, association, omd, cds))
      {
      SENSEI_ERROR("Failed to add \"" << array_name << "\"")
      return -1;
      }

    return 0;
    }

  // read the array from the stream. this will pull data across the wire
  if (this->Internals->DataObject.ReadArray(comm,
    iStream.File, doid, array_name, association, md, cds))
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

// --------------------------------------------------------------------------
int DataObjectCollectionSchema::AddBlockOwnerArray(MPI_Comm comm,
  const std::string &name, int centering, const sensei::MeshMetadataPtr &md,
  vtkCompositeDataSet *dobj)
{
  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  unsigned int num_blocks = md->NumBlocks;
  int array_cen = centering;

  vtkCompositeDataIterator *it = dobj->NewIterator();
  it->SetSkipEmptyNodes(0);
  it->InitTraversal();

  // read each block
  for (unsigned int j = 0; j < num_blocks; ++j)
    {
    // get the block size
    unsigned long long num_elem_local = (array_cen == vtkDataObject::POINT ?
      md->BlockNumPoints[j] : md->BlockNumCells[j]);

    // define the variable for a local block
    vtkDataSet *ds = dynamic_cast<vtkDataSet*>(it->GetCurrentDataObject());
    if (ds)
      {
      // create arrays filled with sender and receiver ranks
      vtkDataArray *bo = vtkIntArray::New();
      bo->SetNumberOfTuples(num_elem_local);
      bo->SetName(name.c_str());
      bo->FillComponent(0, md->BlockOwner[j]);

      vtkDataSetAttributes *dsa = array_cen == vtkDataObject::POINT ?
        dynamic_cast<vtkDataSetAttributes*>(ds->GetPointData()) :
        dynamic_cast<vtkDataSetAttributes*>(ds->GetCellData());

      dsa->AddArray(bo);
      bo->Delete();
      }

    // next block
    it->GoToNextItem();
    }

  it->Delete();

  return 0;
}

}
