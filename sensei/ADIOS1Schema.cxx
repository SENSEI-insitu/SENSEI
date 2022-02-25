#include "ADIOS1Schema.h"
#include "MeshMetadataMap.h"
#include "BinaryStream.h"
#include "Partitioner.h"
#include "SVTKUtils.h"
#include "MPIUtils.h"
#include "Error.h"
#include "Profiler.h"

#include <svtkCellTypes.h>
#include <svtkCellData.h>
#include <svtkCompositeDataIterator.h>
#include <svtkCompositeDataSet.h>
#include <svtkDataSetAttributes.h>
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
#include <svtkCellArray.h>
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
  if (sizeof(svtkIdType) == sizeof(int64_t))
    {
    return adios_long; // 64 bits
    }
  else if(sizeof(svtkIdType) == sizeof(int32_t))
    {
    return adios_integer; // 32 bits
    }
  else
    {
    SENSEI_ERROR("No conversion from svtkIdType to ADIOS_DATATYPES")
    MPI_Abort(MPI_COMM_WORLD, -1);
    }
  return adios_unknown;
}

// --------------------------------------------------------------------------
ADIOS_DATATYPES adiosType(svtkDataArray* da)
{
  if (dynamic_cast<svtkFloatArray*>(da))
    {
    return adios_real;
    }
  else if (dynamic_cast<svtkDoubleArray*>(da))
    {
    return adios_double;
    }
  else if (dynamic_cast<svtkCharArray*>(da))
    {
    return adios_byte;
    }
  else if (dynamic_cast<svtkIntArray*>(da))
    {
    return adios_integer;
    }
  else if (dynamic_cast<svtkLongArray*>(da))
    {
    if (sizeof(long) == 4)
      return adios_integer; // 32 bits
    return adios_long; // 64 bits
    }
  else if (dynamic_cast<svtkLongLongArray*>(da))
    {
    return adios_long; // 64 bits
    }
  else if (dynamic_cast<svtkUnsignedCharArray*>(da))
    {
    return adios_unsigned_byte;
    }
  else if (dynamic_cast<svtkUnsignedIntArray*>(da))
    {
    return adios_unsigned_integer;
    }
  else if (dynamic_cast<svtkUnsignedLongArray*>(da))
    {
    if (sizeof(unsigned long) == 4)
      return adios_unsigned_integer; // 32 bits
    return adios_unsigned_long; // 64 bits
    }
  else if (dynamic_cast<svtkUnsignedLongLongArray*>(da))
    {
    return adios_unsigned_long; // 64 bits
    }
  else if (dynamic_cast<svtkIdTypeArray*>(da))
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
ADIOS_DATATYPES adiosType(int svtkt)
{
  switch (svtkt)
    {
    case SVTK_FLOAT:
      return adios_real;
      break;
    case SVTK_DOUBLE:
      return adios_double;
      break;
    case SVTK_CHAR:
      return adios_byte;
      break;
    case SVTK_UNSIGNED_CHAR:
      return adios_byte;
      break;
    case SVTK_INT:
      return adios_integer;
      break;
    case SVTK_UNSIGNED_INT:
      return adios_unsigned_integer;
      break;
    case SVTK_LONG:
      if (sizeof(long) == 4)
        return adios_integer; // 32 bits
      return adios_long; // 64 bits
      break;
    case SVTK_UNSIGNED_LONG:
      if (sizeof(long) == 4)
        return adios_unsigned_integer; // 32 bits
      return adios_unsigned_long; // 64 bits
      break;
    case SVTK_LONG_LONG:
      return adios_long;
      break;
    case SVTK_UNSIGNED_LONG_LONG:
      return adios_unsigned_long; // 64 bits
      break;
    case SVTK_ID_TYPE:
      return adiosIdType();
      break;
    default:
      {
      SENSEI_ERROR("the adios type for svtk type enumeration " << svtkt
        << " is currently not implemented")
      MPI_Abort(MPI_COMM_WORLD, -1);
      }
    }
  return adios_unknown;
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

// dataset_function takes a svtkDataSet*, it might be nullptr,
// does some processing, and returns and integer code.
//
// return codes:
//  1   : successfully processed the dataset, end the traversal
//  0   : successfully processed the dataset, continue traversal
//  < 0 : an error occured, report it and end traversal
using dataset_function =
  std::function<int(unsigned int,unsigned int,svtkDataSet*)>;

// --------------------------------------------------------------------------
// apply given function to each leaf in the data object.
int apply(unsigned int doid, unsigned int dsid, svtkDataObject* dobj,
  dataset_function &func, int skip_empty=1)
{
  if (svtkCompositeDataSet* cd = dynamic_cast<svtkCompositeDataSet*>(dobj))
    {
    svtkSmartPointer<svtkCompositeDataIterator> iter;
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
  else if (svtkDataSet *ds = dynamic_cast<svtkDataSet*>(dobj))
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
unsigned int getNumberOfDatasets(svtkDataObject *dobj)
{
  unsigned int number_of_datasets = 0;
  if (svtkCompositeDataSet *cd = dynamic_cast<svtkCompositeDataSet*>(dobj))
    {
    // function to count number of datasets of the given type
    dataset_function func = [&number_of_datasets](unsigned int, unsigned int, svtkDataSet *ds) -> int
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
unsigned int getNumberOfDatasets(MPI_Comm comm, svtkDataObject *dobj,
  int local_only)
{
  unsigned int number_of_datasets = 0;
  // function that counts number of datasets of any type
  dataset_function func = [&number_of_datasets](unsigned int, unsigned int, svtkDataSet*) -> int
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
  sensei::TimeEvent<128> mark("senseiADIOS1::BinaryStreamSchema::DefineVariables");

  // a second variable holding the local, global, length is required
  // for writing. according to the docs you could write a constant
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
  sensei::Profiler::StartEvent("senseiADIOS1::BinaryStreamSchema::Read");

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

  sensei::Profiler::EndEvent("senseiADIOS1::BinaryStreamSchema::Read", nbytes);

  return 0;
}

// --------------------------------------------------------------------------
int BinaryStreamSchema::Write(uint64_t fh, const std::string &path,
  const sensei::BinaryStream &str)
{
  sensei::Profiler::StartEvent("senseiADIOS1::BinaryStreamSchema::Write");

  int n = str.Size();
  std::string len = path + "_len";
  if (adios_write(fh, len.c_str(), &n) ||
    adios_write(fh, path.c_str(), str.GetData()))
    {
    SENSEI_ERROR("Failed to write BinaryStream at \"" << path << "\"")
    return -1;
    }

  sensei::Profiler::EndEvent("senseiADIOS1::BinaryStreamSchema::Write", n);
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
  sensei::TimeEvent<128> mark("senseiADIOS1::VersionSchema::DefineVariables");

  // all ranks need to write this info for FLEXPATH method
  // but not the MPI method.
  adios_define_var(gh, "DataObjectSchema", "", adios_unsigned_integer,
    "", "", "");
  return 0;
}

// --------------------------------------------------------------------------
int VersionSchema::Write(int64_t fh)
{
  sensei::Profiler::StartEvent("senseiADIOS1::VersionSchema::Write");

  // all ranks need to write this info for FLEXPATH method
  // but not the MPI method.
  adios_write(fh, "DataObjectSchema", &this->Revision);

  sensei::Profiler::EndEvent("senseiADIOS1::VersionSchema::Write", sizeof(this->Revision));
  return 0;
}

// --------------------------------------------------------------------------
int VersionSchema::Read(InputStream &iStream)
{
  sensei::Profiler::StartEvent("senseiADIOS1::VersionSchema::Read");

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

  sensei::Profiler::EndEvent("senseiADIOS1::VersionSchema::Read", sizeof(revision));
  return 0;
}


// --------------------------------------------------------------------------
int InputStream::SetReadMethod(const std::string &method)
{
  sensei::TimeEvent<128> mark("senseiADIOS1::InputStream::SetReadMethod");

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
  sensei::TimeEvent<128> mark("senseiADIOS1::InputStream::Open");

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
  sensei::TimeEvent<128> mark("senseiADIOS1::InputStream::AdvanceTimeStep");

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
  sensei::TimeEvent<128> mark("senseiADIOS1::InputStream::Close");

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
    const sensei::MeshMetadataPtr &md, svtkCompositeDataSet *dobj);

  int Write(MPI_Comm comm, int64_t fh, unsigned int i,
    const std::string &array_name, int array_cen, svtkCompositeDataSet *dobj,
    unsigned int num_blocks, const std::vector<int> &block_owner,
    const std::vector<int64_t> &writeIds);

  int Read(MPI_Comm comm, ADIOS_FILE *fh, const std::string &ons,
    const std::string &array_name, int centering,
    const sensei::MeshMetadataPtr &md, svtkCompositeDataSet *dobj);

  int Read(MPI_Comm comm, ADIOS_FILE *fh, const std::string &ons,
    unsigned int i, const std::string &array_name, int array_type,
    unsigned long long num_components, int array_cen, unsigned int num_blocks,
    const std::vector<long> &block_num_points,
    const std::vector<long> &block_num_cells, const std::vector<int> &block_owner,
    svtkCompositeDataSet *dobj);

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
  sensei::TimeEvent<128> mark("senseiADIOS1::ArraySchema::DefineVariable");

  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  // validate centering
  if ((array_cen != svtkDataObject::POINT) && (array_cen != svtkDataObject::CELL))
    {
    SENSEI_ERROR("Invalid array centering at array " << i)
    return -1;
    }

  // put each data array in its own namespace
  std::ostringstream ans;
  ans << ons << "data_array_" << i << "/";

  // select global size either point or cell data
  unsigned long long num_elem_total = (array_cen == svtkDataObject::POINT ?
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
    unsigned long long num_elem_local = (array_cen == svtkDataObject::POINT ?
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
  sensei::TimeEvent<128> mark("senseiADIOS1::ArraySchema::DefineVariables");

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
    if (this->DefineVariable(comm, gh, ons, num_arrays, SVTK_UNSIGNED_CHAR,
      1, svtkDataObject::CELL, num_points_total, num_cells_total, num_blocks,
      md->BlockNumPoints, md->BlockNumCells, md->BlockOwner, writeIds))
      return -1;
    num_arrays += 1;
    }

  if (md->NumGhostNodes && this->DefineVariable(comm, gh, ons, num_arrays,
      SVTK_UNSIGNED_CHAR, 1, svtkDataObject::POINT, num_points_total,
      num_cells_total, num_blocks, md->BlockNumPoints, md->BlockNumCells,
      md->BlockOwner, writeIds))
      return -1;

  return 0;
}


// --------------------------------------------------------------------------
int ArraySchema::Write(MPI_Comm comm, int64_t fh, unsigned int i,
  const std::string &array_name, int array_cen, svtkCompositeDataSet *dobj,
  unsigned int num_blocks, const std::vector<int> &block_owner,
  const std::vector<int64_t> &writeIds)
{
  sensei::Profiler::StartEvent("senseiADIOS1::ArraySchema::Write");
  long long numBytes = 0ll;

  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  svtkCompositeDataIterator *it = dobj->NewIterator();
  it->SetSkipEmptyNodes(0);
  it->InitTraversal();

  for (unsigned int j = 0; j < num_blocks; ++j)
    {
    if (block_owner[j] == rank)
      {
      svtkDataSet *ds = dynamic_cast<svtkDataSet*>(it->GetCurrentDataObject());
      if (!ds)
        {
        SENSEI_ERROR("Failed to get block " << j)
        return -1;
        }

      svtkDataSetAttributes *dsa = array_cen == svtkDataObject::POINT ?
        dynamic_cast<svtkDataSetAttributes*>(ds->GetPointData()) :
        dynamic_cast<svtkDataSetAttributes*>(ds->GetCellData());

      svtkDataArray *da = dsa->GetArray(array_name.c_str());
      if (!da)
        {
        SENSEI_ERROR("Failed to get array \"" << array_name << "\"")
        return -1;
        }

      adios_write_byid(fh, writeIds[i*num_blocks + j], da->GetVoidPointer(0));

      numBytes += da->GetNumberOfTuples()*
        da->GetNumberOfComponents()*sensei::SVTKUtils::Size(da->GetDataType());
      }

    it->GoToNextItem();
    }

  it->Delete();

  sensei::Profiler::EndEvent("senseiADIOS1::ArraySchema::Write", numBytes);
  return 0;
}

// --------------------------------------------------------------------------
int ArraySchema::Write(MPI_Comm comm, int64_t fh,
  const sensei::MeshMetadataPtr &md, svtkCompositeDataSet *dobj)
{
  sensei::TimeEvent<128> mark("senseiADIOS1::ArraySchema::Write");

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
    if (this->Write(comm, fh, num_arrays, "svtkGhostType", svtkDataObject::CELL,
      dobj, md->NumBlocks, md->BlockOwner, writeIds))
      return -1;
    num_arrays += 1;
    }

  if (md->NumGhostNodes && this->Write(comm, fh, num_arrays,
    "svtkGhostType", svtkDataObject::POINT, dobj, md->NumBlocks,
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
  svtkCompositeDataSet *dobj)
{
  sensei::Profiler::StartEvent("senseiADIOS1::ArraySchema::Read");
  long long numBytes = 0ll;

  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  // put each data array in its own namespace
  std::ostringstream ans;
  ans << ons << "data_array_" << i << "/";

  svtkCompositeDataIterator *it = dobj->NewIterator();
  it->SetSkipEmptyNodes(0);
  it->InitTraversal();

  // read each block
  unsigned long long block_offset = 0;
  for (unsigned int j = 0; j < num_blocks; ++j)
    {
    // get the block size
    unsigned long long num_elem_local = (array_cen == svtkDataObject::POINT ?
      block_num_points[j] : block_num_cells[j])*num_components;

    // define the variable for a local block
    if (block_owner[j] ==  rank)
      {
      uint64_t start = block_offset;
      uint64_t count = num_elem_local;;
      ADIOS_SELECTION *sel = adios_selection_boundingbox(1, &start, &count);

      svtkDataArray *array = svtkDataArray::CreateDataArray(array_type);
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

      // pass to svtk
      svtkDataSet *ds = dynamic_cast<svtkDataSet*>(it->GetCurrentDataObject());
      if (!ds)
        {
        SENSEI_ERROR("Failed to get block " << j)
        return -1;
        }

      svtkDataSetAttributes *dsa = array_cen == svtkDataObject::POINT ?
        dynamic_cast<svtkDataSetAttributes*>(ds->GetPointData()) :
        dynamic_cast<svtkDataSetAttributes*>(ds->GetCellData());

      dsa->AddArray(array);
      array->Delete();

      numBytes += num_elem_local*sensei::SVTKUtils::Size(array_type);
      }

    // update the block offset
    block_offset += num_elem_local;

    // next block
    it->GoToNextItem();
    }

  it->Delete();

  sensei::Profiler::EndEvent("senseiADIOS1::ArraySchema::Read", numBytes);
  return 0;
}

// --------------------------------------------------------------------------
int ArraySchema::Read(MPI_Comm comm, ADIOS_FILE *fh, const std::string &ons,
  const std::string &name, int centering, const sensei::MeshMetadataPtr &md,
  svtkCompositeDataSet *dobj)
{
  sensei::TimeEvent<128> mark("senseiADIOS1::ArraySchema::Read");

  unsigned int num_blocks = md->NumBlocks;
  unsigned int num_arrays = md->NumArrays;

  // read ghost arrays
  if (name == "svtkGhostType")
    {
    unsigned int i = centering == svtkDataObject::CELL ?
      num_arrays : num_arrays + 1;

    return this->Read(comm, fh, ons, i, "svtkGhostType", SVTK_UNSIGNED_CHAR,
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
    const sensei::MeshMetadataPtr &md, svtkCompositeDataSet *dobj);

  int Read(MPI_Comm comm, ADIOS_FILE *fh, const std::string &ons,
    const sensei::MeshMetadataPtr &md, svtkCompositeDataSet *dobj);

  std::map<std::string, std::vector<int64_t>> WriteIds;
};

// --------------------------------------------------------------------------
int PointSchema::DefineVariables(MPI_Comm comm, int64_t gh,
  const std::string &ons, const sensei::MeshMetadataPtr &md)
{
  if (sensei::SVTKUtils::Unstructured(md) || sensei::SVTKUtils::Structured(md)
    || sensei::SVTKUtils::Polydata(md))
    {
    sensei::TimeEvent<128> mark("senseiADIOS1::PointSchema::DefineVariables");

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
  const sensei::MeshMetadataPtr &md, svtkCompositeDataSet *dobj)
{
  if (sensei::SVTKUtils::Unstructured(md) || sensei::SVTKUtils::Structured(md)
    || sensei::SVTKUtils::Polydata(md))
    {
    sensei::Profiler::StartEvent("senseiADIOS1::PointSchema::Write");
    long long numBytes = 0ll;

    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    std::vector<int64_t> &writeIds = this->WriteIds[md->MeshName];

    svtkCompositeDataIterator *it = dobj->NewIterator();
    it->SetSkipEmptyNodes(0);
    it->InitTraversal();

    unsigned int num_blocks = md->NumBlocks;
    for (unsigned int j = 0; j < num_blocks; ++j)
      {
      if (md->BlockOwner[j] == rank)
        {
        svtkPointSet *ds = dynamic_cast<svtkPointSet*>(it->GetCurrentDataObject());
        if (!ds)
          {
          SENSEI_ERROR("Failed to get block " << j)
          return -1;
          }

        svtkDataArray *da = ds->GetPoints()->GetData();
        adios_write_byid(fh, writeIds[j], da->GetVoidPointer(0));

        numBytes += da->GetNumberOfTuples()*
          da->GetNumberOfComponents()*sensei::SVTKUtils::Size(da->GetDataType());
        }

      it->GoToNextItem();
      }
    it->Delete();

    sensei::Profiler::EndEvent("senseiADIOS1::PointSchema::Write", numBytes);
    }

  return 0;
}

// --------------------------------------------------------------------------
int PointSchema::Read(MPI_Comm comm, ADIOS_FILE *fh, const std::string &ons,
  const sensei::MeshMetadataPtr &md, svtkCompositeDataSet *dobj)
{
  if (sensei::SVTKUtils::Unstructured(md) || sensei::SVTKUtils::Structured(md)
    || sensei::SVTKUtils::Polydata(md))
    {
    sensei::Profiler::StartEvent("senseiADIOS1::PointSchema::Read");
    long long numBytes = 0ll;

    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    svtkCompositeDataIterator *it = dobj->NewIterator();
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

        svtkDataArray *points = svtkDataArray::CreateDataArray(md->CoordinateType);
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

        // pass into svtk
        svtkPoints *pts = svtkPoints::New();
        pts->SetData(points);
        points->Delete();

        svtkPointSet *ds = dynamic_cast<svtkPointSet*>(it->GetCurrentDataObject());
        if (!ds)
          {
          SENSEI_ERROR("Failed to get block " << j)
          return -1;
          }

        ds->SetPoints(pts);
        pts->Delete();

        numBytes += count*sensei::SVTKUtils::Size(md->CoordinateType);
        }

      // update the block offset
      block_offset += num_local;

      // next block
      it->GoToNextItem();
      }

    it->Delete();

    sensei::Profiler::EndEvent("senseiADIOS1::PointSchema::Read", numBytes);
    }

  return 0;
}



struct UnstructuredCellSchema
{
  int DefineVariables(MPI_Comm comm, int64_t gh,
    const std::string &ons, const sensei::MeshMetadataPtr &md);

  int Write(MPI_Comm comm, int64_t fh,
    const sensei::MeshMetadataPtr &md, svtkCompositeDataSet *dobj);

  int Read(MPI_Comm comm, ADIOS_FILE *fh, const std::string &ons,
    const sensei::MeshMetadataPtr &md, svtkCompositeDataSet *dobj);

  std::map<std::string, std::vector<int64_t>> TypeWriteIds;
  std::map<std::string, std::vector<int64_t>> ArrayWriteIds;
};

// --------------------------------------------------------------------------
int UnstructuredCellSchema::DefineVariables(MPI_Comm comm, int64_t gh,
  const std::string &ons, const sensei::MeshMetadataPtr &md)
{
  if (sensei::SVTKUtils::Unstructured(md))
    {
    sensei::TimeEvent<128> mark("senseiADIOS1::UnstructuredCellSchema::DefineVariables");

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
  const sensei::MeshMetadataPtr &md, svtkCompositeDataSet *dobj)
{
  if (sensei::SVTKUtils::Unstructured(md))
    {
    sensei::Profiler::StartEvent("senseiADIOS1::UnstructuredCellSchema");
    long long numBytes = 0ll;

    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    std::vector<int64_t> &arrayWriteIds = this->ArrayWriteIds[md->MeshName];
    std::vector<int64_t> &typeWriteIds = this->TypeWriteIds[md->MeshName];

    svtkCompositeDataIterator *it = dobj->NewIterator();
    it->SetSkipEmptyNodes(0);
    it->InitTraversal();

    unsigned int num_blocks = md->NumBlocks;
    for (unsigned int j = 0; j < num_blocks; ++j)
      {
      // read local block
      if (md->BlockOwner[j] ==  rank)
        {
        svtkUnstructuredGrid *ds =
          dynamic_cast<svtkUnstructuredGrid*>(it->GetCurrentDataObject());
        if (!ds)
          {
          SENSEI_ERROR("Failed to get block " << j)
          return -1;
          }

        svtkDataArray *cta = ds->GetCellTypesArray();
        svtkDataArray *ca = ds->GetCells()->GetData();

        adios_write_byid(fh, typeWriteIds[j], cta->GetVoidPointer(0));
        adios_write_byid(fh, arrayWriteIds[j], ca->GetVoidPointer(0));

        numBytes += cta->GetNumberOfTuples()*sensei::SVTKUtils::Size(cta->GetDataType()) +
          ca->GetNumberOfTuples()*sensei::SVTKUtils::Size(ca->GetDataType());
        }
      it->GoToNextItem();
      }
    it->Delete();

    sensei::Profiler::EndEvent("senseiADIOS1::UnstructuredCellSchema::Write", numBytes);
    }

  return 0;
}

// --------------------------------------------------------------------------
int UnstructuredCellSchema::Read(MPI_Comm comm, ADIOS_FILE *fh,
  const std::string &ons, const sensei::MeshMetadataPtr &md, svtkCompositeDataSet *dobj)
{
  if (sensei::SVTKUtils::Unstructured(md))
    {
    sensei::Profiler::StartEvent("senseiADIOS1::UnstructuredCellSchema::Read");
    long long numBytes = 0ll;

    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    svtkCompositeDataIterator *it = dobj->NewIterator();
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

        svtkUnsignedCharArray *cell_types = svtkUnsignedCharArray::New();
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

        svtkIdTypeArray *cell_array = svtkIdTypeArray::New();
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

        svtkUnstructuredGrid *ds =
          dynamic_cast<svtkUnstructuredGrid*>(it->GetCurrentDataObject());
        if (!ds)
          {
          SENSEI_ERROR("Failed to get block " << j)
          return -1;
          }
        // build locations
        svtkIdTypeArray *cell_locs = svtkIdTypeArray::New();
        cell_locs->SetNumberOfTuples(num_cells_local);
        svtkIdType *p_locs = cell_locs->GetPointer(0);
        svtkIdType *p_cells = cell_array->GetPointer(0);
        p_locs[0] = 0;
        for (unsigned long i = 1; i < num_cells_local; ++i)
          p_locs[i] = p_locs[i-1] + p_cells[p_locs[i-1]] + 1;

        // pass types, cell_locs, and cells
        svtkCellArray *ca = svtkCellArray::New();
        ca->SetCells(num_cells_local, cell_array);
        cell_array->Delete();

        ds->SetCells(cell_types, cell_locs, ca);

        cell_locs->Delete();
        cell_array->Delete();
        cell_types->Delete();

        numBytes += ct_count*sizeof(unsigned char) + ca_count*sizeof(svtkIdType);
        }

      // update the block offset
      cell_types_block_offset += num_cells_local;
      cell_array_block_offset += cell_array_size_local;
      }

    sensei::Profiler::EndEvent("senseiADIOS1::UnstructuredCellSchema::Read", numBytes);
    }

  return 0;
}



struct PolydataCellSchema
{
  int DefineVariables(MPI_Comm comm, int64_t gh,
    const std::string &ons, const sensei::MeshMetadataPtr &md);

  int Write(MPI_Comm comm, int64_t fh,
    const sensei::MeshMetadataPtr &md, svtkCompositeDataSet *dobj);

  int Read(MPI_Comm comm, ADIOS_FILE *fh,
    const std::string &ons, const sensei::MeshMetadataPtr &md,
    svtkCompositeDataSet *dobj);

  std::map<std::string, std::vector<int64_t>> TypeWriteIds;
  std::map<std::string, std::vector<int64_t>> ArrayWriteIds;
};

// --------------------------------------------------------------------------
int PolydataCellSchema::DefineVariables(MPI_Comm comm, int64_t gh,
  const std::string &ons, const sensei::MeshMetadataPtr &md)
{
  if (sensei::SVTKUtils::Polydata(md))
    {
    sensei::TimeEvent<128> mark("senseiADIOS1::PolydataCellSchema::DefineVariables");

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
  const sensei::MeshMetadataPtr &md, svtkCompositeDataSet *dobj)
{
  if (sensei::SVTKUtils::Polydata(md))
    {
    sensei::Profiler::StartEvent("senseiADIOS1::PolydataCellSchema::Write");
    long long numBytes = 0ll;

    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    std::vector<int64_t> &typeWriteIds = this->TypeWriteIds[md->MeshName];
    std::vector<int64_t> &arrayWriteIds = this->ArrayWriteIds[md->MeshName];

    svtkCompositeDataIterator *it = dobj->NewIterator();
    it->SetSkipEmptyNodes(0);
    it->InitTraversal();

    unsigned int num_blocks = md->NumBlocks;
    for (unsigned int j = 0; j < num_blocks; ++j)
      {
      if (md->BlockOwner[j] == rank)
        {
        svtkPolyData *pd = dynamic_cast<svtkPolyData*>(it->GetCurrentDataObject());
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
        std::vector<svtkIdType> cells;

        svtkIdType nv = pd->GetNumberOfVerts();
        if (nv)
          {
          types.insert(types.end(), nv, SVTK_VERTEX);
          svtkIdType *pv = pd->GetVerts()->GetData()->GetPointer(0);
          cells.insert(cells.end(), pv, pv + pd->GetVerts()->GetData()->GetNumberOfTuples());
          }

        svtkIdType nl = pd->GetNumberOfLines();
        if (nl)
          {
          types.insert(types.end(), nl, SVTK_LINE);
          svtkIdType *pl = pd->GetLines()->GetData()->GetPointer(0);
          cells.insert(cells.end(), pl, pl + pd->GetLines()->GetData()->GetNumberOfTuples());
          }

        svtkIdType np = pd->GetNumberOfPolys();
        if (np)
          {
          types.insert(types.end(), np, SVTK_POLYGON);
          svtkIdType *pp = pd->GetPolys()->GetData()->GetPointer(0);
          cells.insert(cells.end(), pp, pp + pd->GetPolys()->GetData()->GetNumberOfTuples());
          }

        svtkIdType ns = pd->GetNumberOfStrips();
        if (ns)
          {
          types.insert(types.end(), ns, SVTK_TRIANGLE_STRIP);
          svtkIdType *ps = pd->GetStrips()->GetData()->GetPointer(0);
          cells.insert(cells.end(), ps, ps + pd->GetStrips()->GetData()->GetNumberOfTuples());
          }

        adios_write_byid(fh, typeWriteIds[j], types.data());
        adios_write_byid(fh, arrayWriteIds[j], cells.data());

        numBytes += types.size()*sizeof(unsigned char) + cells.size()*sizeof(svtkIdType);
        }

      // go to the next block
      it->GoToNextItem();
      }
    it->Delete();

    sensei::Profiler::EndEvent("senseiADIOS1::PolydataCellSchema::Write", numBytes);
    }

  return 0;
}

// --------------------------------------------------------------------------
int PolydataCellSchema::Read(MPI_Comm comm, ADIOS_FILE *fh,
  const std::string &ons, const sensei::MeshMetadataPtr &md,
  svtkCompositeDataSet *dobj)
{
  if (sensei::SVTKUtils::Polydata(md))
    {
    sensei::Profiler::StartEvent("senseiADIOS1::PolydataCellSchema::Read");
    long long numBytes = 0ll;

    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    svtkCompositeDataIterator *it = dobj->NewIterator();
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
        std::vector<svtkIdType> cell_array(cell_array_size_local);
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
        svtkIdType *p_cells = cell_array.data();

        // assumptions made here:
        // data is serialized in the order verts, lines, polys, strips

        // find first and last vert and number of verts
        unsigned long i = 0;
        unsigned long n_verts = 0;
        svtkIdType *vert_begin = p_cells;
        while ((i < num_cells_local) && (p_types[i] == SVTK_VERTEX))
          {
          p_cells += p_cells[0] + 1;
          ++n_verts;
          ++i;
          }
        svtkIdType *vert_end = p_cells;

        // find first and last line and number of lines
        unsigned long n_lines = 0;
        svtkIdType *line_begin = p_cells;
        while ((i < num_cells_local) && (p_types[i] == SVTK_LINE))
          {
          p_cells += p_cells[0] + 1;
          ++n_lines;
          ++i;
          }
        svtkIdType *line_end = p_cells;

        // find first and last poly and number of polys
        unsigned long n_polys = 0;
        svtkIdType *poly_begin = p_cells;
        while ((i < num_cells_local) && (p_types[i] == SVTK_VERTEX))
          {
          p_cells += p_cells[0] + 1;
          ++n_polys;
          ++i;
          }
        svtkIdType *poly_end = p_cells;

        // find first and last strip and number of strips
        unsigned long n_strips = 0;
        svtkIdType *strip_begin = p_cells;
        while ((i < num_cells_local) && (p_types[i] == SVTK_VERTEX))
          {
          p_cells += p_cells[0] + 1;
          ++n_strips;
          ++i;
          }
        svtkIdType *strip_end = p_cells;

        // pass into svtk
        svtkPolyData *pd = dynamic_cast<svtkPolyData*>(it->GetCurrentDataObject());
        if (!pd)
          {
          SENSEI_ERROR("Failed to get block " << j)
          return -1;
          }

        // pass verts
        unsigned long n_tups = vert_end - vert_begin;
        svtkIdTypeArray *verts = svtkIdTypeArray::New();
        verts->SetNumberOfTuples(n_tups);
        svtkIdType *p_verts = verts->GetPointer(0);

        for (unsigned long j = 0; j < n_tups; ++j)
          p_verts[j] = vert_begin[j];

        svtkCellArray *ca = svtkCellArray::New();
        ca->SetCells(n_verts, verts);
        verts->Delete();

        pd->SetVerts(ca);
        ca->Delete();

        // pass lines
        n_tups = line_end - line_begin;
        svtkIdTypeArray *lines = svtkIdTypeArray::New();
        lines->SetNumberOfTuples(n_tups);
        svtkIdType *p_lines = lines->GetPointer(0);

        for (unsigned long j = 0; j < n_tups; ++j)
          p_lines[j] = line_begin[j];

        ca = svtkCellArray::New();
        ca->SetCells(n_lines, lines);
        lines->Delete();

        pd->SetLines(ca);
        ca->Delete();

        // pass polys
        n_tups = poly_end - poly_begin;
        svtkIdTypeArray *polys = svtkIdTypeArray::New();
        polys->SetNumberOfTuples(n_tups);
        svtkIdType *p_polys = polys->GetPointer(0);

        for (unsigned long j = 0; j < n_tups; ++j)
          p_polys[j] = poly_begin[j];

        ca = svtkCellArray::New();
        ca->SetCells(n_polys, polys);
        polys->Delete();

        pd->SetPolys(ca);
        ca->Delete();

        // pass strips
        n_tups = strip_end - strip_begin;
        svtkIdTypeArray *strips = svtkIdTypeArray::New();
        strips->SetNumberOfTuples(n_tups);
        svtkIdType *p_strips = strips->GetPointer(0);

        for (unsigned long j = 0; j < n_tups; ++j)
          p_strips[j] = strip_begin[j];

        ca = svtkCellArray::New();
        ca->SetCells(n_strips, strips);
        strips->Delete();

        pd->SetStrips(ca);
        ca->Delete();

        pd->BuildCells();

        numBytes += ct_count*sizeof(unsigned char) + ca_count*sizeof(svtkIdType);
        }
      // go to the next block
      it->GoToNextItem();

      // update the block offset
      cell_block_offset += num_cells_local;
      cell_array_block_offset += cell_array_size_local;
      }

    it->Delete();

    sensei::Profiler::EndEvent("senseiADIOS1::PolydataCellSchema::Read", numBytes);
    }

  return 0;
}



struct LogicallyCartesianSchema
{
  int DefineVariables(MPI_Comm comm, int64_t gh, const std::string &ons,
    const sensei::MeshMetadataPtr &md);

  int Write(MPI_Comm comm, int64_t fh,
    const sensei::MeshMetadataPtr &md, svtkCompositeDataSet *dobj);

  int Read(MPI_Comm comm, ADIOS_FILE *fh,
    const std::string &ons, const sensei::MeshMetadataPtr &md,
    svtkCompositeDataSet *dobj);

  std::map<std::string, std::vector<int64_t>> WriteIds;
};

// --------------------------------------------------------------------------
int LogicallyCartesianSchema::DefineVariables(MPI_Comm comm, int64_t gh,
  const std::string &ons, const sensei::MeshMetadataPtr &md)
{
  if (sensei::SVTKUtils::LogicallyCartesian(md))
    {
    sensei::TimeEvent<128> mark("senseiADIOS1::LogicallyCartesianSchema::DefineVariables");

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
  const sensei::MeshMetadataPtr &md, svtkCompositeDataSet *dobj)
{
  if (sensei::SVTKUtils::LogicallyCartesian(md))
    {
    sensei::Profiler::StartEvent("senseiADIOS1::LogicallyCartesianSchema::Write");
    long long numBytes = 0ll;

    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    std::vector<int64_t> &writeIds = this->WriteIds[md->MeshName];

    svtkCompositeDataIterator *it = dobj->NewIterator();
    it->SetSkipEmptyNodes(0);
    it->InitTraversal();

    unsigned int num_blocks = md->NumBlocks;
    for (unsigned int j = 0; j < num_blocks; ++j)
      {
      // define the variable for a local block
      if (md->BlockOwner[j] ==  rank)
        {
        svtkDataObject *dobj = it->GetCurrentDataObject();
        if (!dobj)
          {
          SENSEI_ERROR("Failed to get block " << j)
          return -1;
          }
        switch (md->BlockType)
          {
          case SVTK_RECTILINEAR_GRID:
            adios_write_byid(fh, writeIds[j],
              dynamic_cast<svtkRectilinearGrid*>(dobj)->GetExtent());
            break;
          case SVTK_IMAGE_DATA:
          case SVTK_UNIFORM_GRID:
            adios_write_byid(fh, writeIds[j],
              dynamic_cast<svtkImageData*>(dobj)->GetExtent());
            break;
          case SVTK_STRUCTURED_GRID:
            adios_write_byid(fh, writeIds[j],
              dynamic_cast<svtkStructuredGrid*>(dobj)->GetExtent());
            break;
          }

        numBytes += 6*sizeof(int);
        }
      it->GoToNextItem();
      }
    it->Delete();

    sensei::Profiler::EndEvent("senseiADIOS1::LogicallyCartesianSchema::Write", numBytes);
    }

  return 0;
}

// --------------------------------------------------------------------------
int LogicallyCartesianSchema::Read(MPI_Comm comm, ADIOS_FILE *fh,
  const std::string &ons, const sensei::MeshMetadataPtr &md,
  svtkCompositeDataSet *dobj)
{
  if (sensei::SVTKUtils::LogicallyCartesian(md))
    {
    sensei::Profiler::StartEvent("senseiADIOS1::LogicallyCartesianSchema::Read");
    long long numBytes = 0ll;

    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    svtkCompositeDataIterator *it = dobj->NewIterator();
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

        // update the svtk object
        svtkDataObject *dobj = it->GetCurrentDataObject();
        if (!dobj)
          {
          SENSEI_ERROR("Failed to get block " << j)
          return -1;
          }
        switch (md->BlockType)
          {
          case SVTK_RECTILINEAR_GRID:
            dynamic_cast<svtkRectilinearGrid*>(dobj)->SetExtent(ext);
            break;
          case SVTK_IMAGE_DATA:
          case SVTK_UNIFORM_GRID:
              dynamic_cast<svtkImageData*>(dobj)->SetExtent(ext);
            break;
          case SVTK_STRUCTURED_GRID:
              dynamic_cast<svtkStructuredGrid*>(dobj)->SetExtent(ext);
            break;
          }

        numBytes += 6*sizeof(int);
        }
      // next block
      it->GoToNextItem();
      }
    it->Delete();

    sensei::Profiler::EndEvent("senseiADIOS1::LogicallyCartesianSchema::Read", numBytes);
    }

  return 0;
}



struct UniformCartesianSchema
{
  int DefineVariables(MPI_Comm comm, int64_t gh, const std::string &ons,
    const sensei::MeshMetadataPtr &md);

  int Write(MPI_Comm comm, int64_t fh,
    const sensei::MeshMetadataPtr &md, svtkCompositeDataSet *dobj);

  int Read(MPI_Comm comm, ADIOS_FILE *fh, const std::string &ons,
    const sensei::MeshMetadataPtr &md, svtkCompositeDataSet *dobj);

  std::map<std::string, std::vector<int64_t>> OriginWriteIds;
  std::map<std::string, std::vector<int64_t>> SpacingWriteIds;
};

// --------------------------------------------------------------------------
int UniformCartesianSchema::DefineVariables(MPI_Comm comm, int64_t gh,
  const std::string &ons, const sensei::MeshMetadataPtr &md)
{
  if (sensei::SVTKUtils::UniformCartesian(md))
    {
    sensei::TimeEvent<128> mark("senseiADIOS1::UniformCartesianSchema::DefineVariables");

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
  const sensei::MeshMetadataPtr &md, svtkCompositeDataSet *dobj)
{
  if (sensei::SVTKUtils::UniformCartesian(md))
    {
    sensei::Profiler::StartEvent("senseiADIOS1::UniformCartesianSchema::Write");
    long long numBytes = 0ll;

    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    std::vector<int64_t> &originWriteIds = this->OriginWriteIds[md->MeshName];
    std::vector<int64_t> &spacingWriteIds = this->SpacingWriteIds[md->MeshName];

    svtkCompositeDataIterator *it = dobj->NewIterator();
    it->SetSkipEmptyNodes(0);
    it->InitTraversal();

    unsigned int num_blocks = md->NumBlocks;
    for (unsigned int j = 0; j < num_blocks; ++j)
      {
      // define the variable for a local block
      if (md->BlockOwner[j] ==  rank)
        {
        svtkImageData *ds = dynamic_cast<svtkImageData*>(it->GetCurrentDataObject());
        if (!ds)
          {
          SENSEI_ERROR("Failed to get block " << j)
          return -1;
          }

        adios_write_byid(fh, originWriteIds[j], ds->GetOrigin());
        adios_write_byid(fh, spacingWriteIds[j], ds->GetSpacing());

        numBytes += 6*sizeof(double);
        }

      it->GoToNextItem();
      }
    it->Delete();

    sensei::Profiler::EndEvent("senseiADIOS1::UniformCartesianSchema::Write", numBytes);
    }

  return 0;
}

// --------------------------------------------------------------------------
int UniformCartesianSchema::Read(MPI_Comm comm, ADIOS_FILE *fh,
  const std::string &ons, const sensei::MeshMetadataPtr &md,
  svtkCompositeDataSet *dobj)
{
  if (sensei::SVTKUtils::UniformCartesian(md))
    {
    sensei::Profiler::StartEvent("senseiADIOS1::UniformCartesianSchema::Read");
    long long numBytes = 0ll;

    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    svtkCompositeDataIterator *it = dobj->NewIterator();
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

        // update the svtk object
        svtkImageData *ds = dynamic_cast<svtkImageData*>(it->GetCurrentDataObject());
        if (!ds)
          {
          SENSEI_ERROR("Failed to get block " << j << " not image data")
          return -1;
          }

        ds->SetOrigin(x0);
        ds->SetSpacing(dx);

        numBytes += 6*sizeof(double);
        }
      // next block
      it->GoToNextItem();
      }
    it->Delete();

    sensei::Profiler::EndEvent("senseiADIOS1::UniformCartesianSchema::Read", numBytes);
    }

  return 0;
}



struct StretchedCartesianSchema
{
  int DefineVariables(MPI_Comm comm, int64_t gh, const std::string &ons,
    const sensei::MeshMetadataPtr &md);

  int Write(MPI_Comm comm, int64_t fh,
    const sensei::MeshMetadataPtr &md, svtkCompositeDataSet *dobj);

  int Read(MPI_Comm comm, ADIOS_FILE *fh, const std::string &ons,
    const sensei::MeshMetadataPtr &md, svtkCompositeDataSet *dobj);

  std::map<std::string, std::vector<int64_t>> XCoordWriteIds;
  std::map<std::string, std::vector<int64_t>> YCoordWriteIds;
  std::map<std::string, std::vector<int64_t>> ZCoordWriteIds;
};

// --------------------------------------------------------------------------
int StretchedCartesianSchema::DefineVariables(MPI_Comm comm, int64_t gh,
  const std::string &ons, const sensei::MeshMetadataPtr &md)
{
  if (sensei::SVTKUtils::StretchedCartesian(md))
    {
    sensei::TimeEvent<128> mark("senseiADIOS1::StretchedCartesianSchema::DefineVariables");

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
  const sensei::MeshMetadataPtr &md, svtkCompositeDataSet *dobj)
{
  if (sensei::SVTKUtils::StretchedCartesian(md))
    {
    sensei::Profiler::StartEvent("senseiADIOS1::StretchedCartesianSchema");
    long long numBytes = 0ll;

    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    std::vector<int64_t> &xCoordWriteIds = this->XCoordWriteIds[md->MeshName];
    std::vector<int64_t> &yCoordWriteIds = this->YCoordWriteIds[md->MeshName];
    std::vector<int64_t> &zCoordWriteIds = this->ZCoordWriteIds[md->MeshName];

    svtkCompositeDataIterator *it = dobj->NewIterator();
    it->SetSkipEmptyNodes(0);
    it->InitTraversal();

    unsigned int num_blocks = md->NumBlocks;
    for (unsigned int j = 0; j < num_blocks; ++j)
      {
      if (md->BlockOwner[j] ==  rank)
        {
        svtkRectilinearGrid *ds = dynamic_cast<svtkRectilinearGrid*>(it->GetCurrentDataObject());
        if (!ds)
          {
          SENSEI_ERROR("Failed to get block " << j << " not unstructured")
          return -1;
          }

        svtkDataArray *xda = ds->GetXCoordinates();
        svtkDataArray *yda = ds->GetYCoordinates();
        svtkDataArray *zda = ds->GetZCoordinates();

        adios_write_byid(fh, xCoordWriteIds[j], xda->GetVoidPointer(0));
        adios_write_byid(fh, yCoordWriteIds[j], yda->GetVoidPointer(0));
        adios_write_byid(fh, zCoordWriteIds[j], zda->GetVoidPointer(0));

        long long cts = sensei::SVTKUtils::Size(xda->GetDataType());
        numBytes += xda->GetNumberOfTuples()*cts +
          yda->GetNumberOfTuples()*cts + zda->GetNumberOfTuples()*cts;
        }
      it->GoToNextItem();
      }
    it->Delete();

    sensei::Profiler::EndEvent("senseiADIOS1::StretchedCartesianSchema::Write", numBytes);
    }

  return 0;
}

// --------------------------------------------------------------------------
int StretchedCartesianSchema::Read(MPI_Comm comm, ADIOS_FILE *fh,
  const std::string &ons, const sensei::MeshMetadataPtr &md,
  svtkCompositeDataSet *dobj)
{
  if (sensei::SVTKUtils::StretchedCartesian(md))
    {
    sensei::Profiler::StartEvent("senseiADIOS1::StretchedCartesianSchema::Read");
    long long numBytes = 0ll;

    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    svtkCompositeDataIterator *it = dobj->NewIterator();
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

        svtkDataArray *x_coords = svtkDataArray::CreateDataArray(md->CoordinateType);
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

        svtkDataArray *y_coords = svtkDataArray::CreateDataArray(md->CoordinateType);
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

        svtkDataArray *z_coords = svtkDataArray::CreateDataArray(md->CoordinateType);
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

        // update the svtk object
        svtkRectilinearGrid *ds = dynamic_cast<svtkRectilinearGrid*>(it->GetCurrentDataObject());
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

        long long cts = sensei::SVTKUtils::Size(md->CoordinateType);
        numBytes += x_count*cts + y_count*cts + z_count*cts;
        }

      // next block
      it->GoToNextItem();

      // update the block offset
      xc_offset += nx_local;
      yc_offset += ny_local;
      zc_offset += nz_local;
      }

    it->Delete();

    sensei::Profiler::EndEvent("senseiADIOS1::StretchedCartesianSchema::Read", numBytes);
    }

  return 0;
}



struct DataObjectSchema
{
  int DefineVariables(MPI_Comm comm, int64_t gh,
    unsigned int doid,  const sensei::MeshMetadataPtr &md);

  int Write(MPI_Comm comm, int64_t fh, unsigned int doid,
    const sensei::MeshMetadataPtr &md, svtkCompositeDataSet *dobj);

  int ReadMesh(MPI_Comm comm, ADIOS_FILE *fh,
    unsigned int doid, const sensei::MeshMetadataPtr &md,
    svtkCompositeDataSet *&dobj, bool structure_only);

  int ReadArray(MPI_Comm comm, ADIOS_FILE *fh,
    unsigned int doid, const std::string &name, int association,
    const sensei::MeshMetadataPtr &md, svtkCompositeDataSet *dobj);

  int InitializeDataObject(MPI_Comm comm,
    const sensei::MeshMetadataPtr &md, svtkCompositeDataSet *&dobj);

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
  sensei::TimeEvent<128> mark("senseiADIOS1::DataObjectSchema::DefineVariables");

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
  const sensei::MeshMetadataPtr &md, svtkCompositeDataSet *dobj)
{
  sensei::TimeEvent<128> mark("senseiADIOS1::DataObjectSchema::Write");

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
  svtkCompositeDataSet *&dobj, bool structure_only)
{
  sensei::TimeEvent<128> mark("senseiADIOS1::DataObjectSchema::ReadMesh");

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
  const sensei::MeshMetadataPtr &md, svtkCompositeDataSet *dobj)
{
  sensei::TimeEvent<128> mark("senseiADIOS1::DataObjectSchema::ReadArray");

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
  const sensei::MeshMetadataPtr &md, svtkCompositeDataSet *&dobj)
{
  sensei::TimeEvent<128>("senseiADIOS1::DataObjectSchema::InitializeDataObject");

  dobj = nullptr;

  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  // allocate the local dataset
  svtkMultiBlockDataSet *mbds = svtkMultiBlockDataSet::New();
  mbds->SetNumberOfBlocks(md->NumBlocks);
  for (int i = 0; i < md->NumBlocks; ++i)
    {
    if (md->BlockOwner[i] == rank)
      {
      svtkDataObject *ds = sensei::SVTKUtils::NewDataObject(md->BlockType);
      mbds->SetBlock(md->BlockIds[i], ds);
      ds->Delete();
      }
    }

  dobj = mbds;

  return 0;
}



struct DataObjectCollectionSchema::InternalsType
{
  InternalsType() : BlockOwnerArrayMetadata(0) {}
  VersionSchema Version;
  DataObjectSchema DataObject;
  sensei::MeshMetadataMap SenderMdMap;
  sensei::MeshMetadataMap ReceiverMdMap;
  int BlockOwnerArrayMetadata;
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
  sensei::TimeEvent<128> mark("senseiADIOS1::DataObjectCollectionSchema::ReadMeshMetadata");

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

    // FIXME
    // Don't add internally generated arrays, as these
    // interfere with ghost cell/node arrays which are
    // also special cases. adding these breaks the numbering
    // scheme we used but these arrays here are useful for
    // testing and validation eg. visualizing the decomp
    if (this->Internals->BlockOwnerArrayMetadata)
      {
      md->ArrayName.push_back("SenderBlockOwner");
      md->ArrayCentering.push_back(svtkDataObject::CELL);
      md->ArrayComponents.push_back(1);
      md->ArrayType.push_back(SVTK_INT);

      md->ArrayName.push_back("ReceiverBlockOwner");
      md->ArrayCentering.push_back(svtkDataObject::CELL);
      md->ArrayComponents.push_back(1);
      md->ArrayType.push_back(SVTK_INT);

      md->NumArrays += 2;
      }

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
  sensei::TimeEvent<128> mark("senseiADIOS1::DataObjectCollectionSchema::GetSenderMeshMetadata");

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
  sensei::TimeEvent<128>("senseiADIOS1::DataObjectCollectionSchema::SetReceiverMeshMetadata");
  return this->Internals->ReceiverMdMap.SetMeshMetadata(id, md);
}


// --------------------------------------------------------------------------
int DataObjectCollectionSchema::GetReceiverMeshMetadata(unsigned int id,
  sensei::MeshMetadataPtr &md)
{
  sensei::TimeEvent<128>("senseiADIOS1::DataObjectCollectionSchema::GetReceiverMeshMetadata");
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
  sensei::TimeEvent<128>("senseADIOS1::DataObjectCollectionSchema::GetNumberOfObjects");
  num = this->Internals->SenderMdMap.Size();
  return 0;
}

// --------------------------------------------------------------------------
int DataObjectCollectionSchema::GetObjectId(MPI_Comm comm,
  const std::string &object_name, unsigned int &doid)
{
  sensei::TimeEvent<128>("senseiADIOS1::DataObjectCollectionSchema::GetObjectId");

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
  sensei::TimeEvent<128>("senseiADIOS1::DataObjectCollectionSchema::DefineVariables");

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
  const std::vector<svtkCompositeDataSet*> &objects)
{
  sensei::Profiler::StartEvent("senseiADIOS1::DataObjectCollectionSchema::Write");

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

  sensei::Profiler::EndEvent("senseiADIOS1::DataObjectCollectionSchema::Write",
    sizeof(time_step)+sizeof(time));
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
  svtkDataObject *&dobj, bool structure_only)
{
  sensei::TimeEvent<128> mark("senseiADIOS1::DataObjectCollectionSchema::ReadObject");

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

  svtkCompositeDataSet *cd = dynamic_cast<svtkCompositeDataSet*>(dobj);
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
  const std::string &array_name, svtkDataObject *dobj)
{
  sensei::TimeEvent<128> mark("senseiADIOS1::DataObjectCollectionSchema::ReadArray");

  // convert the mesh name into its id
  unsigned int doid = 0;
  if (this->GetObjectId(comm, object_name, doid))
    {
    SENSEI_ERROR("Failed to get object id for \"" << object_name << "\"")
    return -1;
    }

  // our factory will create svtkMultiBlock even if the sender has a legacy
  // dataset type. this enables block based re-partitioning.
  svtkCompositeDataSet *cds = dynamic_cast<svtkCompositeDataSet*>(dobj);
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
      << sensei::SVTKUtils::GetAttributesName(association)
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
  sensei::TimeEvent<128> mark("senseiADIOS1::DataObjectCollectionSchema::ReadTimeStep");

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
  svtkCompositeDataSet *dobj)
{
  sensei::TimeEvent<128> mark("senseiADIOS1::DataObjectCollectionSchema::AddBlockOwnerArray");

  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  unsigned int num_blocks = md->NumBlocks;
  int array_cen = centering;

  svtkCompositeDataIterator *it = dobj->NewIterator();
  it->SetSkipEmptyNodes(0);
  it->InitTraversal();

  // read each block
  for (unsigned int j = 0; j < num_blocks; ++j)
    {
    // get the block size
    unsigned long long num_elem_local = (array_cen == svtkDataObject::POINT ?
      md->BlockNumPoints[j] : md->BlockNumCells[j]);

    // define the variable for a local block
    svtkDataSet *ds = dynamic_cast<svtkDataSet*>(it->GetCurrentDataObject());
    if (ds)
      {
      // create arrays filled with sender and receiver ranks
      svtkDataArray *bo = svtkIntArray::New();
      bo->SetNumberOfTuples(num_elem_local);
      bo->SetName(name.c_str());
      bo->FillComponent(0, md->BlockOwner[j]);

      svtkDataSetAttributes *dsa = array_cen == svtkDataObject::POINT ?
        dynamic_cast<svtkDataSetAttributes*>(ds->GetPointData()) :
        dynamic_cast<svtkDataSetAttributes*>(ds->GetCellData());

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
