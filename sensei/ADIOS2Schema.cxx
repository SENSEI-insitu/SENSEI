#include "ADIOS2Schema.h"
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
#include <adios2_c.h>

#include <vector>
#include <map>
#include <set>
#include <string>
#include <functional>
#include <sstream>
#include <regex>

namespace senseiADIOS2
{

const char *adios2_strerror(adios2_error err)
{
  if (err == adios2_error_none)
    return "adios2_error_none";

  if (err == adios2_error_invalid_argument)
    return "adios2_error_invalid_argument";

  if (err == adios2_error_system_error)
    return "adios2_error_system_error";

  if (err == adios2_error_runtime_error)
    return "adios2_error_runtime_error";

  if (err == adios2_error_exception)
    return "adios2_error_exception";

  return "unknown error";
}

// --------------------------------------------------------------------------
adios2_type adiosIdType()
{
  if (sizeof(svtkIdType) == sizeof(int64_t))
    {
    return adios2_type_int64_t; // 64 bits
    }
  else if(sizeof(svtkIdType) == sizeof(int32_t))
    {
    return adios2_type_int32_t; // 32 bits
    }
  else
    {
    SENSEI_ERROR("No conversion from svtkIdType to ADIOS2_DATATYPES")
    MPI_Abort(MPI_COMM_WORLD, -1);
    }
  return adios2_type_unknown;
}

// --------------------------------------------------------------------------
adios2_type adiosType(svtkDataArray* da)
{
  if (dynamic_cast<svtkFloatArray*>(da))
    {
    return adios2_type_float;
    }
  else if (dynamic_cast<svtkDoubleArray*>(da))
    {
    return adios2_type_double;
    }
  else if (dynamic_cast<svtkCharArray*>(da))
    {
    return adios2_type_uint8_t;
    }
  else if (dynamic_cast<svtkIntArray*>(da))
    {
    return adios2_type_int32_t;
    }
  else if (dynamic_cast<svtkLongArray*>(da))
    {
    if (sizeof(long) == 4)
      return adios2_type_int32_t; // 32 bits
    return adios2_type_int64_t; // 64 bits
    }
  else if (dynamic_cast<svtkLongLongArray*>(da))
    {
    return adios2_type_int64_t; // 64 bits
    }
  else if (dynamic_cast<svtkUnsignedCharArray*>(da))
    {
    return adios2_type_uint8_t;
    }
  else if (dynamic_cast<svtkUnsignedIntArray*>(da))
    {
    return adios2_type_uint32_t;
    }
  else if (dynamic_cast<svtkUnsignedLongArray*>(da))
    {
    if (sizeof(unsigned long) == 4)
      return adios2_type_uint32_t; // 32 bits
    return adios2_type_uint64_t; // 64 bits
    }
  else if (dynamic_cast<svtkUnsignedLongLongArray*>(da))
    {
    return adios2_type_uint64_t; // 64 bits
    }
  else if (dynamic_cast<svtkIdTypeArray*>(da))
    {
    return adiosIdType();
    }
  else
    {
    SENSEI_ERROR("the adios2 type for data array \"" << da->GetClassName()
      << "\" is currently not implemented")
    MPI_Abort(MPI_COMM_WORLD, -1);
    }
  return adios2_type_unknown;
}

// --------------------------------------------------------------------------
adios2_type adiosType(int svtkt)
{
  switch (svtkt)
    {
    case SVTK_FLOAT:
      return adios2_type_float;
      break;
    case SVTK_DOUBLE:
      return adios2_type_double;
      break;
    case SVTK_CHAR:
      return adios2_type_uint8_t;
      break;
    case SVTK_UNSIGNED_CHAR:
      return adios2_type_uint8_t;
      break;
    case SVTK_INT:
      return adios2_type_int32_t;
      break;
    case SVTK_UNSIGNED_INT:
      return adios2_type_uint32_t;
      break;
    case SVTK_LONG:
      if (sizeof(long) == 4)
        return adios2_type_int32_t; // 32 bits
      return adios2_type_int64_t; // 64 bits
      break;
    case SVTK_UNSIGNED_LONG:
      if (sizeof(long) == 4)
        return adios2_type_uint32_t; // 32 bits
      return adios2_type_uint64_t; // 64 bits
      break;
    case SVTK_LONG_LONG:
      return adios2_type_int64_t;
      break;
    case SVTK_UNSIGNED_LONG_LONG:
      return adios2_type_uint64_t; // 64 bits
      break;
    case SVTK_ID_TYPE:
      return adiosIdType();
      break;
    default:
      {
      SENSEI_ERROR("the adios2 type for svtk type enumeration " << svtkt
        << " is currently not implemented")
      MPI_Abort(MPI_COMM_WORLD, -1);
      }
    }
  return adios2_type_unknown;
}

// --------------------------------------------------------------------------
int isLegacyDataObject(int code)
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
svtkDataObject *newDataObject(int code)
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

// --------------------------------------------------------------------------
bool streamIsFileBased(std::string engine)
{
  if (engine == "BPFile" || engine == "HDF5" || engine == "BP3" || engine == "BP4" || engine == "BP5")
    {
    return true;
    }
  else
    {
    return false;
    }
  SENSEI_ERROR("Unknown adios2 read engine " << engine)
  return false;
}

// --------------------------------------------------------------------------
template <typename val_t>
int adiosInq(InputStream &iStream, const std::string &path, val_t &val)
{
  adios2_error aerr = adios2_error_none;
  if ((aerr = adios2_get_by_name(iStream.Handles.engine,
    path.c_str(), &val, adios2_mode_sync)))
    {
    SENSEI_ERROR("adios2_get_by_name \"" << path << "\" failed. "
        << adios2_strerror(aerr))
    return -1;
    }
  return 0;
}



//----------------------------------------------------------------------------
void InputStream::SetFileName(const std::string &fileName)
{
  this->FileName = fileName;
  this->FileSeries = 0;

  // look for for a decimal format specifier in the file name.
  // this is used to detect file series.
  std::regex decFmtSpec("%[0-9]*[diuoxX]", std::regex_constants::basic);
  if (std::regex_search(this->FileName.c_str(), decFmtSpec))
    {
    this->FileSeries = 1;
    }
}

//----------------------------------------------------------------------------
void InputStream::AddParameter(const std::string &key, const std::string &value)
{
  this->Parameters.emplace_back(key, value);
}

// --------------------------------------------------------------------------
int InputStream::Initialize(MPI_Comm comm)
{
  sensei::TimeEvent<128> mark("senseiADIOS2::InputStream::Initialize");

  // initialize adios2
#if ADIOS2_VERSION_MAJOR > 2 || (ADIOS2_VERSION_MAJOR == 2 && ADIOS2_VERSION_MINOR >= 9)
  // adios2_init()'s signature changed in version 2.9.0
  this->Adios = adios2_init(comm);
#else
  this->Adios = adios2_init(comm, adios2_debug_mode(this->DebugMode));
#endif
  if (this->Adios == nullptr)
    {
    SENSEI_ERROR("adios2_init failed")
    return -1;
    }

  return 0;
}

// --------------------------------------------------------------------------
int InputStream::EndOfStream()
{
  adios2_error aerr = adios2_error_none;

  // check for end of stream signal.
  uint64_t time_step = 0;
  if ((aerr = adios2_get_by_name(this->Handles.engine,
    "time_step", &time_step, adios2_mode_sync)))
    {
    SENSEI_ERROR("adios2_get_by_name time_step failed."
        << adios2_strerror(aerr))
    return -1;
    }

  if (time_step == std::numeric_limits<uint64_t>::max())
    {
    SENSEI_STATUS("End of stream detected")
    return 1;
    }

  return 0;
}

// --------------------------------------------------------------------------
int InputStream::Open()
{
  sensei::TimeEvent<128> mark("senseiADIOS2::InputStream::Open");

  adios2_error aerr;

  // format the file name
  char buffer[1024];
  buffer[1023] = '\0';
  if (this->FileSeries)
    {
    snprintf(buffer, 1023, this->FileName.c_str(), this->FileIndex);
    ++this->FileIndex;
    }
  else
    {
    strncpy(buffer, this->FileName.c_str(), 1023);
    }

  // clear existing state
  if (this->Handles.io)
    {
    adios2_bool result = adios2_false;
    if ((aerr = adios2_remove_io(&result, this->Adios, "SENSEI")))
      {
      SENSEI_ERROR("adios2_remove_io failed. " << adios2_strerror(aerr))
      return -1;
      }
    }

  // open the state object
  if (!(this->Handles.io = adios2_declare_io(this->Adios, "SENSEI")))
    {
    SENSEI_ERROR("adios2_declare_io failed")
    return -1;
    }

  // pass additional engine control parameters
  unsigned int nParms = this->Parameters.size();
  for (unsigned int j = 0; j < nParms; ++j)
    {
    std::pair<std::string,std::string> &parm = this->Parameters[j];

    if ((aerr = adios2_set_parameter(this->Handles.io,
      parm.first.c_str(), parm.second.c_str())))
      {
      SENSEI_ERROR("adios2_set_paramter " << parm.first
        << " = " << parm.second << " failed. " << adios2_strerror(aerr))
      return -1;
      }
    }

  // set the engine
  if ((aerr = adios2_set_engine(this->Handles.io, this->ReadEngine.c_str())))
    {
    SENSEI_ERROR("adios2_set_engine \"" << this->ReadEngine
      << "\" failed. " << adios2_strerror(aerr))
    return -1;
    }

  // open the file
  if (!(this->Handles.engine = adios2_open(this->Handles.io,
    buffer, adios2_mode_read)))
    {
    SENSEI_ERROR("adios2_open \"" << this->FileName
      << "\" for reading failed")
    return -1;
    }

  // determine the number of steps available
  if (this->FileSeries)
    {
    size_t nSteps = 0;
    if ((aerr = adios2_steps(&nSteps, this->Handles.engine)))
      {
      SENSEI_ERROR("Failed to determin the number of steps in \""
        << buffer  << "\"")
      return -1;
      }
    this->StepsPerFile = nSteps;
    }

  return 0;
}

// --------------------------------------------------------------------------
int InputStream::BeginStep()
{
  // begin step
  adios2_step_status status = adios2_step_status_ok;

  adios2_error err = adios2_begin_step(this->Handles.engine,
    adios2_step_mode_read, -1, &status);

  // in practice this happens when we actually should continue. BP4
  // engine only? TODO -- document the specifics here.
  if (err != 0)
    {
    SENSEI_WARNING("adios2_begin_step for read failed. status=" << status)
    }

  if (err != 0 && status == adios2_step_status::adios2_step_status_other_error)
    {
    SENSEI_ERROR("adios2_begin_step failed and reports error status")
    return -1;
    }

  // Check if the status says we are at the end or no step is ready, if so, just leave
  if (status == adios2_step_status::adios2_step_status_end_of_stream ||
      status == adios2_step_status::adios2_step_status_not_ready)
    {
    return -1;
    }

  return 0;
}

// --------------------------------------------------------------------------
int InputStream::AdvanceTimeStep()
{
  sensei::TimeEvent<128> mark("senseiADIOS2::InputStream::AdvanceTimeStep");

  // end the previous time step
  adios2_error endErr = adios2_end_step(this->Handles.engine);
    if (endErr != 0)
    {
    SENSEI_ERROR("adios2_end_step failed")
    return -1;
    }

  // check for multiple time steps per file, and if it is time to
  // open the next file
  ++this->StepIndex;
  if (this->FileSeries && ((this->StepIndex % this->StepsPerFile) == 0))
    {
    if (this->Close())
      return -1;

    if (this->Open())
      return -1;
    }

  // begin the next time step
  if (this->BeginStep())
    return -1;

  // check for more data to come
  if (this->EndOfStream())
    return 1;

  return 0;
}

// --------------------------------------------------------------------------
int InputStream::Close()
{
  sensei::TimeEvent<128> mark("senseiADIOS2::InputStream::Close");

  if (this->Handles.engine)
    {
    adios2_error err = adios2_close(this->Handles.engine);
    if (err != 0)
      {
      SENSEI_ERROR("adios2_close failed. " << err)
      return -1;
      }
    }

  this->Handles.engine = nullptr;

  return 0;
}

// --------------------------------------------------------------------------
int InputStream::Finalize()
{
  sensei::TimeEvent<128> mark("senseiADIOS2::InputStream::Finalize");

  this->Close();

  if (this->Adios)
    {
    adios2_error err = adios2_finalize(this->Adios);
    if (err != 0)
      {
      SENSEI_ERROR("adios2_finalize failed. " << err)
      return -1;
      }
    }

  this->Adios = nullptr;
  this->Handles.engine = nullptr;
  this->Handles.io = nullptr;
  this->ReadEngine = "";

  return 0;
}




// helper for writing binary streams of data. binary stream is a sequence
// of bytes that has externally defined meaning.
class BinaryStreamSchema
{
public:
  static int DefineVariables(AdiosHandle handles, const std::string &path);

  static int Write(AdiosHandle handles, const std::string &path,
    const sensei::BinaryStream &md);

  static int Read(MPI_Comm comm, InputStream &iStream,
    const std::string &path, sensei::BinaryStream &md);
};

// --------------------------------------------------------------------------
int BinaryStreamSchema::DefineVariables(AdiosHandle handles,
  const std::string &path)
{
  sensei::TimeEvent<128> mark(
    "senseiADIOS2::BinaryStreamSchema::DefineVariables");

  // define the stream
  size_t defaultSize = 1024;
  if (!adios2_define_variable(handles.io, path.c_str(),
    adios2_type_int8_t, 1, &defaultSize, &defaultSize,
    &defaultSize, adios2_constant_dims_false))
    {
    SENSEI_ERROR("adios2_define_variable \"" << path << "\" failed")
    return -1;
    }
  return 0;
}

// --------------------------------------------------------------------------
int BinaryStreamSchema::Read(MPI_Comm comm, InputStream &iStream,
  const std::string &path, sensei::BinaryStream &str)
{
  sensei::Profiler::StartEvent("senseiADIOS2::BinaryStreamSchema::Read");

  // get metadata

  adios2_variable *vinfo = adios2_inquire_variable(iStream.Handles.io, path.c_str());
  if (!vinfo)
    {
    SENSEI_ERROR("adios2_inquire_variable \"" << path << "\" failed")
    return -1;
    }

  size_t nbytes = 0;
  adios2_error shapeErr = adios2_variable_shape(&nbytes, vinfo);
  if (shapeErr != 0)
    {
    SENSEI_ERROR("adios2_variable_shape failed w/ " << shapeErr)
    return -1;
    }

  // allocate a buffer
  str.Resize(nbytes);
  str.SetReadPos(0);
  str.SetWritePos(nbytes);

  if (!streamIsFileBased(iStream.ReadEngine))
    {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    adios2_error selErr = adios2_set_block_selection(vinfo, rank);
    if (selErr != 0)
      {
      SENSEI_ERROR("adios2_set_block_selection failed")
      return -1;
      }
    }

  // read it
  adios2_error readErr = adios2_get(iStream.Handles.engine,
    vinfo, str.GetData(), adios2_mode_sync);

  if (readErr != 0)
    {
    SENSEI_ERROR("adios2_get \"" << path << "\" failed")
    return -1;
    }

  sensei::Profiler::EndEvent("senseiADIOS2::BinaryStreamSchema::Read", nbytes);

  return 0;
}

// --------------------------------------------------------------------------
int BinaryStreamSchema::Write(AdiosHandle handles, const std::string &path,
  const sensei::BinaryStream &str)
{
  sensei::Profiler::StartEvent("senseiADIOS2::BinaryStreamSchema::Write");

  adios2_variable *internalBinVar = adios2_inquire_variable(handles.io, path.c_str());

  size_t n = str.Size();
  size_t selectionStart = 0;

  if (adios2_set_shape(internalBinVar, 1, &n) ||
      adios2_set_selection(internalBinVar, 1, &selectionStart, &n) ||
      adios2_put_by_name(handles.engine, path.c_str(), str.GetData(), adios2_mode_sync))
    {
    SENSEI_ERROR("Failed to write BinaryStream at \"" << path << "\"")
    return -1;
    }

  sensei::Profiler::EndEvent("senseiADIOS2::BinaryStreamSchema::Write", n);
  return 0;
}



class VersionSchema
{
public:
  VersionSchema() : Revision(3), LowestCompatibleRevision(3) {}

  int DefineVariables(AdiosHandle handles);

  int Write(AdiosHandle handles);

  int Read(InputStream &iStream);

private:
  unsigned int Revision;
  unsigned int LowestCompatibleRevision;
};

// --------------------------------------------------------------------------
int VersionSchema::DefineVariables(AdiosHandle handles)
{
  sensei::TimeEvent<128> mark(
    "senseiADIOS2::VersionSchema::DefineVariables");

  if (!adios2_define_variable(handles.io, "DataObjectSchema",
    adios2_type_uint32_t, 0, NULL, NULL, NULL, adios2_constant_dims_true))
    {
    SENSEI_ERROR("adios2_define_variable DataObjectSchema failed")
    return -1;
    }

  return 0;
}

// --------------------------------------------------------------------------
int VersionSchema::Write(AdiosHandle handles)
{
  sensei::Profiler::StartEvent("senseiADIOS2::VersionSchema::Write");

  if (adios2_put_by_name(handles.engine, "DataObjectSchema",
    &this->Revision, adios2_mode_sync))
    {
    SENSEI_ERROR("adios2_put_by_name DataObjectSchema failed")
    return -1;
    }

  sensei::Profiler::EndEvent("senseiADIOS2::VersionSchema::Write",
     sizeof(this->Revision));

  return 0;
}

// --------------------------------------------------------------------------
int VersionSchema::Read(InputStream &iStream)
{
  sensei::Profiler::StartEvent("senseiADIOS2::VersionSchema::Read");

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

  sensei::Profiler::EndEvent("senseiADIOS2::VersionSchema::Read",
    sizeof(revision));

  return 0;
}


struct ArraySchema
{
  int DefineVariables(MPI_Comm comm, AdiosHandle handles,
    const std::string &ons, const sensei::MeshMetadataPtr &md);

  int DefineVariable(MPI_Comm comm, AdiosHandle handles, const std::string &ons,
    int i, int array_type, int num_components, int array_cen,
    unsigned long long num_points_total, unsigned long long num_cells_total,
    unsigned int num_blocks, const std::vector<long> &block_num_points,
    const std::vector<long> &block_num_cells,
    const std::vector<int> &block_owner, std::vector<size_t> &putVarsStart,
    std::vector<size_t> &putVarsCount, adios2_variable *&putVar);

  int Write(MPI_Comm comm, AdiosHandle handles,
    const sensei::MeshMetadataPtr &md, svtkCompositeDataSet *dobj);

  int Write(MPI_Comm comm, AdiosHandle handles, unsigned int i,
    const std::string &array_name, int array_cen, svtkCompositeDataSet *dobj,
    unsigned int num_blocks, const std::vector<int> &block_owner,
    const std::vector<size_t> &putVarsStart, const std::vector<size_t> &putVarsCount,
    adios2_variable *putVar);

  int Read(MPI_Comm comm, AdiosHandle handles, const std::string &ons,
    const std::string &array_name, int centering,
    const sensei::MeshMetadataPtr &md, svtkCompositeDataSet *dobj);

  int Read(MPI_Comm comm, AdiosHandle handles , const std::string &ons,
    unsigned int i, const std::string &array_name, int array_type,
    unsigned long long num_components, int array_cen, unsigned int num_blocks,
    const std::vector<long> &block_num_points,
    const std::vector<long> &block_num_cells, const std::vector<int> &block_owner,
    svtkCompositeDataSet *dobj);

  std::map<std::string,std::vector<size_t>> PutVarsStart;
  std::map<std::string,std::vector<size_t>> PutVarsCount;
  std::map<std::string,std::vector<adios2_variable*>> PutVars;
};


// --------------------------------------------------------------------------
int ArraySchema::DefineVariable(MPI_Comm comm, AdiosHandle handles,
  const std::string &ons, int i, int array_type, int num_components,
  int array_cen, unsigned long long num_points_total,
  unsigned long long num_cells_total, unsigned int num_blocks,
  const std::vector<long> &block_num_points,
  const std::vector<long> &block_num_cells,
  const std::vector<int> &block_owner,
  std::vector<size_t> &putVarsStart,
  std::vector<size_t> &putVarsCount,
  adios2_variable *&putVar)
{
  sensei::TimeEvent<128> mark("senseiADIOS2::ArraySchema::DefineVariable");

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

   // /data_object_<id>/data_array_<id>/data
  std::string path = ans.str() + "data";

  // select global size either point or cell data
  unsigned long num_elem_total = (array_cen == svtkDataObject::POINT ?
    num_points_total : num_cells_total)*num_components;

  // adios2 type of the array
  adios2_type elem_type = adiosType(array_type);

  // unlike ADIOS1 in ADIOS2 you can't define the variable once for
  // each write. Instead define it once with an empty size. and calculate
  // all the book keeping info to later write each block's chunk of
  // the array in the correct spot.

  size_t localStart = 0;
  size_t localCount = 0;

  putVar = adios2_define_variable(handles.io,
     path.c_str(), elem_type, 1, &num_elem_total, &localStart,
     &localCount, adios2_constant_dims_false);

  if (!putVar)
    {
    SENSEI_ERROR("adios2_define_variable failed with "
      << "num_elem_total=" << num_elem_total << " path=\""
      << path << "\"")
    }

  unsigned long block_offset = 0;
  for (unsigned int j = 0; j < num_blocks; ++j)
    {
    // get the block size
    unsigned long num_elem_local = (array_cen == svtkDataObject::POINT ?
      block_num_points[j] : block_num_cells[j])*num_components;

    // define the variable for a local block
    if (block_owner[j] == rank)
      {
      // save the var attr. to use later for putting
      putVarsStart[i*num_blocks + j] = block_offset;
      putVarsCount[i*num_blocks + j] = num_elem_local;
      }

    // update the block offset
    block_offset += num_elem_local;
    }

  return 0;
}

// --------------------------------------------------------------------------
int ArraySchema::DefineVariables(MPI_Comm comm, AdiosHandle handles,
  const std::string &ons, const sensei::MeshMetadataPtr &md)
{
  sensei::TimeEvent<128> mark(
    "senseiADIOS2::ArraySchema::DefineVariables");

  std::vector<size_t> &putVarsStart = this->PutVarsStart[md->MeshName];
  std::vector<size_t> &putVarsCount = this->PutVarsCount[md->MeshName];
  std::vector<adios2_variable*> &putVars = this->PutVars[md->MeshName];

  // allocate write ids
  unsigned int num_blocks = md->NumBlocks;
  unsigned int num_arrays = md->NumArrays;

  bool have_ghost_cells = md->NumGhostCells || sensei::SVTKUtils::AMR(md);

  unsigned int num_ghost_arrays =
    (have_ghost_cells ? 1 : 0) + (md->NumGhostNodes ? 1 : 0);

  unsigned int num_arrays_total = num_arrays + num_ghost_arrays;

  putVarsStart.resize(num_blocks*num_arrays_total);
  putVarsCount.resize(num_blocks*num_arrays_total);
  putVars.resize(num_arrays_total);

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
    if (this->DefineVariable(comm, handles, ons, i, md->ArrayType[i],
      md->ArrayComponents[i], md->ArrayCentering[i], num_points_total,
      num_cells_total, num_blocks, md->BlockNumPoints, md->BlockNumCells,
      md->BlockOwner, putVarsStart, putVarsCount, putVars[i]))
      return -1;
    }

  // define ghost arrays
  if (have_ghost_cells && this->DefineVariable(comm, handles, ons,
      num_arrays, SVTK_UNSIGNED_CHAR, 1, svtkDataObject::CELL, num_points_total,
      num_cells_total, num_blocks, md->BlockNumPoints, md->BlockNumCells,
      md->BlockOwner, putVarsStart, putVarsCount, putVars[num_arrays]))
      return -1;

  if (md->NumGhostNodes && this->DefineVariable(comm, handles, ons,
      num_arrays, SVTK_UNSIGNED_CHAR, 1, svtkDataObject::POINT, num_points_total,
      num_cells_total, num_blocks, md->BlockNumPoints, md->BlockNumCells,
      md->BlockOwner, putVarsStart, putVarsCount,
      putVars[num_arrays + (have_ghost_cells ? 1 : 0)]))
      return -1;

  return 0;
}

// --------------------------------------------------------------------------
int ArraySchema::Write(MPI_Comm comm, AdiosHandle handles, unsigned int i,
  const std::string &array_name, int array_cen, svtkCompositeDataSet *dobj,
  unsigned int num_blocks, const std::vector<int> &block_owner,
  const std::vector<size_t> &putVarsStart,
  const std::vector<size_t> &putVarsCount,
  adios2_variable *putVar)
{
  sensei::Profiler::StartEvent("senseiADIOS2::ArraySchema::Write");
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
        SENSEI_ERROR("Failed to get array \"" << array_name
          << "\" block " << j << " array " << i)
        return -1;
        }

      // select the spot in the global array that this block's
      // data will land
      size_t start = putVarsStart[i*num_blocks + j];
      size_t count = putVarsCount[i*num_blocks + j];
      if (adios2_set_selection(putVar, 1, &start, &count))
        {
        SENSEI_ERROR("adios2_set_selection start=" << start
          << " count=" << count << " block " << j << " array "
          << i << " failed")
        return -1;
        }

      // do the write
      if (adios2_put(handles.engine, putVar,
        da->GetVoidPointer(0), adios2_mode_sync))
        {
        SENSEI_ERROR("adios2_put block " << j << " array "
          << i << " failed")
        return -1;
        }

      numBytes += count*sensei::SVTKUtils::Size(da->GetDataType());
      }

    it->GoToNextItem();
    }

  it->Delete();

  sensei::Profiler::EndEvent("senseiADIOS2::ArraySchema::Write", numBytes);
  return 0;
}

// --------------------------------------------------------------------------
int ArraySchema::Write(MPI_Comm comm, AdiosHandle handles,
  const sensei::MeshMetadataPtr &md, svtkCompositeDataSet *dobj)
{
  sensei::TimeEvent<128> mark("senseiADIOS2::ArraySchema::Write");

  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  std::vector<size_t> &putVarsStart = this->PutVarsStart[md->MeshName];
  std::vector<size_t> &putVarsCount = this->PutVarsCount[md->MeshName];
  std::vector<adios2_variable*> &putVars = this->PutVars[md->MeshName];

  // write data arrays
  unsigned int num_arrays = md->NumArrays;
  bool have_ghost_cells = md->NumGhostCells || sensei::SVTKUtils::AMR(md);

  for (unsigned int i = 0; i < num_arrays; ++i)
    {
    if (this->Write(comm, handles, i, md->ArrayName[i], md->ArrayCentering[i],
      dobj, md->NumBlocks, md->BlockOwner, putVarsStart, putVarsCount, putVars[i]))
      return -1;
    }

  // write ghost arrays
  if (have_ghost_cells && this->Write(comm, handles, num_arrays, "svtkGhostType",
    svtkDataObject::CELL, dobj, md->NumBlocks, md->BlockOwner, putVarsStart,
    putVarsCount, putVars[num_arrays]))
      return -1;

  if (md->NumGhostNodes && this->Write(comm, handles, num_arrays,
    "svtkGhostType", svtkDataObject::POINT, dobj, md->NumBlocks,
    md->BlockOwner, putVarsStart, putVarsCount,
    putVars[num_arrays + (have_ghost_cells ? 1 : 0)]))
    return -1;

  return 0;
}

// --------------------------------------------------------------------------
int ArraySchema::Read(MPI_Comm comm, AdiosHandle handles, const std::string &ons,
  unsigned int i, const std::string &array_name, int array_type,
  unsigned long long num_components, int array_cen, unsigned int num_blocks,
  const std::vector<long> &block_num_points,
  const std::vector<long> &block_num_cells, const std::vector<int> &block_owner,
  svtkCompositeDataSet *dobj)
{
  sensei::Profiler::StartEvent("senseiADIOS2::ArraySchema::Read");
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
    std::string path = ans.str() + "data";

    // get the block size
    unsigned long long num_elem_local = (array_cen == svtkDataObject::POINT ?
      block_num_points[j] : block_num_cells[j])*num_components;

    // define the variable for a local block
    if (block_owner[j] ==  rank)
      {
      adios2_variable *vinfo = adios2_inquire_variable(handles.io, path.c_str());
      if (!vinfo)
        {
        SENSEI_ERROR("adios2_inquire_variable \"" << path
          << "\" block " << j << " array " << i << " failed")
        return -1;
        }

      size_t start = block_offset;
      size_t count = num_elem_local;
      if (adios2_set_selection(vinfo, 1, &start, &count))
        {
        SENSEI_ERROR("adios2_set_selection start=" << start
          << " count=" << count << " block " << j << " array " << i << " failed")
        return -1;
        }

      svtkDataArray *array = svtkDataArray::CreateDataArray(array_type);
      array->SetNumberOfComponents(num_components);
      array->SetNumberOfTuples(num_elem_local);
      array->SetName(array_name.c_str());

      // /data_object_<id>/data_array_<id>/data
      if (adios2_get(handles.engine, vinfo, array->GetVoidPointer(0),
        adios2_mode_sync))
        {
        SENSEI_ERROR("adios2_get \"" << array_name
          << "\" block " << j << " array " << i << " failed")
        return -1;
        }

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

  sensei::Profiler::EndEvent("senseiADIOS2::ArraySchema::Read", numBytes);
  return 0;
}

// --------------------------------------------------------------------------
int ArraySchema::Read(MPI_Comm comm, AdiosHandle handles, const std::string &ons,
  const std::string &name, int centering, const sensei::MeshMetadataPtr &md,
  svtkCompositeDataSet *dobj)
{
  sensei::TimeEvent<128> mark("senseiADIOS2::ArraySchema::Read");

  unsigned int num_blocks = md->NumBlocks;
  unsigned int num_arrays = md->NumArrays;

  bool have_ghost_cells = md->NumGhostCells || sensei::SVTKUtils::AMR(md);

  // read ghost arrays
  if (name == "svtkGhostType")
    {
    unsigned int i = (centering == svtkDataObject::CELL ?
      num_arrays : num_arrays + (have_ghost_cells ? 1 : 0));

    return this->Read(comm, handles, ons, i, "svtkGhostType",
      SVTK_UNSIGNED_CHAR, 1, centering, num_blocks, md->BlockNumPoints,
      md->BlockNumCells, md->BlockOwner, dobj);
    }

  // read data arrays
  for (unsigned int i = 0; i < num_arrays; ++i)
    {
    const std::string &array_name = md->ArrayName[i];
    int array_cen = md->ArrayCentering[i];

    // skip all but the requested array
    if ((centering != array_cen) || (name != array_name))
      continue;

    return this->Read(comm, handles, ons, i, array_name, md->ArrayType[i],
      md->ArrayComponents[i], array_cen, num_blocks, md->BlockNumPoints,
      md->BlockNumCells, md->BlockOwner, dobj);
    }

  return 0;
}


struct PointSchema
{
  int DefineVariables(MPI_Comm comm, AdiosHandle handles,
    const std::string &ons, const sensei::MeshMetadataPtr &md);

  int Write(MPI_Comm comm, AdiosHandle handles,
    const sensei::MeshMetadataPtr &md, svtkCompositeDataSet *dobj);

  int Read(MPI_Comm comm, AdiosHandle handles, const std::string &ons,
    const sensei::MeshMetadataPtr &md, svtkCompositeDataSet *dobj);

  std::map<std::string, std::vector<size_t>> Starts;
  std::map<std::string, std::vector<size_t>> Counts;
  std::map<std::string, adios2_variable*> PutVars;
};

// --------------------------------------------------------------------------
int PointSchema::DefineVariables(MPI_Comm comm, AdiosHandle handles,
  const std::string &ons, const sensei::MeshMetadataPtr &md)
{
  (void)comm;

  if (sensei::SVTKUtils::Unstructured(md) || sensei::SVTKUtils::Structured(md)
    || sensei::SVTKUtils::Polydata(md))
    {
    sensei::TimeEvent<128> mark("senseiADIOS2::PointSchema::DefineVariables");

    // calc global size
    unsigned int num_blocks = md->NumBlocks;
    unsigned long long num_total = 0;
    for (unsigned int j = 0; j < num_blocks; ++j)
      {
      num_total += md->BlockNumPoints[j];
      }

    // data type for points
    adios2_type type = adiosType(md->CoordinateType);

    // global size
    size_t gdims = 3*num_total;
    size_t ldims = 0;
    size_t loffs = 0;

    // /data_object_<id>/points
    std::string path_pts = ons + "points";

    adios2_variable *var = adios2_define_variable(
      handles.io, path_pts.c_str(), type, 1,  &gdims, &loffs,
      &ldims, adios2_constant_dims_false);

    if (var == nullptr)
      {
      SENSEI_ERROR("adios2_define_variable \"" << path_pts << "\" failed")
      return -1;
      }

    // save the id for subsequent write
    this->PutVars[md->MeshName] = var;

    // calculate the starts and counts for each write
    std::vector<size_t> &starts = this->Starts[md->MeshName];
    std::vector<size_t> &counts = this->Counts[md->MeshName];

    starts.resize(num_blocks);
    counts.resize(num_blocks);

    unsigned long long block_offset = 0;

    for (unsigned int j = 0; j < num_blocks; ++j)
      {
      unsigned long long num_local = md->BlockNumPoints[j];

      counts[j] = 3*num_local;
      starts[j] = 3*block_offset;

      block_offset += num_local;
      }
    }

  return 0;
}

// --------------------------------------------------------------------------
int PointSchema::Write(MPI_Comm comm, AdiosHandle handles,
  const sensei::MeshMetadataPtr &md, svtkCompositeDataSet *dobj)
{
  if (sensei::SVTKUtils::Unstructured(md) || sensei::SVTKUtils::Structured(md)
    || sensei::SVTKUtils::Polydata(md))
    {
    sensei::Profiler::StartEvent("senseiADIOS2::PointSchema::Write");
    long long numBytes = 0ll;

    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    const std::vector<size_t> &starts = this->Starts[md->MeshName];
    const std::vector<size_t> &counts = this->Counts[md->MeshName];
    adios2_variable *putVar = this->PutVars[md->MeshName];

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

        // select the spot in the global array that this block's
        // data will land
        size_t start = starts[j];
        size_t count = counts[j];
        if (adios2_set_selection(putVar, 1, &start, &count))
          {
          SENSEI_ERROR("adios2_set_selection start=" << start
            << " count=" << count << " block " << j << " failed")
          return -1;
          }

        svtkDataArray *da = ds->GetPoints()->GetData();
        if (adios2_put(handles.engine, putVar,
          da->GetVoidPointer(0), adios2_mode_sync))
          {
          SENSEI_ERROR("adios2_put \"" << md->MeshName
            << "\" block " << j << " points failed")
          return -1;
          }

        numBytes += count*sensei::SVTKUtils::Size(da->GetDataType());
        }

      it->GoToNextItem();
      }
    it->Delete();

    sensei::Profiler::EndEvent("senseiADIOS2::PointSchema::Write", numBytes);
    }

  return 0;
}

// --------------------------------------------------------------------------
int PointSchema::Read(MPI_Comm comm, AdiosHandle handles, const std::string &ons,
  const sensei::MeshMetadataPtr &md, svtkCompositeDataSet *dobj)
{
  if (sensei::SVTKUtils::Unstructured(md) || sensei::SVTKUtils::Structured(md)
    || sensei::SVTKUtils::Polydata(md))
    {
    sensei::Profiler::StartEvent("senseiADIOS2::PointSchema::Read");
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
        std::string path = ons + "points";
        adios2_variable *vinfo = adios2_inquire_variable(handles.io, path.c_str());
        if (!vinfo)
          {
          SENSEI_ERROR("ADIOS2 stream is missing \"" << path << "\"")
          return -1;
          }

        size_t start = 3*block_offset;
        size_t count = 3*num_local;
        if (adios2_set_selection(vinfo, 1, &start, &count))
          {
          SENSEI_ERROR("adios2_set_selection points block " << j
            << " start=" << start << " count=" << count << " failed")
          return -1;
          }

        svtkDataArray *points = svtkDataArray::CreateDataArray(md->CoordinateType);
        points->SetNumberOfComponents(3);
        points->SetNumberOfTuples(num_local);
        points->SetName("points");

        adios2_error getErr = adios2_get(handles.engine,
          vinfo, points->GetVoidPointer(0), adios2_mode_sync);

        if (getErr != 0)
          {
          SENSEI_ERROR("adios2_get points "
            << " block " << j <<  " failed. " << getErr)
          return -1;
          }

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

    sensei::Profiler::EndEvent("senseiADIOS2::PointSchema::Read", numBytes);
    }

  return 0;
}



struct UnstructuredCellSchema
{
  int DefineVariables(MPI_Comm comm, AdiosHandle handles,
    const std::string &ons, const sensei::MeshMetadataPtr &md);

  int Write(MPI_Comm comm, AdiosHandle handles,
    const sensei::MeshMetadataPtr &md, svtkCompositeDataSet *dobj);

  int Read(MPI_Comm comm, AdiosHandle handles, const std::string &ons,
    const sensei::MeshMetadataPtr &md, svtkCompositeDataSet *dobj);

  std::map<std::string, adios2_variable*> CellTypeVars;
  std::map<std::string, std::vector<size_t>> CellTypeStarts;
  std::map<std::string, std::vector<size_t>> CellTypeCounts;

  std::map<std::string, adios2_variable*> CellOffsVars;
  std::map<std::string, adios2_variable*> CellConnVars;
  std::map<std::string, std::vector<size_t>> CellConnStarts;
  std::map<std::string, std::vector<size_t>> CellConnCounts;
};

// --------------------------------------------------------------------------
int UnstructuredCellSchema::DefineVariables(MPI_Comm comm, AdiosHandle handles,
  const std::string &ons, const sensei::MeshMetadataPtr &md)
{
  (void)comm;

  if (sensei::SVTKUtils::Unstructured(md))
    {
    sensei::TimeEvent<128> mark(
      "senseiADIOS2::UnstructuredCellSchema::DefineVariables");

    // allocate write ids
    unsigned int num_blocks = md->NumBlocks;

    // calculate global size
    unsigned long long num_cells_total = 0;
    unsigned long long cell_array_size_total = 0;

    for (unsigned int j = 0; j < num_blocks; ++j)
      {
      num_cells_total += md->BlockNumCells[j];
      cell_array_size_total += md->BlockCellArraySize[j];
      }

    // get the data type for cells
    adios2_type cell_array_type = adios2_type_int64_t;
    long long elemSize = sensei::SVTKUtils::Size(md->CellArrayType);
    if (elemSize == 8)
      {
      cell_array_type = adios2_type_int64_t;
      }
    else if (elemSize == 4)
      {
      cell_array_type = adios2_type_int32_t;
      }
    else
      {
      SENSEI_ERROR("Invalid cell array type " << md->CellArrayType)
      return -1;
      }

    // global sizes
    // the new SVTK offsets layout uses 1 extra value per block
    size_t cell_type_gdmins = num_cells_total;
    size_t cell_offset_gdims = num_cells_total + num_blocks;
    size_t cell_connect_gdims = cell_array_size_total;

    size_t start = 0;
    size_t count = 0;

    // /data_object_<id>/cell_types
    std::string path_ct = ons + "cell_types";

    adios2_variable *var = adios2_define_variable(handles.io, path_ct.c_str(),
      adios2_type_uint8_t, 1, &cell_type_gdmins, &start, &count,
      adios2_constant_dims_false);

    if (var == nullptr)
      {
      SENSEI_ERROR("adios2_define_variable " << path_ct << "\" failed")
      return -1;
      }

    // save the variable for later writes
    this->CellTypeVars[md->MeshName] = var;

    // /data_object_<id>/cell_offsets
    std::string path_offs = ons + "cell_offsets";

    var = adios2_define_variable(handles.io,
      path_offs.c_str(), cell_array_type, 1, &cell_offset_gdims,
      &start, &count, adios2_constant_dims_false);

    if (var == nullptr)
      {
      SENSEI_ERROR("adios2_define_variable \"" << path_offs << "\" failed")
      return -1;
      }

    // save the variable for later writes
    this->CellOffsVars[md->MeshName] = var;

    // /data_object_<id>/cell_connectivity
    std::string path_conn = ons + "cell_connectivity";

    var = adios2_define_variable(handles.io,
      path_conn.c_str(), cell_array_type, 1, &cell_connect_gdims,
      &start, &count, adios2_constant_dims_false);

    if (var == nullptr)
      {
      SENSEI_ERROR("adios2_define_variable \"" << path_conn << "\" failed")
      return -1;
      }

    // save the variable for later writes
    this->CellConnVars[md->MeshName] = var;

    // calculate start and count for writing each block.
    // in the new SVTK layout, the offsets are 1 longer than the types, and
    // start elements rank further than types
    std::vector<size_t> &cellConnStarts = this->CellConnStarts[md->MeshName];
    std::vector<size_t> &cellConnCounts = this->CellConnCounts[md->MeshName];
    cellConnStarts.resize(num_blocks);
    cellConnCounts.resize(num_blocks);

    std::vector<size_t> &cellTypeStarts = this->CellTypeStarts[md->MeshName];
    std::vector<size_t> &cellTypeCounts = this->CellTypeCounts[md->MeshName];
    cellTypeStarts.resize(num_blocks);
    cellTypeCounts.resize(num_blocks);

    size_t cellTypesStart = 0;
    size_t cellConnStart = 0;
    for (unsigned int j = 0; j < num_blocks; ++j)
      {
      // get the block size
      size_t numCellsLocal = md->BlockNumCells[j];
      size_t cellConnSizeLocal = md->BlockCellArraySize[j];

      // local size & offset
      cellConnCounts[j]  = cellConnSizeLocal;
      cellConnStarts[j] = cellConnStart;

      cellTypeCounts[j]  = numCellsLocal;
      cellTypeStarts[j] = cellTypesStart;

      // update the block offset
      cellTypesStart += numCellsLocal;
      cellConnStart += cellConnSizeLocal;
      }
    }

  return 0;
}

// --------------------------------------------------------------------------
int UnstructuredCellSchema::Write(MPI_Comm comm, AdiosHandle handles,
  const sensei::MeshMetadataPtr &md, svtkCompositeDataSet *dobj)
{
  if (sensei::SVTKUtils::Unstructured(md))
    {
    sensei::Profiler::StartEvent("senseiADIOS2::UnstructuredCellSchema::Write");
    long long numBytes = 0ll;
    long long elemSize = sensei::SVTKUtils::Size(md->CellArrayType);

    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    adios2_variable *cellOffsVar = this->CellOffsVars[md->MeshName];
    adios2_variable *cellConnVar = this->CellConnVars[md->MeshName];

    std::vector<size_t> &cellConnStarts = this->CellConnStarts[md->MeshName];
    std::vector<size_t> &cellConnCounts = this->CellConnCounts[md->MeshName];

    adios2_variable *cellTypeVar = this->CellTypeVars[md->MeshName];
    std::vector<size_t> &cellTypeStarts = this->CellTypeStarts[md->MeshName];
    std::vector<size_t> &cellTypeCounts = this->CellTypeCounts[md->MeshName];

    svtkCompositeDataIterator *it = dobj->NewIterator();
    it->SetSkipEmptyNodes(0);
    it->InitTraversal();

    unsigned int num_blocks = md->NumBlocks;
    for (unsigned int j = 0; j < num_blocks; ++j)
      {
      // write local block
      if (md->BlockOwner[j] == rank)
        {
        svtkUnstructuredGrid *ds =
          dynamic_cast<svtkUnstructuredGrid*>(it->GetCurrentDataObject());

        if (!ds)
          {
          SENSEI_ERROR("Failed to get block " << j)
          return -1;
          }

        // write cell cellTypes
        size_t ctStart = cellTypeStarts[j];
        size_t ctCount = cellTypeCounts[j];

        if (adios2_set_selection(cellTypeVar, 1, &ctStart, &ctCount))
          {
          SENSEI_ERROR("adios2_set_selection start=" << ctStart
            << " count=" << ctCount << " block " << j << " failed")
          return -1;
          }

        svtkDataArray *cta = ds->GetCellTypesArray();
        if (adios2_put(handles.engine, cellTypeVar,
          cta->GetVoidPointer(0), adios2_mode_sync))
          {
          SENSEI_ERROR("adios2_put cell types for mesh \""
            << md->MeshName << "\" block " << j << " failed")
          return -1;
          }

        // write cell cell offset array
        // In the new SVTK layout cell offsets are always 1 element longer than
        // cell types
        size_t coStart = ctStart + j;
        size_t coCount = ctCount + 1;

        if (adios2_set_selection(cellOffsVar, 1, &coStart, &coCount))
          {
          SENSEI_ERROR("adios2_set_selection start=" << coStart
            << " count=" << coCount << " block " << j << " failed")
          return -1;
          }

        svtkDataArray *co = ds->GetCells()->GetOffsetsArray();

        if (adios2_put(handles.engine, cellOffsVar,
          co->GetVoidPointer(0), adios2_mode_sync))
          {
          SENSEI_ERROR("adios2_put cell offsets for mesh \""
            << md->MeshName << "\" block " << j << " failed")
          return -1;
          }

        // write cell cell connectivity array
        size_t ccStart = cellConnStarts[j];
        size_t ccCount = cellConnCounts[j];

        if (adios2_set_selection(cellConnVar, 1, &ccStart, &ccCount))
          {
          SENSEI_ERROR("adios2_set_selection start=" << ccStart
            << " count=" << ccCount << " block " << j << " failed")
          return -1;
          }

        svtkDataArray *cc = ds->GetCells()->GetConnectivityArray();

        if (adios2_put(handles.engine, cellConnVar,
          cc->GetVoidPointer(0), adios2_mode_sync))
          {
          SENSEI_ERROR("adios2_put cell offsets for mesh \""
            << md->MeshName << "\" block " << j << " failed")
          return -1;
          }

        // track number of bytes for profiling
        numBytes += (ccCount + coCount) * elemSize + ctCount*sizeof(unsigned char);
        }

      it->GoToNextItem();
      }

    it->Delete();

    sensei::Profiler::EndEvent("senseiADIOS2::UnstructuredCellSchema::Write",
      numBytes);
    }

  return 0;
}

// --------------------------------------------------------------------------
int UnstructuredCellSchema::Read(MPI_Comm comm, AdiosHandle handles,
  const std::string &ons, const sensei::MeshMetadataPtr &md, svtkCompositeDataSet *dobj)
{
  if (sensei::SVTKUtils::Unstructured(md))
    {
    sensei::Profiler::StartEvent("senseiADIOS2::UnstructuredCellSchema::Read");

    long long numBytes = 0ll;
    long long elemSize = sensei::SVTKUtils::Size(md->CellArrayType);

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
      if (md->BlockOwner[j] == rank)
        {
        // /data_object_<id>/cell_types
        std::string ctPath = ons + "cell_types";
        adios2_variable *ctVar = adios2_inquire_variable(handles.io, ctPath.c_str());
        if (!ctVar)
          {
          SENSEI_ERROR("ADIOS2 stream is missing \"" << ctPath << "\"")
          return -1;
          }

        size_t ctStart = cell_types_block_offset;
        size_t ctCount = num_cells_local;

        if (adios2_set_selection(ctVar, 1, &ctStart, &ctCount))
          {
          SENSEI_ERROR("adios2_set_selection start=" << ctStart
            << " count=" << ctCount << " failed")
          return -1;
          }

        svtkUnsignedCharArray *ct = svtkUnsignedCharArray::New();
        ct->SetNumberOfComponents(1);
        ct->SetNumberOfTuples(ctCount);
        ct->SetName("CellTypes");

        adios2_error ctErr = adios2_get(handles.engine,
          ctVar, ct->GetVoidPointer(0), adios2_mode_sync);

        if (ctErr != 0)
          {
          SENSEI_ERROR("adios2_get cell types block " << j <<  " failed")
          return -1;
          }

        // /data_object_<id>/cell_offsets
        svtkDataArray *co = nullptr;
        if (md->CellArrayType == SVTK_TYPE_INT64)
          {
          co = svtkAOSDataArrayTemplate<svtkTypeInt64>::New();
          }
        else if (md->CellArrayType == SVTK_TYPE_INT32)
          {
          co = svtkAOSDataArrayTemplate<svtkTypeInt32>::New();
          }
        else
          {
          SENSEI_ERROR("Invalid CellArrayType " << md->CellArrayType)
          return -1;
          }

        // in the new SVTK layout cell offsets are 1 longer than cell types
        // and start rank elements beyond the cell types start
        size_t coStart = ctStart + j;
        size_t coCount = ctCount + 1;

        co->SetNumberOfComponents(1);
        co->SetNumberOfTuples(coCount);
        co->SetName("CellOffsets");

        std::string coPath = ons + "cell_offsets";
        adios2_variable *coVar = adios2_inquire_variable(handles.io, coPath.c_str());
        if (coVar == nullptr)
          {
          SENSEI_ERROR("ADIOS2 stream is missing \"" << coPath << "\"")
          return -1;
          }

        if (adios2_set_selection(coVar, 1, &coStart, &coCount))
          {
          SENSEI_ERROR("adios2_set_selection start=" << coStart
            << " count=" << coCount << " failed")
          return -1;
          }

        adios2_error coErr = adios2_get(handles.engine,
          coVar, co->GetVoidPointer(0), adios2_mode_sync);

        if (coErr != 0)
          {
          SENSEI_ERROR("adios2_get cell types block " << j <<  " failed")
          return -1;
          }

        // /data_object_<id>/cell_connectivity
        size_t ccStart = cell_array_block_offset;
        size_t ccCount = cell_array_size_local;

        svtkDataArray *cc = nullptr;
        if (md->CellArrayType == SVTK_TYPE_INT64)
          {
          cc = svtkAOSDataArrayTemplate<svtkTypeInt64>::New();
          }
        else if (md->CellArrayType == SVTK_TYPE_INT32)
          {
          cc = svtkAOSDataArrayTemplate<svtkTypeInt32>::New();
          }
        else
          {
          SENSEI_ERROR("Invalid CellArrayType " << md->CellArrayType)
          return -1;
          }

        cc->SetNumberOfComponents(1);
        cc->SetNumberOfTuples(cell_array_size_local);
        cc->SetName("CellConnectivity");

        std::string ccPath = ons + "cell_connectivity";
        adios2_variable *ccVar = adios2_inquire_variable(handles.io, ccPath.c_str());
        if (!ccVar)
          {
          SENSEI_ERROR("adios2_inquire_variable \"" << ccPath
            << "\" block " << j <<  " failed")
          return -1;
          }

        if (adios2_set_selection(ccVar, 1, &ccStart, &ccCount))
          {
          SENSEI_ERROR("adios2_set_selection start=" << ccStart
            << " count=" << ccCount << " block " << j <<  " failed")
          return -1;
          }

        adios2_error caErr = adios2_get(handles.engine,
          ccVar, cc->GetVoidPointer(0), adios2_mode_sync);

        if (caErr)
          {
          SENSEI_ERROR("adios2_get cell_array block " << j <<  " failed")
          return -1;
          }

        // package cells
        svtkCellArray *ca = svtkCellArray::New();
        ca->SetData(co, cc);

        // pass cells into the dataset
        svtkUnstructuredGrid *ds =
          dynamic_cast<svtkUnstructuredGrid*>(it->GetCurrentDataObject());

        if (!ds)
          {
          SENSEI_ERROR("Failed to get block " << j)
          return -1;
          }

        ds->SetCells(ct, ca);

        ca->Delete();
        ct->Delete();
        co->Delete();
        cc->Delete();

        numBytes += ctCount * sizeof(unsigned char) + (coCount + ccCount) * elemSize;
        }

      // update the block offset
      cell_types_block_offset += num_cells_local;
      cell_array_block_offset += cell_array_size_local;

      it->GoToNextItem();
      }

    sensei::Profiler::EndEvent("senseiADIOS2::UnstructuredCellSchema::Read", numBytes);
    }

  return 0;
}







struct PolydataCellSchema
{
  int DefineVariables(MPI_Comm comm, AdiosHandle handles,
    const std::string &ons, const sensei::MeshMetadataPtr &md);

  int Write(MPI_Comm comm, AdiosHandle handles,
    const sensei::MeshMetadataPtr &md, svtkCompositeDataSet *dobj);

  int Read(MPI_Comm comm, AdiosHandle handles, const std::string &ons,
    const sensei::MeshMetadataPtr &md, svtkCompositeDataSet *dobj);

  std::map<std::string, adios2_variable*> CellTypeVars;
  std::map<std::string, std::vector<size_t>> CellTypeStarts;
  std::map<std::string, std::vector<size_t>> CellTypeCounts;

  std::map<std::string, adios2_variable*> CellOffsVars;
  std::map<std::string, adios2_variable*> CellConnVars;
  std::map<std::string, std::vector<size_t>> CellConnStarts;
  std::map<std::string, std::vector<size_t>> CellConnCounts;
};

// --------------------------------------------------------------------------
int PolydataCellSchema::DefineVariables(MPI_Comm comm, AdiosHandle handles,
  const std::string &ons, const sensei::MeshMetadataPtr &md)
{
  (void)comm;

  if (sensei::SVTKUtils::Polydata(md))
    {
    sensei::TimeEvent<128> mark(
      "senseiADIOS2::PolydataCellSchema::DefineVariables");

    // allocate write ids
    unsigned int num_blocks = md->NumBlocks;

    // calculate global size
    unsigned long long num_cells_total = 0;
    unsigned long long cell_array_size_total = 0;

    for (unsigned int j = 0; j < num_blocks; ++j)
      {
      num_cells_total += md->BlockNumCells[j];
      cell_array_size_total += md->BlockCellArraySize[j];
      }

    // get the data type for cells
    adios2_type cell_array_type = adios2_type_int64_t;
    long long elemSize = sensei::SVTKUtils::Size(md->CellArrayType);
    if (elemSize == 8)
      {
      cell_array_type = adios2_type_int64_t;
      }
    else if (elemSize == 4)
      {
      cell_array_type = adios2_type_int32_t;
      }
    else
      {
      SENSEI_ERROR("Invalid cell array type " << md->CellArrayType)
      return -1;
      }

    // global sizes
    // new SVTK layout always has an extra element per cell type
    size_t cell_type_gdmins = 4 * num_blocks;
    size_t cell_offset_gdims = num_cells_total + 4 * num_blocks;
    size_t cell_connect_gdims = cell_array_size_total;

    size_t start = 0;
    size_t count = 0;

    // /data_object_<id>/cell_types
    std::string path_ct = ons + "cell_types";

    adios2_variable *var = adios2_define_variable(handles.io, path_ct.c_str(),
      adios2_type_int64_t, 1, &cell_type_gdmins, &start, &count,
      adios2_constant_dims_false);

    if (var == nullptr)
      {
      SENSEI_ERROR("adios2_define_variable " << path_ct << "\" failed")
      return -1;
      }

    // save the variable for later writes
    this->CellTypeVars[md->MeshName] = var;

    // /data_object_<id>/cell_offsets
    std::string path_offs = ons + "cell_offsets";

    var = adios2_define_variable(handles.io,
      path_offs.c_str(), cell_array_type, 1, &cell_offset_gdims,
      &start, &count, adios2_constant_dims_false);

    if (var == nullptr)
      {
      SENSEI_ERROR("adios2_define_variable \"" << path_offs << "\" failed")
      return -1;
      }

    // save the variable for later writes
    this->CellOffsVars[md->MeshName] = var;

    // /data_object_<id>/cell_connectivity
    std::string path_conn = ons + "cell_connectivity";

    var = adios2_define_variable(handles.io,
      path_conn.c_str(), cell_array_type, 1, &cell_connect_gdims,
      &start, &count, adios2_constant_dims_false);

    if (var == nullptr)
      {
      SENSEI_ERROR("adios2_define_variable \"" << path_conn << "\" failed")
      return -1;
      }

    // save the variable for later writes
    this->CellConnVars[md->MeshName] = var;

    // calculate start and count for writing each block.
    // in the new SVTK layout, the offsets are 1 longer than the types, and
    // start elements rank further than types
    std::vector<size_t> &cellConnStarts = this->CellConnStarts[md->MeshName];
    std::vector<size_t> &cellConnCounts = this->CellConnCounts[md->MeshName];
    cellConnStarts.resize(num_blocks);
    cellConnCounts.resize(num_blocks);

    std::vector<size_t> &cellOffsStarts = this->CellTypeStarts[md->MeshName];
    std::vector<size_t> &cellOffsCounts = this->CellTypeCounts[md->MeshName];
    cellOffsStarts.resize(num_blocks);
    cellOffsCounts.resize(num_blocks);

    size_t cellOffsStart = 0;
    size_t cellConnStart = 0;
    for (unsigned int j = 0; j < num_blocks; ++j)
      {
      // get the block size
      // new SVTK layout uses an extra element per cell type
      size_t numCellsLocal = md->BlockNumCells[j] + 4;
      size_t cellConnSizeLocal = md->BlockCellArraySize[j];

      // local size & offset
      cellConnCounts[j]  = cellConnSizeLocal;
      cellConnStarts[j] = cellConnStart;

      cellOffsCounts[j]  = numCellsLocal;
      cellOffsStarts[j] = cellOffsStart;

      // update the block offset
      cellOffsStart += numCellsLocal;
      cellConnStart += cellConnSizeLocal;
      }
    }

  return 0;
}

// --------------------------------------------------------------------------
int PolydataCellSchema::Write(MPI_Comm comm, AdiosHandle handles,
  const sensei::MeshMetadataPtr &md, svtkCompositeDataSet *dobj)
{
  if (sensei::SVTKUtils::Polydata(md))
    {
    sensei::Profiler::StartEvent("senseiADIOS2::PolydataCellSchema");
    long long numBytes = 0ll;
    long long elemSize = sensei::SVTKUtils::Size(md->CellArrayType);

    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    adios2_variable *cellOffsVar = this->CellOffsVars[md->MeshName];
    adios2_variable *cellConnVar = this->CellConnVars[md->MeshName];

    std::vector<size_t> &cellConnStarts = this->CellConnStarts[md->MeshName];
    std::vector<size_t> &cellConnCounts = this->CellConnCounts[md->MeshName];

    adios2_variable *cellTypeVar = this->CellTypeVars[md->MeshName];
    std::vector<size_t> &cellOffsStarts = this->CellTypeStarts[md->MeshName];
    std::vector<size_t> &cellOffsCounts = this->CellTypeCounts[md->MeshName];

    svtkCompositeDataIterator *it = dobj->NewIterator();
    it->SetSkipEmptyNodes(0);
    it->InitTraversal();

    // write local blocks
    unsigned int num_blocks = md->NumBlocks;
    for (unsigned int j = 0; j < num_blocks; ++j)
      {
      // get local block
      if (md->BlockOwner[j] == rank)
        {
        svtkPolyData *ds =
          dynamic_cast<svtkPolyData*>(it->GetCurrentDataObject());

        if (!ds)
          {
          SENSEI_ERROR("Failed to get block " << j)
          return -1;
          }

        // cell type lengths
        size_t ctStart = 4*j;
        size_t ctCount = 4;

        int64_t ct[4] = {ds->GetNumberOfVerts(), ds->GetNumberOfLines(),
            ds->GetNumberOfPolys(),  ds->GetNumberOfStrips()};

        // write the cell types
        if (adios2_set_selection(cellTypeVar, 1, &ctStart, &ctCount))
          {
          SENSEI_ERROR("adios2_set_selection start=" << ctStart
            << " count=" << ctCount << " block " << j << " failed")
          return -1;
          }

        if (adios2_put(handles.engine, cellTypeVar, ct, adios2_mode_sync))
          {
          SENSEI_ERROR("adios2_put cell types for mesh \""
            << md->MeshName << "\" block " << j << " failed")
          return -1;
          }

        // get the total block sizes
        size_t coStart = cellOffsStarts[j];
        size_t coCount = cellOffsCounts[j];

        size_t ccStart = cellConnStarts[j];
        size_t ccCount = cellConnCounts[j];

        svtkDataArray *co = nullptr;
        svtkDataArray *cc = nullptr;

        // concatenate cell arrays
        switch (md->CellArrayType)
          {
          svtkCellTemplateMacro(

            using ARRAY_TT = svtkAOSDataArrayTemplate<SVTK_TT>;

            ARRAY_TT *tco = ARRAY_TT::New();
            tco->SetNumberOfTuples(coCount);

            ARRAY_TT *tcc = ARRAY_TT::New();
            tcc->SetNumberOfTuples(ccCount);

            size_t coId = 0;
            size_t ccId = 0;

            sensei::SVTKUtils::PackCells<SVTK_TT>(
              dynamic_cast<ARRAY_TT*>(ds->GetVerts()->GetOffsetsArray()),
              dynamic_cast<ARRAY_TT*>(ds->GetVerts()->GetConnectivityArray()),
              tco, tcc, coId, ccId);

            sensei::SVTKUtils::PackCells<SVTK_TT>(
              dynamic_cast<ARRAY_TT*>(ds->GetLines()->GetOffsetsArray()),
              dynamic_cast<ARRAY_TT*>(ds->GetLines()->GetConnectivityArray()),
              tco, tcc, coId, ccId);

            sensei::SVTKUtils::PackCells<SVTK_TT>(
              dynamic_cast<ARRAY_TT*>(ds->GetPolys()->GetOffsetsArray()),
              dynamic_cast<ARRAY_TT*>(ds->GetPolys()->GetConnectivityArray()),
              tco, tcc, coId, ccId);

            sensei::SVTKUtils::PackCells<SVTK_TT>(
              dynamic_cast<ARRAY_TT*>(ds->GetStrips()->GetOffsetsArray()),
              dynamic_cast<ARRAY_TT*>(ds->GetStrips()->GetConnectivityArray()),
              tco, tcc, coId, ccId);

            co = tco;
            cc = tcc;
            )
          default:
            {
            SENSEI_ERROR("Unsupported CellArrayType " << md->CellArrayType)
            return -1;
            }
          }

        // write cell cell offset array
        if (adios2_set_selection(cellOffsVar, 1, &coStart, &coCount))
          {
          SENSEI_ERROR("adios2_set_selection start=" << coStart
            << " count=" << coCount << " block " << j << " failed")
          return -1;
          }

        if (adios2_put(handles.engine, cellOffsVar,
          co->GetVoidPointer(0), adios2_mode_sync))
          {
          SENSEI_ERROR("adios2_put cell offsets for mesh \""
            << md->MeshName << "\" block " << j << " failed")
          return -1;
          }

        // write cell cell connectivity array
        if (adios2_set_selection(cellConnVar, 1, &ccStart, &ccCount))
          {
          SENSEI_ERROR("adios2_set_selection start=" << ccStart
            << " count=" << ccCount << " block " << j << " failed")
          return -1;
          }

        if (adios2_put(handles.engine, cellConnVar,
          cc->GetVoidPointer(0), adios2_mode_sync))
          {
          SENSEI_ERROR("adios2_put cell offsets for mesh \""
            << md->MeshName << "\" block " << j << " failed")
          return -1;
          }

        co->Delete();
        cc->Delete();

        // track number of bytes for profiling
        numBytes += 4 *sizeof(uint64_t) + (coCount + ccCount) * elemSize;
        }

      it->GoToNextItem();
      }

    it->Delete();

    sensei::Profiler::EndEvent("senseiADIOS2::PolydataCellSchema::Write",
      numBytes);
    }

  return 0;
}

// --------------------------------------------------------------------------
int PolydataCellSchema::Read(MPI_Comm comm, AdiosHandle handles,
  const std::string &ons, const sensei::MeshMetadataPtr &md, svtkCompositeDataSet *dobj)
{
  if (sensei::SVTKUtils::Polydata(md))
    {
    sensei::Profiler::StartEvent("senseiADIOS2::PolydataCellSchema::Read");

    long long numBytes = 0ll;
    long long elemSize = sensei::SVTKUtils::Size(md->CellArrayType);

    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    svtkCompositeDataIterator *it = dobj->NewIterator();
    it->SetSkipEmptyNodes(0);
    it->InitTraversal();

    // calc block offsets
    unsigned long long cell_offsets_block_offset = 0;
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
        std::string ctPath = ons + "cell_types";
        adios2_variable *ctVar = adios2_inquire_variable(handles.io, ctPath.c_str());
        if (!ctVar)
          {
          SENSEI_ERROR("ADIOS2 stream is missing \"" << ctPath << "\"")
          return -1;
          }

        size_t ctStart = 4 * j;
        size_t ctCount = 4;
        int64_t ct[4] = {0};

        if (adios2_set_selection(ctVar, 1, &ctStart, &ctCount))
          {
          SENSEI_ERROR("adios2_set_selection start=" << ctStart
            << " count=" << ctCount << " failed")
          return -1;
          }

        adios2_error ctErr = adios2_get(handles.engine,
          ctVar, ct, adios2_mode_sync);

        if (ctErr != 0)
          {
          SENSEI_ERROR("adios2_get cell types block " << j <<  " failed")
          return -1;
          }

        // /data_object_<id>/cell_offsets
        size_t coStart = cell_offsets_block_offset;
        size_t coCount = num_cells_local + 4;

        // allocate arrays
        svtkDataArray *co = nullptr;
        switch (md->CellArrayType)
          {
          svtkCellTemplateMacro(
            using ARRAY_TT = svtkAOSDataArrayTemplate<SVTK_TT>;
            co = ARRAY_TT::New();
            )
          }
        co->SetNumberOfTuples(coCount);
        co->SetName("CellOffsets");

        // read offsets
        std::string coPath = ons + "cell_offsets";
        adios2_variable *coVar = adios2_inquire_variable(handles.io, coPath.c_str());
        if (coVar == nullptr)
          {
          SENSEI_ERROR("ADIOS2 stream is missing \"" << coPath << "\"")
          return -1;
          }

        if (adios2_set_selection(coVar, 1, &coStart, &coCount))
          {
          SENSEI_ERROR("adios2_set_selection start=" << ctStart
            << " count=" << ctCount << " failed")
          return -1;
          }

        adios2_error coErr = adios2_get(handles.engine,
          coVar, co->GetVoidPointer(0), adios2_mode_sync);

        if (coErr != 0)
          {
          SENSEI_ERROR("adios2_get cell types block " << j <<  " failed")
          return -1;
          }

        // /data_object_<id>/cell_connectivity
        size_t ccStart = cell_array_block_offset;
        size_t ccCount = cell_array_size_local;

        // allocate arrays
        svtkDataArray *cc = nullptr;
        switch (md->CellArrayType)
          {
          svtkCellTemplateMacro(
            using ARRAY_TT = svtkAOSDataArrayTemplate<SVTK_TT>;
            cc = ARRAY_TT::New();
            )
          }
        cc->SetNumberOfTuples(ccCount);
        cc->SetName("CellConnectivity");

        std::string ccPath = ons + "cell_connectivity";
        adios2_variable *ccVar = adios2_inquire_variable(handles.io, ccPath.c_str());
        if (!ccVar)
          {
          SENSEI_ERROR("adios2_inquire_variable \"" << ccPath
            << "\" block " << j <<  " failed")
          return -1;
          }

        if (adios2_set_selection(ccVar, 1, &ccStart, &ccCount))
          {
          SENSEI_ERROR("adios2_set_selection start=" << ccStart
            << " count=" << ccCount << " block " << j <<  " failed")
          return -1;
          }

        adios2_error caErr = adios2_get(handles.engine,
          ccVar, cc->GetVoidPointer(0), adios2_mode_sync);

        if (caErr)
          {
          SENSEI_ERROR("adios2_get cell_array block " << j <<  " failed")
          return -1;
          }

        // unpack cells
        svtkCellArray *verts = svtkCellArray::New();
        svtkCellArray *lines = svtkCellArray::New();
        svtkCellArray *polys = svtkCellArray::New();
        svtkCellArray *strips = svtkCellArray::New();

        switch (md->CellArrayType)
          {
          svtkCellTemplateMacro(

            using ARRAY_TT = svtkAOSDataArrayTemplate<SVTK_TT>;

            ARRAY_TT *tco = dynamic_cast<ARRAY_TT*>(co);
            ARRAY_TT *tcc = dynamic_cast<ARRAY_TT*>(cc);

            size_t coId = 0;
            size_t ccId = 0;

            sensei::SVTKUtils::UnpackCells<SVTK_TT>(ct[0], tco, tcc, verts, coId, ccId);
            sensei::SVTKUtils::UnpackCells<SVTK_TT>(ct[1], tco, tcc, lines, coId, ccId);
            sensei::SVTKUtils::UnpackCells<SVTK_TT>(ct[2], tco, tcc, polys, coId, ccId);
            sensei::SVTKUtils::UnpackCells<SVTK_TT>(ct[3], tco, tcc, strips, coId, ccId);
            )
          }

        co->Delete();
        cc->Delete();

        // pass cells into the dataset
        svtkPolyData *ds =
          dynamic_cast<svtkPolyData*>(it->GetCurrentDataObject());

        if (!ds)
          {
          SENSEI_ERROR("Failed to get block " << j)
          return -1;
          }

        ds->SetVerts(verts);
        ds->SetLines(lines);
        ds->SetPolys(polys);
        ds->SetStrips(strips);

        verts->Delete();
        lines->Delete();
        polys->Delete();
        strips->Delete();

        numBytes += (ccCount + coCount) * elemSize + 4 *sizeof(uint64_t);
        }

      // update the block offset
      cell_offsets_block_offset += num_cells_local + 4;
      cell_array_block_offset += cell_array_size_local;

      it->GoToNextItem();
      }

    sensei::Profiler::EndEvent("senseiADIOS2::PolydataCellSchema::Read", numBytes);
    }

  return 0;
}


struct LogicallyCartesianSchema
{
  int DefineVariables(MPI_Comm comm, AdiosHandle handles, const std::string &ons,
    const sensei::MeshMetadataPtr &md);

  int Write(MPI_Comm comm, AdiosHandle handles,
    const sensei::MeshMetadataPtr &md, svtkCompositeDataSet *dobj);

  int Read(MPI_Comm comm, AdiosHandle handles,
    const std::string &ons, const sensei::MeshMetadataPtr &md,
    svtkCompositeDataSet *dobj);

  std::map<std::string, adios2_variable*> WriteVars;
};

// --------------------------------------------------------------------------
int LogicallyCartesianSchema::DefineVariables(MPI_Comm comm, AdiosHandle handles,
  const std::string &ons, const sensei::MeshMetadataPtr &md)
{
  (void)comm;

  if (sensei::SVTKUtils::LogicallyCartesian(md))
    {
    sensei::TimeEvent<128> mark("senseiADIOS2::"
      "LogicallyCartesianSchema::DefineVariables");

    // global sizes
    unsigned int num_blocks = md->NumBlocks;
    size_t gdims = 6*num_blocks;
    size_t start = 0;
    size_t count = 0;

    // /data_object_<id>/extent
    std::string path_extent = ons + "extent";

    adios2_variable *var = adios2_define_variable(handles.io,
       path_extent.c_str(), adios2_type_int32_t, 1, &gdims,
       &start, &count, adios2_constant_dims_false);

    if (var == nullptr)
      {
      SENSEI_ERROR("adios2_define_variable \"" << path_extent
        << "\"  failed")
      return -1;
      }

    // save the id for subsequent write
    this->WriteVars[md->MeshName] = var;
    }

  return 0;
}

// --------------------------------------------------------------------------
int LogicallyCartesianSchema::Write(MPI_Comm comm, AdiosHandle handles,
  const sensei::MeshMetadataPtr &md, svtkCompositeDataSet *dobj)
{
  if (sensei::SVTKUtils::LogicallyCartesian(md))
    {
    sensei::Profiler::StartEvent("senseiADIOS2::"
      "LogicallyCartesianSchema::Write");

    long long numBytes = 0ll;

    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    adios2_variable *writeVar = this->WriteVars[md->MeshName];

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

        // select the spot in the global array that this block's
        // data will land
        size_t start = 6*j;
        size_t count = 6;
        if (adios2_set_selection(writeVar, 1, &start, &count))
          {
          SENSEI_ERROR("adios2_set_selection start=" << start
            << " count=" << count << " block " << j << " failed")
          return -1;
          }

        int ierr = 0;
        switch (md->BlockType)
          {
          case SVTK_RECTILINEAR_GRID:
            ierr = adios2_put(handles.engine, writeVar,
              dynamic_cast<svtkRectilinearGrid*>(dobj)->GetExtent(),
              adios2_mode_sync);
            break;

          case SVTK_IMAGE_DATA:
          case SVTK_UNIFORM_GRID:
            ierr = adios2_put(handles.engine, writeVar,
              dynamic_cast<svtkImageData*>(dobj)->GetExtent(), adios2_mode_sync);
            break;

          case SVTK_STRUCTURED_GRID:
            ierr = adios2_put(handles.engine, writeVar,
              dynamic_cast<svtkStructuredGrid*>(dobj)->GetExtent(), adios2_mode_sync);
            break;
          }

        if (ierr)
          {
          SENSEI_ERROR("adios2_put extent for block " << j << " failed")
          return -1;
          }

        numBytes += 6*sizeof(int);
        }

      it->GoToNextItem();
      }

    it->Delete();

    sensei::Profiler::EndEvent("senseiADIOS2::LogicallyCartesianSchema::Write",
      numBytes);
    }

  return 0;
}

// --------------------------------------------------------------------------
int LogicallyCartesianSchema::Read(MPI_Comm comm, AdiosHandle handles,
  const std::string &ons, const sensei::MeshMetadataPtr &md,
  svtkCompositeDataSet *dobj)
{
  if (sensei::SVTKUtils::LogicallyCartesian(md))
    {
    sensei::Profiler::StartEvent("senseiADIOS2::LogicallyCartesianSchema::Read");
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
        std::string extent_path = ons + "extent";
        adios2_variable *vinfo = adios2_inquire_variable(handles.io, extent_path.c_str());
        if (!vinfo)
          {
          SENSEI_ERROR("adios2_inquire_variable \"" << extent_path
            << "\" block " << j <<  " failed")
          return -1;
          }

        // /data_object_<id>/extent
        size_t hexplet_start = 6*j;
        size_t hexplet_count = 6;
        if (adios2_set_selection(vinfo, 1, &hexplet_start, &hexplet_count))
          {
          SENSEI_ERROR("adios2_set_selection start=" << hexplet_start
            << " count=" << hexplet_count << " block " << j <<  " failed")
          return -1;
          }

        int ext[6] = {0};
        adios2_error getErr = adios2_get(handles.engine, vinfo, ext, adios2_mode_sync);
        if (getErr != 0)
          {
          SENSEI_ERROR("adios2_get extent block " << j << " failed")
          return -1;
          }

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

    sensei::Profiler::EndEvent("senseiADIOS2::LogicallyCartesianSchema::Read", numBytes);
    }

  return 0;
}



struct UniformCartesianSchema
{
  int DefineVariables(MPI_Comm comm, AdiosHandle handles, const std::string &ons,
    const sensei::MeshMetadataPtr &md);

  int Write(MPI_Comm comm, AdiosHandle handles,
    const sensei::MeshMetadataPtr &md, svtkCompositeDataSet *dobj);

  int Read(MPI_Comm comm, AdiosHandle handles, const std::string &ons,
    const sensei::MeshMetadataPtr &md, svtkCompositeDataSet *dobj);

  std::map<std::string, adios2_variable*> OriginWriteVar;
  std::map<std::string, adios2_variable*> SpacingWriteVar;
};

// --------------------------------------------------------------------------
int UniformCartesianSchema::DefineVariables(MPI_Comm comm, AdiosHandle handles,
  const std::string &ons, const sensei::MeshMetadataPtr &md)
{
  (void)comm;

  if (sensei::SVTKUtils::UniformCartesian(md))
    {
    sensei::TimeEvent<128> mark(
      "senseiADIOS2::UniformCartesianSchema::DefineVariables");

    // allocate write ids
    unsigned int num_blocks = md->NumBlocks;

    // global and local sizes. in adios2 only global sizes are specified
    // upfront. local sizes are specified at the time of the write.
    size_t gdims = 3*num_blocks;
    size_t ldims = 0;
    size_t loffs = 0;

    // /data_object_<id>/origin
    std::string path_origin = ons + "origin";

    adios2_variable *var = adios2_define_variable(handles.io,
      path_origin.c_str(), adios2_type_double, 1, &gdims,
      &loffs, &ldims, adios2_constant_dims_false);

    if (var == nullptr)
      {
      SENSEI_ERROR("adios2_define_variable \"" << path_origin << "\" failed")
      return -1;
      }

    // save the id for subsequent write
    this->OriginWriteVar[md->MeshName] = var;

    // /data_object_<id>/spacing
    std::string path_spacing = ons + "spacing";

    var = adios2_define_variable(handles.io, path_spacing.c_str(),
       adios2_type_double, 1, &gdims, &loffs,  &ldims,
       adios2_constant_dims_false);

    if (var == nullptr)
      {
      SENSEI_ERROR("adios2_define_variable \"" << path_spacing << "\" failed")
      return -1;
      }

    // save the id for subsequent write
    this->SpacingWriteVar[md->MeshName] = var;
    }

  return 0;
}

// --------------------------------------------------------------------------
int UniformCartesianSchema::Write(MPI_Comm comm, AdiosHandle handles,
  const sensei::MeshMetadataPtr &md, svtkCompositeDataSet *dobj)
{
  if (sensei::SVTKUtils::UniformCartesian(md))
    {
    sensei::Profiler::StartEvent("senseiADIOS2::UniformCartesianSchema::Write");
    long long numBytes = 0ll;

    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    adios2_variable *originWriteVar = this->OriginWriteVar[md->MeshName];
    adios2_variable *spacingWriteVar = this->SpacingWriteVar[md->MeshName];

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

        // set the spot to write this block's contribution to the global array
        size_t start = 3*j;
        size_t count = 3;
        if (adios2_set_selection(originWriteVar, 1, &start, &count))
          {
          SENSEI_ERROR("adios2_set_selection start=" << start
            << " count=" << count << " block " << j << " failed")
          return -1;
          }

        if (adios2_put(handles.engine, originWriteVar,
          ds->GetOrigin(), adios2_mode_sync))
          {
          SENSEI_ERROR("adios2_put origin block " << j << " failed")
          return -1;
          }

        if (adios2_set_selection(spacingWriteVar, 1, &start, &count))
          {
          SENSEI_ERROR("adios2_set_selection start=" << start
            << " count=" << count << " block " << j << " failed")
          return -1;
          }

        if (adios2_put(handles.engine, spacingWriteVar,
          ds->GetSpacing(), adios2_mode_sync))
          {
          SENSEI_ERROR("adios2_put spacing block " << j << " failed")
          return -1;
          }

        numBytes += 6*sizeof(double);
        }

      it->GoToNextItem();
      }
    it->Delete();

    sensei::Profiler::EndEvent("senseiADIOS2::UniformCartesianSchema::Write", numBytes);
    }

  return 0;
}

// --------------------------------------------------------------------------
int UniformCartesianSchema::Read(MPI_Comm comm, AdiosHandle handles,
  const std::string &ons, const sensei::MeshMetadataPtr &md,
  svtkCompositeDataSet *dobj)
{
  if (sensei::SVTKUtils::UniformCartesian(md))
    {
    sensei::Profiler::StartEvent("senseiADIOS2::UniformCartesianSchema::Read");
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
        std::string origin_path = ons + "origin";
        adios2_variable *origin_vinfo = adios2_inquire_variable(handles.io, origin_path.c_str());
        if (!origin_vinfo)
          {
          SENSEI_ERROR("adios2_inquire_variable \"" << origin_path
            << "\" block " << j <<  " failed")
          return -1;
          }

        size_t triplet_start = 3*j;
        size_t triplet_count = 3;
        if (adios2_set_selection(origin_vinfo, 1, &triplet_start, &triplet_count))
          {
          SENSEI_ERROR("adios2_set_selection block " << j << " start=" << triplet_start
            << " count=" << triplet_count << " failed")
          return -1;
          }

        // /data_object_<id>/data_array_<id>/origin
        double x0[3] = {0.0};

        if (adios2_get(handles.engine, origin_vinfo, x0, adios2_mode_sync))
          {
          SENSEI_ERROR("adios2_get origin block " << j << " failed")
          return -1;
          }

        // /data_object_<id>/data_array_<id>/spacing
        double dx[3] = {0.0};
        std::string spacing_path = ons + "spacing";
        adios2_variable *spacing_vinfo = adios2_inquire_variable(handles.io, spacing_path.c_str());
        if (!spacing_vinfo)
          {
          SENSEI_ERROR("ADIOS2 stream is missing \"" << spacing_path << "\"")
          return -1;
          }

        if (adios2_set_selection(spacing_vinfo, 1, &triplet_start, &triplet_count))
          {
          SENSEI_ERROR("adios2_set_selection block " << j << " start=" << triplet_start
            << " count=" << triplet_count << " failed")
          return -1;
          }

        if (adios2_get(handles.engine, spacing_vinfo, dx, adios2_mode_sync))
          {
          SENSEI_ERROR("adios2_get spacing block " << j << " failed")
          return -1;
          }

        if (adios2_perform_gets(handles.engine))
          {
          SENSEI_ERROR("adios2_perform_gets block " << j << " failed")
          return -1;
          }

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

    sensei::Profiler::EndEvent("senseiADIOS2::UniformCartesianSchema::Read", numBytes);
    }

  return 0;
}



struct StretchedCartesianSchema
{
  int DefineVariables(MPI_Comm comm, AdiosHandle handles, const std::string &ons,
    const sensei::MeshMetadataPtr &md);

  int Write(MPI_Comm comm, AdiosHandle handles,
    const sensei::MeshMetadataPtr &md, svtkCompositeDataSet *dobj);

  int Read(MPI_Comm comm, AdiosHandle handles, const std::string &ons,
    const sensei::MeshMetadataPtr &md, svtkCompositeDataSet *dobj);

  std::map<std::string, adios2_variable*> XCoordWriteVars;
  std::map<std::string, std::vector<size_t>> XCoordStarts;
  std::map<std::string, std::vector<size_t>> XCoordCounts;

  std::map<std::string, adios2_variable*> YCoordWriteVars;
  std::map<std::string, std::vector<size_t>> YCoordStarts;
  std::map<std::string, std::vector<size_t>> YCoordCounts;

  std::map<std::string, adios2_variable*> ZCoordWriteVars;
  std::map<std::string, std::vector<size_t>> ZCoordStarts;
  std::map<std::string, std::vector<size_t>> ZCoordCounts;
};

// --------------------------------------------------------------------------
int StretchedCartesianSchema::DefineVariables(MPI_Comm comm, AdiosHandle handles,
  const std::string &ons, const sensei::MeshMetadataPtr &md)
{
  if (sensei::SVTKUtils::StretchedCartesian(md))
    {
    sensei::TimeEvent<128> mark(
      "senseiADIOS2::StretchedCartesianSchema::DefineVariables");

    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    unsigned int num_blocks = md->NumBlocks;

    // allocate write ids

    // calc global size
    size_t nx_total = 0;
    size_t ny_total = 0;
    size_t nz_total = 0;
    for (unsigned int j = 0; j < num_blocks; ++j)
      {
      int *ext = md->BlockExtents[j].data();
      nx_total += ext[1] - ext[0] + 2;
      ny_total += ext[3] - ext[2] + 2;
      nz_total += ext[5] - ext[4] + 2;
      }

    // data type for points
    adios2_type point_type = adiosType(md->CoordinateType);

    // size and offset calculated later
    size_t start = 0;
    size_t count = 0;

    // /data_object_<id>/x_coords
    std::string path_xc = ons + "x_coords";

    adios2_variable *var = adios2_define_variable(handles.io,
       path_xc.c_str(), point_type, 1, &nx_total, &start, &count,
       adios2_constant_dims_false);

    if (var == nullptr)
      {
      SENSEI_ERROR("adios2_define_variable \"" << path_xc << "\" failed")
      return -1;
      }

    // save the id for subsequent write
    this->XCoordWriteVars[md->MeshName] = var;

    // /data_object_<id>/y_coords
    std::string path_yc = ons + "y_coords";

    var = adios2_define_variable(handles.io, path_yc.c_str(),
       point_type, 1, &ny_total, &start, &count,
       adios2_constant_dims_false);

    if (var == nullptr)
      {
      SENSEI_ERROR("adios2_define_variable \"" << path_yc << "\" failed")
      return -1;
      }

    // save the id for subsequent write
    this->YCoordWriteVars[md->MeshName] = var;

    // /data_object_<id>/data_array_<id>/z_coords
    std::string path_zc = ons + "z_coords";

    var = adios2_define_variable(handles.io,
      path_zc.c_str(), point_type, 1, &nz_total, &start, &count,
      adios2_constant_dims_false);

    if (var == nullptr)
      {
      SENSEI_ERROR("adios2_define_variable \"" << path_zc << "\" failed")
      return -1;
      }

    // save the id for subsequent write
    this->ZCoordWriteVars[md->MeshName] = var;

    // calculate the write selections for each block
    std::vector<size_t> &xcStarts = this->XCoordStarts[md->MeshName];
    std::vector<size_t> &xcCounts = this->XCoordCounts[md->MeshName];
    xcStarts.resize(num_blocks);
    xcCounts.resize(num_blocks);

    std::vector<size_t> &ycStarts = this->YCoordStarts[md->MeshName];
    std::vector<size_t> &ycCounts = this->YCoordCounts[md->MeshName];
    ycStarts.resize(num_blocks);
    ycCounts.resize(num_blocks);

    std::vector<size_t> &zcStarts = this->ZCoordStarts[md->MeshName];
    std::vector<size_t> &zcCounts = this->ZCoordCounts[md->MeshName];
    zcStarts.resize(num_blocks);
    zcCounts.resize(num_blocks);

    size_t x_block_offset = 0;
    size_t y_block_offset = 0;
    size_t z_block_offset = 0;

    for (unsigned int j = 0; j < num_blocks; ++j)
      {
      // get the block size
      int *ext = md->BlockExtents[j].data();
      size_t nx_local = ext[1] - ext[0] + 2;
      size_t ny_local = ext[3] - ext[2] + 2;
      size_t nz_local = ext[5] - ext[4] + 2;

      // local size
      xcCounts[j] = nx_local;
      ycCounts[j] = ny_local;
      zcCounts[j] = nz_local;

      // offset
      xcStarts[j] = x_block_offset;
      ycStarts[j] = y_block_offset;
      zcStarts[j] = z_block_offset;

      // update the block offset
      x_block_offset += nx_local;
      y_block_offset += ny_local;
      z_block_offset += nz_local;
      }
    }

  return 0;
}

// --------------------------------------------------------------------------
int StretchedCartesianSchema::Write(MPI_Comm comm, AdiosHandle handles,
  const sensei::MeshMetadataPtr &md, svtkCompositeDataSet *dobj)
{
  if (sensei::SVTKUtils::StretchedCartesian(md))
    {
    sensei::Profiler::StartEvent("senseiADIOS2::StretchedCartesianSchema");
    long long numBytes = 0ll;

    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    adios2_variable *xcVar = this->XCoordWriteVars[md->MeshName];
    std::vector<size_t> &xcStarts = this->XCoordStarts[md->MeshName];
    std::vector<size_t> &xcCounts = this->XCoordCounts[md->MeshName];

    adios2_variable *ycVar = this->YCoordWriteVars[md->MeshName];
    std::vector<size_t> &ycStarts = this->YCoordStarts[md->MeshName];
    std::vector<size_t> &ycCounts = this->YCoordCounts[md->MeshName];

    adios2_variable *zcVar = this->ZCoordWriteVars[md->MeshName];
    std::vector<size_t> &zcStarts = this->ZCoordStarts[md->MeshName];
    std::vector<size_t> &zcCounts = this->ZCoordCounts[md->MeshName];

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

        svtkDataArray *zda = ds->GetZCoordinates();

        // write x-coords
        size_t xcStart = xcStarts[j];
        size_t xcCount = xcCounts[j];
        if (adios2_set_selection(xcVar, 1, &xcStart, &xcCount))
          {
          SENSEI_ERROR("adios2_set_selection start=" << xcStart
            << " count=" << xcCount << " block " << j << " failed")
          return -1;
          }

        svtkDataArray *xda = ds->GetXCoordinates();
        if (adios2_put(handles.engine, xcVar,
          xda->GetVoidPointer(0), adios2_mode_sync))
          {
          SENSEI_ERROR("adios2_put x-coordinates block " << j << " failed")
          return -1;
          }

        // write y-coords
        size_t ycStart = ycStarts[j];
        size_t ycCount = ycCounts[j];
        if (adios2_set_selection(ycVar, 1, &ycStart, &ycCount))
          {
          SENSEI_ERROR("adios2_set_selection start=" << ycStart
            << " count=" << ycCount << " block " << j << " failed")
          return -1;
          }

        svtkDataArray *yda = ds->GetYCoordinates();
        if (adios2_put(handles.engine, ycVar,
          yda->GetVoidPointer(0), adios2_mode_sync))
          {
          SENSEI_ERROR("adios2_put y-coordinates block " << j << " failed")
          return -1;
          }

        // write z-coords
        size_t zcStart = zcStarts[j];
        size_t zcCount = zcCounts[j];
        if (adios2_set_selection(zcVar, 1, &zcStart, &zcCount))
          {
          SENSEI_ERROR("adios2_set_selection start=" << zcStart
            << " count=" << zcCount << " block " << j << " failed")
          return -1;
          }

        if (adios2_put(handles.engine, zcVar,
          zda->GetVoidPointer(0), adios2_mode_sync))
          {
          SENSEI_ERROR("adios2_put y-coordinates block " << j << " failed")
          return -1;
          }

        numBytes += (xcCount + ycCount +zcCount)*sensei::SVTKUtils::Size(xda->GetDataType());
        }

      it->GoToNextItem();
      }

    it->Delete();

    sensei::Profiler::EndEvent("senseiADIOS2::StretchedCartesianSchema::Write",
      numBytes);
    }

  return 0;
}

// --------------------------------------------------------------------------
int StretchedCartesianSchema::Read(MPI_Comm comm, AdiosHandle handles,
  const std::string &ons, const sensei::MeshMetadataPtr &md,
  svtkCompositeDataSet *dobj)
{
  if (sensei::SVTKUtils::StretchedCartesian(md))
    {
    sensei::Profiler::StartEvent("senseiADIOS2::StretchedCartesianSchema::Read");
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
        std::string xc_path = ons + "x_coords";
        adios2_variable *xc_vinfo = adios2_inquire_variable(handles.io, xc_path.c_str());
        if (!xc_vinfo)
          {
          SENSEI_ERROR("adios2_inquire_variable \"" << xc_path
            << "\" block " << j <<  " failed")
          return -1;
          }

        // /data_object_<id>/data_array_<id>/x_coords
        size_t x_start = xc_offset;
        size_t x_count = nx_local;
        if (adios2_set_selection(xc_vinfo, 1, &x_start, &x_count))
          {
          SENSEI_ERROR("adios2_set_selection block " << j << " start=" << x_start
            << " count=" << x_count << " failed")
          return -1;
          }

        svtkDataArray *x_coords = svtkDataArray::CreateDataArray(md->CoordinateType);
        x_coords->SetNumberOfComponents(1);
        x_coords->SetNumberOfTuples(nx_local);
        x_coords->SetName("x_coords");

        if (adios2_get(handles.engine, xc_vinfo,
          x_coords->GetVoidPointer(0), adios2_mode_sync))
          {
          SENSEI_ERROR("adios2_get x_coords block " << j << " failed")
          return -1;
          }

        std::string yc_path = ons + "y_coords";
        adios2_variable *yc_vinfo = adios2_inquire_variable(handles.io, yc_path.c_str());
        if (!yc_vinfo)
          {
          SENSEI_ERROR("adios2_inquire_variable \"" << yc_path
            << "\" block " << j <<  " failed")
          return -1;
          }

        // /data_object_<id>/data_array_<id>/y_coords
        size_t y_start = yc_offset;
        size_t y_count = ny_local;
        if (adios2_set_selection(yc_vinfo, 1, &y_start, &y_count))
          {
          SENSEI_ERROR("adios2_set_selection block " << j << " start=" << y_start
            << " count=" << y_count << " failed")
          return -1;
          }

        svtkDataArray *y_coords = svtkDataArray::CreateDataArray(md->CoordinateType);
        y_coords->SetNumberOfComponents(1);
        y_coords->SetNumberOfTuples(ny_local);
        y_coords->SetName("y_coords");

        if (adios2_get(handles.engine, yc_vinfo,
          y_coords->GetVoidPointer(0), adios2_mode_sync))
          {
          SENSEI_ERROR("adios2_get y_coords block " << j << " failed")
          return -1;
          }

        std::string zc_path = ons + "z_coords";
        adios2_variable *zc_vinfo = adios2_inquire_variable(handles.io, zc_path.c_str());
        if (!zc_vinfo)
          {
          SENSEI_ERROR("adios2_inquire_variable \"" << zc_path
            << "\" block " << j <<  " failed")
          return -1;
          }

        // /data_object_<id>/data_array_<id>/z_coords
        size_t z_start = zc_offset;
        size_t z_count = nz_local;
        if (adios2_set_selection(zc_vinfo, 1, &z_start, &z_count))
          {
          SENSEI_ERROR("adios2_set_selection block " << j << " start=" << z_start
            << " count=" << z_count << " failed")
          return -1;
          }

        svtkDataArray *z_coords = svtkDataArray::CreateDataArray(md->CoordinateType);
        z_coords->SetNumberOfComponents(1);
        z_coords->SetNumberOfTuples(nz_local);
        z_coords->SetName("z_coords");

        if (adios2_get(handles.engine, zc_vinfo,
          z_coords->GetVoidPointer(0), adios2_mode_sync))
          {
          SENSEI_ERROR("adios2_get z_coords block " << j << " failed")
          return -1;
          }

        if (adios2_perform_gets(handles.engine))
          {
          SENSEI_ERROR("Failed to read stretched Cartesian block " << j)
          return -1;
          }

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

    sensei::Profiler::EndEvent("senseiADIOS2::StretchedCartesianSchema::Read", numBytes);
    }

  return 0;
}



struct DataObjectSchema
{
  int DefineVariables(MPI_Comm comm, AdiosHandle handles,
    unsigned int doid,  const sensei::MeshMetadataPtr &md);

  int Write(MPI_Comm comm, AdiosHandle handles, unsigned int doid,
    const sensei::MeshMetadataPtr &md, svtkCompositeDataSet *dobj);

  int ReadMesh(MPI_Comm comm, AdiosHandle handles,
    unsigned int doid, const sensei::MeshMetadataPtr &md,
    svtkCompositeDataSet *&dobj, bool structure_only);

  int ReadArray(MPI_Comm comm, AdiosHandle handles,
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
int DataObjectSchema::DefineVariables(MPI_Comm comm, AdiosHandle handles,
  unsigned int doid, const sensei::MeshMetadataPtr &md)
{
  sensei::TimeEvent<128> mark(
    "senseiADIOS2::DataObjectSchema::DefineVariables");

  // put each data object in its own namespace
  std::ostringstream ons;
  ons << "data_object_" << doid << "/";

  if (this->DataArrays.DefineVariables(comm, handles, ons.str(), md) ||
    this->Points.DefineVariables(comm, handles, ons.str(), md) ||
    this->UnstructuredCells.DefineVariables(comm, handles, ons.str(), md) ||
    this->PolydataCells.DefineVariables(comm, handles, ons.str(), md) ||
    this->UniformCartesian.DefineVariables(comm, handles, ons.str(), md) ||
    this->StretchedCartesian.DefineVariables(comm, handles, ons.str(), md) ||
    this->LogicallyCartesian.DefineVariables(comm, handles, ons.str(), md))
    {
    SENSEI_ERROR("Failed to define variables for object "
      << doid << " \"" << md->MeshName << "\"")
    return -1;
    }

  return 0;
}

// --------------------------------------------------------------------------
int DataObjectSchema::Write(MPI_Comm comm, AdiosHandle handles, unsigned int doid,
  const sensei::MeshMetadataPtr &md, svtkCompositeDataSet *dobj)
{
  sensei::TimeEvent<128> mark("senseiADIOS2::DataObjectSchema::Write");

  if (this->DataArrays.Write(comm, handles, md, dobj) ||
    this->Points.Write(comm, handles, md, dobj) ||
    this->UnstructuredCells.Write(comm, handles, md, dobj) ||
    this->PolydataCells.Write(comm, handles, md, dobj) ||
    this->UniformCartesian.Write(comm, handles, md, dobj) ||
    this->StretchedCartesian.Write(comm, handles, md, dobj) ||
    this->LogicallyCartesian.Write(comm, handles, md, dobj))
    {
    SENSEI_ERROR("Failed to write for object "
      << doid << " \"" << md->MeshName << "\"")
    return -1;
    }

  return 0;
}

// --------------------------------------------------------------------------
int DataObjectSchema::ReadMesh(MPI_Comm comm, AdiosHandle handles,
  unsigned int doid, const sensei::MeshMetadataPtr &md,
  svtkCompositeDataSet *&dobj, bool structure_only)
{
  sensei::TimeEvent<128> mark(
    "senseiADIOS2::DataObjectSchema::ReadMesh");

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
    (this->Points.Read(comm, handles, ons.str(), md, dobj) ||
    this->UnstructuredCells.Read(comm, handles, ons.str(), md, dobj) ||
    this->PolydataCells.Read(comm, handles, ons.str(), md, dobj))) ||
    this->UniformCartesian.Read(comm, handles, ons.str(), md, dobj) ||
    this->StretchedCartesian.Read(comm, handles, ons.str(), md, dobj) ||
    this->LogicallyCartesian.Read(comm, handles, ons.str(), md, dobj))
    {
    SENSEI_ERROR("Failed to define variables for object "
      << doid << " \"" << md->MeshName << "\"")
    return -1;
    }

  return 0;
}

// --------------------------------------------------------------------------
int DataObjectSchema::ReadArray(MPI_Comm comm, AdiosHandle handles,
  unsigned int doid, const std::string &name, int association,
  const sensei::MeshMetadataPtr &md, svtkCompositeDataSet *dobj)
{
  sensei::TimeEvent<128> mark(
    "senseiADIOS2::DataObjectSchema::ReadArray");

  std::ostringstream ons;
  ons << "data_object_" << doid << "/";

  if (this->DataArrays.Read(comm, handles, ons.str(), name, association, md, dobj))
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
  sensei::TimeEvent<128>("DataObjectSchema::InitializeDataObject");

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
      svtkDataObject *ds = newDataObject(md->BlockType);
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
  sensei::TimeEvent<128> mark(
    "senseiADIOS2::DataObjectCollectionSchema::ReadMeshMetadata");

  this->Internals->SenderMdMap.Clear();
  this->Internals->ReceiverMdMap.Clear();

  // /number_of_data_objects
  unsigned int n_objects = 0;
  if (adiosInq(iStream, "number_of_data_objects", n_objects))
    return -1;

   // read the sender mesh metadta
  for (unsigned int i = 0; i < n_objects; ++i)
    {
    std::ostringstream oss;
    oss << "data_object_" << i << "/";
    std::string data_object_id = oss.str();

    // /data_object_<id>/metadata
    sensei::BinaryStream bs;
    std::string path = data_object_id + "metadata";
    if (BinaryStreamSchema::Read(comm, iStream, path, bs))
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

  return 0;
}

// --------------------------------------------------------------------------
int DataObjectCollectionSchema::GetSenderMeshMetadata(unsigned int id,
  sensei::MeshMetadataPtr &md)
{
  sensei::TimeEvent<128> mark(
    "senseiADIOS2::DataObjectCollectionSchema::GetSenderMeshMetadata");

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
  sensei::TimeEvent<128>("DataObjectCollectionSchema::SetReceiverMeshMetadata");
  return this->Internals->ReceiverMdMap.SetMeshMetadata(id, md);
}


// --------------------------------------------------------------------------
int DataObjectCollectionSchema::GetReceiverMeshMetadata(unsigned int id,
  sensei::MeshMetadataPtr &md)
{
  sensei::TimeEvent<128>("DataObjectCollectionSchema::GetReceiverMeshMetadata");
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
  sensei::TimeEvent<128>("DataObjectCollectionSchema::GetNumberOfObjects");
  num = this->Internals->SenderMdMap.Size();
  return 0;
}

// --------------------------------------------------------------------------
int DataObjectCollectionSchema::GetObjectId(MPI_Comm comm,
  const std::string &object_name, unsigned int &doid)
{
  sensei::TimeEvent<128>("DataObjectCollectionSchema::GetObjectId");

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
int DataObjectCollectionSchema::DefineVariables(MPI_Comm comm, AdiosHandle handles,
  const std::vector<sensei::MeshMetadataPtr> &metadata)
{
  sensei::TimeEvent<128>("DataObjectCollectionSchema::DefineVariables");

  // mark the file as ours and declare version it is written with
  this->Internals->Version.DefineVariables(handles);

  // /time_step
  if (!adios2_define_variable(handles.io, "time_step",
    adios2_type_uint64_t, 0, NULL, NULL, NULL, adios2_constant_dims_true))
    {
    SENSEI_ERROR("adios2_define_variable time_step failed")
    return -1;
    }

  // /time
  if (!adios2_define_variable(handles.io, "time",
    adios2_type_double, 0, NULL, NULL, NULL, adios2_constant_dims_true))
    {
    SENSEI_ERROR("adios2_define_variable time failed")
    return -1;
    }

  // /number_of_data_objects
  unsigned int n_objects = metadata.size();
  if (!adios2_define_variable(handles.io, "number_of_data_objects",
    adios2_type_int32_t, 0, NULL, NULL, NULL, adios2_constant_dims_true))
    {
    SENSEI_ERROR("adios2_define_variable number_of_data_objects")
    return -1;
    }

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
    BinaryStreamSchema::DefineVariables(handles, object_id + "metadata");

    if (this->Internals->DataObject.DefineVariables(comm, handles, i, metadata[i]))
      {
      SENSEI_ERROR("Failed to define variables for object "
        << i << " " << metadata[i]->MeshName)
      return -1;
      }
    }

  return 0;
}

// --------------------------------------------------------------------------
int DataObjectCollectionSchema::Write(MPI_Comm comm, AdiosHandle handles,
  unsigned long time_step, double time,
  const std::vector<sensei::MeshMetadataPtr> &metadata,
  const std::vector<svtkCompositeDataSetPtr> &objects)
{
  sensei::Profiler::StartEvent("senseiADIOS2::DataObjectCollectionSchema::Write");

  unsigned int n_objects = objects.size();
  if (n_objects != metadata.size())
    {
    SENSEI_ERROR("Missing metadata for some objects. "
      << n_objects << " data objects and " << metadata.size()
      << " metadata")
    return -1;
    }

  // write the schema version
  if (this->Internals->Version.Write(handles))
    {
    SENSEI_ERROR("Failed to write schema version")
    return -1;
    }

  // /time_step
  if (adios2_put_by_name(handles.engine, "time_step", &time_step, adios2_mode_sync))
    {
    SENSEI_ERROR("adios_put_by_name time_step failed")
    return -1;
    }

  // /time
  if (adios2_put_by_name(handles.engine, "time", &time, adios2_mode_sync))
    {
    SENSEI_ERROR("adios_put_by_name time failed")
    return -1;
    }

  // /number_of_data_objects
  std::string path = "number_of_data_objects";
  if (adios2_put_by_name(handles.engine, path.c_str(), &n_objects, adios2_mode_sync))
    {
    SENSEI_ERROR("adios_put_by_name number_of_data_objects failed")
    return -1;
    }

  for (unsigned int i = 0; i < n_objects; ++i)
    {
    sensei::BinaryStream bs;
    metadata[i]->ToStream(bs);

    std::ostringstream oss;
    oss << "data_object_" << i << "/";
    std::string object_id = oss.str();

    // /data_object_<id>/metadata
    path = object_id + "metadata";
    if (BinaryStreamSchema::Write(handles, path, bs))
      {
      SENSEI_ERROR("Failed to write metadata for object " << i)
      return -1;
      }

    // write the object
    if (this->Internals->DataObject.Write(comm, handles, i,
      metadata[i], objects[i].Get()))
      {
      SENSEI_ERROR("Failed to write object " << i << " \""
        << metadata[i]->MeshName << "\"")
      return -1;
      }
    }

  sensei::Profiler::EndEvent("senseiADIOS2::DataObjectCollectionSchema::Write",
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
  sensei::TimeEvent<128> mark(
    "senseiADIOS2::DataObjectCollectionSchema::ReadObject");

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
    iStream.Handles, doid, md, cd, structure_only))
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
  sensei::TimeEvent<128> mark(
    "senseiADIOS2::DataObjectCollectionSchema::ReadArray");

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
    iStream.Handles, doid, array_name, association, md, cds))
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

  sensei::TimeEvent<128> mark(
    "senseiADIOS2::DataObjectCollectionSchema::ReadTimeStep");

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
  sensei::TimeEvent<128> mark(
    "senseiADIOS2::DataObjectCollectionSchema::AddBlockOwnerArray");

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
