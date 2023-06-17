#include "DataBinning.h"
#include "DataAdaptor.h"
#include "MeshMetadata.h"
#include "MeshMetadataMap.h"
#include "Profiler.h"
#include "SVTKUtils.h"
#include "XMLUtils.h"
#include "MPIUtils.h"
#include "STLUtils.h"
#include "SVTKDataAdaptor.h"
#include "Error.h"

using namespace sensei::STLUtils;

#include <pugixml.hpp>

#include <svtkObjectFactory.h>
#include <svtkDataObject.h>
#include <svtkImageData.h>
#include <svtkCompositeDataSet.h>
#include <svtkMultiBlockDataSet.h>
#include <svtkDataSetAttributes.h>
#include <svtkTable.h>
#include <svtkPointData.h>
#include <svtkCompositeDataIterator.h>
#include <svtkSmartPointer.h>
#include <svtkUnsignedCharArray.h>
#include <svtkHAMRDataArray.h>

#if defined(SENSEI_ENABLE_CUDA)
#include "CUDAUtils.h"
#include <thrust/sequence.h>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#endif

#include <algorithm>
#include <vector>
#include <sys/time.h>

namespace
{
#if defined(SENSEI_ENABLE_CUDA)
namespace CudaImpl
{

// **************************************************************************
__device__ double atomicMin(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        double oldVal = __longlong_as_double(old);
        double newVal = val < oldVal ? val : oldVal;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(newVal));
    } while (assumed != old);
    return __longlong_as_double(old);
}

// **************************************************************************
__device__ double atomicMax(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        double oldVal = __longlong_as_double(old);
        double newVal = val > oldVal ? val : oldVal;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(newVal));
    } while (assumed != old);
    return __longlong_as_double(old);
}

// **************************************************************************
__device__ float atomicMin(float* address, float val)
{
    unsigned int* address_as_ui = (unsigned int*)address;
    unsigned int old = *address_as_ui, assumed;
    do {
        assumed = old;
        float oldVal = __uint_as_float(old);
        float newVal = val < oldVal ? val : oldVal;
        old = atomicCAS(address_as_ui, assumed, __float_as_uint(newVal));
    } while (assumed != old);
    return __uint_as_float(old);
}

// **************************************************************************
__device__ float atomicMax(float* address, float val)
{
    unsigned int* address_as_ui = (unsigned int*)address;
    unsigned int old = *address_as_ui, assumed;
    do {
        assumed = old;
        float oldVal = __uint_as_float(old);
        float newVal = val > oldVal ? val : oldVal;
        old = atomicCAS(address_as_ui, assumed, __float_as_uint(newVal));
    } while (assumed != old);
    return __uint_as_float(old);
}


/** count the number of values in each bin. The output must be
 * pre-initialized to zero multiple invokations of the kernel accumulate
 * results for new data.
 *
 * @param[in] datX    the x coordinate array
 * @param[in] datY    the y coordinate array
 * @param[in] nVals   the length of the data arrays
 * @param[in] minX    the axis minimum in the x-direction
 * @param[in] minY    the axis minimum in the y-direction
 * @param[in] dx      the grid spacing in the x-direction
 * @param[in] dy      the grid spacing in the y-direction
 * @param[in] resX    the grid resolution in the x direction.
 * @param[in] resX    the grid resolution in the y direction.
 * @param[in,out] cnt the number of values in the grid cell
 */
template <typename coord_t, typename grid_t>
__global__
void count(const coord_t *datX, const coord_t *datY, long nVals,
  grid_t minX, grid_t minY, grid_t dx, grid_t dy, long resX, long resY,
  long *cnt)
{
  unsigned long q = sensei::CUDAUtils::ThreadIdToArrayIndex();

  if (q >= nVals)
    return;

  // look up the coordinates
  coord_t x = datX[q];
  coord_t y = datY[q];

  // calculate the cell the point falls into. the logic here handles a point
  // that is out of bounds by adding its contribution to the first or last bin.
  long i = 0;
  if (minX < x)
    i = (x - minX) / dx;
  if (i >= resX)
    i = resX - 1;

  long j = 0;
  if (minY < y)
    j = (y - minY) / dy;
  if (j >= resY)
    j = resY - 1;

  // the array index of the cell
  size_t qq = j * resX + i;

  // update the bin count
  atomicAdd((unsigned long long*)&(cnt[qq]), 1l);
}

/** launch the count kernel */
template <typename coord_t, typename grid_t>
int blockCount(cudaStream_t strm,
  const coord_t *datX, const coord_t *datY, long nVals,
  grid_t minX, grid_t minY, grid_t dx, grid_t dy,
  long resX, long resY, long *cnt)
{
  // determine kernel launch parameters
  dim3 blockGrid;
  int nBlocks = 0;
  dim3 threadGrid;
  if (sensei::CUDAUtils::PartitionThreadBlocks(0,
   nVals, 8, blockGrid, nBlocks, threadGrid))
  {
    SENSEI_ERROR("Failed to partition thread blocks")
    return -1;
  }

  // compute the histgram for this block's worth of data on the GPU. It is
  // left on the GPU until data for all blocks has been processed.
  count<<<blockGrid, threadGrid, 0, strm>>>(datX, datY,
    nVals, minX, minY, dx, dy, resX, resY, cnt);

  cudaError_t ierr = cudaGetLastError();
  if (ierr != cudaSuccess)
  {
    SENSEI_ERROR("Failed to launch the kernel." << cudaGetErrorString(ierr))
    return -1;
  }

  return 0;
}


/// update the sum
template <typename T>
struct SumOp {
  static constexpr T initial_value() { return T(0); }
  __device__ void operator()(T *dest, T src) const { atomicAdd(dest, src); }
};

/// update the min
template <typename T>
struct MinOp {
  static constexpr T initial_value() { return std::numeric_limits<T>::max(); }
  __device__ void operator()(T *dest, T src) const { atomicMin(dest, src); }
};

/// update the max
template <typename T>
struct MaxOp {
  static constexpr T initial_value() { return std::numeric_limits<T>::lowest(); }
  __device__ void operator()(T *dest, T src) const { atomicMax(dest, src); }
};

/** bin the values in each grid cell using the specified operator. The output
 * must be pre-initialized multiple invokations of the kernel accumulate
 * results for new data.
 *
 * @param[in] datX    the x coordinate array
 * @param[in] datY    the y coordinate array
 * @param[in] datF    the array to sum
 * @param[in] nVals   the length of the data arrays
 * @param[in] minX    the axis minimum in the x-direction
 * @param[in] minY    the axis minimum in the y-direction
 * @param[in] dx      the grid spacing in the x-direction
 * @param[in] dy      the grid spacing in the y-direction
 * @param[in] resX    the grid resolution in the x direction.
 * @param[in] resX    the grid resolution in the y direction.
 * @param[in,out] binF the sum of values in the grid cell
 */
template <typename coord_t, typename grid_t, typename array_t, typename op_t>
__global__
void bin(const coord_t *datX, const coord_t *datY, const array_t *datF,
  const long nVals, grid_t minX, grid_t minY, grid_t dx, grid_t dy,
  long resX, long resY, const op_t &binOp, array_t *binF)
{
  unsigned long q = sensei::CUDAUtils::ThreadIdToArrayIndex();

  if (q >= nVals)
    return;

  // look up the coordinates
  coord_t x = datX[q];
  coord_t y = datY[q];

  // calculate the cell the point falls into. the logic here handles a point
  // that is out of bounds by adding its contribution to the first or last bin.
  long i = 0;
  if (minX < x)
    i = (x - minX) / dx;
  if (i >= resX)
    i = resX - 1;

  long j = 0;
  if (minY < y)
    j = (y - minY) / dy;
  if (j >= resY)
    j = resY - 1;

  // the array index of the cell
  size_t qq = j * resX + i;

  // update the bin total
  binOp(&binF[qq], datF[q]);
}

/** launch the bin kernel */
template <typename coord_t, typename grid_t, typename array_t,
  template<typename> typename op_t>
int blockBin(cudaStream_t strm,
  const coord_t *datX, const coord_t *datY, const array_t *datF,
  const long nVals, grid_t minX, grid_t minY, grid_t dx, grid_t dy,
  long resX, long resY, const op_t<array_t> &binOp, array_t *binF)
{
  // determine kernel launch parameters
  dim3 blockGrid;
  int nBlocks = 0;
  dim3 threadGrid;
  if (sensei::CUDAUtils::PartitionThreadBlocks(0,
   nVals, 8, blockGrid, nBlocks, threadGrid))
  {
    SENSEI_ERROR("Failed to partition thread blocks")
    return -1;
  }

  // compute the histgram for this block's worth of data on the GPU. It is
  // left on the GPU until data for all blocks has been processed.
  bin<<<blockGrid, threadGrid, 0, strm>>>(datX, datY, datF,
    nVals, minX, minY, dx, dy, resX, resY, binOp, binF);

  cudaError_t ierr = cudaGetLastError();
  if (ierr != cudaSuccess)
  {
    SENSEI_ERROR("Failed to launch the kernel." << cudaGetErrorString(ierr))
    return -1;
  }

  return 0;
}


// **************************************************************************
template <typename array_t>
__global__
void maskGreater(array_t *datF, long nVals, array_t threshold, array_t value)
{
  unsigned long q = sensei::CUDAUtils::ThreadIdToArrayIndex();

  if (q >= nVals)
    return;

  datF[q] = datF[q] > threshold ? value : datF[q];
}

// **************************************************************************
template <typename array_t>
__global__
void maskLess(array_t *datF, long nVals, array_t threshold, array_t value)
{
  unsigned long q = sensei::CUDAUtils::ThreadIdToArrayIndex();

  if (q >= nVals)
    return;

  datF[q] = datF[q] < threshold ? value : datF[q];
}

// **************************************************************************
template <typename array_t, typename scale_t>
__global__
void scaleElement(array_t *datF, const scale_t * scale, long nVals)
{
  unsigned long q = sensei::CUDAUtils::ThreadIdToArrayIndex();

  if (q >= nVals)
    return;

  datF[q] = scale[q] ? datF[q] / ((array_t)scale[q]) : datF[q];
}


// **************************************************************************
template <typename array_t>
int maskGreater(cudaStream_t strm, array_t *datF, long nVals,
                array_t threshold, array_t value)
{
  // determine kernel launch parameters
  dim3 blockGrid;
  int nBlocks = 0;
  dim3 threadGrid;
  if (sensei::CUDAUtils::PartitionThreadBlocks(0,
   nVals, 8, blockGrid, nBlocks, threadGrid))
  {
    SENSEI_ERROR("Failed to partition thread blocks")
    return -1;
  }

  // launch the kernel
  maskGreater<<<blockGrid, threadGrid, 0, strm>>>(datF, nVals, threshold, value);

  cudaError_t ierr = cudaGetLastError();
  if (ierr != cudaSuccess)
  {
    SENSEI_ERROR("Failed to launch the kernel." << cudaGetErrorString(ierr))
    return -1;
  }

  return 0;
}

// **************************************************************************
template <typename array_t>
int maskLess(cudaStream_t strm, array_t *datF, long nVals,
             array_t threshold, array_t value)
{
  // determine kernel launch parameters
  dim3 blockGrid;
  int nBlocks = 0;
  dim3 threadGrid;
  if (sensei::CUDAUtils::PartitionThreadBlocks(0,
   nVals, 8, blockGrid, nBlocks, threadGrid))
  {
    SENSEI_ERROR("Failed to partition thread blocks")
    return -1;
  }

  // launch the kernel
  maskLess<<<blockGrid, threadGrid, 0, strm>>>(datF, nVals, threshold, value);

  cudaError_t ierr = cudaGetLastError();
  if (ierr != cudaSuccess)
  {
    SENSEI_ERROR("Failed to launch the kernel." << cudaGetErrorString(ierr))
    return -1;
  }

  return 0;
}

// **************************************************************************
template <typename array_t, typename scale_t>
int scaleElement(cudaStream_t strm, array_t *datF, const scale_t * scale,
                 long nVals)
{
  // determine kernel launch parameters
  dim3 blockGrid;
  int nBlocks = 0;
  dim3 threadGrid;
  if (sensei::CUDAUtils::PartitionThreadBlocks(0,
   nVals, 8, blockGrid, nBlocks, threadGrid))
  {
    SENSEI_ERROR("Failed to partition thread blocks")
    return -1;
  }

  // launch the kernel
  scaleElement<<<blockGrid, threadGrid, 0, strm>>>(datF, scale, nVals);

  cudaError_t ierr = cudaGetLastError();
  if (ierr != cudaSuccess)
  {
    SENSEI_ERROR("Failed to launch the kernel." << cudaGetErrorString(ierr))
    return -1;
  }

  return 0;
}
}
#endif

namespace HostImpl
{
/** count the number of values in each bin. The output must be
 * pre-initialized to zero multiple invokations of the kernel accumulate
 * results for new data.
 *
 * @param[in] datX    the x coordinate array
 * @param[in] datY    the y coordinate array
 * @param[in] nVals   the length of the data arrays
 * @param[in] minX    the axis minimum in the x-direction
 * @param[in] minY    the axis minimum in the y-direction
 * @param[in] dx      the grid spacing in the x-direction
 * @param[in] dy      the grid spacing in the y-direction
 * @param[in] resX    the grid resolution in the x direction.
 * @param[in] resX    the grid resolution in the y direction.
 * @param[in,out] cnt the number of values in the grid cell
 */
template <typename coord_t, typename grid_t>
int blockCount(const coord_t *datX, const coord_t *datY, long nVals,
  grid_t minX, grid_t minY, grid_t dx, grid_t dy, long resX, long resY,
  long *cnt)
{
  for (long q = 0; q < nVals; ++q)
  {
    // look up the coordinates
    coord_t x = datX[q];
    coord_t y = datY[q];

    // calculate the cell the point falls into. the logic here handles a point
    // that is out of bounds by adding its contribution to the first or last bin.
    long i = 0;
    if (minX < x)
      i = (x - minX) / dx;
    if (i >= resX)
      i = resX - 1;

    long j = 0;
    if (minY < y)
      j = (y - minY) / dy;
    if (j >= resY)
      j = resY - 1;

    // array index of the cell
    size_t qq = j * resX + i;

    // update the bin count
    cnt[qq] += 1;
  }
  return 0;
}


/// update the sum
template <typename T>
struct SumOp {
  static constexpr T initial_value() { return T(0); }
  void operator()(T &dest, const T & src) const { dest += src; }
};

/// update the min
template <typename T>
struct MinOp {
  static constexpr T initial_value() { return std::numeric_limits<T>::max(); }
  void operator()(T &dest, const T & src) const { std::min(dest, src); }
};

/// update the max
template <typename T>
struct MaxOp {
  static constexpr T initial_value() { return std::numeric_limits<T>::lowest(); }
  void operator()(T &dest, const T & src) const { std::max(dest, src); }
};

// given a string naming the operation get the initial value
template <typename T>
int GetInitialValue(int op, T &value)
{
  int ret = 0;
  if (op == sensei::DataBinning::BIN_SUM)
    value = SumOp<T>::initial_value();
  if (op == sensei::DataBinning::BIN_AVG)
    value = SumOp<T>::initial_value();
  else if (op == sensei::DataBinning::BIN_MIN)
    value = MinOp<T>::initial_value();
  else if (op == sensei::DataBinning::BIN_MAX)
    value = MaxOp<T>::initial_value();
  else
    ret = -1;
  return ret;
}

// given a string naming the operation get the initial value
template <typename T>
int GetFinalValue(int op, T &value)
{
  int ret = 0;
  if ((op == sensei::DataBinning::BIN_SUM) ||
   (op == sensei::DataBinning::BIN_AVG))
    ret = 0;
  else if (op == sensei::DataBinning::BIN_MIN)
    value = MinOp<T>::initial_value() - MinOp<T>::initial_value() * 0.001;
  else if (op == sensei::DataBinning::BIN_MAX)
    value = MaxOp<T>::initial_value() + MaxOp<T>::initial_value() * 0.001;
  else
    ret = -1;
  return ret;
}

/** bin the values in each grid cell using the specified operator. The output
 * must be pre-initialized multiple invokations of the kernel accumulate
 * results for new data.
 *
 * @param[in] datX    the x coordinate array
 * @param[in] datY    the y coordinate array
 * @param[in] datF    the array to sum
 * @param[in] nVals   the length of the data arrays
 * @param[in] minX    the axis minimum in the x-direction
 * @param[in] minY    the axis minimum in the y-direction
 * @param[in] dx      the grid spacing in the x-direction
 * @param[in] dy      the grid spacing in the y-direction
 * @param[in] resX    the grid resolution in the x direction.
 * @param[in] resX    the grid resolution in the y direction.
 * @param[in] binOp   binary operator that updates the bin value
 * @param[in,out] binF the sum of values in the grid cell
 */
template <typename coord_t, typename grid_t, typename array_t,
  template<typename> typename op_t>
int blockBin(const coord_t *datX, const coord_t *datY, const array_t *datF,
  const long nVals, grid_t minX, grid_t minY, grid_t dx, grid_t dy,
  long resX, long resY, const op_t<array_t> &binOp, array_t *binF)
{
  for (long q = 0; q < nVals; ++q)
  {
    // look up the coordinates
    coord_t x = datX[q];
    coord_t y = datY[q];

    // calculate the cell the point falls into. the logic here handles a point
    // that is out of bounds by adding its contribution to the first or last bin.
    long i = 0;
    if (minX < x)
      i = (x - minX) / dx;
    if (i >= resX)
      i = resX - 1;

    long j = 0;
    if (minY < y)
      j = (y - minY) / dy;
    if (j >= resY)
      j = resY - 1;

    // array index of the cell
    size_t qq = j * resX + i;

    // update the bin total
    binOp(binF[qq], datF[q]);
  }
  return 0;
}

// **************************************************************************
template <typename array_t, typename scale_t = long>
struct ElementScale
{
  ElementScale(const scale_t *scal) : scale(scal) {}

  void operator()(unsigned long i, array_t *el) const
  { el[i] = scale[i] != scale_t(0) ? el[i] / ((array_t)scale[i]) : el[i]; }

  const scale_t *scale;
};

// **************************************************************************
template <typename array_t>
struct MaskGreater
{
  MaskGreater(array_t thresh, array_t val) : threshold(thresh), value(val) {}

  void operator()(unsigned long i, array_t *el) const
  { el[i] = el[i] > threshold ? value : el[i]; }

  array_t threshold;
  array_t value;
};

// **************************************************************************
template <typename array_t>
struct MaskLess
{
  MaskLess(array_t thresh, array_t val) : threshold(thresh), value(val) {}

  void operator()(unsigned long i, array_t *el) const
  { el[i] = el[i] < threshold ? value : el[i]; }

  array_t threshold;
  array_t value;
};

/** launch the finalize kernel */
template <typename array_t, typename op_t>
int finalize(array_t *datF, long nVals, const op_t &finOp)
{
  for (long i = 0; i < nVals; ++i)
    finOp(i, datF);
  return 0;
}
}

// **************************************************************************
svtkDataArray *GetColumn(svtkTable *tab, const std::string &colName)
{
  return dynamic_cast<svtkDataArray*>(tab->GetColumnByName(colName.c_str()));
}
}

namespace sensei
{

//-----------------------------------------------------------------------------
senseiNewMacro(DataBinning);

//-----------------------------------------------------------------------------
DataBinning::DataBinning() : XRes(128), YRes(128), OutDir("./"),
  Iteration(0), AxisFactor(0.1), MeshName(), XAxisArray(), YAxisArray(),
  BinnedArray(), Operation(), ReturnData(0)
{
}

//-----------------------------------------------------------------------------
DataBinning::~DataBinning()
{
}

//-----------------------------------------------------------------------------
int DataBinning::GetOperation(const std::string &opName, int &opCode)
{
  int ret = 0;
  if (opName == "sum")
    opCode = BIN_SUM;
  else if (opName == "avg")
    opCode = BIN_AVG;
  else if (opName == "min")
    opCode = BIN_MIN;
  else if (opName == "max")
    opCode = BIN_MAX;
  else
    ret = -1;
  return ret;
}

//-----------------------------------------------------------------------------
int DataBinning::GetOperation(int opCode, std::string &opName)
{
  int ret = 0;
  if (opCode == BIN_SUM)
    opName = "sum";
  else if (opCode == BIN_AVG)
    opName = "avg";
  else if (opCode == BIN_MIN)
    opName = "min";
  else if (opCode == BIN_MAX)
    opName = "max";
  else
    ret = -1;
  return ret;
}

//-----------------------------------------------------------------------------
int DataBinning::Initialize(pugi::xml_node node)
{
  // get the required attributes
  if (XMLUtils::RequireAttribute(node, "mesh") ||
      XMLUtils::RequireAttribute(node, "x_axis") ||
      XMLUtils::RequireAttribute(node, "y_axis"))
    return -1;

  std::string mesh = node.attribute("mesh").value();
  std::string xAxis = node.attribute("x_axis").value();
  std::string yAxis = node.attribute("y_axis").value();

  // get the optional attributes
  std::string oDir = node.attribute("out_dir").as_string("./");
  int xRes = node.attribute("x_res").as_int(128);
  int yRes = node.attribute("y_res").as_int(-1);
  int retDat = node.attribute("return_data").as_int(0);

  // get arrays to bin
  std::vector<std::string> arrays;
  XMLUtils::ParseList(node.child("arrays"), arrays);

  // and coresponding operations
  std::vector<std::string> ops;
  XMLUtils::ParseList(node.child("operations"), ops);

  return this->Initialize(mesh, xAxis, yAxis, arrays, ops,
                          xRes, yRes, oDir, retDat);
}

//-----------------------------------------------------------------------------
int DataBinning::Initialize(const std::string &meshName,
  const std::string &xAxisArray, const std::string &yAxisArray,
  const std::vector<std::string> &binnedArray,
  const std::vector<std::string> &operation,
  long xres, long yres, const std::string &outDir,
  int returnData)
{
  this->MeshName = meshName;
  this->XAxisArray = xAxisArray;
  this->YAxisArray = yAxisArray;
  this->BinnedArray = binnedArray;

  // check that an operation for each array was provided
  unsigned int nBinnedArrays = binnedArray.size();
  if (operation.size() != nBinnedArrays)
  {
    SENSEI_ERROR("There must be an operation for each binned array")
    return -1;
  }

  // convert the operation string to an enumeration
  this->Operation.resize(nBinnedArrays);
  for (unsigned int i = 0; i < nBinnedArrays; ++i)
  {
    if (this->GetOperation(operation[i], this->Operation[i]))
    {
      SENSEI_ERROR("Operation " << i << " \"" << operation[i] << "\" is invalid")
      return -1;
    }
  }

  this->XRes = xres;
  this->YRes = yres;
  this->OutDir = outDir;
  this->ReturnData = returnData;

  int rank = 0;
  MPI_Comm_rank(this->GetCommunicator(), &rank);
  if (rank == 0)
  {
    SENSEI_STATUS(<< "Configured DataBinning: MeshName=" << meshName
      << " XAxisArray=" << xAxisArray << " YAxisArray=" << yAxisArray
      << " BinnedArray={" << binnedArray << "} Operations={" << operation << "}"
      << " XRes=" << xres << " YRes=" << yres << " OutDir=" << outDir
      << "ReturnData=" << returnData)
  }

  return 0;
}

//-----------------------------------------------------------------------------
bool DataBinning::Execute(DataAdaptor* daIn, DataAdaptor** dataOut)
{
  TimeEvent<128> mark("DataBinning::Execute");

  int rank = 0;
  int n_ranks = 1;
  MPI_Comm comm = this->GetCommunicator();
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &n_ranks);

  timeval startTime{};
  if (rank == 0)
    gettimeofday(&startTime, nullptr);

  if (dataOut)
    *dataOut = nullptr;

  // see what the simulation is providing
  MeshMetadataMap mdMap;

  MeshMetadataFlags mdFlags;
  mdFlags.SetBlockBounds();

  if (mdMap.Initialize(daIn, mdFlags))
  {
    SENSEI_ERROR("Failed to get metadata")
    MPI_Abort(comm, -1);
    return false;
  }

  // get the mesh metadata object
  MeshMetadataPtr mmd;
  if (mdMap.GetMeshMetadata(this->MeshName, mmd))
  {
    SENSEI_ERROR("Failed to get metadata for mesh \"" << this->MeshName << "\"")
    MPI_Abort(comm, -1);
    return false;
  }

  // get the global coordinate axis bounds from the metadata
  if (!mmd->GlobalView)
      mmd->GlobalizeView(comm);

  // build a mapping from array name to its metadata
  std::map<std::string, int> arrayMmdId;
  for (int i = 0; i < mmd->NumArrays; ++i)
    arrayMmdId[mmd->ArrayName[i]] = i;

  // check the coordinate axis arrays have metadata
  if (!arrayMmdId.count(this->XAxisArray) || !arrayMmdId.count(this->YAxisArray))
  {
    SENSEI_ERROR("Failed to get metadata for coordinate arrays \""
      << this->XAxisArray << "\", \"" << this->YAxisArray << "\"")
    MPI_Abort(comm, -1);
    return false;
  }

  // check that the binned arrays have metadata
  int nBinnedArrays = this->BinnedArray.size();
  for (int i = 0; i < nBinnedArrays; ++i)
  {
    if (!arrayMmdId.count(this->BinnedArray[i]))
    {
      SENSEI_ERROR("Failed to get metadata for binned array \""
        << this->BinnedArray[i] << "\"")
      MPI_Abort(comm, -1);
      return false;
    }
  }

  int xAxisArrayId = arrayMmdId[this->XAxisArray];
  int yAxisArrayId = arrayMmdId[this->YAxisArray];

  // check that the coordinate arrays are not multi-component. supporting
  // multi-component data is something that could be added later if needed
  int xAxisArrayComps = mmd->ArrayComponents[xAxisArrayId];
  int yAxisArrayComps = mmd->ArrayComponents[yAxisArrayId];
  if ((xAxisArrayComps != 1) || (yAxisArrayComps != 1))
  {
    SENSEI_ERROR("Coordinate axes are required to have only one component "
      "but the x cooridnate has " << xAxisArrayComps << " and the y coordinate "
      "has " << yAxisArrayComps << " components.")
    MPI_Abort(comm, -1);
    return false;
  }

  // check that the coordinates have the same type
  int xAxisArrayType = mmd->ArrayType[xAxisArrayId];
  int yAxisArrayType = mmd->ArrayType[yAxisArrayId];
  if (xAxisArrayType != yAxisArrayType)
  {
    SENSEI_ERROR("Coordinate arrays do not have the same data type.")
    MPI_Abort(comm, -1);
    return false;
  }

  // check that arrays are floating point
  if ((xAxisArrayType != SVTK_DOUBLE) && (xAxisArrayType != SVTK_FLOAT))
  {
    SENSEI_ERROR("Coordinate arrays are required to be floating point")
    MPI_Abort(comm, -1);
    return false;
  }

  // get the coordinate axis range
  auto [minX, maxX] = mmd->ArrayRange[xAxisArrayId];
  auto [minY, maxY] = mmd->ArrayRange[yAxisArrayId];

  // compute the grid spacing
  long xRes = this->XRes;
  long yRes = this->YRes;

  double dx = 0.;
  double dy = 0.;

  dx = maxX - minX;
  dy = maxY - minY;

  // setting either x or y res negative tell us to make a grid of square cells
  // using the non-negative res
  if ((xRes < 0) && (yRes < 0))
      xRes = 128;

  if (xRes < 0)
      xRes = yRes * dx / dy;

  if (yRes < 0)
      yRes = xRes * dy / dx;

  dx /= xRes;
  dy /= yRes;

  long xyRes = xRes * yRes;

  // fetch the mesh object from the simulation
  svtkDataObject *dobj = nullptr;
  if (daIn->GetMesh(this->MeshName, true, dobj))
  {
    SENSEI_ERROR("Failed to get mesh \"" << this->MeshName << "\"")
    MPI_Abort(comm, -1);
    return false;
  }

  if (!dobj)
  {
    SENSEI_ERROR("DataBinning requires all ranks to have data")
    MPI_Abort(comm, -1);
    return false;
  }

  // this lets one load balance across multiple GPU's and CPU's
  // set -1 to execute on the CPU and 0 to N_CUDA_DEVICES -1 to specify
  // the specific GPU to run on.
#if defined(SENSEI_ENABLE_CUDA)
  int deviceId = 0;
  const char *aDevId = getenv("SENSEI_DEVICE_ID");
  if (aDevId)
    deviceId = atoi(aDevId);
#else
  int deviceId = -1;
#endif

  // get the current time and step
  int step = daIn->GetDataTimeStep();
  double time = daIn->GetDataTime();

  // fetch the cooridnate axes from the simulation
  int xAxisArrayCen = mmd->ArrayCentering[xAxisArrayId];
  int yAxisArrayCen = mmd->ArrayCentering[yAxisArrayId];
  if (daIn->AddArray(dobj, this->MeshName, xAxisArrayCen, this->XAxisArray) ||
    daIn->AddArray(dobj, this->MeshName, yAxisArrayCen, this->YAxisArray))
  {
    SENSEI_ERROR(<< daIn->GetClassName()
      << " failed to fetch the coordinate axis arrays \""
      << this->XAxisArray << "\", \"" << this->YAxisArray << "\"" )

    MPI_Abort(comm, -1);
    return false;
  }

  // determine the allocator and stream to use
  svtkAllocator alloc = svtkAllocator::malloc;
  auto smode = svtkStreamMode::async;
  int nStream = 4;
#if defined(SENSEI_ENABLE_CUDA)
  // allocate some streams for data movement and computation
  std::vector<cudaStream_t> calcStr(nStream);
  for (int i = 0; i < nStream; ++i)
    cudaStreamCreate(&calcStr[i]);

  // if we are assigned to a specific GPU make it active and use a GPU
  // allocator
  if (deviceId >= 0)
  {
    alloc = svtkAllocator::cuda;
    sensei::CUDAUtils::SetDevice(deviceId);
  }
#else
  std::vector<svtkStream> calcStr(nStream);
#endif

  // allocate the count result array
  auto countDa = svtkHAMRLongArray::New("count", xyRes, 1,
                                        alloc, calcStr[0], smode, 0);

  // fetch the arrays to bin from the simulation and allocate the binned result arrays
  std::vector<svtkDataArray*> binnedDa(nBinnedArrays);
  for (int i = 0; i < nBinnedArrays; ++i)
  {
    const std::string &arrayName = this->BinnedArray[i];
    int opId = this->Operation[i];

    int arrayId = arrayMmdId[arrayName];
    int arrayComp = mmd->ArrayComponents[arrayId];
    int arrayCen = mmd->ArrayCentering[arrayId];
    int arrayType = mmd->ArrayType[arrayId];

    // check that arrays are floating point
    if ((arrayType != SVTK_DOUBLE) && (arrayType != SVTK_FLOAT))
    {
      SENSEI_ERROR("Binned arrays are required to be floating point")
      MPI_Abort(comm, -1);
      return false;
    }

    // check that the binned arrays are not multi-component. supporting
    // multi-component data is something that could be added if needed
    if ((arrayComp != 1))
    {
      SENSEI_ERROR("Binned arrays are required to have only one component. \""
        << arrayName << "\" has " << arrayComp << " components")
      MPI_Abort(comm, -1);
      return false;
    }

    // fetch the data from the simulation
    if (daIn->AddArray(dobj, this->MeshName, arrayCen, arrayName))
    {
      SENSEI_ERROR(<< daIn->GetClassName()
        << " failed to fetch the " << i << "th array to bin  \""
        << this->BinnedArray[i] << "\"")

      MPI_Abort(comm, -1);
      return false;
    }

    // allocate the result array
    switch (mmd->ArrayType[arrayId])
    {
      svtkTemplateMacroFloat(

        using elem_t = SVTK_TT;
        using array_t = svtkHAMRDataArray<elem_t>;

        elem_t iValue{};
        HostImpl::GetInitialValue(opId, iValue);

        std::string opName;
        GetOperation(opId, opName);

        binnedDa[i] = array_t::New(arrayName + "_" + opName, xyRes, 1, alloc,
                                   calcStr[(i+1)%nStream], smode, iValue);
      );
    }
  }

  // for temporaries if they are needed
  std::vector<svtkDataArray*> deleteDa;

  // process the blocks of data
  svtkCompositeDataSetPtr mesh = SVTKUtils::AsCompositeData(comm, dobj, true);
  svtkSmartPointer<svtkCompositeDataIterator> iter;
  iter.TakeReference(mesh->NewIterator());
  for (iter->InitTraversal(); !iter->IsDoneWithTraversal(); iter->GoToNextItem())
  {
    // get the next local block of data.
    auto curObj = iter->GetCurrentDataObject();

    // downcast to the block type. for simplicity of this example we require
    // tabular data.
    auto tab = dynamic_cast<svtkTable*>(curObj);
    if (!tab)
    {
      SENSEI_ERROR("Unsupported dataset type "
        << (curObj ? curObj->GetClassName() : "nullptr"))
      MPI_Abort(comm, -1);
      return false;
    }


    switch (xAxisArrayType)
    {
    svtkNestedTemplateMacroFloat(_COORDS,

      using coord_t = SVTK_TT_COORDS;
      using coord_array_t = svtkHAMRDataArray<coord_t>;

      // get the x-coordinate arrays.
      auto xCol = GetColumn(tab, this->XAxisArray);
      if (!xCol)
      {
        SENSEI_ERROR("Failed to get column \"" << this->XAxisArray << "\" from table")
        MPI_Abort(comm, -1);
        return false;
      }

      // handle the case when the simulation gives us vtkAOSDataArrayTemplate
      // instances by making a deep copy.
      auto xCoord = dynamic_cast<coord_array_t*>(xCol);
      if (!xCoord)
      {
        xCoord = coord_array_t::New(xCol, alloc, calcStr[1], smode);
        deleteDa.push_back(xCoord);
      }

      // get the y-coordinate arrays.
      auto yCol = GetColumn(tab, this->YAxisArray);
      if (!yCol)
      {
        SENSEI_ERROR("Failed to get column \"" << this->YAxisArray << "\" from table")
        MPI_Abort(comm, -1);
        return false;
      }

      // handle the case when the simulation gives us vtkAOSDataArrayTemplate
      // instances by making a deep copy.
      auto yCoord = dynamic_cast<coord_array_t*>(yCol);
      if (!yCoord)
      {
        yCoord = coord_array_t::New(yCol, alloc, calcStr[2], smode);
        deleteDa.push_back(yCoord);
      }

      // compute the count
      std::shared_ptr<const coord_t> spXCoord;
      std::shared_ptr<const coord_t> spYCoord;

      long nVals = xCoord->GetNumberOfTuples();
#if defined(SENSEI_ENABLE_CUDA)
      if (deviceId >= 0)
      {
        // make sure the data is on the active GPU
        spXCoord = xCoord->GetDeviceAccessible();
        spYCoord = yCoord->GetDeviceAccessible();

        // sync here so we know that incoming data is ready
        xCoord->Synchronize();
        yCoord->Synchronize();

        // compute the block's contribution
        if (::CudaImpl::blockCount(calcStr[0], spXCoord.get(), spYCoord.get(),
          nVals, minX, minY, dx, dy, xRes, yRes, countDa->GetData()))
        {
          SENSEI_ERROR("Failed to compute the count on the GPU")
          MPI_Abort(comm, -1);
          return false;
        }
      }
      else
      {
#endif
        // make sure the data is on the host
        spXCoord = xCoord->GetHostAccessible();
        spYCoord = yCoord->GetHostAccessible();

        // sync here so we know that incoming data is ready
        xCoord->Synchronize();
        yCoord->Synchronize();

        // compute the projections on the host
        if (::HostImpl::blockCount(spXCoord.get(), spYCoord.get(),
          nVals, minX, minY, dx, dy, xRes, yRes, countDa->GetData()))
        {
          SENSEI_ERROR("Failed to compute the count on the host")
          MPI_Abort(comm, -1);
          return false;
        }
#if defined(SENSEI_ENABLE_CUDA)
      }
#endif

      // process each array to bin
      for (int i = 0; i < nBinnedArrays; ++i)
      {
        const std::string &arrayName = this->BinnedArray[i];
        int opId = this->Operation[i];

        int arrayId = arrayMmdId[arrayName];
        int arrayType = mmd->ArrayType[arrayId];

        switch (arrayType)
        {
        svtkNestedTemplateMacroFloat(_DATA,

          using elem_t = SVTK_TT_DATA;
          using array_t = svtkHAMRDataArray<elem_t>;

          // get the array to bin.
          auto col = GetColumn(tab, arrayName);
          if (!col)
          {
            SENSEI_ERROR("Failed to get column \"" << arrayName << "\" from table")
            MPI_Abort(comm, -1);
            return false;
          }

          // handle the case when the simulation gives us
          // vtkAOSDataArrayTemplate instances by making a deep copy.
          auto arrayIn = dynamic_cast<array_t*>(col);
          if (!arrayIn)
          {
            arrayIn = array_t::New(col, alloc, calcStr[(i+1)%nStream], smode);
            deleteDa.push_back(arrayIn);
          }

          // get the output
          array_t *arrayOut = static_cast<array_t*>(binnedDa[i]);

          int iret = -1;
#if defined(SENSEI_ENABLE_CUDA)
          if (deviceId >= 0)
          {
            // make sure the data is on the active GPU
            auto spArrayIn = arrayIn->GetDeviceAccessible();

            // sync here so we know that incoming data is ready
            arrayIn->Synchronize();

            // compute the block's contribution on the GPU
            switch (opId)
            {
              case BIN_SUM:
              case BIN_AVG:
                iret = ::CudaImpl::blockBin(calcStr[(i+1)%nStream], spXCoord.get(),
                                            spYCoord.get(), spArrayIn.get(), nVals,
                                            minX, minY, dx, dy, xRes, yRes,
                                            ::CudaImpl::SumOp<elem_t>(),
                                            arrayOut->GetData());
                break;
              case BIN_MIN:
                iret = ::CudaImpl::blockBin(calcStr[(i+1)%nStream], spXCoord.get(),
                                            spYCoord.get(), spArrayIn.get(), nVals,
                                            minX, minY, dx, dy, xRes, yRes,
                                            ::CudaImpl::MinOp<elem_t>(),
                                            arrayOut->GetData());
                break;
              case BIN_MAX:
                iret = ::CudaImpl::blockBin(calcStr[(i+1)%nStream], spXCoord.get(),
                                            spYCoord.get(), spArrayIn.get(), nVals,
                                            minX, minY, dx, dy, xRes, yRes,
                                            ::CudaImpl::MaxOp<elem_t>(),
                                            arrayOut->GetData());
                break;
            }
          }
          else
          {
#endif
            // make sure the data is on the active CPU
            auto spArrayIn = arrayIn->GetHostAccessible();

            // sync here so we know that incoming data is ready
            arrayIn->Synchronize();

            // compute the block's contribution on the CPU
            switch (opId)
            {
              case BIN_SUM:
              case BIN_AVG:
                iret = ::HostImpl::blockBin(spXCoord.get(), spYCoord.get(), spArrayIn.get(),
                                            nVals, minX, minY, dx, dy, xRes, yRes,
                                            ::HostImpl::SumOp<elem_t>(),
                                            arrayOut->GetData());
                break;
              case BIN_MIN:
                iret = ::HostImpl::blockBin(spXCoord.get(), spYCoord.get(), spArrayIn.get(),
                                            nVals, minX, minY, dx, dy, xRes, yRes,
                                            ::HostImpl::MinOp<elem_t>(),
                                            arrayOut->GetData());
                break;
              case BIN_MAX:
                iret = ::HostImpl::blockBin(spXCoord.get(), spYCoord.get(), spArrayIn.get(),
                                            nVals, minX, minY, dx, dy, xRes, yRes,
                                            ::HostImpl::MaxOp<elem_t>(),
                                            arrayOut->GetData());
                break;
            }
#if defined(SENSEI_ENABLE_CUDA)
          }
#endif

          // check for error in the calculation
          if (iret)
          {
            std::string opName;
            GetOperation(opId, opName);
            SENSEI_ERROR("bin " << opName << " failed on array " << i
              << " \"" << arrayName << "\"")
            MPI_Abort(comm, -1);
            return false;
          }
        );}
      }
    );}
  }

  // finalize the calculations
  countDa->Synchronize();

  for (int i = 0; i < nBinnedArrays; ++i)
  {
    const std::string &arrayName = this->BinnedArray[i];
    int opId = this->Operation[i];

    switch (binnedDa[i]->GetDataType())
    {
    svtkTemplateMacroFloat(
      using elem_t = SVTK_TT;
      using array_t = svtkHAMRDataArray<SVTK_TT>;

      // get the output
      array_t *arrayOut = static_cast<array_t*>(binnedDa[i]);

      int iret = 0;
#if defined(SENSEI_ENABLE_CUDA)
      if (deviceId >= 0)
      {
        // finalize on the GPU, the same stream is used as before so no
        // synchronization should be needed.
        switch (opId)
        {
          case BIN_AVG:
            {
            iret = ::CudaImpl::scaleElement(calcStr[(i+1)%nStream],
                                            arrayOut->GetData(), countDa->GetData(),
                                            xyRes);
            }
            break;
          case BIN_MIN:
            {
            elem_t thresh = 0.9999 * CudaImpl::MinOp<elem_t>::initial_value();
            elem_t qnan = std::numeric_limits<elem_t>::quiet_NaN();


            iret = ::CudaImpl::maskGreater(calcStr[(i+1)%nStream],
                                           arrayOut->GetData(), xyRes, thresh, qnan);
            }
            break;
          case BIN_MAX:
            {
            elem_t thresh = 0.9999 * CudaImpl::MaxOp<elem_t>::initial_value();
            elem_t qnan = std::numeric_limits<elem_t>::quiet_NaN();

            iret = ::CudaImpl::maskLess(calcStr[(i+1)%nStream],
                                        arrayOut->GetData(), xyRes, thresh, qnan);
            }
            break;
        }
      }
      else
      {
#endif
        // finalize on the host, the same stream is used as before so no
        // synchronization should be needed.
        switch (opId)
        {
          case BIN_AVG:
            {
            auto finOp = ::HostImpl::ElementScale<elem_t>(countDa->GetData());

            iret = ::HostImpl::finalize(arrayOut->GetData(), xyRes, finOp);
            }
            break;
          case BIN_MIN:
            {
            elem_t thresh = 0.9999 * HostImpl::MinOp<elem_t>::initial_value();
            elem_t qnan = std::numeric_limits<elem_t>::quiet_NaN();

            auto finOp = ::HostImpl::MaskGreater<elem_t>(thresh, qnan);

            iret = ::HostImpl::finalize(arrayOut->GetData(), xyRes, finOp);
            }
            break;
          case BIN_MAX:
            {
            elem_t thresh = 0.9999 * HostImpl::MaxOp<elem_t>::initial_value();
            elem_t qnan = std::numeric_limits<elem_t>::quiet_NaN();

            auto finOp = ::HostImpl::MaskLess<elem_t>(thresh, qnan);

            iret = ::HostImpl::finalize(arrayOut->GetData(), xyRes, finOp);
            }
            break;
        }
#if defined(SENSEI_ENABLE_CUDA)
      }
#endif

      // check for error in the calculation
      if (iret)
      {
        std::string opName;
        GetOperation(opId, opName);
        SENSEI_ERROR("finalize " << opName << " failed on array " << i
          << " \"" << arrayName << "\"")
        MPI_Abort(comm, -1);
        return false;
      }

    );}
  }

  // move the results to the CPU for comm and I/O
#if defined(SENSEI_ENABLE_CUDA)
  if (deviceId >= 0)
  {
    // wait until all calculations are complete
    for (int i = 0; i < nStream; ++i)
      cudaStreamSynchronize(calcStr[i]);

    // explcitly move the data to the host
    alloc = svtkAllocator::cuda_host;

    countDa->SetAllocator(alloc);

    for (int i = 0; i < nBinnedArrays; ++i)
    {
      switch (binnedDa[i]->GetDataType())
      {
      svtkTemplateMacroFloat(
        using array_t = svtkHAMRDataArray<SVTK_TT>;
        static_cast<array_t*>(binnedDa[i])->SetAllocator(alloc);
      );}
    }
  }
#endif

  // accumulate contributions from all ranks
  countDa->Synchronize();
  MPI_Reduce(MPI_IN_PLACE, countDa->GetData(), xyRes,
             MPI_UNSIGNED_LONG, MPI_SUM, 0, comm);

  for (int i = 0; i < nBinnedArrays; ++i)
  {
    MPI_Op redOp = MPI_OP_NULL;
    switch (this->Operation[i])
    {
      case BIN_SUM: redOp = MPI_SUM; break;
      case BIN_AVG: redOp = MPI_SUM; break;
      case BIN_MIN: redOp = MPI_MIN; break;
      case BIN_MAX: redOp = MPI_MAX; break;
    }

    switch(binnedDa[i]->GetDataType())
    {
    svtkTemplateMacroFloat(
      using elem_t = SVTK_TT;
      using array_t = svtkHAMRDataArray<elem_t>;

      auto array = dynamic_cast<array_t*>(binnedDa[i]);
      auto redType = MPIUtils::mpi_tt<elem_t>::datatype();

      array->Synchronize();

      MPI_Reduce(MPI_IN_PLACE, array->GetData(),
                 xyRes, redType, redOp, 0, comm);
    );}
  }

  // write the results
  if (rank == 0)
  {
    char fn[512];
    fn[511] = '\0';

    snprintf(fn, 511, "%s/data_bin_%s_%06ld.vtk", this->OutDir.c_str(),
             this->MeshName.c_str(), this->Iteration);

    std::vector<svtkDataArray*> cellData(binnedDa.begin(), binnedDa.end());
    cellData.push_back(countDa);

    if (SVTKUtils::WriteVTK(fn, xRes + 1, yRes + 1, 1, minX,
                            minY, 0., dx, dy, 1., cellData, {}))
    {
      SENSEI_ERROR("Failed to write file \"" << fn << "\"")
      return false;
    }

    this->Iteration += 1;
  }

  if (this->ReturnData && dataOut)
  {
    auto mbo = svtkMultiBlockDataSet::New();
    mbo->SetNumberOfBlocks(n_ranks);

    if (rank == 0)
    {
      auto imo = svtkImageData::New();
      imo->SetOrigin(minX, minY, 0.0);
      imo->SetSpacing(dx, dy, 0.0);
      imo->SetDimensions(xRes, yRes, 1);
      imo->GetPointData()->AddArray(countDa);
      for (int i = 0; i < nBinnedArrays; ++i)
        imo->GetPointData()->AddArray(binnedDa[i]);

      mbo->SetBlock(0, imo);

      imo->Delete();
    }

    auto va = SVTKDataAdaptor::New();
    va->SetDataObject(this->MeshName, mbo);

    mbo->Delete();

    *dataOut = va;
  }

  // delete any temporaries
  int nDeleteDa = deleteDa.size();
  for (int i = 0; i < nDeleteDa; ++i)
    deleteDa[i]->Delete();

  countDa->Delete();

  for (int i = 0; i < nBinnedArrays; ++i)
    binnedDa[i]->Delete();

#if defined(SENSEI_ENABLE_CUDA)
  // clean up streams
  for (int i = 0; i < nStream; ++i)
  {
    cudaStreamSynchronize(calcStr[i]);
    cudaStreamDestroy(calcStr[i]);
  }
#endif

  if (rank == 0)
  {
    timeval endTime{};
    gettimeofday(&endTime, nullptr);

    double runTimeUs = (endTime.tv_sec * 1e6 + endTime.tv_usec) -
      (startTime.tv_sec * 1e6 + startTime.tv_usec);

    SENSEI_STATUS("DataBinning::Execute Step = " << step << " Time = " << time
      << "\" using " << (deviceId < 0 ? "the host" : "CUDA GPU")
      << "(" << deviceId << ") in " << runTimeUs / 1e6 << " s")
  }

  return true;
}

//-----------------------------------------------------------------------------
int DataBinning::Finalize()
{
  return 0;
}

}
