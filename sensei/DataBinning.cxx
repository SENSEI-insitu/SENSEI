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

// lets the compiler find certain overloads
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
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#endif

#include <algorithm>
#include <vector>
#include <sys/time.h>
#include <chrono>

// lets the compiler find the time literals
using namespace std::chrono_literals;

// undefine this to use default streams
#define DATA_BIN_STREAMS

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
  void operator()(T &dest, const T & src) const { dest = std::min(dest, src); }
};

/// update the max
template <typename T>
struct MaxOp {
  static constexpr T initial_value() { return std::numeric_limits<T>::lowest(); }
  void operator()(T &dest, const T & src) const { dest = std::max(dest, src); }
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
  BinnedArray(), Operation(), ReturnData(0), MaxThreads(4)
{
}

//-----------------------------------------------------------------------------
DataBinning::~DataBinning()
{
  int n = this->ThreadComm.size();
  for (int i = 0; i < n; ++i)
    MPI_Comm_free(&this->ThreadComm[i]);
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
void DataBinning::SetAsynchronous(int val)
{
  sensei::AnalysisAdaptor::SetAsynchronous(val);
  this->InitializeThreads();
}

//-----------------------------------------------------------------------------
void DataBinning::InitializeThreads()
{
  bool async = this->GetAsynchronous() && !this->ReturnData;
  if (async && (int(this->Threads.size()) != this->MaxThreads))
  {
    this->Threads.resize(this->MaxThreads);

    // make a communicator for each threads
    MPI_Comm comm = this->GetCommunicator();
    this->ThreadComm.resize(this->MaxThreads);
    for (int i = 0; i < this->MaxThreads; ++i)
      MPI_Comm_dup(comm, &this->ThreadComm[i]);
  }
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
  std::string oDir = node.attribute("out_dir").as_string(this->OutDir.c_str());
  int xRes = node.attribute("x_res").as_int(this->XRes);
  int yRes = node.attribute("y_res").as_int(this->YRes);
  int retData = node.attribute("return_data").as_int(this->ReturnData);
  int maxThreads = node.attribute("max_threads").as_int(this->MaxThreads);

  // get arrays to bin
  std::vector<std::string> arrays;
  XMLUtils::ParseList(node.child("arrays"), arrays);

  // and coresponding operations
  std::vector<std::string> ops;
  XMLUtils::ParseList(node.child("operations"), ops);

  return this->Initialize(mesh, xAxis, yAxis, arrays, ops,
                          xRes, yRes, oDir, retData, maxThreads);
}

//-----------------------------------------------------------------------------
int DataBinning::Initialize(const std::string &meshName,
  const std::string &xAxisArray, const std::string &yAxisArray,
  const std::vector<std::string> &binnedArray,
  const std::vector<std::string> &operation,
  long xres, long yres, const std::string &outDir,
  int returnData, int maxThreads)
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
  this->MaxThreads = maxThreads;

  // each thread needs a communicator
  this->InitializeThreads();

  int rank = 0;
  MPI_Comm_rank(this->GetCommunicator(), &rank);
  int verbose = this->GetVerbose();
  if ((verbose && (rank == 0)) || (verbose > 1))
  {
    SENSEI_STATUS(<< "Configured DataBinning: MeshName=" << meshName
      << " XAxisArray=" << xAxisArray << " YAxisArray=" << yAxisArray
      << " BinnedArray=" << binnedArray << " Operations=" << operation
      << " XRes=" << xres << " YRes=" << yres << " OutDir=" << outDir
      << " ReturnData=" << returnData << " MaxThreads=" << maxThreads
      << " Asynchronous=" << this->GetAsynchronous()
      << " DeviceId=" << this->GetDeviceId())
  }

  return 0;
}


/// functor for computing the data binning in the back ground
struct DataBin
{
  ~DataBin() {}

  DataBin() :
    XRes{}, YRes{}, OutDir{}, Iteration{}, MeshName{}, XAxisArray{},
    YAxisArray{}, BinnedArray{}, Operation{}, ReturnData{}, Rank{},
    NRanks{}, DeviceId{}, Asynchronous{}, Verbose{}, Alloc{}, SMode{},
    NStream{}, CalcStr{}, Mmd{}, ArrayMmdId{}, Mesh{}, MeshOut{},
    Step{}, Time{}, Error{}
 {}

 DataBin(const DataBin &) = delete;
 void operator=(const DataBin &) = delete;
 DataBin(const DataBin &&) = delete;
 void operator=(const DataBin &&) = delete;

  int Initialize(const std::string &meshName,
    const std::string &xAxisArray, const std::string &yAxisArray,
    const std::vector<std::string> &binnedArray, const std::vector<int> &operation,
    long xres, long yres, const std::string &outDir, int returnData, MPI_Comm
    comm, int rank, int nRanks, int deviceId, int async, unsigned long iteration,
   int verbose, sensei::DataAdaptor *daIn);

  void Compute();


  long XRes;
  long YRes;
  std::string OutDir;
  unsigned long Iteration;
  std::string MeshName;
  std::string XAxisArray;
  std::string YAxisArray;
  std::vector<std::string> BinnedArray;
  std::vector<int> Operation;
  int ReturnData;

  MPI_Comm Comm;
  int Rank;
  int NRanks;
  int DeviceId;
  int Asynchronous;
  int Verbose;
  svtkAllocator Alloc;
  svtkStreamMode SMode;
  int NStream;
  std::vector<svtkStream> CalcStr;
  MeshMetadataPtr Mmd;
  std::map<std::string, int> ArrayMmdId;
  svtkCompositeDataSetPtr Mesh;
  svtkCompositeDataSetPtr MeshOut;
  long Step;
  double Time;
  int Error;
};

int DataBin::Initialize(const std::string &meshName,
  const std::string &xAxisArray, const std::string &yAxisArray,
  const std::vector<std::string> &binnedArray, const std::vector<int> &operation,
  long xres, long yres, const std::string &outDir, int returnData, MPI_Comm comm,
  int rank, int nRanks, int deviceId, int async, unsigned long iteration, int verbose,
  sensei::DataAdaptor *daIn)
{
  this->MeshName = meshName;
  this->XAxisArray = xAxisArray;
  this->YAxisArray = yAxisArray;
  this->BinnedArray = binnedArray;
  this->Operation = operation;
  this->XRes = xres;
  this->YRes = yres;
  this->OutDir = outDir;
  this->ReturnData = returnData;
  this->Comm = comm;
  this->Rank = rank;
  this->NRanks = nRanks;
  this->DeviceId = deviceId;
  this->Asynchronous = async;
  this->Iteration = iteration;
  this->Verbose = verbose;
  this->Error = 0;

  // determine the allocator and stream to use
  this->Alloc = svtkAllocator::malloc;
  this->SMode = svtkStreamMode::async;
  this->NStream = 4;
  this->CalcStr.resize(this->NStream);

#if defined(SENSEI_ENABLE_CUDA)
  // if we are assigned to a specific GPU make it active and use a GPU
  // allocator
  if (this->DeviceId >= 0)
  {
    // use a CUDA allocator
    this->Alloc = svtkAllocator::cuda;

    // activate the assigned device
    cudaSetDevice(this->DeviceId);

    // allocate some streams for data movement and computation
    for (int i = 0; i < this->NStream; ++i)
    {
#if defined(DATA_BIN_STREAMS)
      cudaStream_t strm;
      cudaStreamCreate(&strm);
      this->CalcStr[i] = strm;
#else
      this->CalcStr[i] = cudaStreamPerThread;
#endif
    }
  }
#endif

  // see what the simulation is providing
  MeshMetadataMap mdMap;
  MeshMetadataFlags mdFlags;
  if (mdMap.Initialize(daIn, mdFlags))
  {
    SENSEI_ERROR("Failed to get metadata")
    return -1;
  }

  // get the this->Mesh metadata object
  if (mdMap.GetMeshMetadata(this->MeshName, this->Mmd))
  {
    SENSEI_ERROR("Failed to get metadata for this->Mesh \"" << this->MeshName << "\"")
    return -1;
  }

  // build a mapping from array name to its metadata
  for (int i = 0; i < this->Mmd->NumArrays; ++i)
    this->ArrayMmdId[this->Mmd->ArrayName[i]] = i;

  // check the coordinate axis arrays have metadata
  if (!this->ArrayMmdId.count(this->XAxisArray) || !this->ArrayMmdId.count(this->YAxisArray))
  {
    SENSEI_ERROR("Failed to get metadata for coordinate arrays \""
      << this->XAxisArray << "\", \"" << this->YAxisArray << "\"")
    return -1;
  }

  // check that the binned arrays have metadata
  int nBinnedArrays = this->BinnedArray.size();
  for (int i = 0; i < nBinnedArrays; ++i)
  {
    if (!this->ArrayMmdId.count(this->BinnedArray[i]))
    {
      SENSEI_ERROR("Failed to get metadata for binned array \""
        << this->BinnedArray[i] << "\"")
      return -1;
    }
  }

  int xAxisArrayId = this->ArrayMmdId[this->XAxisArray];
  int yAxisArrayId = this->ArrayMmdId[this->YAxisArray];

  // check that the coordinate arrays are not multi-component. supporting
  // multi-component data is something that could be added later if needed
  int xAxisArrayComps = this->Mmd->ArrayComponents[xAxisArrayId];
  int yAxisArrayComps = this->Mmd->ArrayComponents[yAxisArrayId];
  if ((xAxisArrayComps != 1) || (yAxisArrayComps != 1))
  {
    SENSEI_ERROR("Coordinate axes are required to have only one component "
      "but the x cooridnate has " << xAxisArrayComps << " and the y coordinate "
      "has " << yAxisArrayComps << " components.")
    return -1;
  }

  // check that the coordinates have the same type
  int xAxisArrayType = this->Mmd->ArrayType[xAxisArrayId];
  int yAxisArrayType = this->Mmd->ArrayType[yAxisArrayId];
  if (xAxisArrayType != yAxisArrayType)
  {
    SENSEI_ERROR("Coordinate arrays do not have the same data type.")
    return -1;
  }

  // check that arrays are floating point
  if ((xAxisArrayType != SVTK_DOUBLE) && (xAxisArrayType != SVTK_FLOAT))
  {
    SENSEI_ERROR("Coordinate arrays are required to be floating point")
    return -1;
  }

  // fetch the this->Mesh object from the simulation
  svtkDataObject *dobj = nullptr;
  if (daIn->GetMesh(this->MeshName, true, dobj))
  {
    SENSEI_ERROR("Failed to get this->Mesh \"" << this->MeshName << "\"")
    return -1;
  }

  if (!dobj)
  {
    SENSEI_ERROR("DataBinning requires all ranks to have data")
    return -1;
  }

  // get the current this->Time and this->Step
  this->Step = daIn->GetDataTimeStep();
  this->Time = daIn->GetDataTime();

  // fetch the cooridnate axes from the simulation
  int xAxisArrayCen = this->Mmd->ArrayCentering[xAxisArrayId];
  int yAxisArrayCen = this->Mmd->ArrayCentering[yAxisArrayId];
  if (daIn->AddArray(dobj, this->MeshName, xAxisArrayCen, this->XAxisArray) ||
    daIn->AddArray(dobj, this->MeshName, yAxisArrayCen, this->YAxisArray))
  {
    SENSEI_ERROR(<< daIn->GetClassName()
      << " failed to fetch the coordinate axis arrays \""
      << this->XAxisArray << "\", \"" << this->YAxisArray << "\"" )
    return -1;
  }

  // fetch the arrays to bin from the simulation
  for (int i = 0; i < nBinnedArrays; ++i)
  {
    const std::string &arrayName = this->BinnedArray[i];

    int arrayId = this->ArrayMmdId[arrayName];
    int arrayComp = this->Mmd->ArrayComponents[arrayId];
    int arrayCen = this->Mmd->ArrayCentering[arrayId];
    int arrayType = this->Mmd->ArrayType[arrayId];

    // check that arrays are floating point
    if ((arrayType != SVTK_DOUBLE) && (arrayType != SVTK_FLOAT))
    {
      SENSEI_ERROR("Binned arrays are required to be floating point")
      return -1;
    }

    // check that the binned arrays are not multi-component. supporting
    // multi-component data is something that could be added if needed
    if ((arrayComp != 1))
    {
      SENSEI_ERROR("Binned arrays are required to have only one component. \""
        << arrayName << "\" has " << arrayComp << " components")
      return -1;
    }

    // fetch the data from the simulation
    if (daIn->AddArray(dobj, this->MeshName, arrayCen, arrayName))
    {
      SENSEI_ERROR(<< daIn->GetClassName()
        << " failed to fetch the " << i << "th array to bin  \""
        << this->BinnedArray[i] << "\"")
      return -1;
    }
  }

  // work with composite data from here on out
  this->Mesh = SVTKUtils::AsCompositeData(this->Comm, dobj, true);

  // if we are running asynchronously or any of the arrays are not HAMR data
  // arrays make a deep copy
  svtkSmartPointer<svtkCompositeDataIterator> iter;
  iter.TakeReference(this->Mesh->NewIterator());
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
      return -1;
    }

    switch (xAxisArrayType)
    {
    svtkNestedTemplateMacroFloat(_COORDS,

      using coord_array_t = svtkHAMRDataArray<SVTK_TT_COORDS>;

      // get the x-coordinate arrays.
      auto xCol = GetColumn(tab, this->XAxisArray);
      if (!xCol)
      {
        SENSEI_ERROR("Failed to get column \"" << this->XAxisArray << "\" from table")
        return -1;
      }

      // when changing devices, use streams from the active device
      coord_array_t *xCoordIn = dynamic_cast<coord_array_t*>(xCol);
      if (xCoordIn && (xCoordIn->GetOwner() != this->DeviceId))
      {
        xCoordIn->Synchronize();
        xCoordIn->SetStream(this->CalcStr[1], this->SMode);
      }

      // make the copy and update the this->Mesh data
      if (async || !xCoordIn)
      {
        auto xCoord = coord_array_t::New(xCol, this->Alloc, this->CalcStr[1], this->SMode);
        xCoord->Synchronize();
        tab->RemoveColumnByName(this->XAxisArray.c_str());
        tab->AddColumn(xCoord);
        xCoord->Delete();
      }

      // get the y-coordinate arrays.
      auto yCol = GetColumn(tab, this->YAxisArray);
      if (!yCol)
      {
        SENSEI_ERROR("Failed to get column \"" << this->YAxisArray << "\" from table")
        return -1;
      }

      // when changing devices, use streams from the active device
      coord_array_t *yCoordIn = dynamic_cast<coord_array_t*>(yCol);
      if (yCoordIn && (yCoordIn->GetOwner() != this->DeviceId))
      {
        yCoordIn->Synchronize();
        yCoordIn->SetStream(this->CalcStr[2], this->SMode);
      }

      // make the copy and update the this->Mesh data
      if (async || !yCoordIn)
      {
        auto yCoord = coord_array_t::New(yCol, this->Alloc, this->CalcStr[2], this->SMode);
        yCoord->Synchronize();
        tab->RemoveColumnByName(this->YAxisArray.c_str());
        tab->AddColumn(yCoord);
        yCoord->Delete();
      }

      // process each array to bin
      for (int i = 0; i < nBinnedArrays; ++i)
      {
        const std::string &arrayName = this->BinnedArray[i];
        int arrayId = this->ArrayMmdId[arrayName];
        int arrayType = this->Mmd->ArrayType[arrayId];

        switch (arrayType)
        {
        svtkNestedTemplateMacroFloat(_DATA,

          using array_t = svtkHAMRDataArray<SVTK_TT_DATA>;

          // get the array to bin.
          auto col = GetColumn(tab, arrayName);
          if (!col)
          {
            SENSEI_ERROR("Failed to get column \"" << arrayName << "\" from table")
            return -1;
          }

          // when changing devices, use streams from the active device
          auto strm = this->CalcStr[(i+1)%this->NStream];

          coord_array_t *colIn = dynamic_cast<coord_array_t*>(col);
          if (colIn && (colIn->GetOwner() != this->DeviceId))
          {
            colIn->Synchronize();
            colIn->SetStream(strm, this->SMode);
          }

          // make the copy and update the this->Mesh data. sync up before deleteing the source!
          if (async || !colIn)
          {
            auto arrayIn = array_t::New(col, this->Alloc, strm, this->SMode);
            arrayIn->Synchronize();
            tab->RemoveColumnByName(arrayName.c_str());
            tab->AddColumn(arrayIn);
            arrayIn->Delete();
          }

        );}
      }
    );}
  }

  return 0;
}

void DataBin::Compute()
{
  timeval startTime{};
  if (this->Rank == 0)
    gettimeofday(&startTime, nullptr);

#if defined(SENSEI_ENABLE_CUDA)
  if (this->DeviceId >= 0)
    cudaSetDevice(this->DeviceId);
#endif

  // get the block min/max
  int xAxisArrayId = this->ArrayMmdId[this->XAxisArray];
  int xAxisArrayType = this->Mmd->ArrayType[xAxisArrayId];

  double axesMin[2] = {std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
  double axesMax[2] = {std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest()};

  svtkSmartPointer<svtkCompositeDataIterator> iter;
  iter.TakeReference(this->Mesh->NewIterator());
  for (iter->InitTraversal(); !iter->IsDoneWithTraversal(); iter->GoToNextItem())
  {
    // get the next local block of data.
    auto curObj = iter->GetCurrentDataObject();
    auto tab = static_cast<svtkTable*>(curObj);

    switch (xAxisArrayType)
    {
    svtkNestedTemplateMacroFloat(_COORDS,
      using coord_t = SVTK_TT_COORDS;
      using coord_array_t = svtkHAMRDataArray<SVTK_TT_COORDS>;

      // get the x-coordinate arrays.
      auto xCol = GetColumn(tab, this->XAxisArray);
      auto xCoord = static_cast<coord_array_t*>(xCol);
      long nVals = xCoord->GetNumberOfTuples();

      // get the y-coordinate arrays.
      auto yCol = GetColumn(tab, this->YAxisArray);
      auto yCoord = static_cast<coord_array_t*>(yCol);

#if defined(SENSEI_ENABLE_CUDA)
      if (this->DeviceId >= 0)
      {
        // compute the min/max x-axis
        auto spx = xCoord->GetDeviceAccessible();
        auto px = spx.get();

        axesMin[0] = thrust::reduce(thrust::device, px, px + nVals,
                                    coord_t(axesMin[0]), thrust::minimum<coord_t>());

        axesMax[0] = thrust::reduce(thrust::device, px, px + nVals,
                                    coord_t(axesMax[0]), thrust::maximum<coord_t>());

        // compute the min/max y-axis
        auto spy = yCoord->GetDeviceAccessible();
        auto py = spy.get();

        axesMin[1] = thrust::reduce(thrust::device, py, py + nVals,
                                    coord_t(axesMin[1]), thrust::minimum<coord_t>());

        axesMax[1] = thrust::reduce(thrust::device, py, py + nVals,
                                    coord_t(axesMax[1]), thrust::maximum<coord_t>());
      }
      else
      {
#endif
        // compute the min/max x-axis
        auto spx = xCoord->GetHostAccessible();
        auto px = spx.get();
        for (long i = 0; i < nVals; ++i)
        {
            axesMin[0] = std::min(coord_t(axesMin[0]), px[i]);
            axesMax[0] = std::max(coord_t(axesMax[0]), px[i]);
        }

        // compute the min/max y-axis
        auto spy = yCoord->GetHostAccessible();
        auto py = spy.get();
        for (long i = 0; i < nVals; ++i)
        {
            axesMin[1] = std::min(coord_t(axesMin[1]), py[i]);
            axesMax[1] = std::max(coord_t(axesMax[1]), py[i]);
        }
#if defined(SENSEI_ENABLE_CUDA)
      }
#endif
    );}
  }

  // get the global min/max
  switch (xAxisArrayType)
  {
  svtkNestedTemplateMacroFloat(_COORDS,
    using coord_t = SVTK_TT_COORDS;
    auto redType = MPIUtils::mpi_tt<coord_t>::datatype();
    MPI_Allreduce(MPI_IN_PLACE, axesMin, 2, redType, MPI_MIN, this->Comm);
    MPI_Allreduce(MPI_IN_PLACE, axesMax, 2, redType, MPI_MAX, this->Comm);
  );}

  // compute the grid spacing
  long xRes = this->XRes;
  long yRes = this->YRes;

  double dx = 0.;
  double dy = 0.;

  dx = axesMax[0] - axesMin[0];
  dy = axesMax[1] - axesMin[1];

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

  // allocate the count result array
  auto countDa = svtkHAMRLongArray::New("count", xyRes, 1,
                                        this->Alloc, this->CalcStr[0], this->SMode, 0);

  // allocate the binned result arrays
  int nBinnedArrays = this->BinnedArray.size();
  std::vector<svtkDataArray*> binnedDa(nBinnedArrays);
  for (int i = 0; i < nBinnedArrays; ++i)
  {
    const std::string &arrayName = this->BinnedArray[i];
    int arrayId = this->ArrayMmdId[arrayName];

    int opId = this->Operation[i];

    // allocate the result array
    switch (this->Mmd->ArrayType[arrayId])
    {
      svtkTemplateMacroFloat(

        using elem_t = SVTK_TT;
        using array_t = svtkHAMRDataArray<elem_t>;

        elem_t iValue{};
        HostImpl::GetInitialValue(opId, iValue);

        std::string opName;
        DataBinning::GetOperation(opId, opName);

        binnedDa[i] = array_t::New(arrayName + "_" + opName, xyRes, 1, this->Alloc,
                                   this->CalcStr[(i+1)%this->NStream], this->SMode, iValue);
      );
    }
  }

  // process the blocks of data
  for (iter->InitTraversal(); !iter->IsDoneWithTraversal(); iter->GoToNextItem())
  {
    // get the next local block of data.
    auto curObj = iter->GetCurrentDataObject();

    // downcast to the block type. for simplicity of this example we require
    // tabular data.
    auto tab = static_cast<svtkTable*>(curObj);

    switch (xAxisArrayType)
    {
    svtkNestedTemplateMacroFloat(_COORDS,

      using coord_t = SVTK_TT_COORDS;
      using coord_array_t = svtkHAMRDataArray<coord_t>;

      // get the x-coordinate arrays.
      auto xCol = GetColumn(tab, this->XAxisArray);
      auto xCoord = static_cast<coord_array_t*>(xCol);
      long nVals = xCoord->GetNumberOfTuples();

      // get the y-coordinate arrays.
      auto yCol = GetColumn(tab, this->YAxisArray);
      auto yCoord = static_cast<coord_array_t*>(yCol);

      // compute the count
      std::shared_ptr<const coord_t> spXCoord;
      std::shared_ptr<const coord_t> spYCoord;

#if defined(SENSEI_ENABLE_CUDA)
      if (this->DeviceId >= 0)
      {
        // make sure the data is on the active GPU
        spXCoord = xCoord->GetDeviceAccessible();
        spYCoord = yCoord->GetDeviceAccessible();

        // sync here so we know that incoming data is ready
        xCoord->Synchronize();
        yCoord->Synchronize();

        // compute the block's contribution
        if (::CudaImpl::blockCount(this->CalcStr[0], spXCoord.get(), spYCoord.get(),
          nVals, axesMin[0], axesMin[1], dx, dy, xRes, yRes, countDa->GetData()))
        {
          SENSEI_ERROR("Failed to compute the count on the GPU")
          this->Error = 1;
          return;
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
          nVals, axesMin[0], axesMin[1], dx, dy, xRes, yRes, countDa->GetData()))
        {
          SENSEI_ERROR("Failed to compute the count on the host")
          this->Error = 1;
          return;
        }
#if defined(SENSEI_ENABLE_CUDA)
      }
#endif

      // process each array to bin
      for (int i = 0; i < nBinnedArrays; ++i)
      {
        const std::string &arrayName = this->BinnedArray[i];
        int opId = this->Operation[i];

        int arrayId = this->ArrayMmdId[arrayName];
        int arrayType = this->Mmd->ArrayType[arrayId];

        switch (arrayType)
        {
        svtkNestedTemplateMacroFloat(_DATA,

          using elem_t = SVTK_TT_DATA;
          using array_t = svtkHAMRDataArray<elem_t>;

          // get the array to bin.
          auto col = GetColumn(tab, arrayName);
          auto arrayIn = static_cast<array_t*>(col);

          // get the output
          array_t *arrayOut = static_cast<array_t*>(binnedDa[i]);

          int iret = -1;
#if defined(SENSEI_ENABLE_CUDA)
          if (this->DeviceId >= 0)
          {
            // make sure the data is on the active GPU
            auto spArrayIn = arrayIn->GetDeviceAccessible();

            // sync here so we know that incoming data is ready
            arrayIn->Synchronize();

            // compute the block's contribution on the GPU
            switch (opId)
            {
              case DataBinning::BIN_SUM:
              case DataBinning::BIN_AVG:
                iret = ::CudaImpl::blockBin(this->CalcStr[(i+1)%this->NStream], spXCoord.get(),
                                            spYCoord.get(), spArrayIn.get(), nVals,
                                            axesMin[0], axesMin[1], dx, dy, xRes, yRes,
                                            ::CudaImpl::SumOp<elem_t>(),
                                            arrayOut->GetData());
                break;
              case DataBinning::BIN_MIN:
                iret = ::CudaImpl::blockBin(this->CalcStr[(i+1)%this->NStream], spXCoord.get(),
                                            spYCoord.get(), spArrayIn.get(), nVals,
                                            axesMin[0], axesMin[1], dx, dy, xRes, yRes,
                                            ::CudaImpl::MinOp<elem_t>(),
                                            arrayOut->GetData());
                break;
              case DataBinning::BIN_MAX:
                iret = ::CudaImpl::blockBin(this->CalcStr[(i+1)%this->NStream], spXCoord.get(),
                                            spYCoord.get(), spArrayIn.get(), nVals,
                                            axesMin[0], axesMin[1], dx, dy, xRes, yRes,
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
              case DataBinning::BIN_SUM:
              case DataBinning::BIN_AVG:
                iret = ::HostImpl::blockBin(spXCoord.get(), spYCoord.get(), spArrayIn.get(),
                                            nVals, axesMin[0], axesMin[1], dx, dy, xRes, yRes,
                                            ::HostImpl::SumOp<elem_t>(),
                                            arrayOut->GetData());
                break;
              case DataBinning::BIN_MIN:
                iret = ::HostImpl::blockBin(spXCoord.get(), spYCoord.get(), spArrayIn.get(),
                                            nVals, axesMin[0], axesMin[1], dx, dy, xRes, yRes,
                                            ::HostImpl::MinOp<elem_t>(),
                                            arrayOut->GetData());
                break;
              case DataBinning::BIN_MAX:
                iret = ::HostImpl::blockBin(spXCoord.get(), spYCoord.get(), spArrayIn.get(),
                                            nVals, axesMin[0], axesMin[1], dx, dy, xRes, yRes,
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
            DataBinning::GetOperation(opId, opName);
            SENSEI_ERROR("bin " << opName << " failed on array " << i
              << " \"" << arrayName << "\"")
            this->Error = 1;
            return;
          }
        );}
      }
    );}
  }

  // move the results to the CPU for communication and I/O
#if defined(SENSEI_ENABLE_CUDA)
  if (this->DeviceId >= 0)
  {
    // wait until all calculations are complete
    for (int i = 0; i < this->NStream; ++i)
      cudaStreamSynchronize(this->CalcStr[i]);

#if !defined(SENSEI_ENABLE_CUDA_MPI)
    // explcitly move the data to the host
    auto alloc = svtkAllocator::cuda_host;
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
#endif
  }
#endif

  // accumulate contributions from all ranks
  countDa->Synchronize();

  if (this->Rank == 0)
  {
    MPI_Reduce(MPI_IN_PLACE, countDa->GetData(), xyRes,
               MPI_UNSIGNED_LONG, MPI_SUM, 0, this->Comm);
  }
  else
  {
    MPI_Reduce(countDa->GetData(), nullptr, xyRes,
               MPI_UNSIGNED_LONG, MPI_SUM, 0, this->Comm);
  }

  for (int i = 0; i < nBinnedArrays; ++i)
  {
    MPI_Op redOp = MPI_OP_NULL;
    switch (this->Operation[i])
    {
      case DataBinning::BIN_SUM: redOp = MPI_SUM; break;
      case DataBinning::BIN_AVG: redOp = MPI_SUM; break;
      case DataBinning::BIN_MIN: redOp = MPI_MIN; break;
      case DataBinning::BIN_MAX: redOp = MPI_MAX; break;
    }

    switch(binnedDa[i]->GetDataType())
    {
    svtkTemplateMacroFloat(
      using elem_t = SVTK_TT;
      using array_t = svtkHAMRDataArray<elem_t>;

      auto array = dynamic_cast<array_t*>(binnedDa[i]);
      auto redType = MPIUtils::mpi_tt<elem_t>::datatype();

      array->Synchronize();

      if (this->Rank == 0)
      {
        MPI_Reduce(MPI_IN_PLACE, array->GetData(),
                   xyRes, redType, redOp, 0, this->Comm);
      }
      else
      {
        MPI_Reduce(array->GetData(), nullptr,
                   xyRes, redType, redOp, 0, this->Comm);
      }

    );}
  }

  // finalize the calculation, and write the results on rank 0
  if (this->Rank == 0)
  {
    // move the results to the GPU for finalization
#if defined(SENSEI_ENABLE_CUDA)
    if (this->DeviceId >= 0)
    {
#if !defined(SENSEI_ENABLE_CUDA_MPI)
      auto alloc = svtkAllocator::cuda_async;
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
#endif
      countDa->Synchronize();
    }
#endif

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
        arrayOut->Synchronize();
        if (this->DeviceId >= 0)
        {
          // finalize on the GPU
          switch (opId)
          {
            case DataBinning::BIN_AVG:
              {
              iret = ::CudaImpl::scaleElement(this->CalcStr[(i+1)%this->NStream],
                                              arrayOut->GetData(), countDa->GetData(),
                                              xyRes);
              }
              break;
            case DataBinning::BIN_MIN:
              {
              elem_t thresh = 0.9000 * CudaImpl::MinOp<elem_t>::initial_value();
              elem_t qnan = std::numeric_limits<elem_t>::quiet_NaN();


              iret = ::CudaImpl::maskGreater(this->CalcStr[(i+1)%this->NStream],
                                             arrayOut->GetData(), xyRes, thresh, qnan);
              }
              break;
            case DataBinning::BIN_MAX:
              {
              elem_t thresh = 0.9000 * CudaImpl::MaxOp<elem_t>::initial_value();
              elem_t qnan = std::numeric_limits<elem_t>::quiet_NaN();

              iret = ::CudaImpl::maskLess(this->CalcStr[(i+1)%this->NStream],
                                          arrayOut->GetData(), xyRes, thresh, qnan);
              }
              break;
          }
        }
        else
        {
#endif
          // finalize on the host
          switch (opId)
          {
            case DataBinning::BIN_AVG:
              {
              auto finOp = ::HostImpl::ElementScale<elem_t>(countDa->GetData());

              iret = ::HostImpl::finalize(arrayOut->GetData(), xyRes, finOp);
              }
              break;
            case DataBinning::BIN_MIN:
              {
              elem_t thresh = 0.9000 * HostImpl::MinOp<elem_t>::initial_value();
              elem_t qnan = std::numeric_limits<elem_t>::quiet_NaN();

              auto finOp = ::HostImpl::MaskGreater<elem_t>(thresh, qnan);

              iret = ::HostImpl::finalize(arrayOut->GetData(), xyRes, finOp);
              }
              break;
            case DataBinning::BIN_MAX:
              {
              elem_t thresh = 0.9000 * HostImpl::MaxOp<elem_t>::initial_value();
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
          DataBinning::GetOperation(opId, opName);
          SENSEI_ERROR("finalize " << opName << " failed on array " << i
            << " \"" << arrayName << "\"")
          this->Error = 1;
          return;
        }

      );}
    }

    // write the results
    char fn[512];
    fn[511] = '\0';

    snprintf(fn, 511, "%s/data_bin_%s_%s_%s_%06ld.vtk",
             this->OutDir.c_str(), this->MeshName.c_str(),
             this->XAxisArray.c_str(), this->YAxisArray.c_str(),
             this->Iteration);

    std::vector<svtkDataArray*> cellData(binnedDa.begin(), binnedDa.end());
    cellData.push_back(countDa);

    if (SVTKUtils::WriteVTK(fn, xRes + 1, yRes + 1, 1, axesMin[0],
                            axesMin[1], 0., dx, dy, 1., cellData, {}))
    {
      SENSEI_ERROR("Failed to write file \"" << fn << "\"")
      this->Error = 1;
      return;
    }
  }

  if (this->ReturnData)
  {
    auto mbo = svtkMultiBlockDataSet::New();
    mbo->SetNumberOfBlocks(this->NRanks);

    if (this->Rank == 0)
    {
      auto imo = svtkImageData::New();
      imo->SetOrigin(axesMin[0], axesMin[1], 0.0);
      imo->SetSpacing(dx, dy, 0.0);
      imo->SetDimensions(xRes, yRes, 1);
      imo->GetPointData()->AddArray(countDa);
      for (int i = 0; i < nBinnedArrays; ++i)
        imo->GetPointData()->AddArray(binnedDa[i]);

      mbo->SetBlock(0, imo);

      imo->Delete();
    }

    this->MeshOut.Take(mbo);
  }

  countDa->Delete();

  for (int i = 0; i < nBinnedArrays; ++i)
    binnedDa[i]->Delete();

#if defined(SENSEI_ENABLE_CUDA)
#if defined(DATA_BIN_STREAMS)
  if (this->DeviceId >= 0)
  {
    // clean up streams
    for (int i = 0; i < this->NStream; ++i)
    {
      cudaStreamSynchronize(this->CalcStr[i]);
      cudaStreamDestroy(this->CalcStr[i]);
    }
  }
#endif
#endif

  this->Mesh = nullptr;

  if (this->Asynchronous && (this->Verbose && (this->Rank == 0)) || (this->Verbose > 1))
  {
    timeval endTime{};
    gettimeofday(&endTime, nullptr);

    double runTimeUs = (endTime.tv_sec * 1e6 + endTime.tv_usec) -
      (startTime.tv_sec * 1e6 + startTime.tv_usec);

    SENSEI_STATUS_ALL("thread:" << std::hex << std::this_thread::get_id()
      << std::dec << "  step:" << this->Step << "  time:" << this->Time
      << "  device:" << (this->DeviceId < 0 ? "host" : "CUDA GPU")
      << "(" << this->DeviceId << ")  t_bin:" << runTimeUs / 1e6 << "s")
  }
}

//-----------------------------------------------------------------------------
bool DataBinning::Execute(DataAdaptor* daIn, DataAdaptor** daOut)
{
  TimeEvent<128> mark("DataBinning::Execute");

  int rank = 0;
  int n_ranks = 1;
  MPI_Comm comm = this->GetCommunicator();
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &n_ranks);

  timeval startExec{}, startFetch{}, endFetch{}, endBin{}, endExec{};

  if (rank == 0)
    gettimeofday(&startExec, nullptr);

  // always zero this out, if somethig goes wrong the caller will not get and
  // invalid value
  if (daOut)
    *daOut = nullptr;

  int retData = this->ReturnData && daOut;
  int async = this->GetAsynchronous() && !retData;

  // if this thread is pending wait for it and check for an error
  int threadId = this->Iteration % this->MaxThreads;
  if (async && this->Threads[threadId].valid() && this->Threads[threadId].get())
  {
    SENSEI_ERROR("Async binning failed at iteration " << this->Iteration
      << " thread " << threadId)
    MPI_Abort(comm, -1);
    return false;
  }

  // get this thread's communicator
  MPI_Comm threadComm = async ? this->ThreadComm[threadId] : comm;

  if (rank == 0)
    gettimeofday(&startFetch, nullptr);

  auto binner = std::make_shared<DataBin>();

  // fetch data from the simulation
  int verbose = this->GetVerbose();
  if (binner->Initialize(this->MeshName, this->XAxisArray, this->YAxisArray,
    this->BinnedArray, this->Operation, this->XRes, this->YRes, this->OutDir,
    this->ReturnData, threadComm, rank, n_ranks, this->GetDeviceId(),
    async, this->Iteration, verbose, daIn))
  {
    SENSEI_ERROR("Failed to intialize the binner")
    MPI_Abort(comm, -1);
    return false;
  }

  if (rank == 0)
    gettimeofday(&endFetch, nullptr);

  // launch a thread to do the calculation
  auto pending = std::async(std::launch::async, [binner]() -> int { binner->Compute(); return binner->Error; });

  if (async)
  {
    // keep track of the threads so we don't exit before they are complete
    this->Threads[threadId] = std::move(pending);
  }
  else
  {
    // wait here for the thread to complete
    if (pending.get())
    {
      SENSEI_ERROR("Binning failed at iteration " << this->Iteration)
      MPI_Abort(comm, -1);
      return false;
    }
  }

  if (rank == 0)
    gettimeofday(&endBin, nullptr);

  // wrap the returned data in an adaptor
  if (retData)
  {
    auto va = SVTKDataAdaptor::New();
    va->SetDataObject(this->MeshName, binner->MeshOut.Get());
    *daOut = va;
  }

  if ((verbose && (rank == 0)) || (verbose > 1))
  {
    gettimeofday(&endExec, nullptr);

    double fetchTimeUs = (endFetch.tv_sec * 1e6 + endFetch.tv_usec) -
      (startFetch.tv_sec * 1e6 + startFetch.tv_usec);

    double binTimeUs = (endBin.tv_sec * 1e6 + endBin.tv_usec) -
      (endFetch.tv_sec * 1e6 + endFetch.tv_usec);

    double runTimeUs = (endExec.tv_sec * 1e6 + endExec.tv_usec) -
      (startExec.tv_sec * 1e6 + startExec.tv_usec);

    int deviceId = this->GetDeviceId();

    SENSEI_STATUS_ALL("DataBinning::Execute  iteration:"
      << this->Iteration << " mode:" << (async ? "async" : "sync")
      << "  device:" << (deviceId < 0 ? "host" : "CUDA GPU")
      << "(" << deviceId << ")  ret_data:" << (retData ? "yes":"no")
      << "  t_total:" << runTimeUs / 1e6 << "s  t_fetch:"
      << fetchTimeUs / 1e6 << "s  t_bin:" << binTimeUs / 1e6 << "s")
  }

  this->Iteration += 1;

  return true;
}

//-----------------------------------------------------------------------------
int DataBinning::WaitThreads()
{
  int iret = 0;
  if (this->GetAsynchronous() && !this->ReturnData)
  {
    int nThreads = this->Threads.size();
    for (int i = 0; i < nThreads; ++i)
    {
      if (this->Threads[i].valid() && this->Threads[i].get())
      {
        SENSEI_ERROR("Asynchronous binning failed at iteration " << this->Iteration
          << " thread id " << i)
        iret = -1;
      }
    }
  }
  return iret;
}

//-----------------------------------------------------------------------------
int DataBinning::Finalize()
{
  timeval startExec{}, endExec{};
  gettimeofday(&startExec, nullptr);

  int rank = 0;
  MPI_Comm_rank(this->GetCommunicator(), &rank);

  // wait for the last thread to finish
  this->WaitThreads();

  // send status message
  if (this->GetVerbose() && (rank == 0))
  {
    gettimeofday(&endExec, nullptr);

    double runTimeUs = (endExec.tv_sec * 1e6 + endExec.tv_usec) -
      (startExec.tv_sec * 1e6 + startExec.tv_usec);

    SENSEI_STATUS("DataBinning::Finalize  t_total:" << runTimeUs / 1e6 << "s")
  }

  return 0;
}

}
