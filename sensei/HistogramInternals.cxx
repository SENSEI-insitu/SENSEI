#include "senseiConfig.h"
#include "HistogramInternals.h"
#include "SVTKUtils.h"
#include "MemoryUtils.h"
#include "Error.h"

#if defined(SENSEI_ENABLE_CUDA)
#include "CUDAUtils.h"
#include <thrust/sequence.h>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#endif

#include <algorithm>
#include <vector>
#include <cstdio>
#include <cstring>
#include <errno.h>
#include <limits>

#include <svtkSmartPointer.h>
#include <svtkDataArray.h>
#include <svtkFloatArray.h>
#include <svtkDoubleArray.h>
#include <svtkUnsignedCharArray.h>
#include <svtkAOSDataArrayTemplate.h>
#include <svtkSOADataArrayTemplate.h>
#include <svtkHAMRDataArray.h>
#include <svtkCompositeDataIterator.h>
#include <svtkCompositeDataSet.h>
#include <svtkDataObject.h>
#include <svtkFieldData.h>
#include <svtkObjectFactory.h>

namespace sensei
{
#if defined(SENSEI_ENABLE_CUDA)
namespace HistogramInternalsCUDA
{
/** given a data array a ghost array and two indices, looks up
 * values in the array and computes less than accounting for
 * ghosted values.
 */
template <typename data_t>
struct indirectGhostMin
{
    indirectGhostMin(const data_t *data, const unsigned char *ghosts) :
      Data(data), Ghosts(ghosts) {}

    __device__
    size_t operator()(const size_t &l, const size_t &r) const
    {
      if (Ghosts[l] != 0)         // left is ghosted
          return r;
      else if (Ghosts[r] != 0)    // right is ghosted
          return l;
      else if (Data[l] < Data[r]) // left is less than right
          return l;
      else
          return r;               // right is less than left
    }

    const data_t *Data;
    const unsigned char *Ghosts;
};

/** given a data array a ghost array and two indices, looks up
 * values in the array and computes less than accounting for
 * ghosted values.
 */
template <typename data_t>
struct indirectGhostMax
{
    indirectGhostMax(const data_t *data, const unsigned char *ghosts) :
      Data(data), Ghosts(ghosts) {}

    __device__
    size_t operator()(const size_t &l, const size_t &r) const
    {
      if (Ghosts[l] != 0)         // left is ghosted
          return r;
      else if (Ghosts[r] != 0)    // right is ghosted
          return l;
      else if (Data[l] > Data[r]) // left is greater than right
          return l;
      else
          return r;               // right is greater than left
    }

    const data_t *Data;
    const unsigned char *Ghosts;
};


// **************************************************************************
template <typename data_t>
int ComputeRange(std::shared_ptr<const data_t> &pdata,
    std::shared_ptr<const unsigned char> &pGhosts, size_t nVals,
    data_t &dataMin, data_t &dataMax)
{
  // wrap pointers for thrust
  thrust::device_ptr<const data_t> tpdata(pdata.get());
  thrust::device_ptr<const unsigned char> tpGhosts(pGhosts.get());

  // generate ordinal sequence
  thrust::device_vector<size_t> indices(nVals);
  thrust::sequence(indices.begin(), indices.end());

  // compute the minimum taking into account ghost zones
  size_t minId = thrust::reduce(indices.begin(), indices.end(), 0,
    indirectGhostMin<data_t>(pdata.get(), pGhosts.get()));

  cudaError_t ierr = cudaSuccess;
  if ((ierr = cudaMemcpy(&dataMin, pdata.get() + minId,
    sizeof(data_t), cudaMemcpyDeviceToHost)) != cudaSuccess)
    {
    SENSEI_ERROR("Failed to read min from CUDA. "
      << cudaGetErrorString(ierr))
    return -1;
    }

  // compute the maximum taking into account ghost zones
  size_t maxId = thrust::reduce(indices.begin(), indices.end(), 0,
    indirectGhostMax<data_t>(pdata.get(), pGhosts.get()));

  if ((ierr = cudaMemcpy(&dataMax, pdata.get() + maxId,
    sizeof(data_t), cudaMemcpyDeviceToHost)) != cudaSuccess)
    {
    SENSEI_ERROR("Failed to read max from CUDA"
      << cudaGetErrorString(ierr))
    return -1;
    }

  return 0;
}

/** Computes a histogram on the GPU. The histgoram must be pre-initialized to
 * zero multiple invokations of the kernel accumulate results for new data.
 *
 * @param[in] data      the array to calculate the histogram for
 * @param[in] ghosts    an array of 0 and non zero, 0 where data is valid
 * @param[in] nVals     the length of the array
 * @param[in] minVal    the minimum bin value
 * @param[in] width     the width of histogram bins
 * @param[in] nBins     the number of bins + 1.
 * @param[in,out] hist  the histogram
 */
template <typename data_t>
__global__
void histogram(const data_t *data, const unsigned char *ghosts,
  size_t nVals, data_t minVal, data_t width, unsigned int *hist,
  size_t nBins)
{
  // per thread block local/temporary copy of the histogram.
  // this shared array must be allocated to store nBins values.
  // nBins is 1 longer than the desired output to handle binning
  // the maximum value.
  extern __shared__ unsigned int tmp[];

  unsigned long i = sensei::CUDAUtils::ThreadIdToArrayIndex();

  if (i >= nVals)
    return;

  // initialize the per thread block local histogram
  if (threadIdx.x < nBins)
    tmp[threadIdx.x] = 0u;

  __syncthreads();

  // find the bin for this value
  unsigned long j = (data[i] - minVal) / width;

  // update the bin count if the data point is not from a ghost zone
  unsigned int inc_valid = ghosts[i] ? 0 : 1;
  atomicAdd(&(tmp[j]), inc_valid);

#if SENSEI_DEBUG > 1
  printf("data[%lu] = %g bin %lu ghost %c inc_valid %u tmp[%lu] = %u\n",
    i, data[i], j, ghosts[i], inc_valid, j, tmp[j]);
#endif

  __syncthreads();

  // finalize from per thread block local results into the global output array
  if (threadIdx.x < nBins)
    atomicAdd(&(hist[threadIdx.x]), tmp[threadIdx.x]);
}

/** launch the histogram kernel */
template <typename data_t>
int block_local_histogram(const data_t *data, const unsigned char *ghosts,
  size_t nVals, data_t minVal, data_t width, unsigned int *hist,
  size_t nBins)
{
  // determine kernel launch parameters
  dim3 blockGrid;
  int nBlocks = 0;
  dim3 threadGrid;
  if (sensei::CUDAUtils::PartitionThreadBlocks(0, nVals,
      8, blockGrid, nBlocks, threadGrid))
  {
      SENSEI_ERROR("Failed to partition thread blocks")
      return -1;
  }

  // this is the ammount of shared memory we need for the kernel
  size_t histBytes = nBins*sizeof(unsigned int);

  // compute the histgram for this block's worth of data on the GPU. It is
  // left on the GPU until data for all blocks has been processed.
  histogram<<<blockGrid, threadGrid, histBytes>>>(
      data, ghosts, nVals, minVal, width, hist, nBins);

  cudaDeviceSynchronize();

  return 0;
}
}
#endif

namespace HistogramInternalsCPU
{
/** Computes a histogram on the CPU. The histgoram must be pre-initialized to
 * zero multiple invokations of the kernel accumulate results for new data.
 *
 * @param[in] data      the array to calculate the histogram for
 * @param[in] ghosts    an array of 0 and non zero, 0 where data is valid
 * @param[in] nVals     the length of the array
 * @param[in] minVal    the minimum bin value
 * @param[in] width     the width of histogram bins
 * @param[in] nBins     the number of bins + 1.
 * @param[in,out] hist  the histogram
 */
template <typename data_t>
void block_local_histogram(const data_t *data, const unsigned char *ghosts,
  size_t nVals, data_t minVal, data_t width, unsigned int *hist,
  size_t nBins)
{
  (void) nBins;

  for (size_t i = 0; i < nVals; ++i)
    {
    // find the bin for this value
    size_t j = (data[i] - minVal) / width;

    // update the bin count if the data point is not from a ghost zone
    unsigned int inc_valid = ghosts[i] ? 0 : 1;
    hist[j] += inc_valid;
    }
}
}

// --------------------------------------------------------------------------
HistogramInternals::~HistogramInternals()
{}

// --------------------------------------------------------------------------
int HistogramInternals::Clear()
{
  this->Min = std::numeric_limits<double>::max();
  this->Max = std::numeric_limits<double>::lowest();
  this->Width = 1.0;
  this->DataCache.clear();
  this->GhostCache.clear();
  this->Histogram = nullptr;
  return 0;
}

// --------------------------------------------------------------------------
int HistogramInternals::Initialize()
{
  return this->Clear();
}

// --------------------------------------------------------------------------
int HistogramInternals::AddLocalData(svtkDataArray *da,
  svtkUnsignedCharArray *ghosts)
{
  // validate the input
  if (!da)
    {
    SENSEI_ERROR("AddLocalData failed, null data array")
    return -1;
    }

  if (da->GetNumberOfComponents() != 1)
    {
    SENSEI_ERROR("Histogram on array \""
      << (da->GetName() ? da->GetName() : "")
      << "\" cannot be computed because the array has "
      << da->GetNumberOfComponents() << " components")
    return -1;
    }

  // if ghost zones were provided use them, otherwise generate
  size_t nVals = da->GetNumberOfTuples();
  std::shared_ptr<const unsigned char> pGhosts;
  if (ghosts)
    {
    // we have ghosts
#if defined(SENSEI_ENABLE_CUDA)
    if (this->DeviceId >= 0)
      {
#if defined(SENSEI_DEBUG)
      std::cerr << "HistogramInternals::AddLocalData ghosts CUDA" << std::endl;
#endif
      // make the requested GPU the active one
      sensei::CUDAUtils::SetDevice(this->DeviceId);

      // get a pointer accessible on the GPU
      if (dynamic_cast<svtkHAMRDataArray<unsigned char>*>(((svtkDataArray*)ghosts)))
        {
        svtkHAMRDataArray<unsigned char> *tGhosts =
          static_cast<svtkHAMRDataArray<unsigned char>*>(((svtkDataArray*)ghosts));

        pGhosts = tGhosts->GetCUDAAccessible();
        }
      else
        {
        pGhosts = sensei::MemoryUtils::MakeCudaAccessible(ghosts->GetPointer(0), nVals);
        }
      }
    else
      {
#endif
#if defined(SENSEI_DEBUG)
      std::cerr << "HistogramInternals::AddLocalData ghosts CPU" << std::endl;
#endif
      // get a pointer accessible on the CPU
      if (dynamic_cast<svtkHAMRDataArray<unsigned char>*>((svtkDataArray*)ghosts))
        {
        svtkHAMRDataArray<unsigned char> *tGhosts =
          static_cast<svtkHAMRDataArray<unsigned char>*>((svtkDataArray*)ghosts);

        pGhosts = tGhosts->GetCPUAccessible();
        }
      else
        {
        pGhosts = sensei::MemoryUtils::MakeCpuAccessible(ghosts->GetPointer(0), nVals);
        }
#if defined(SENSEI_ENABLE_CUDA)
      }
#endif
    }
  else
    {
    // we don't have ghosts
#if defined(SENSEI_ENABLE_CUDA)
    if (this->DeviceId >= 0)
      {
#if defined(SENSEI_DEBUG)
      std::cerr << "HistogramInternals::AddLocalData ghosts were not provided,"
          " allocating on CUDA" << std::endl;
#endif
      // make the requested GPU the active one
      sensei::CUDAUtils::SetDevice(this->DeviceId);

      // generate ghosts on the GPU, and get a shared pointer to the buffer
      auto tGhosts = svtkHAMRUnsignedCharArray::New("vtkGhostType",
        nVals, 1, svtkAllocator::cuda_async, svtkStream(), svtkStreamMode::sync_cpu, 0);

      pGhosts = tGhosts->GetDataPointer();

      tGhosts->Delete();
      }
    else
      {
#endif
#if defined(SENSEI_DEBUG)
      std::cerr << "HistogramInternals::AddLocalData ghosts were not provided,"
        " allocating on the CPU" << std::endl;
#endif
      // generate ghosts on the CPU, and get a shared pointer to the buffer
      auto tGhosts = svtkHAMRUnsignedCharArray::New("vtkGhostType",
        nVals, 1, svtkAllocator::malloc, svtkStream(), svtkStreamMode::sync, 0);

      pGhosts = tGhosts->GetDataPointer();

      tGhosts->Delete();
#if defined(SENSEI_ENABLE_CUDA)
      }
#endif
    }

  // cache the GPU accessible pointer for use in the histogram calculation
  this->GhostCache[da] = pGhosts;

  // compute the block min and max
  switch (da->GetDataType())
    {
    svtkTemplateMacro(

      std::shared_ptr<const SVTK_TT> pDa;
#if defined(SENSEI_ENABLE_CUDA)
      if (this->DeviceId >= 0)
        {
#if defined(SENSEI_DEBUG)
        std::cerr << "HistogramInternals::AddLocalData "
          << (da->GetName() ? da->GetName() : "\"\"") << " CUDA" << std::endl;
#endif
        // get a pointer to the data that's usable on the GPU
        if (dynamic_cast<svtkHAMRDataArray<SVTK_TT>*>(da))
          {
          auto tDa = static_cast<svtkHAMRDataArray<SVTK_TT>*>(da);
          pDa = tDa->GetCUDAAccessible();
          }
        else
          {
          pDa = sensei::MemoryUtils::MakeCudaAccessible(
            sensei::SVTKUtils::GetPointer<SVTK_TT>(da), nVals);
          }
        }
      else
        {
#endif
#if defined(SENSEI_DEBUG)
        std::cerr << "HistogramInternals::AddLocalData "
          << (da->GetName() ? da->GetName() : "\"\"") << " CPU" << std::endl;
#endif
        if (dynamic_cast<svtkHAMRDataArray<SVTK_TT>*>(da))
          {
          auto tDa = static_cast<svtkHAMRDataArray<SVTK_TT>*>(da);
          pDa = tDa->GetCPUAccessible();
          }
        else
          {
          pDa = sensei::MemoryUtils::MakeCpuAccessible(
            sensei::SVTKUtils::GetPointer<SVTK_TT>(da), nVals);
          }
#if defined(SENSEI_ENABLE_CUDA)
        }
#endif
      // cache the GPU accessible pointer for use in the histogram calculation
      this->DataCache[da] = pDa;
    );
    default:
      {
      SENSEI_ERROR("Unsupported dispatch " << da->GetClassName());
      return -1;
      }
    }

  return 0;
}

// --------------------------------------------------------------------------
int HistogramInternals::ComputeRange()
{
  this->Min = std::numeric_limits<double>::max();
  this->Max = std::numeric_limits<double>::lowest();

  auto dit = this->DataCache.begin();
  auto git = this->GhostCache.begin();

  for (; dit != this->DataCache.end(); ++dit, ++git)
    {
    // get the data array. arrays in the cache have already been moved to the
    // GPU if that was neccessary.
    std::shared_ptr<const unsigned char> pGhosts = git->second;

    svtkDataArray *da = dit->first;
    std::shared_ptr<const void> pvDa = dit->second;

    size_t nVals = da->GetNumberOfTuples();

    // compute the block min and max
    switch (da->GetDataType())
      {
      svtkTemplateMacro(

        SVTK_TT blockMin = std::numeric_limits<SVTK_TT>::max();
        SVTK_TT blockMax = std::numeric_limits<SVTK_TT>::lowest();

        // cast to the correct type. The data will already be in the right place
        // data movement is handled in AddLocalData
        std::shared_ptr<const SVTK_TT> pDa = std::static_pointer_cast<const SVTK_TT>(pvDa);

#if defined(SENSEI_ENABLE_CUDA)
        if (this->DeviceId >= 0)
          {
          // make the requested GPU the active one
          sensei::CUDAUtils::SetDevice(this->DeviceId);
          // calculate range taking into account ghost zones on the GPU
          HistogramInternalsCUDA::ComputeRange<SVTK_TT>(pDa, pGhosts, nVals, blockMin, blockMax);
#if defined(SENSEI_DEBUG)
          std::cerr << "HistogramInternals::ComputeRange CUDA ["
             << blockMin << ", " << blockMax << "]" << std::endl;
#endif
          }
        else
          {
#endif
          // calculate range taking into account ghost zones on the CPU
          const SVTK_TT *rpDa = pDa.get();
          const unsigned char *rpGhosts = pGhosts.get();
          for (size_t i = 0; i < nVals; ++i)
            {
            if (rpGhosts[i] == 0)
              {
              SVTK_TT value = rpDa[i];
              blockMin = std::min(blockMin, value);
              blockMax = std::max(blockMax, value);
              }
            }
#if defined(SENSEI_DEBUG)
          std::cerr << "HistogramInternals::ComputeRange CPU ["
             << blockMin << ", " << blockMax << "]" << std::endl;
#endif
#if defined(SENSEI_ENABLE_CUDA)
          }
#endif
        // accumulate the min/max
        this->Min = std::min(this->Min, double(blockMin));
        this->Max = std::max(this->Max, double(blockMax));
        );
      default:
        {
        SENSEI_ERROR("Unsupported dispatch " << da->GetClassName());
        return -1;
        }
      }
    }

  // check the result
  if (fabs(this->Max - this->Min) < 1.0e-6)
    {
    SENSEI_ERROR("Invalid range detected ["
      << this->Min << ", " << this->Max << "]")
    return -1;
    }

  // compute the min and max across all MPI ranks
  double gMin = 0.0;
  double gMax = 0.0;

  MPI_Allreduce(&this->Min, &gMin, 1, MPI_DOUBLE, MPI_MIN, this->Comm);
  MPI_Allreduce(&this->Max, &gMax, 1, MPI_DOUBLE, MPI_MAX, this->Comm);

  this->Min = gMin;
  this->Max = gMax;

#if defined(SENSEI_DEBUG)
   std::cerr << "HistogramInternals::ComputeRange global range ["
      << gMin << ", " << gMax << "]" << std::endl;
#endif

  return 0;
}

// --------------------------------------------------------------------------
int HistogramInternals::ComputeHistogram()
{
  if (this->ComputeRange()
    || this->InitializeHistogram()
    || this->ComputeLocalHistogram()
    || this->FinalizeHistogram())
    return -1;
  return 0;
}

// --------------------------------------------------------------------------
int HistogramInternals::InitializeHistogram()
{
  // now with the min and amax in hand we can calculate the bin width.
  this->Width = (this->Max - this->Min) / this->NumberOfBins;

  // allocate space for the histogram and initialize the first time
  // through. NOTE: There is an extra bin allocated to deal with out-of-bounds
  // when binning the maximum value. This bin is merged in after the calculations
  size_t nBins = this->NumberOfBins + 1;
  size_t histBytes = nBins*sizeof(unsigned int);
  unsigned int *pHist = nullptr;

#if defined(SENSEI_ENABLE_CUDA)
  if (this->DeviceId >= 0)
    {
#if defined(SENSEI_DEBUG)
    std::cerr << "InitializeHistogram initializing "
      << this->NumberOfBins << " bins on the GPU" << std::endl;
#endif
    // make the requested GPU the active one
    sensei::CUDAUtils::SetDevice(this->DeviceId);

    cudaError_t ierr = cudaSuccess;
    if ((ierr = cudaMalloc(&pHist, histBytes)) != cudaSuccess)
    {
        SENSEI_ERROR("Failed to allocate space for the histogram on the GPU. "
            << cudaGetErrorString(ierr))
        return -1;
    }

    if ((ierr = cudaMemset(pHist, 0, histBytes)) != cudaSuccess)
    {
        SENSEI_ERROR("Failed to initialize the histogram on the GPU. "
            << cudaGetErrorString(ierr))
        return -1;
    }

    // save the pointer for calculations of subsequent blocks
    this->Histogram = std::shared_ptr<unsigned int>(pHist,
      sensei::MemoryUtils::FreeCudaPtr);
    }
  else
    {
#endif
#if defined(SENSEI_DEBUG)
    std::cerr << "InitializeHistogram initializing "
      << this->NumberOfBins << " bins on the CPU" << std::endl;
#endif
    pHist = (unsigned int*)malloc(histBytes);
    memset(pHist, 0, histBytes);

    // save the pointer for calculations of subsequent blocks
    this->Histogram = std::shared_ptr<unsigned int>(pHist,
      sensei::MemoryUtils::FreeCpuPtr);
#if defined(SENSEI_ENABLE_CUDA)
    }
#endif

  return 0;
}

// --------------------------------------------------------------------------
int HistogramInternals::ComputeLocalHistogram()
{
  // validate the histogram. it should have been pre-allocated
  if (!this->Histogram)
    {
    SENSEI_ERROR("Histogram was not pre-allocated. Did you forget"
      " to call InitializeHistogram?")
    return -1;
    }

  auto dit = this->DataCache.begin();
  auto git = this->GhostCache.begin();

  for (; dit != this->DataCache.end(); ++dit, ++git)
    {
    // get the data array. arrays in the cache have already been moved to the
    // GPU if that was neccessary.
    std::shared_ptr<const unsigned char> pGhosts = git->second;

    svtkDataArray *da = dit->first;
    std::shared_ptr<const void> pDa = dit->second;

    // get the sizes of the datat array
    size_t nVals = da->GetNumberOfTuples();

    // get the size of per thread block shared memory. NOte an extra bin is used
    // to handle binning of the maximum value. it is merged after the calculations
    // complete.
    size_t nBins = this->NumberOfBins + 1;

    switch (da->GetDataType())
      {
      svtkTemplateMacro(
#if defined(SENSEI_ENABLE_CUDA)
        if (this->DeviceId >= 0)
          {
#if defined(SENSEI_DEBUG)
          std::cerr << "HistogramInternals::ComputeLocalHistogram CUDA" << std::endl;
#endif
          // make the requested GPU the active one
          sensei::CUDAUtils::SetDevice(this->DeviceId);

          // compute the histgram for this block's worth of data on the GPU. It is
          // left on the GPU until data for all blocks has been processed.
          // data is already in the right place, it is moved in AddLocalData
          if (HistogramInternalsCUDA::block_local_histogram<SVTK_TT>((SVTK_TT*)pDa.get(),
            pGhosts.get(), nVals, this->Min, this->Width, this->Histogram.get(), nBins))
            return -1;
          }
        else
          {
#endif
#if defined(SENSEI_DEBUG)
          std::cerr << "HistogramInternals::ComputeLocalHistogram CPU" << std::endl;
#endif
          // compute the histgram for this block's worth of data on the CPU
          // data is already in the right place, it is moved in AddLocalData
          HistogramInternalsCPU::block_local_histogram<SVTK_TT>((SVTK_TT*)pDa.get(),
            pGhosts.get(), nVals, this->Min, this->Width, this->Histogram.get(), nBins);
#if defined(SENSEI_ENABLE_CUDA)
          }
#endif
        );
      default:
        {
        SENSEI_ERROR("Unsupported dispatch " << da->GetClassName());
        return -1;
        }
      }
    }
  return 0;
}

// --------------------------------------------------------------------------
int HistogramInternals::FinalizeHistogram()
{
#if defined(SENSEI_DEBUG)
  std::cerr << "HistogramInternals::FinalizeHistogram" << std::endl;
#endif

  int rank = 0;
  MPI_Comm_rank(this->Comm, &rank);

  size_t nBins = this->NumberOfBins + 1;
  size_t histBytes = nBins*sizeof(unsigned int);

#if defined(SENSEI_ENABLE_CUDA)
  // make the requested GPU the active one
  if (this->DeviceId >= 0)
    sensei::CUDAUtils::SetDevice(this->DeviceId);
#endif

  // fetch result from the GPU for the MPI parallel part of the reduction
  // this call synchronizes CUDA kernels
  std::shared_ptr<unsigned int> pHist =
    sensei::MemoryUtils::MakeCpuAccessible(this->Histogram.get(), nBins);

  // allocate a buffer on teh CPU for the result of the MPI parallel reduction
  // the result of the histogram is always coppied to the CPU
  unsigned int *tmp = nullptr;
  if (rank == 0)
    {
    tmp = (unsigned int *)malloc(histBytes);
    memset(tmp, 0, histBytes);
    }

  // finalize the histogram calculation by summing up contributions from each
  // MPI rank to MPI rank 0
  MPI_Reduce(pHist.get(), tmp, nBins, MPI_UNSIGNED, MPI_SUM, 0, this->Comm);

  // merge in the extra bin (see earlier comments)
  if (rank == 0)
    tmp[this->NumberOfBins - 1] += tmp[this->NumberOfBins];

  // Replace the internal copy of the histogram with the finalized result.
  // only MPI rank 0 has the result after this
#if defined(SENSEI_CLANG_CUDA)
  // work around an issue on clang17 cuda wrappers as of June 9 2023
  this->Histogram = std::shared_ptr<unsigned int>(tmp,
    (void (*)(void *) noexcept(true)) free);
#else
  this->Histogram = std::shared_ptr<unsigned int>(tmp, free);
#endif

  return 0;
}

// --------------------------------------------------------------------------
int HistogramInternals::GetHistogram(int &nBins, double &binMin, double &binMax,
  double &binWidth, std::vector<unsigned int> &histogram)
{
  int rank = 0;
  MPI_Comm_rank(this->Comm, &rank);

  if (rank == 0)
    {
    if (!this->Histogram)
      {
      SENSEI_ERROR("Failed calculation detected. MPI rank 0 has no histogram to return.")
      return -1;
      }

    nBins = this->NumberOfBins;
    binMin = this->Min;
    binMax = this->Max;
    binWidth = this->Width;

    unsigned int *pHist = this->Histogram.get();
    histogram.assign(pHist, pHist + nBins);
    }

  return 0;
}

}
