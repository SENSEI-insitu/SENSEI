#ifndef MemoryUtils_h
#define MemoryUtils_h

///@file

#include <memory>

//#define SENSEI_DEBUG 1

/// SENSEI
namespace sensei
{
/// Functions for dealing with memory access across heterogeneous architectures
namespace MemoryUtils
{

/// return true if the pointer is accessible by code running on a CUDA GPU
int CudaAccessible(const void *ptr);

/// return true if the pointer is accessible by code running on the CPU
int CpuAccessible(const void *ptr);

/// callback that can free memory managed by CUDA
void FreeCudaPtr(void *ptr);

/// callback that can free memory managed by malloc
void FreeCpuPtr(void *ptr);

/// A callback that does not free the pointer
void DontFreePtr(void *ptr);

/** @name MakedCudaAccessible
 * returns a pointer to data that is accessible from CUDA kernels. If the data
 * is already accessible, this call is a noop. On the other hand if the data is
 * not accessible it will be moved. The accessible data is returned as a shared
 * pointer so that in the case data needed to  be moved, the buffers on the GPU
 * will automatically be released.
 */
///@{
std::shared_ptr<void> MakeCudaAccessible_(void *ptr, size_t nBytes);

template <typename data_t>
std::shared_ptr<data_t> MakeCudaAccessible(data_t *ptr, size_t nVals)
{
  return std::static_pointer_cast<data_t>(
    sensei::MemoryUtils::MakeCudaAccessible_(
      (void*)ptr, nVals*sizeof(data_t)));
}
///@}

/** @name MakeCpuAccessible
 * returns a pointer to data that is accessible from CPU codes. If the data is
 * already accessible, this call is a noop. On the other hand if the data is
 * not accessible it will be moved. The accessible data is returned as a shared
 * pointer so that in the case data needed to  be moved, the buffers on the CPU
 * will automatically be released.
 */
///@{
std::shared_ptr<void> MakeCpuAccessible_(void *ptr, size_t nBytes);

template <typename data_t>
std::shared_ptr<data_t> MakeCpuAccessible(data_t *ptr, size_t nVals)
{
  return std::static_pointer_cast<data_t>(
    sensei::MemoryUtils::MakeCpuAccessible_(
      (void*)ptr, nVals*sizeof(data_t)));
}
///@}

}
}
#endif
