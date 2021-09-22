#include "MemoryUtils.h"
#include "senseiConfig.h"
#include "Error.h"
#include "CUDAUtils.h"

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

namespace sensei
{
namespace MemoryUtils
{
// **************************************************************************
int cudaAccessible(const void *ptr)
{
    // this requires at least CUDA 11
    cudaError_t ierr = cudaSuccess;
    cudaPointerAttributes ptrAtts;
    ierr = cudaPointerGetAttributes(&ptrAtts, ptr);
    cudaGetLastError();

    // these types of pointers are NOT accessible on the GPU
    // cudaErrorInValue occurs when the pointer is unknown to CUDA, as is
    // the case with pointers allocated by malloc or new.
    if ((ierr == cudaErrorInvalidValue)
      || ((ierr == cudaSuccess) && ((ptrAtts.type == cudaMemoryTypeHost)
      || (ptrAtts.type == cudaMemoryTypeUnregistered))))
      return 0;

    // an error occurred
    if (ierr != cudaSuccess)
      {
      SENSEI_ERROR("Failed to get pointer attributes " << ptr
        << ". " << cudaGetErrorString(ierr))
      return 0;
      }

    return 1;
}

// **************************************************************************
int cpuAccessible(const void *ptr)
{
    // this requires at least CUDA 11
    cudaError_t ierr = cudaSuccess;
    cudaPointerAttributes ptrAtts;
    ierr = cudaPointerGetAttributes(&ptrAtts, ptr);
    cudaGetLastError();

    // cudaErrorInValue occurs when the pointer is unknown to CUDA, as is
    // the case with pointers allocated by malloc or new.
    if (ierr == cudaErrorInvalidValue)
      return 1;

    if (ierr != cudaSuccess)
      {
      SENSEI_ERROR("Failed to get pointer attributes " << ptr
        << ". " << cudaGetErrorString(ierr))
      return 0;
      }

    // these types of pointers are NOT accessible on the CPU
    if (ptrAtts.type == cudaMemoryTypeDevice)
      return 0;

    return 1;
}

// **************************************************************************
void FreeCudaPtr(void *ptr)
{
#if defined(SENSEI_DEBUG)
    std::cerr << "FreeCudaPtr(" << ptr << ")" << std::endl;
#endif
    cudaFree(ptr);
}

// **************************************************************************
std::shared_ptr<void> MakeCudaAccessible_(void *ptr, size_t nBytes)
{
  if (!cudaAccessible(ptr))
    {
#if defined(SENSEI_DEBUG)
    std::cerr << "Moving " << ptr
      << " " << nBytes << " to CUDA" << std::endl;
#endif
    // not on CUDA, move to CUDA
    void *devPtr = nullptr;
    cudaError_t ierr = cudaSuccess;

    if ((ierr = cudaMalloc(&devPtr, nBytes)) != cudaSuccess)
      {
      SENSEI_ERROR("Failed to allocate a buffer of "
        << nBytes << " on CUDA. " << cudaGetErrorString(ierr))
      return nullptr;
      }

    if ((ierr = cudaMemcpy(devPtr, ptr, nBytes,
      cudaMemcpyHostToDevice)) != cudaSuccess)
      {
      SENSEI_ERROR("Failed to copy data to CUDA. "
        << cudaGetErrorString(ierr))
      return nullptr;
      }

    return std::shared_ptr<void>(devPtr, sensei::MemoryUtils::FreeCudaPtr);
    }

  // this pointer can be accessed on CUDA
  // return a shared pointer with no deleter
#if defined(SENSEI_DEBUG)
  std::cerr << "Pointer " << ptr << " already is on CUDA" << std::endl;
#endif
  return std::shared_ptr<void>(ptr, sensei::MemoryUtils::DontFreePtr);
}

// **************************************************************************
std::shared_ptr<void> MakeCpuAccessible_(void *ptr, size_t nBytes)
{
  if (!cpuAccessible(ptr))
    {
    // not on the CPU, move to the CPU
#if defined(SENSEI_DEBUG)
    std::cerr << "Moving " << ptr << " " << nBytes << " to the CPU" << std::endl;
#endif

    cudaError_t ierr = cudaSuccess;

    // synchronize here to make certain that all operations on the data have
    // completed before fetching data from the CPU
    if ((ierr = cudaDeviceSynchronize()))
      {
      SENSEI_ERROR("Error detected during device synchronization. "
        << cudaGetErrorString(ierr))
      return nullptr;
      }

    // allocate a buffer on the CPU
    void *cpuPtr = (void*)malloc(nBytes);

    // copy the data from the device
    if ((ierr = cudaMemcpy(cpuPtr, ptr, nBytes,
      cudaMemcpyDeviceToHost)) != cudaSuccess)
      {
      SENSEI_ERROR("Failed to move data to the CPU. "
        << cudaGetErrorString(ierr))
      return nullptr;
      }

    // return a shared pointer that will free up the CPU buffer
    return std::shared_ptr<void>(cpuPtr, MemoryUtils::FreeCpuPtr);
    }

  // this pointer can be accessed on the CPU
#if defined(SENSEI_DEBUG)
  std::cerr << "Pointer " << ptr << " already is on the CPU" << std::endl;
#endif
  // return a shared pointer with a noop deleter
  return std::shared_ptr<void>(ptr, MemoryUtils::DontFreePtr);
}
}
}
