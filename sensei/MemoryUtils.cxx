#include "MemoryUtils.h"
#include "senseiConfig.h"
#include "Error.h"

#include <iostream>

namespace sensei
{
namespace MemoryUtils
{

// **************************************************************************
void FreeCpuPtr(void *ptr)
{
#if defined(SENSEI_DEBUG)
    std::cerr << "FreeCpuPtr(" << ptr << ")" << std::endl;
#endif
    free(ptr);
}

// **************************************************************************
void DontFreePtr(void *ptr)
{
  (void) ptr;
#if defined(SENSEI_DEBUG)
  std::cerr << "DontFreePtr(" << ptr << ")" << std::endl;
#endif
}

#if !defined(SENSEI_ENABLE_CUDA)
// **************************************************************************
void FreeCudaPtr(void *ptr)
{
  (void) ptr;

  SENSEI_ERROR("CUDA pointer cannot be free'd"
   " because CUDA is not enabled")
}

// **************************************************************************
std::shared_ptr<void> MakeCudaAccessible_(void *ptr, size_t nBytes)
{
  (void) ptr;
  (void) nBytes;

  SENSEI_ERROR("Pointer cannot be accessed"
   " from CUDA because CUDA is not enabled")

  return nullptr;
}

// **************************************************************************
std::shared_ptr<void> MakeCpuAccessible_(void *ptr, size_t nBytes)
{
  (void) nBytes;

  // this pointer can be accessed on the CPU
  // return a shared pointer with no deleter
#if defined(SENSEI_DEBUG)
  std::cerr << "Pointer " << ptr << " already is on the CPU" << std::endl;
#endif
  return std::shared_ptr<void>(ptr, MemoryUtils::DontFreePtr);
}
#endif

}
}
