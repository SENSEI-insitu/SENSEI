#include "svtkHAMRDataArray.h"
#include "svtkImageData.h"
#include "svtkPointData.h"

#include "senseiConfig.h"

#if defined(SENSEI_ENABLE_CUDA)
#include <cuda.h>
#include <cuda_runtime.h>
#endif


void print(std::ostream & os, svtkHAMRDoubleArray *da)
{
    // get a view of the data on the host
    auto spDa = da->GetHostAccessible();

    size_t nElem = da->GetNumberOfTuples();
    auto pDa = spDa.get();

    os << da->GetName() << "(owner=" << da->GetOwner() << ") = ";

    da->Synchronize();

    os << pDa[0];
    for (size_t i = 1; i < nElem; ++i)
        os << ", " << pDa[i];

    os << std::endl;
}



int main(int argc, char **argv)
{
    size_t nTups = 64;

    int nDev = 0;
    cudaGetDeviceCount(&nDev);

    std::cerr << "a total of " << nDev << " devices are present" << std::endl;

    std::cerr << "creating some data on device 0 ... ";

    // create an array on GPU 0
    cudaSetDevice(0);
    cudaStream_t s0 = cudaStreamPerThread;
    cudaStreamCreate(&s0);

    auto a0 = svtkHAMRDoubleArray::New("data", nTups, 1, svtkAllocator::cuda_async,
                                       s0, svtkStreamMode::async, -3.14);

    // make sure data is ready
    a0->Synchronize();

    std::cerr << "OK!" << std::endl;

    // copy construct on each device
    cudaStream_t strm[nDev];
    for (int i = 0; i < nDev; ++i)
      strm[i] = cudaStreamPerThread;

    svtkHAMRDoubleArray *src = a0;
    svtkHAMRDoubleArray *dest = nullptr;

    for (int i = 0; i < nDev; ++i)
    {
      std::cerr << "copy construct on device " << i << " ... ";

      // copy construct on GPU i
      cudaSetDevice(i);
      cudaStreamCreate(&strm[i]);
      dest = svtkHAMRDoubleArray::New(src, svtkAllocator::cuda_async,
                                      strm[i], svtkStreamMode::async);

      // make sure copy finished
      dest->Synchronize();

      // get rid of the source
      src->Delete();

      std::cerr << "OK!" << std::endl;

      // print the data
      print(std::cerr, dest);
      /*dest->Print(std::cerr);
      dest->Synchronize();*/

      src = dest;
    }

    // copy construct on host
    //cudaSetDevice(0);
    cudaStream_t s2 = cudaStreamPerThread;
    cudaStreamCreate(&s2);

    auto a2 = svtkHAMRDoubleArray::New(dest, svtkAllocator::cuda_host,
                                       s2, svtkStreamMode::async);

    // make sure data movement finished
    a2->Synchronize();

    // print the data
    a2->Print(std::cerr);
    a2->Synchronize();

    // clean up
    dest->Delete();

    cudaStreamDestroy(s0);
    cudaStreamDestroy(s2);
    for (int i = 0; i < nDev; ++i)
      cudaStreamDestroy(strm[i]);

    return 0;
}
