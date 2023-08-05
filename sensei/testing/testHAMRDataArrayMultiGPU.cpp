#include "svtkHAMRDataArray.h"
#include "svtkImageData.h"
#include "svtkPointData.h"

#include "senseiConfig.h"

#if defined(SENSEI_ENABLE_CUDA)
#include <cuda.h>
#include <cuda_runtime.h>
#endif


int main(int argc, char **argv)
{
    /*if (argc != 2)
    {
        std::cerr << "usage: testAOSDataArray [device id]" << std::endl;
        return -1;
    }*/

    size_t nTups = 64;

    int nDev = 0;
    cudaGetDeviceCount(&nDev);

    std::cerr << "a total of " << nDev << " devices are present" << std::endl;

    if (nDev < 4)
    {
        std::cerr << "test skipped: 4 devices are needed for this test" << std::endl;
        return 0;
    }

    // create an array on GPU 0
    cudaSetDevice(0);

    cudaStream_t s0 = cudaStreamPerThread;
    cudaStreamCreate(&s0);

    auto a0 = svtkHAMRDoubleArray::New("host", nTups, 1, svtkAllocator::cuda_async,
                                       s0, svtkStreamMode::async, -3.14);

    a0->Synchronize();

    // copy construct on GPU 1
    cudaSetDevice(1);

    cudaStream_t s1 = cudaStreamPerThread;
    cudaStreamCreate(&s1);

    auto a1 = svtkHAMRDoubleArray::New(a0, svtkAllocator::cuda_async,
                                       s1, svtkStreamMode::async);

    a1->Synchronize();

    // copy construct on GPU 2
    cudaSetDevice(2);

    cudaStream_t s2 = cudaStreamPerThread;
    cudaStreamCreate(&s2);

    auto a2 = svtkHAMRDoubleArray::New(a1, svtkAllocator::cuda_async,
                                       s2, svtkStreamMode::async);

    a2->Synchronize();

    // copy construct on host
    cudaStream_t s3 = cudaStreamPerThread;
    cudaStreamCreate(&s3);

    auto a3 = svtkHAMRDoubleArray::New(a2, svtkAllocator::cuda_host,
                                       s3, svtkStreamMode::async);

    a3->Synchronize();


    // print
    auto sp3 = a3->GetHostAccessible();
    auto p3 = sp3.get();

    a3->Synchronize();

    for (int i = 0; i < nTups; ++i)
    {
        std::cerr << p3[i] << ", ";
    }
    std::cerr << std::endl;


    // clean up
    a0->Delete();
    a1->Delete();
    a2->Delete();
    a3->Delete();

    cudaStreamDestroy(s0);
    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);
    cudaStreamDestroy(s3);

    return 0;
}
