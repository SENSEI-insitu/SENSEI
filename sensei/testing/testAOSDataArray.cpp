#include "svtkHAMRDataArray.h"
#include "svtkImageData.h"
#include "svtkPointData.h"

#include "senseiConfig.h"

#if defined(ENABLE_CUDA)
#include <cuda.h>
#include <cuda_runtime.h>

template <typename T>
__global__
void initialize(T *ptr, size_t nTups, size_t nComps)
{
    size_t i = threadIdx.x + blockDim.x*(blockIdx.x + blockIdx.y * gridDim.x
        + blockIdx.z * gridDim.x * gridDim.y);

    if (i >= nTups)
        return;

    for (int j = 0; j < nComps; ++j)
    {
        ptr[nComps*i + j] = nComps*i + j;
    }
}

template<typename T>
svtkHAMRDataArray<T> *initializeCUDA(size_t nTups, size_t nComps)
{
    T *ptr = nullptr;
    cudaMalloc(&ptr, nTups*nComps*sizeof(T));

    dim3 thread_grid(128,1,1);
    dim3 block_grid(nTups / 128 + (nTups % 128 ? 1 : 0), 1, 1);

    initialize<<<block_grid, thread_grid>>>(ptr, nTups, nComps);

    svtkHAMRDataArray<T> *da = svtkHAMRDataArray<T>::New("foo", ptr, nTups,
                                   nComps, svtkAllocator::cuda, 0, 1);

    return da;
}

#endif

template <typename T>
svtkHAMRDataArray<T> *initializeCPU(size_t nTups, size_t nComps)
{
    T *ptr = (T*)malloc(nTups*nComps*sizeof(T));
    for (size_t i = 0; i < nTups; ++i)
    {
        for (size_t j = 0; j < nComps; ++j)
        {
            ptr[nComps*i + j] = nComps*i + j;
        }
    }

    svtkHAMRDataArray<T> *da = svtkHAMRDataArray<T>::New("foo", ptr, nTups,
                                   nComps, svtkAllocator::malloc, -1, 1);

    return da;
}



int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "usage: testAOSDataArray [use cuda]" << std::endl;
        return -1;
    }

    int device = atoi(argv[1]);

    int nx = 2;
    int ny = 2;
    int nz = 1;

    size_t nTups = nx*ny*nz;
    int nComps = 2;

    svtkImageData *id = svtkImageData::New();
    id->SetDimensions(nx, ny, nz);

    svtkHAMRDataArray<double> *da = nullptr;
#if defined(ENABLE_CUDA)
    if (device >= 0)
    {
        da = initializeCUDA<double>(nTups, nComps);
    }
    else
#endif
    {
        da = initializeCPU<double>(nTups, nComps);
    }


    da->Print(std::cerr);

    id->GetPointData()->AddArray(da);

    da->Delete();

    id->Print(std::cerr);

    id->Delete();

    return 0;
}
