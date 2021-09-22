#include "BlockInternals.h"

namespace BlockInternals
{
#if defined(OSCILLATOR_CUDA)
namespace CUDA
{
// **************************************************************************
/// calculate oscillator contributions on the GPU
__global__
void UpdateFields(
  float t,
  const Oscillator *oscillators,
  int nOscillators,
  int ni, int nj, int nk,
  int i0, int j0, int k0,
  float x0, float y0, float z0,
  float dx, float dy, float dz,
  float *pdata)
{
    unsigned long ii = threadIdx.x + blockDim.x*blockIdx.x;

    unsigned long nVals = ni*nj*nk;

    if (ii >= nVals)
      return;

    // 1D index to 3D index
    long nij = ni*nj;
    long k = ii / nij;
    long j = (ii - k * nij) / ni;;
    long i = ii % ni;

    // mesh position
    float x = x0 + dx*float(i0 + i);
    float y = y0 + dy*float(j0 + j);
    float z = z0 + dz*float(k0 + k);

    // evaluate the oscillators
    pdata[ii] = float(0);

    for (int q = 0; q < nOscillators; ++q)
      pdata[ii] += oscillators[q].evaluate(x,y,z, t);

    /*printf("ii=%lu i=%ld j=%ld k=%ld x=%g y=%g z=%g data[ii]=%g \n",
      ii,  i, j, k,  x, y, z,  pdata[ii]);*/
}
}
#endif

namespace CPU
{
/// calculate oscillator contributions on the CPU
void UpdateFields(
  float t,
  const Oscillator *oscillators,
  int nOscillators,
  int ni, int nj, int nk,
  int i0, int j0, int k0,
  float x0, float y0, float z0,
  float dx, float dy, float dz,
  float *pdata)
{
    int nij = ni*nj;

    for (int k = 0; k < nk; ++k)
    {
        float z = z0 + dz*(k0 + k);
        float *pdk = pdata + k*nij;
        for (int j = 0; j < nj; ++j)
        {
            float y = y0 + dy*(j0 + j);
            float *pd = pdk + j*ni;
            for (int i = 0; i < ni; ++i)
            {
                float x = x0 + dx*(i0 + i);
                pd[i] = 0.f;

                for (int q = 0; q < nOscillators; ++q)
                    pd[i] += oscillators[q].evaluate(x,y,z, t);
            }
        }
    }
}
}

// **************************************************************************
int UpdateFields(int deviceId, float t, const Oscillator *oscillators,
  int nOscillators, int ni, int nj, int nk, int i0, int j0, int k0,
  float x0, float y0, float z0, float dx, float dy, float dz, float *pdata)
{
  (void) deviceId;

#if defined(OSCILLATOR_CUDA)
  if (deviceId < 0)
  {
#endif
#if defined(OSCILLATOR_DEBUG)
    std::cerr << "BlockInternals::CPU::UpdateFields" << std::endl;
#endif
    // run on the CPU
    BlockInternals::CPU::UpdateFields(
      t, oscillators, nOscillators,
      ni,nj,nk, i0,j0,k0, x0,y0,z0, dx,dy,dz, pdata);
#if defined(OSCILLATOR_CUDA)
  }
  else
  {
#if defined(OSCILLATOR_DEBUG)
    std::cerr << "BlockInternals::CUDA::UpdateFields deviceId=" << deviceId << std::endl;
#endif
    // set the device
    cudaError_t ierr = cudaSuccess;
    if ((ierr = cudaSetDevice(deviceId)) != cudaSuccess)
    {
        std::cerr << "Failed to set the CUDA device to " << deviceId
            << ". " << cudaGetErrorString(ierr) << std::endl;
        return -1;
    }

    cudaDeviceSynchronize();

    // determine kernel launch parameters
    size_t nVals = ni*nj*nk;
    int nWarpsPerBlock = 16;
    int warpSize = 32;
    int threadsPerBlock = nWarpsPerBlock*warpSize;
    int nBlocks = nVals / threadsPerBlock;
    if (nVals % threadsPerBlock)
        nBlocks += 1;
    dim3 threadGrid(threadsPerBlock);
    dim3 blockGrid(nBlocks);

#if defined(OSCILLATOR_DEBUG)
    std::cerr << "launching the kernel on " << nBlocks
      << " blocks each with " << threadsPerBlock << " threads" << std::endl;
#endif

    // launch the kernel
    BlockInternals::CUDA::UpdateFields<<<blockGrid, threadGrid>>>(t,
      oscillators, nOscillators, ni,nj,nk, i0,j0,k0, x0,y0,z0, dx,dy,dz,
      pdata);

    cudaDeviceSynchronize();
  }
#endif
  return 0;
}
}
