#ifndef CUDAUtils_h
#define CUDAUtils_h

/// @file

#include "senseiConfig.h"
#include "Error.h"

#include <mpi.h>

#include <deque>
#include <memory>

#include <cuda.h>
#include <cuda_runtime.h>

namespace sensei
{

/// A collection of utility classes and functions for intergacing with CUDA
namespace CUDAUtils
{

/** query the system for the locally available(on this rank) CUDA device count.
 * this is an MPI collective call which returns a set of device ids that can be
 * used locally. If there are as many (or more than) devices on the node than
 * the number of MPI ranks assigned to the node the list of devicce ids will be
 * unique across MPI ranks on the node. Otherwise devices are assigned round
 * robbin fashion.
 *
 * @param[in]  comm      MPI communicator defining a set of nodes on which need
 *                       access to the available GPUS
 * @param[out] localDev a list of device ids that can be used my the calling
 *                       MPI rank.
 * @returns              non-zero on error.
 */
SENSEI_EXPORT
int GetLocalCudaDevices(MPI_Comm comm, std::deque<int> &localDev);

/// set the CUDA device. returns non-zero on error
int SetDevice(int deviceId);

/// stop and wait for previuoiusly launched kernels to complete
SENSEI_EXPORT
int Synchronize();

/// querry properties for the named CUDA device. retruns non-zero on error
SENSEI_EXPORT
int GetLaunchProps(int deviceId,
    int *blockGridMax, int &warpSize,
    int &maxWarpsPerBlock);

/** A flat array is broken into blocks of number of threads where each adjacent
 * thread accesses adjacent memory locations. To accomplish this we might need
 * a large number of blocks. If the number of blocks exceeds the max block
 * dimension in the first and or second block grid dimension then we need to
 * use a 2d or 3d block grid.
 *
 * partitionThreadBlocks - decides on a partitioning of the data based on
 * warpsPerBlock parameter. The resulting decomposition will be either 1,2,
 * or 3D as needed to accomodate the number of fixed sized blocks. It can
 * happen that max grid dimensions are hit, in which case you'll need to
 * increase the number of warps per block.
 *
 * threadIdToArrayIndex - given a thread and block id gets the
 * array index to update. _this may be out of bounds so be sure
 * to validate before using it.
 *
 * indexIsValid - test an index for validity.
*/
/// @name CUDA indexing scheme
///@{

/** convert a CUDA index into a flat array index using the paritioning scheme
 * defined in partitionThreadBlocks
 */
inline
__device__
unsigned long ThreadIdToArrayIndex()
{
    return threadIdx.x + blockDim.x*(blockIdx.x + blockIdx.y * gridDim.x
        + blockIdx.z * gridDim.x * gridDim.y);
}

/// bounds check the flat index
inline
__device__
int IndexIsValid(unsigned long index, unsigned long maxIndex)
{
    return index < maxIndex;
}

/** calculate CUDA launch paramters for an arbitrarily large flat array
 *
 * @param[in] arraySize     the length of the array being processed
 * @param[in] warpsPerBlock number of warps to use per block (your choice)
 * @param[out] blockGrid    block dimension kernel launch control
 * @param[out] nBlocks      number of blocks
 * @param[out] threadGrid   thread dimension kernel launch control
 *
 * @returns non zero on error
 */
SENSEI_EXPORT
int PartitionThreadBlocks(int deviceId, size_t arraySize,
    int warpsPerBlock, dim3 &blockGrid, int &nBlocks,
    dim3 &threadGrid);

/** calculate CUDA launch paramters for an arbitrarily large flat array
 *
 * @param[in] arraySize  the length of the array being processed
 * @param[in] warpSize  number of threads per warp supported on the device
 * @param[in] warpsPerBlock  number of warps to use per block (your choice)
 * @param[out] blockGridMax  maximum number of blocks supported by the device
 * @param[out] blockGrid  block dimension kernel launch control
 * @param[out] nBlocks  number of blocks
 * @param[out] threadGrid  thread dimension kernel launch control
 *
 * @returns non zero on error
 */
SENSEI_EXPORT
int PartitionThreadBlocks(size_t arraySize,
    int warpsPerBlock, int warpSize, int *blockGridMax,
    dim3 &blockGrid, int &nBlocks, dim3 &threadGrid);
}
}

#endif
