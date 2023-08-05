#include "CUDAUtils.h"

namespace sensei
{
namespace CUDAUtils
{

// **************************************************************************
int Synchronize()
{
    cudaError_t ierr = cudaSuccess;
    if ((ierr = cudaDeviceSynchronize()) != cudaSuccess)
    {
        SENSEI_ERROR("Failed to synchronize CUDA execution. "
            << cudaGetErrorString(ierr))
        return -1;
    }
    return 0;
}

// **************************************************************************
int GetLocalCudaDevices(MPI_Comm comm, std::deque<int> &localDev)
{
    cudaError_t ierr = cudaSuccess;

    // get the number of CUDA GPU's available on this node
    int nNodeDev = 0;
    if ((ierr = cudaGetDeviceCount(&nNodeDev)) != cudaSuccess)
    {
        SENSEI_ERROR("Failed to get the number of CUDA devices. "
            << cudaGetErrorString(ierr))
        return -1;
    }

    // if there are no GPU's error out
    if (nNodeDev < 1)
    {
        SENSEI_ERROR("No CUDA devices found")
        return -1;
    }

    // get the number of MPI ranks on this node, and their core id's
    int nNodeRanks = 1;
    int nodeRank = 0;

    int isInit = 0;
    MPI_Initialized(&isInit);
    if (isInit)
    {
        // get node local rank and num ranks
        MPI_Comm nodeComm;
        MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED,
            0, MPI_INFO_NULL, &nodeComm);

        MPI_Comm_size(nodeComm, &nNodeRanks);
        MPI_Comm_rank(nodeComm, &nodeRank);

        if (nNodeDev >= nNodeRanks)
        {
            // assign devices evenly between ranks
            int maxDev = nNodeDev - 1;
            int nPerRank = std::max(nNodeDev / nNodeRanks, 1);
            int nLarger = nNodeDev % nNodeRanks;

            int firstDev = nPerRank * nodeRank + (nodeRank < nLarger ? nodeRank : nLarger);
            firstDev = std::min(maxDev, firstDev);

            int lastDev = firstDev + nPerRank - 1 + (nodeRank < nLarger ? 1 : 0);
            lastDev = std::min(maxDev, lastDev);

            for (int i = firstDev; i <= lastDev; ++i)
                localDev.push_back(i);
        }
        else
        {
            // round robbin assignment
            localDev.push_back( nodeRank % nNodeDev );
        }

        MPI_Comm_free(&nodeComm);

        return 0;
    }

    // without MPI this process can use all CUDA devices
    for (int i = 0; i < nNodeDev; ++i)
        localDev.push_back(i);
    return 0;
}


//-----------------------------------------------------------------------------
int SetDevice(int deviceId)
{
    int nDevices = 0;

    cudaError_t ierr = cudaGetDeviceCount(&nDevices);
    if (ierr != cudaSuccess)
    {
        SENSEI_ERROR("Failed to get CUDA device count. "
            << cudaGetErrorString(ierr))
        return -1;
    }


    if (deviceId >= nDevices)
    {
        SENSEI_ERROR("Attempt to select invalid device "
            << deviceId << " of " << nDevices)
        return -1;
    }

    ierr = cudaSetDevice(deviceId);
    if (ierr)
    {
        SENSEI_ERROR("Failed to select device " << deviceId << ". "
            <<  cudaGetErrorString(ierr))
        return -1;
    }

    return 0;
}

// --------------------------------------------------------------------------
int GetLaunchProps(int deviceId,
    int *blockGridMax, int &warpSize,
    int &warpsPerBlockMax)
{
    cudaError_t ierr = cudaSuccess;

    if (((ierr = cudaDeviceGetAttribute(&blockGridMax[0], cudaDevAttrMaxGridDimX, deviceId)) != cudaSuccess)
        || ((ierr = cudaDeviceGetAttribute(&blockGridMax[1], cudaDevAttrMaxGridDimY, deviceId)) != cudaSuccess)
        || ((ierr = cudaDeviceGetAttribute(&blockGridMax[2], cudaDevAttrMaxGridDimZ, deviceId)) != cudaSuccess))
    {
        SENSEI_ERROR("Failed to get CUDA max grid dim. " << cudaGetErrorString(ierr))
        return -1;
    }

    if ((ierr = cudaDeviceGetAttribute(&warpSize, cudaDevAttrWarpSize, deviceId)) != cudaSuccess)
    {
        SENSEI_ERROR("Failed to get CUDA warp size. " << cudaGetErrorString(ierr))
        return -1;
    }

    int threadsPerBlockMax = 0;

    if ((ierr = cudaDeviceGetAttribute(&threadsPerBlockMax,
        cudaDevAttrMaxThreadsPerBlock, deviceId)) != cudaSuccess)
    {
        SENSEI_ERROR("Failed to get CUDA max threads per block. " << cudaGetErrorString(ierr))
        return -1;
    }

    warpsPerBlockMax = threadsPerBlockMax / warpSize;

    return 0;
}

// --------------------------------------------------------------------------
int PartitionThreadBlocks(size_t arraySize,
    int warpsPerBlock, int warpSize, int *blockGridMax,
    dim3 &blockGrid, int &nBlocks, dim3 &threadGrid)
{
    unsigned long threadsPerBlock = warpsPerBlock * warpSize;

    threadGrid.x = threadsPerBlock;
    threadGrid.y = 1;
    threadGrid.z = 1;

    unsigned long blockSize = threadsPerBlock;
    nBlocks = arraySize / blockSize;

    if (arraySize % blockSize)
        ++nBlocks;

    if (nBlocks > blockGridMax[0])
    {
        // multi-d decomp required
        blockGrid.x = blockGridMax[0];
        blockGrid.y = nBlocks / blockGridMax[0];
        if (nBlocks % blockGridMax[0])
        {
            ++blockGrid.y;
        }

        if (blockGrid.y > (unsigned)blockGridMax[1])
        {
            // 3d decomp
            unsigned long blockGridMax01 = blockGridMax[0] * blockGridMax[1];
            blockGrid.y = blockGridMax[1];
            blockGrid.z = nBlocks / blockGridMax01;

            if (nBlocks % blockGridMax01)
                ++blockGrid.z;

            if (blockGrid.z > (unsigned)blockGridMax[2])
            {
                SENSEI_ERROR("Too many blocks " << nBlocks << " of size " << blockSize
                    << " are required for a grid of (" << blockGridMax[0] << ", "
                    << blockGridMax[1] << ", " << blockGridMax[2]
                    << ") blocks. Hint: increase the number of warps per block.");
                return -1;
            }
        }
        else
        {
            // 2d decomp
            blockGrid.z = 1;
        }
    }
    else
    {
        // 1d decomp
        blockGrid.x = nBlocks;
        blockGrid.y = 1;
        blockGrid.z = 1;
    }

    return 0;
}

// --------------------------------------------------------------------------
int PartitionThreadBlocks(int deviceId, size_t arraySize,
    int warpsPerBlock, dim3 &blockGrid, int &nBlocks,
    dim3 &threadGrid)
{
    int blockGridMax[3] = {0};
    int warpSize = 0;
    int warpsPerBlockMax = 0;
    if (CUDAUtils::GetLaunchProps(deviceId, blockGridMax,
        warpSize, warpsPerBlockMax))
    {
        SENSEI_ERROR("Failed to get launch properties")
        return -1;
    }

    return CUDAUtils::PartitionThreadBlocks(arraySize, warpsPerBlock,
        warpSize, blockGridMax, blockGrid, nBlocks,
        threadGrid);
}

}
}
