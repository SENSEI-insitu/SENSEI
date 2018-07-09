#include "BinaryStream.h"
#include <mpi.h>

namespace sensei
{

//-----------------------------------------------------------------------------
BinaryStream::BinaryStream()
   : mSize(0), mData(nullptr), mReadPtr(nullptr), mWritePtr(nullptr)
{}

//-----------------------------------------------------------------------------
BinaryStream::~BinaryStream() noexcept
{
  this->Clear();
}

//-----------------------------------------------------------------------------
BinaryStream::BinaryStream(const BinaryStream &other)
   : mSize(0), mData(nullptr), mReadPtr(nullptr), mWritePtr(nullptr)
{ *this = other; }

//-----------------------------------------------------------------------------
BinaryStream::BinaryStream(BinaryStream &&other) noexcept
   : mSize(0), mData(nullptr), mReadPtr(nullptr), mWritePtr(nullptr)
{ this->Swap(other); }

//-----------------------------------------------------------------------------
const BinaryStream &BinaryStream::operator=(
  const BinaryStream &other)
{
  if (&other == this)
    return *this;

  this->Resize(other.mSize);
  unsigned long inUse = other.mWritePtr - other.mData;
  memcpy(mData, other.mData, inUse);
  mWritePtr = mData + inUse;
  mReadPtr = mData + (other.mReadPtr - other.mData);

  return *this;
}

//-----------------------------------------------------------------------------
const BinaryStream &BinaryStream::operator=(
  BinaryStream &&other) noexcept
{
  BinaryStream tmp(std::move(other));
  this->Swap(tmp);
  return *this;
}

//-----------------------------------------------------------------------------
void BinaryStream::Clear() noexcept
{
  free(mData);
  mData = nullptr;
  mReadPtr = nullptr;
  mWritePtr = nullptr;
  mSize = 0;
}

//-----------------------------------------------------------------------------
void BinaryStream::Resize(unsigned long nBytes)
{
  // no change
  if (nBytes == mSize)
    return;

  // free
  if (nBytes == 0)
    {
    this->Clear();
    return;
    }

  // shrink
  if (nBytes < mSize)
    {
    unsigned char *end =  mData + nBytes;
    if (mWritePtr >= end)
      mWritePtr = end;
    return;
    }

  // grow
  unsigned char *origMData = mData;
  mData = (unsigned char *)realloc(mData, nBytes);

  // update the stream pointer
  if (mData != origMData)
    {
    mWritePtr = mData + (mWritePtr - origMData);
    mReadPtr = mData + (mReadPtr - origMData);
    }

  mSize = nBytes;
}

//-----------------------------------------------------------------------------
void BinaryStream::Grow(unsigned long nBytes)
{
  unsigned long nBytesNeeded = this->Size() + nBytes;
  if (nBytesNeeded > mSize)
    {
    unsigned long newSize = mSize + this->GetBlockSize();
    while (newSize < nBytesNeeded)
      newSize += this->GetBlockSize();
    this->Resize(newSize);
    }
}

//-----------------------------------------------------------------------------
void BinaryStream::Swap(BinaryStream &other) noexcept
{
  std::swap(mData, other.mData);
  std::swap(mWritePtr, other.mWritePtr);
  std::swap(mReadPtr, other.mReadPtr);
  std::swap(mSize, other.mSize);
}

//-----------------------------------------------------------------------------
int BinaryStream::Broadcast(int rootRank)
{
  int init = 0;
  int rank = 0;
  MPI_Initialized(&init);
  if (init)
    {
    unsigned long nbytes = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == rootRank)
      {
      nbytes = this->Size();
      MPI_Bcast(&nbytes, 1, MPI_UNSIGNED_LONG, rootRank, MPI_COMM_WORLD);
      MPI_Bcast(this->GetData(), nbytes, MPI_BYTE, rootRank, MPI_COMM_WORLD);
      }
    else
      {
      MPI_Bcast(&nbytes, 1, MPI_UNSIGNED_LONG, rootRank, MPI_COMM_WORLD);
      this->Resize(nbytes);
      MPI_Bcast(this->GetData(), nbytes, MPI_BYTE, rootRank, MPI_COMM_WORLD);
      this->SetReadPos(0);
      this->SetWritePos(nbytes);
      }
    }
  return 0;
}

}
