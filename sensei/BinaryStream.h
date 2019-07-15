#ifndef BinaryStream_H
#define BinaryStream_H

#include "senseiConfig.h"
#include "Error.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <map>
#include <vector>
#include <array>
#include <type_traits>

namespace sensei
{

// Serialize objects into a binary stream.
class BinaryStream
{
public:
  // construct
  BinaryStream();
  ~BinaryStream() noexcept;

  // copy
  BinaryStream(const BinaryStream &s);
  const BinaryStream &operator=(const BinaryStream &other);

  // move
  BinaryStream(BinaryStream &&s) noexcept;
  const BinaryStream &operator=(BinaryStream &&other) noexcept;

  // evaluate to true when the stream is not empty.
  operator bool()
  { return mSize != 0; }

  // Release all resources, set to a uninitialized
  // state.
  void Clear() noexcept;

  // Allocate nBytes for the stream.
  void Resize(unsigned long nBytes);

  // ensures space for nBytes more to the stream.
  void Grow(unsigned long nBytes);

  // Get a pointer to the stream internal representation.
  unsigned char *GetData() noexcept
  { return mData; }

  const unsigned char *GetData() const noexcept
  { return mData; }

  // Get the size of the valid data in the stream.
  // note: the internal buffer may be larger.
  unsigned long Size() const noexcept
  { return mWritePtr - mData; }

  // Get the sise of the internal buffer allocated
  // for the stream.
  unsigned long Capacity() const noexcept
  { return mSize; }

  // set the stream position to n bytes from the head
  // of the stream
  void SetReadPos(unsigned long n) noexcept
  { mReadPtr = mData + n; }

  void SetWritePos(unsigned long n) noexcept
  { mWritePtr = mData + n; }

  // swap the two objects
  void Swap(BinaryStream &other) noexcept;

  // Insert/Extract to/from the stream.
  template <typename T> void Pack(T *val);

  template <typename T> void Pack(const T &val, typename std::enable_if<!std::is_class<T>::value>::type* = 0);
  template <typename T> void Unpack(T &val, typename std::enable_if<!std::is_class<T>::value>::type* = 0);

  template <typename T> void Pack(const T *val, unsigned long n);
  template <typename T> void Unpack(T *val, unsigned long n);

  // specializations
  void Pack(const std::string &str);
  void Unpack(std::string &str);

  template <typename T, unsigned long N> void Pack(const std::array<T,N> &arr);
  template <typename T, unsigned long N> void Unpack(std::array<T,N> &arr);

  template <typename K, typename V> void Pack(const std::map<K,V> &amap);
  template <typename K, typename V> void Unpack(std::map<K,V> &amap);

  template<typename T> void Pack(const std::vector<T> &v,
    typename std::enable_if<std::is_class<T>::value>::type* = 0);

  template<typename T> void Unpack(std::vector<T> &v,
    typename std::enable_if<std::is_class<T>::value>::type* = 0);

#if !defined(SWIG)
  template<typename T> void Pack(const std::vector<T> &v,
    typename std::enable_if<!std::is_class<T>::value>::type* = 0);

  template<typename T> void Unpack(std::vector<T> &v,
    typename std::enable_if<!std::is_class<T>::value>::type* = 0);
#endif

  // broadcast the stream from the root process to all other processes
  int Broadcast(int rootRank=0);

private:
  // re-allocation size
  static
  constexpr unsigned int GetBlockSize()
  { return 512; }

private:
  unsigned long mSize;
  unsigned char *mData;
  unsigned char *mReadPtr;
  unsigned char *mWritePtr;
};

//-----------------------------------------------------------------------------
template <typename T>
void BinaryStream::Pack(T *val)
{
  (void)val;
  SENSEI_ERROR("You con't pack a pointer.");
}

//-----------------------------------------------------------------------------
template <typename T>
void BinaryStream::Pack(const T &val, typename std::enable_if<!std::is_class<T>::value>::type*)
{
  this->Grow(sizeof(T));
  *((T *)mWritePtr) = val;
  mWritePtr += sizeof(T);
}

//-----------------------------------------------------------------------------
template <typename T>
void BinaryStream::Unpack(T &val, typename std::enable_if<!std::is_class<T>::value>::type*)
{
  val = *((T *)mReadPtr);
  mReadPtr += sizeof(T);
}

//-----------------------------------------------------------------------------
template <typename T>
void BinaryStream::Pack(const T *val, unsigned long n)
{
  unsigned long nBytes = n*sizeof(T);
  this->Grow(nBytes);

  unsigned long nn = n*sizeof(T);
  memcpy(mWritePtr, val, nn);
  mWritePtr += nn;
}

//-----------------------------------------------------------------------------
template <typename T>
void BinaryStream::Unpack(T *val, unsigned long n)
{
  unsigned long nn = n*sizeof(T);
  memcpy(val, mReadPtr, nn);
  mReadPtr += nn;
}

//-----------------------------------------------------------------------------
inline
void BinaryStream::Pack(const std::string &str)
{
  unsigned long slen = str.size();
  this->Pack(slen);
  this->Pack(str.c_str(), slen);
}

//-----------------------------------------------------------------------------
inline
void BinaryStream::Unpack(std::string &str)
{
  unsigned long slen = 0;
  this->Unpack(slen);

  str.resize(slen);
  str.assign(reinterpret_cast<char*>(mReadPtr), slen);

  mReadPtr += slen;
}

//-----------------------------------------------------------------------------
template <typename T, unsigned long N>
void BinaryStream::Pack(const std::array<T,N> &arr)
{
  this->Pack(arr.data(), N);
}

//-----------------------------------------------------------------------------
template <typename T, unsigned long N>
void BinaryStream::Unpack(std::array<T,N> &arr)
{
  this->Unpack(arr.data(), N);
}

//-----------------------------------------------------------------------------
template <typename K, typename V>
void BinaryStream::Pack(const std::map<K,V> &amap)
{
  unsigned long len = amap.size();
  this->Pack(len);

  typename std::map<K,V>::const_iterator it = amap.begin();
  for (unsigned long i = 0; i < len; ++i, ++it)
    {
    this->Pack(it->first);
    this->Pack(it->second);
    }
}

//-----------------------------------------------------------------------------
template <typename K, typename V>
void BinaryStream::Unpack(std::map<K,V> &amap)
{
  unsigned long len = 0;
  this->Unpack(len);

  for (unsigned long i = 0; i < len; ++i)
    {
    K key;
    V val;

    this->Unpack(key);
    this->Unpack(val);

    amap[key] = std::move(val);
    }
}

//-----------------------------------------------------------------------------
template<typename T>
void BinaryStream::Pack(const std::vector<T> &v, typename std::enable_if<std::is_class<T>::value>::type*)
{
  unsigned long vlen = v.size();
  this->Pack(vlen);
  for (unsigned long i = 0; i < vlen; ++i)
    this->Pack(v[i]);
}

//-----------------------------------------------------------------------------
template<typename T>
void BinaryStream::Unpack(std::vector<T> &v, typename std::enable_if<std::is_class<T>::value>::type*)
{
  unsigned long vlen;
  this->Unpack(vlen);

  v.resize(vlen);
  for (unsigned long i = 0; i < vlen; ++i)
    this->Unpack(v[i]);
}

//-----------------------------------------------------------------------------
template<typename T>
void BinaryStream::Pack(const std::vector<T> &v, typename std::enable_if<!std::is_class<T>::value>::type*)
{
  const unsigned long vlen = v.size();
  this->Pack(vlen);
  this->Pack(v.data(), vlen);
}

//-----------------------------------------------------------------------------
template<typename T>
void BinaryStream::Unpack(std::vector<T> &v, typename std::enable_if<!std::is_class<T>::value>::type*)
{
  unsigned long vlen;
  this->Unpack(vlen);

  v.resize(vlen);
  this->Unpack(v.data(), vlen);
}


}

#endif
