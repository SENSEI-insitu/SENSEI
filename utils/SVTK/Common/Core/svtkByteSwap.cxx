/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkByteSwap.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkByteSwap.h"
#include "svtkObjectFactory.h"
#include <memory.h>

svtkStandardNewMacro(svtkByteSwap);

//----------------------------------------------------------------------------
svtkByteSwap::svtkByteSwap() = default;

//----------------------------------------------------------------------------
svtkByteSwap::~svtkByteSwap() = default;

//----------------------------------------------------------------------------
// Define swap functions for each type size.
template <size_t s>
struct svtkByteSwapper;
template <>
struct svtkByteSwapper<1>
{
  static inline void Swap(char*) {}
};
template <>
struct svtkByteSwapper<2>
{
  static inline void Swap(char* data)
  {
    char one_byte;
    one_byte = data[0];
    data[0] = data[1];
    data[1] = one_byte;
  }
};
template <>
struct svtkByteSwapper<4>
{
  static inline void Swap(char* data)
  {
    char one_byte;
    one_byte = data[0];
    data[0] = data[3];
    data[3] = one_byte;
    one_byte = data[1];
    data[1] = data[2];
    data[2] = one_byte;
  }
};
template <>
struct svtkByteSwapper<8>
{
  static inline void Swap(char* data)
  {
    char one_byte;
    one_byte = data[0];
    data[0] = data[7];
    data[7] = one_byte;
    one_byte = data[1];
    data[1] = data[6];
    data[6] = one_byte;
    one_byte = data[2];
    data[2] = data[5];
    data[5] = one_byte;
    one_byte = data[3];
    data[3] = data[4];
    data[4] = one_byte;
  }
};

//----------------------------------------------------------------------------
// Define range swap functions.
template <class T>
inline void svtkByteSwapRange(T* first, size_t num)
{
  // Swap one value at a time.
  T* last = first + num;
  for (T* p = first; p != last; ++p)
  {
    svtkByteSwapper<sizeof(T)>::Swap(reinterpret_cast<char*>(p));
  }
}
inline bool svtkByteSwapRangeWrite(const char* first, size_t num, FILE* f, int)
{
  // No need to swap segments of 1 byte.
  size_t status = fwrite(first, sizeof(char), static_cast<size_t>(num), f);
  return status == static_cast<size_t>(num);
}
inline bool svtkByteSwapRangeWrite(const signed char* first, size_t num, FILE* f, int)
{
  // No need to swap segments of 1 byte.
  size_t status = fwrite(first, sizeof(signed char), static_cast<size_t>(num), f);
  return status == static_cast<size_t>(num);
}
inline bool svtkByteSwapRangeWrite(const unsigned char* first, size_t num, FILE* f, int)
{
  // No need to swap segments of 1 byte.
  size_t status = fwrite(first, sizeof(unsigned char), static_cast<size_t>(num), f);
  return status == static_cast<size_t>(num);
}
template <class T>
inline bool svtkByteSwapRangeWrite(const T* first, size_t num, FILE* f, long)
{
  // Swap and write one value at a time.  We do not need to do this in
  // blocks because the file stream is already buffered.
  const T* last = first + num;
  bool result = true;
  for (const T* p = first; p != last && result; ++p)
  {
    // Use a union to avoid breaking C++ aliasing rules.
    union {
      T value;
      char data[sizeof(T)];
    } temp = { *p };
    svtkByteSwapper<sizeof(T)>::Swap(temp.data);
    size_t status = fwrite(temp.data, sizeof(T), 1, f);
    result = status == 1;
  }
  return result;
}
inline void svtkByteSwapRangeWrite(const char* first, size_t num, ostream* os, int)
{
  // No need to swap segments of 1 byte.
  os->write(first, num * static_cast<size_t>(sizeof(char)));
}
inline void svtkByteSwapRangeWrite(const signed char* first, size_t num, ostream* os, int)
{
  // No need to swap segments of 1 byte.
  os->write(reinterpret_cast<const char*>(first), num * static_cast<size_t>(sizeof(signed char)));
}
inline void svtkByteSwapRangeWrite(const unsigned char* first, size_t num, ostream* os, int)
{
  // No need to swap segments of 1 byte.
  os->write(reinterpret_cast<const char*>(first), num * static_cast<size_t>(sizeof(unsigned char)));
}
template <class T>
inline void svtkByteSwapRangeWrite(const T* first, size_t num, ostream* os, long)
{
  // Swap and write one value at a time.  We do not need to do this in
  // blocks because the file stream is already buffered.
  const T* last = first + num;
  for (const T* p = first; p != last; ++p)
  {
    // Use a union to avoid breaking C++ aliasing rules.
    union {
      T value;
      char data[sizeof(T)];
    } temp = { *p };
    svtkByteSwapper<sizeof(T)>::Swap(temp.data);
    os->write(temp.data, sizeof(T));
  }
}

//----------------------------------------------------------------------------
// Define swap functions for each endian-ness.
#if defined(SVTK_WORDS_BIGENDIAN)
template <class T>
inline void svtkByteSwapBE(T*)
{
}
template <class T>
inline void svtkByteSwapBERange(T*, size_t)
{
}
template <class T>
inline bool svtkByteSwapBERangeWrite(const T* p, size_t num, FILE* f)
{
  size_t status = fwrite(p, sizeof(T), static_cast<size_t>(num), f);
  return status == static_cast<size_t>(num);
}
template <class T>
inline void svtkByteSwapBERangeWrite(const T* p, size_t num, ostream* os)
{
  os->write((char*)p, sizeof(T) * num);
}
template <class T>
inline void svtkByteSwapLE(T* p)
{
  svtkByteSwapper<sizeof(T)>::Swap(reinterpret_cast<char*>(p));
}
template <class T>
inline void svtkByteSwapLERange(T* p, size_t num)
{
  svtkByteSwapRange(p, num);
}
template <class T>
inline bool svtkByteSwapLERangeWrite(const T* p, size_t num, FILE* f)
{
  return svtkByteSwapRangeWrite(p, num, f, 1);
}
template <class T>
inline void svtkByteSwapLERangeWrite(const T* p, size_t num, ostream* os)
{
  svtkByteSwapRangeWrite(p, num, os, 1);
}
#else
template <class T>
inline void svtkByteSwapBE(T* p)
{
  svtkByteSwapper<sizeof(T)>::Swap(reinterpret_cast<char*>(p));
}
template <class T>
inline void svtkByteSwapBERange(T* p, size_t num)
{
  svtkByteSwapRange(p, num);
}
template <class T>
inline bool svtkByteSwapBERangeWrite(const T* p, size_t num, FILE* f)
{
  return svtkByteSwapRangeWrite(p, num, f, 1);
}
template <class T>
inline void svtkByteSwapBERangeWrite(const T* p, size_t num, ostream* os)
{
  svtkByteSwapRangeWrite(p, num, os, 1);
}
template <class T>
inline void svtkByteSwapLE(T*)
{
}
template <class T>
inline void svtkByteSwapLERange(T*, size_t)
{
}
template <class T>
inline bool svtkByteSwapLERangeWrite(const T* p, size_t num, FILE* f)
{
  size_t status = fwrite(p, sizeof(T), static_cast<size_t>(num), f);
  return status == static_cast<size_t>(num);
}
template <class T>
inline void svtkByteSwapLERangeWrite(const T* p, size_t num, ostream* os)
{
  os->write(reinterpret_cast<const char*>(p), static_cast<size_t>(sizeof(T)) * num);
}
#endif

//----------------------------------------------------------------------------
#define SVTK_BYTE_SWAP_IMPL(T)                                                                      \
  void svtkByteSwap::SwapLE(T* p) { svtkByteSwapLE(p); }                                             \
  void svtkByteSwap::SwapBE(T* p) { svtkByteSwapBE(p); }                                             \
  void svtkByteSwap::SwapLERange(T* p, size_t num) { svtkByteSwapLERange(p, num); }                  \
  void svtkByteSwap::SwapBERange(T* p, size_t num) { svtkByteSwapBERange(p, num); }                  \
  bool svtkByteSwap::SwapLERangeWrite(const T* p, size_t num, FILE* file)                           \
  {                                                                                                \
    return svtkByteSwapLERangeWrite(p, num, file);                                                  \
  }                                                                                                \
  bool svtkByteSwap::SwapBERangeWrite(const T* p, size_t num, FILE* file)                           \
  {                                                                                                \
    return svtkByteSwapBERangeWrite(p, num, file);                                                  \
  }                                                                                                \
  void svtkByteSwap::SwapLERangeWrite(const T* p, size_t num, ostream* os)                          \
  {                                                                                                \
    svtkByteSwapLERangeWrite(p, num, os);                                                           \
  }                                                                                                \
  void svtkByteSwap::SwapBERangeWrite(const T* p, size_t num, ostream* os)                          \
  {                                                                                                \
    svtkByteSwapBERangeWrite(p, num, os);                                                           \
  }
SVTK_BYTE_SWAP_IMPL(float)
SVTK_BYTE_SWAP_IMPL(double)
SVTK_BYTE_SWAP_IMPL(char)
SVTK_BYTE_SWAP_IMPL(short)
SVTK_BYTE_SWAP_IMPL(int)
SVTK_BYTE_SWAP_IMPL(long)
SVTK_BYTE_SWAP_IMPL(long long)
SVTK_BYTE_SWAP_IMPL(signed char)
SVTK_BYTE_SWAP_IMPL(unsigned char)
SVTK_BYTE_SWAP_IMPL(unsigned short)
SVTK_BYTE_SWAP_IMPL(unsigned int)
SVTK_BYTE_SWAP_IMPL(unsigned long)
SVTK_BYTE_SWAP_IMPL(unsigned long long)
#undef SVTK_BYTE_SWAP_IMPL

#if SVTK_SIZEOF_SHORT == 2
typedef short svtkByteSwapType2;
#else
#error "..."
#endif

#if SVTK_SIZEOF_INT == 4
typedef int svtkByteSwapType4;
#else
#error "..."
#endif

#if SVTK_SIZEOF_DOUBLE == 8
typedef double svtkByteSwapType8;
#else
#error "..."
#endif

//----------------------------------------------------------------------------
#define SVTK_BYTE_SWAP_SIZE(S)                                                                      \
  void svtkByteSwap::Swap##S##LE(void* p)                                                           \
  {                                                                                                \
    svtkByteSwap::SwapLE(static_cast<svtkByteSwapType##S*>(p));                                      \
  }                                                                                                \
  void svtkByteSwap::Swap##S##BE(void* p)                                                           \
  {                                                                                                \
    svtkByteSwap::SwapBE(static_cast<svtkByteSwapType##S*>(p));                                      \
  }                                                                                                \
  void svtkByteSwap::Swap##S##LERange(void* p, size_t n)                                            \
  {                                                                                                \
    svtkByteSwap::SwapLERange(static_cast<svtkByteSwapType##S*>(p), n);                              \
  }                                                                                                \
  void svtkByteSwap::Swap##S##BERange(void* p, size_t n)                                            \
  {                                                                                                \
    svtkByteSwap::SwapBERange(static_cast<svtkByteSwapType##S*>(p), n);                              \
  }                                                                                                \
  bool svtkByteSwap::SwapWrite##S##LERange(void const* p, size_t n, FILE* f)                        \
  {                                                                                                \
    return svtkByteSwap::SwapLERangeWrite(static_cast<const svtkByteSwapType##S*>(p), n, f);         \
  }                                                                                                \
  bool svtkByteSwap::SwapWrite##S##BERange(void const* p, size_t n, FILE* f)                        \
  {                                                                                                \
    return svtkByteSwap::SwapBERangeWrite(static_cast<const svtkByteSwapType##S*>(p), n, f);         \
  }                                                                                                \
  void svtkByteSwap::SwapWrite##S##LERange(void const* p, size_t n, ostream* os)                    \
  {                                                                                                \
    svtkByteSwap::SwapLERangeWrite(static_cast<const svtkByteSwapType##S*>(p), n, os);               \
  }                                                                                                \
  void svtkByteSwap::SwapWrite##S##BERange(void const* p, size_t n, ostream* os)                    \
  {                                                                                                \
    svtkByteSwap::SwapBERangeWrite(static_cast<const svtkByteSwapType##S*>(p), n, os);               \
  }
SVTK_BYTE_SWAP_SIZE(2)
SVTK_BYTE_SWAP_SIZE(4)
SVTK_BYTE_SWAP_SIZE(8)
#undef SVTK_BYTE_SWAP_SIZE

//----------------------------------------------------------------------------
// Swaps the bytes of a buffer.  Uses an arbitrary word size, but
// assumes the word size is divisible by two.
void svtkByteSwap::SwapVoidRange(void* buffer, size_t numWords, size_t wordSize)
{
  unsigned char temp, *out, *buf;
  size_t idx1, idx2, inc, half;

  half = wordSize / 2;
  inc = wordSize - 1;
  buf = static_cast<unsigned char*>(buffer);

  for (idx1 = 0; idx1 < numWords; ++idx1)
  {
    out = buf + inc;
    for (idx2 = 0; idx2 < half; ++idx2)
    {
      temp = *out;
      *out = *buf;
      *buf = temp;
      ++buf;
      --out;
    }
    buf += half;
  }
}
