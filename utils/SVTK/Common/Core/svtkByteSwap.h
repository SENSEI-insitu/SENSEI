/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkByteSwap.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkByteSwap
 * @brief   perform machine dependent byte swapping
 *
 * svtkByteSwap is used by other classes to perform machine dependent byte
 * swapping. Byte swapping is often used when reading or writing binary
 * files.
 */

#ifndef svtkByteSwap_h
#define svtkByteSwap_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkObject.h"

class SVTKCOMMONCORE_EXPORT svtkByteSwap : public svtkObject
{
public:
  static svtkByteSwap* New();
  svtkTypeMacro(svtkByteSwap, svtkObject);

  //@{
  /**
   * Type-safe swap signatures to swap for storage in either Little
   * Endian or Big Endian format.  Swapping is performed according to
   * the true size of the type given.
   */
#define SVTK_BYTE_SWAP_DECL(T)                                                                      \
  static void SwapLE(T* p);                                                                        \
  static void SwapBE(T* p);                                                                        \
  static void SwapLERange(T* p, size_t num);                                                       \
  static void SwapBERange(T* p, size_t num);                                                       \
  static bool SwapLERangeWrite(const T* p, size_t num, FILE* file);                                \
  static bool SwapBERangeWrite(const T* p, size_t num, FILE* file);                                \
  static void SwapLERangeWrite(const T* p, size_t num, ostream* os);                               \
  static void SwapBERangeWrite(const T* p, size_t num, ostream* os)
  SVTK_BYTE_SWAP_DECL(float);
  SVTK_BYTE_SWAP_DECL(double);
  SVTK_BYTE_SWAP_DECL(char);
  SVTK_BYTE_SWAP_DECL(short);
  SVTK_BYTE_SWAP_DECL(int);
  SVTK_BYTE_SWAP_DECL(long);
  SVTK_BYTE_SWAP_DECL(long long);
  SVTK_BYTE_SWAP_DECL(signed char);
  SVTK_BYTE_SWAP_DECL(unsigned char);
  SVTK_BYTE_SWAP_DECL(unsigned short);
  SVTK_BYTE_SWAP_DECL(unsigned int);
  SVTK_BYTE_SWAP_DECL(unsigned long);
  SVTK_BYTE_SWAP_DECL(unsigned long long);
#undef SVTK_BYTE_SWAP_DECL
  //@}

  //@{
  /**
   * Swap 2, 4, or 8 bytes for storage as Little Endian.
   */
  static void Swap2LE(void* p);
  static void Swap4LE(void* p);
  static void Swap8LE(void* p);
  //@}

  //@{
  /**
   * Swap a block of 2-, 4-, or 8-byte segments for storage as Little Endian.
   */
  static void Swap2LERange(void* p, size_t num);
  static void Swap4LERange(void* p, size_t num);
  static void Swap8LERange(void* p, size_t num);
  //@}

  //@{
  /**
   * Swap a block of 2-, 4-, or 8-byte segments for storage as Little Endian.
   * The results are written directly to a file to avoid temporary storage.
   */
  static bool SwapWrite2LERange(void const* p, size_t num, FILE* f);
  static bool SwapWrite4LERange(void const* p, size_t num, FILE* f);
  static bool SwapWrite8LERange(void const* p, size_t num, FILE* f);
  static void SwapWrite2LERange(void const* p, size_t num, ostream* os);
  static void SwapWrite4LERange(void const* p, size_t num, ostream* os);
  static void SwapWrite8LERange(void const* p, size_t num, ostream* os);
  //@}

  //@{
  /**
   * Swap 2, 4, or 8 bytes for storage as Big Endian.
   */
  static void Swap2BE(void* p);
  static void Swap4BE(void* p);
  static void Swap8BE(void* p);
  //@}

  //@{
  /**
   * Swap a block of 2-, 4-, or 8-byte segments for storage as Big Endian.
   */
  static void Swap2BERange(void* p, size_t num);
  static void Swap4BERange(void* p, size_t num);
  static void Swap8BERange(void* p, size_t num);
  //@}

  //@{
  /**
   * Swap a block of 2-, 4-, or 8-byte segments for storage as Big Endian.
   * The results are written directly to a file to avoid temporary storage.
   */
  static bool SwapWrite2BERange(void const* p, size_t num, FILE* f);
  static bool SwapWrite4BERange(void const* p, size_t num, FILE* f);
  static bool SwapWrite8BERange(void const* p, size_t num, FILE* f);
  static void SwapWrite2BERange(void const* p, size_t num, ostream* os);
  static void SwapWrite4BERange(void const* p, size_t num, ostream* os);
  static void SwapWrite8BERange(void const* p, size_t num, ostream* os);
  //@}

  /**
   * Swaps the bytes of a buffer.  Uses an arbitrary word size, but
   * assumes the word size is divisible by two.
   */
  static void SwapVoidRange(void* buffer, size_t numWords, size_t wordSize);

protected:
  svtkByteSwap();
  ~svtkByteSwap() override;

private:
  svtkByteSwap(const svtkByteSwap&) = delete;
  void operator=(const svtkByteSwap&) = delete;
};

#endif
// SVTK-HeaderTest-Exclude: svtkByteSwap.h
