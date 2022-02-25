/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkHeap.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkHeap
 * @brief   replacement for malloc/free and new/delete
 *
 * This class is a replacement for malloc/free and new/delete for software
 * that has inherent memory leak or performance problems. For example,
 * external software such as the PLY library (svtkPLY) and VRML importer
 * (svtkVRMLImporter) are often written with lots of malloc() calls but
 * without the corresponding free() invocations. The class
 * svtkOrderedTriangulator may create and delete millions of new/delete calls.
 * This class allows the overloading of the C++ new operator (or other memory
 * allocation requests) by using the method AllocateMemory(). Memory is
 * deleted with an invocation of CleanAll() (which deletes ALL memory; any
 * given memory allocation cannot be deleted). Note: a block size can be used
 * to control the size of each memory allocation. Requests for memory are
 * fulfilled from the block until the block runs out, then a new block is
 * created.
 *
 * @warning
 * Do not use this class as a general replacement for system memory
 * allocation.  This class should be used only as a last resort if memory
 * leaks cannot be tracked down and eliminated by conventional means. Also,
 * deleting memory from svtkHeap is not supported. Only the deletion of
 * the entire heap is. (A Reset() method allows you to reuse previously
 * allocated memory.)
 *
 * @sa
 * svtkVRMLImporter svtkPLY svtkOrderedTriangulator
 */

#ifndef svtkHeap_h
#define svtkHeap_h

#include "svtkCommonMiscModule.h" // For export macro
#include "svtkObject.h"

class svtkHeapBlock; // forward declaration

class SVTKCOMMONMISC_EXPORT svtkHeap : public svtkObject
{
public:
  static svtkHeap* New();
  svtkTypeMacro(svtkHeap, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Allocate the memory requested.
   */
  void* AllocateMemory(size_t n);

  //@{
  /**
   * Set/Get the size at which blocks are allocated. If a memory
   * request is bigger than the block size, then that size
   * will be allocated.
   */
  virtual void SetBlockSize(size_t);
  virtual size_t GetBlockSize() { return this->BlockSize; }
  //@}

  //@{
  /**
   * Get the number of allocations thus far.
   */
  svtkGetMacro(NumberOfBlocks, int);
  svtkGetMacro(NumberOfAllocations, int);
  //@}

  /**
   * This methods resets the current allocation location
   * back to the beginning of the heap. This allows
   * reuse of previously allocated memory which may be
   * beneficial to performance in many cases.
   */
  void Reset();

  /**
   * Convenience method performs string duplication.
   */
  char* StringDup(const char* str);

protected:
  svtkHeap();
  ~svtkHeap() override;

  void Add(size_t blockSize);
  void CleanAll();
  svtkHeapBlock* DeleteAndNext();

  size_t BlockSize;
  int NumberOfAllocations;
  int NumberOfBlocks;
  size_t Alignment;

  // Manage the blocks
  svtkHeapBlock* First;
  svtkHeapBlock* Last;
  svtkHeapBlock* Current;
  // Manage the memory in the block
  size_t Position; // the position in the Current block

private:
  svtkHeap(const svtkHeap&) = delete;
  void operator=(const svtkHeap&) = delete;
};

#endif
