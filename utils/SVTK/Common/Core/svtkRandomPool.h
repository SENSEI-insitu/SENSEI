/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkRandomPool.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkRandomPool
 * @brief   convenience class to quickly generate a pool of random numbers
 *
 * svtkRandomPool generates random numbers, and can do so using
 * multithreading.  It supports parallel applications where generating random
 * numbers on the fly is difficult (i.e., non-deterministic). Also, it can be
 * used to populate svtkDataArrays in an efficient manner. By default it uses
 * an instance of svtkMersenneTwister to generate random sequences, but any
 * subclass of svtkRandomSequence may be used. It also supports simple methods
 * to generate, access, and pass random memory pools between objects.
 *
 * In threaded applications, these class may be conveniently used to
 * pre-generate a sequence of random numbers, followed by the use of
 * deterministic accessor methods to produce random sequences without
 * problems etc. due to unpredictable work load and order of thread
 * execution.
 *
 * @warning
 * The class uses svtkMultiThreader if the size of the pool is larger than
 * the specified chunk size. Also, svtkSMPTools may be used to scale the
 * components in the method PopulateDataArray().
 */

#ifndef svtkRandomPool_h
#define svtkRandomPool_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkObject.h"

class svtkRandomSequence;
class svtkDataArray;

class SVTKCOMMONCORE_EXPORT svtkRandomPool : public svtkObject
{
public:
  //@{
  /**
   * Standard methods for instantiation, type information, and printing.
   */
  static svtkRandomPool* New();
  svtkTypeMacro(svtkRandomPool, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;
  //@}

  //@{
  /**
   * Specify the random sequence generator used to produce the random pool.
   * By default svtkMersenneTwister is used.
   */
  virtual void SetSequence(svtkRandomSequence* seq);
  svtkGetObjectMacro(Sequence, svtkRandomSequence);
  //@}

  //@{
  /**
   * Methods to set and get the size of the pool. The size must be specified
   * before invoking GeneratePool(). Note the number of components will
   * affect the total size (allocated memory is Size*NumberOfComponents).
   */
  svtkSetClampMacro(Size, svtkIdType, 1, SVTK_ID_MAX);
  svtkGetMacro(Size, svtkIdType);
  //@}

  //@{
  /**
   * Methods to set and get the number of components in the pool. This is a
   * convenience capability and can be used to interface with
   * svtkDataArrays. By default the number of components is =1.
   */
  svtkSetClampMacro(NumberOfComponents, svtkIdType, 1, SVTK_INT_MAX);
  svtkGetMacro(NumberOfComponents, svtkIdType);
  //@}

  /**
   * This convenience method returns the total size of the memory pool, i.e.,
   * Size*NumberOfComponents.
   */
  svtkIdType GetTotalSize() { return (this->Size * this->NumberOfComponents); }

  //@{
  /**
   * These methods provide access to the raw random pool as a double
   * array. The size of the array is Size*NumberOfComponents. Each x value
   * ranges between (0<=x<=1). The class will generate the pool as necessary
   * (a modified time for generation is maintained). Also a method is
   * available for getting the value at the ith pool position and compNum
   * component. Finally, note that the GetValue() method uses modulo
   * reduction to ensure that the request remains inside of the pool. Two
   * forms are provided, the first assumes NumberOfComponents=1; the second
   * allows access to a particular component. The GetPool() and GetValue()
   * methods should only be called after GeneratePool() has been invoked;
   */
  const double* GeneratePool();
  const double* GetPool() { return this->Pool; }
  double GetValue(svtkIdType i) { return this->Pool[(i % this->TotalSize)]; }
  double GetValue(svtkIdType i, int compNum)
  {
    return this->Pool[(compNum + this->NumberOfComponents * i) % this->TotalSize];
  }
  //@}

  //@{
  /**
   * Methods to populate data arrays of various types with values within a
   * specified (min,max) range. Note that compNumber is used to specify the
   * range for a particular component; otherwise all generated components are
   * within the (min,max) range specified. (Thus it is possible to make
   * multiple calls to generate random numbers for each component with
   * different ranges.) Internally the type of the data array passed in is
   * used to cast to the appropriate type. Also the size and number of
   * components of the svtkDataArray controls the total number of random
   * numbers generated; so the input data array should be pre-allocated with
   * (SetNumberOfComponents, SetNumberOfTuples).
   */
  void PopulateDataArray(svtkDataArray* da, double minRange, double maxRange);
  void PopulateDataArray(svtkDataArray* da, int compNumber, double minRange, double maxRange);
  //@}

  //@{
  /**
   * Specify the work chunk size at which point multithreading kicks in. For small
   * memory pools < ChunkSize, no threading is used. Larger pools are computed using
   * svtkMultiThreader.
   */
  svtkSetClampMacro(ChunkSize, svtkIdType, 1000, SVTK_INT_MAX);
  svtkGetMacro(ChunkSize, svtkIdType);
  //@}

protected:
  svtkRandomPool();
  ~svtkRandomPool() override;

  // Keep track of last generation time
  svtkTimeStamp GenerateTime;

  // Data members to support public API
  svtkRandomSequence* Sequence;
  svtkIdType Size;
  int NumberOfComponents;
  svtkIdType ChunkSize;

  // Internal data members
  svtkIdType TotalSize;
  double* Pool;

private:
  svtkRandomPool(const svtkRandomPool&) = delete;
  void operator=(const svtkRandomPool&) = delete;
};

#endif
// SVTK-HeaderTest-Exclude: svtkRandomPool.h
