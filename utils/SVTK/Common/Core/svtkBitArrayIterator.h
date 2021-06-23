/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkBitArrayIterator.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkBitArrayIterator
 * @brief   Iterator for svtkBitArray.
 * This iterator iterates over a svtkBitArray. It uses the double interface
 * to get/set bit values.
 */

#ifndef svtkBitArrayIterator_h
#define svtkBitArrayIterator_h

#include "svtkArrayIterator.h"
#include "svtkCommonCoreModule.h" // For export macro

class svtkBitArray;
class SVTKCOMMONCORE_EXPORT svtkBitArrayIterator : public svtkArrayIterator
{
public:
  static svtkBitArrayIterator* New();
  svtkTypeMacro(svtkBitArrayIterator, svtkArrayIterator);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Set the array this iterator will iterate over.
   * After Initialize() has been called, the iterator is valid
   * so long as the Array has not been modified
   * (except using the iterator itself).
   * If the array is modified, the iterator must be re-initialized.
   */
  void Initialize(svtkAbstractArray* array) override;

  /**
   * Get the array.
   */
  svtkAbstractArray* GetArray();

  /**
   * Must be called only after Initialize.
   */
  int* GetTuple(svtkIdType id);

  /**
   * Must be called only after Initialize.
   */
  int GetValue(svtkIdType id);

  /**
   * Must be called only after Initialize.
   */
  svtkIdType GetNumberOfTuples();

  /**
   * Must be called only after Initialize.
   */
  svtkIdType GetNumberOfValues();

  /**
   * Must be called only after Initialize.
   */
  int GetNumberOfComponents();

  /**
   * Get the data type from the underlying array.
   */
  int GetDataType() const override;

  /**
   * Get the data type size from the underlying array.
   */
  int GetDataTypeSize() const;

  /**
   * Sets the value at the index. This does not verify if the index is valid.
   * The caller must ensure that id is less than the maximum number of values.
   */
  void SetValue(svtkIdType id, int value);

  /**
   * Data type of a value.
   */
  typedef int ValueType;

protected:
  svtkBitArrayIterator();
  ~svtkBitArrayIterator() override;

  int* Tuple;
  int TupleSize;
  void SetArray(svtkBitArray* b);
  svtkBitArray* Array;

private:
  svtkBitArrayIterator(const svtkBitArrayIterator&) = delete;
  void operator=(const svtkBitArrayIterator&) = delete;
};

#endif
