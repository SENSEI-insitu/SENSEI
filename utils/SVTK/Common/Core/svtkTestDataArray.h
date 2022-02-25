/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkTestDataArray.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkTestDataArray
 * @brief   An implementation of svtkGenericDataArray for testing
 * fallback algorithms.
 *
 *
 * svtkTestDataArray is derived from svtkGenericDataArray, and is deliberately
 * omitted from SVTK's whitelist of dispatchable data arrays. It is used to test
 * the fallback mechanisms of algorithms in the case that array dispatch fails.
 *
 * @sa
 * svtkGenericDataArray
 */

#ifndef svtkTestDataArray_h
#define svtkTestDataArray_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkGenericDataArray.h"
#include "svtkObjectFactory.h" // For SVTK_STANDARD_NEW_BODY

template <class ArrayT>
class svtkTestDataArray
  : public svtkGenericDataArray<svtkTestDataArray<ArrayT>, typename ArrayT::ValueType>
{
public:
  typedef ArrayT ArrayType;
  typedef typename ArrayType::ValueType ValueType;
  typedef svtkTestDataArray<ArrayT> SelfType;
  typedef svtkGenericDataArray<svtkTestDataArray<ArrayT>, ValueType> GenericDataArrayType;
  friend class svtkGenericDataArray<svtkTestDataArray<ArrayT>, ValueType>;

  svtkAbstractTemplateTypeMacro(SelfType, GenericDataArrayType);
  svtkAOSArrayNewInstanceMacro(SelfType);

  static svtkTestDataArray<ArrayType>* New() { SVTK_STANDARD_NEW_BODY(svtkTestDataArray<ArrayType>); }

  void PrintSelf(ostream& os, svtkIndent indent) override
  {
    GenericDataArrayType::PrintSelf(os, indent);
  }

  ValueType GetValue(svtkIdType valueIdx) const { return this->Array->GetValue(valueIdx); }
  void SetValue(svtkIdType valueIdx, ValueType value) { this->Array->SetValue(valueIdx, value); }

  void GetTypedTuple(svtkIdType tupleIdx, ValueType* tuple) const
  {
    this->Array->SetTypedTuple(tupleIdx, tuple);
  }
  void SetTypedTuple(svtkIdType tupleIdx, const ValueType* tuple)
  {
    this->Array->SetTypedTuple(tupleIdx, tuple);
  }

  ValueType GetTypedComponent(svtkIdType tupleIdx, int compIdx) const
  {
    return this->Array->GetTypedComponent(tupleIdx, compIdx);
  }
  void SetTypedComponent(svtkIdType tupleIdx, int compIdx, ValueType value)
  {
    this->Array->SetTypedComponent(tupleIdx, compIdx, value);
  }

  void* GetVoidPointer(svtkIdType valueIdx) override
  {
    return this->Array->GetVoidPointer(valueIdx);
  }

protected:
  svtkTestDataArray() { this->Array = ArrayType::New(); }
  ~svtkTestDataArray() override { this->Array->Delete(); }

  bool AllocateTuples(svtkIdType numTuples) { return this->Array->Allocate(numTuples) != 0; }
  bool ReallocateTuples(svtkIdType numTuples) { return this->Array->Allocate(numTuples) != 0; }

private:
  ArrayType* Array;

  svtkTestDataArray(const svtkTestDataArray&) = delete;
  void operator=(const svtkTestDataArray&) = delete;
};

#endif
// SVTK-HeaderTest-Exclude: svtkTestDataArray.h
