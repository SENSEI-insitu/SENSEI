/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkUnicodeStringArray.h

-------------------------------------------------------------------------
  Copyright 2008 Sandia Corporation.
  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
  the U.S. Government retains certain rights in this software.
-------------------------------------------------------------------------

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/**
 * @class   svtkUnicodeStringArray
 * @brief   Subclass of svtkAbstractArray that holds svtkUnicodeStrings
 *
 *
 *
 * @par Thanks:
 * Developed by Timothy M. Shead (tshead@sandia.gov) at Sandia National Laboratories.
 */

#ifndef svtkUnicodeStringArray_h
#define svtkUnicodeStringArray_h

#include "svtkAbstractArray.h"
#include "svtkCommonCoreModule.h" // For export macro
#include "svtkUnicodeString.h"    // For value type

class SVTKCOMMONCORE_EXPORT svtkUnicodeStringArray : public svtkAbstractArray
{
public:
  static svtkUnicodeStringArray* New();
  svtkTypeMacro(svtkUnicodeStringArray, svtkAbstractArray);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  svtkTypeBool Allocate(svtkIdType sz, svtkIdType ext = 1000) override;
  void Initialize() override;
  int GetDataType() const override;
  int GetDataTypeSize() const override;
  int GetElementComponentSize() const override;
  void SetNumberOfTuples(svtkIdType number) override;
  void SetTuple(svtkIdType i, svtkIdType j, svtkAbstractArray* source) override;
  void InsertTuple(svtkIdType i, svtkIdType j, svtkAbstractArray* source) override;
  void InsertTuples(svtkIdList* dstIds, svtkIdList* srcIds, svtkAbstractArray* source) override;
  void InsertTuples(
    svtkIdType dstStart, svtkIdType n, svtkIdType srcStart, svtkAbstractArray* source) override;
  svtkIdType InsertNextTuple(svtkIdType j, svtkAbstractArray* source) override;
  void* GetVoidPointer(svtkIdType id) override;
  void DeepCopy(svtkAbstractArray* da) override;
  void InterpolateTuple(
    svtkIdType i, svtkIdList* ptIndices, svtkAbstractArray* source, double* weights) override;
  void InterpolateTuple(svtkIdType i, svtkIdType id1, svtkAbstractArray* source1, svtkIdType id2,
    svtkAbstractArray* source2, double t) override;
  void Squeeze() override;
  svtkTypeBool Resize(svtkIdType numTuples) override;
  void SetVoidArray(void* array, svtkIdType size, int save) override;
  void SetVoidArray(void* array, svtkIdType size, int save, int deleteMethod) override;
  void SetArrayFreeFunction(void (*callback)(void*)) override;
  unsigned long GetActualMemorySize() const override; // in bytes
  int IsNumeric() const override;
  SVTK_NEWINSTANCE svtkArrayIterator* NewIterator() override;
  svtkVariant GetVariantValue(svtkIdType idx) override;
  svtkIdType LookupValue(svtkVariant value) override;
  void LookupValue(svtkVariant value, svtkIdList* ids) override;

  void SetVariantValue(svtkIdType idx, svtkVariant value) override;
  void InsertVariantValue(svtkIdType idx, svtkVariant value) override;
  void DataChanged() override;
  void ClearLookup() override;

  svtkIdType InsertNextValue(const svtkUnicodeString&);
  void InsertValue(svtkIdType idx, const svtkUnicodeString&); // Ranged checked
  void SetValue(svtkIdType i, const svtkUnicodeString&);      // Not ranged checked
  svtkUnicodeString& GetValue(svtkIdType i);

  void InsertNextUTF8Value(const char*);
  void SetUTF8Value(svtkIdType i, const char*);
  const char* GetUTF8Value(svtkIdType i);

protected:
  svtkUnicodeStringArray();
  ~svtkUnicodeStringArray() override;

private:
  svtkUnicodeStringArray(const svtkUnicodeStringArray&) = delete;
  void operator=(const svtkUnicodeStringArray&) = delete;

  class Implementation;
  Implementation* Internal;
};

#endif
