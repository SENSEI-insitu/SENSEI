/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkArrayIteratorTemplate.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkArrayIteratorTemplate
 * @brief   Implementation template for a array
 * iterator.
 *
 *
 * This is implementation template for a array iterator. It only works
 * with arrays that have a contiguous internal storage of values (as in
 * svtkDataArray, svtkStringArray).
 */

#ifndef svtkArrayIteratorTemplate_h
#define svtkArrayIteratorTemplate_h

#include "svtkArrayIterator.h"
#include "svtkCommonCoreModule.h" // For export macro

#include "svtkStdString.h"     // For template instantiation
#include "svtkUnicodeString.h" // For template instantiation
#include "svtkVariant.h"       // For template instantiation

template <class T>
class SVTKCOMMONCORE_EXPORT svtkArrayIteratorTemplate : public svtkArrayIterator
{
public:
  static svtkArrayIteratorTemplate<T>* New();
  svtkTemplateTypeMacro(svtkArrayIteratorTemplate<T>, svtkArrayIterator);
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
  svtkAbstractArray* GetArray() { return this->Array; }

  /**
   * Must be called only after Initialize.
   */
  T* GetTuple(svtkIdType id);

  /**
   * Must be called only after Initialize.
   */
  T& GetValue(svtkIdType id) { return this->Pointer[id]; }

  /**
   * Sets the value at the index. This does not verify if the index is
   * valid.  The caller must ensure that id is less than the maximum
   * number of values.
   */
  void SetValue(svtkIdType id, T value) { this->Pointer[id] = value; }

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
   * This is the data type for the value.
   */
  typedef T ValueType;

protected:
  svtkArrayIteratorTemplate();
  ~svtkArrayIteratorTemplate() override;

  T* Pointer;

private:
  svtkArrayIteratorTemplate(const svtkArrayIteratorTemplate&) = delete;
  void operator=(const svtkArrayIteratorTemplate&) = delete;

  void SetArray(svtkAbstractArray*);
  svtkAbstractArray* Array;
};

#ifdef SVTK_USE_EXTERN_TEMPLATE
#ifndef svtkArrayIteratorTemplateInstantiate_cxx
#ifdef _MSC_VER
#pragma warning(push)
// The following is needed when the svtkArrayIteratorTemplate is declared
// dllexport and is used from another class in svtkCommonCore
#pragma warning(disable : 4910) // extern and dllexport incompatible
#endif
svtkInstantiateTemplateMacro(extern template class SVTKCOMMONCORE_EXPORT svtkArrayIteratorTemplate);
extern template class SVTKCOMMONCORE_EXPORT svtkArrayIteratorTemplate<svtkStdString>;
extern template class SVTKCOMMONCORE_EXPORT svtkArrayIteratorTemplate<svtkUnicodeString>;
extern template class SVTKCOMMONCORE_EXPORT svtkArrayIteratorTemplate<svtkVariant>;
#ifdef _MSC_VER
#pragma warning(pop)
#endif
#endif
#endif // SVTK_USE_EXTERN_TEMPLATE

#endif

// SVTK-HeaderTest-Exclude: svtkArrayIteratorTemplate.h
