/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkScaledSOADataArrayTemplate.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkScaledSOADataArrayTemplate
 * @brief   Struct-Of-Arrays implementation of
 * svtkGenericDataArray with a scaling factor.
 *
 *
 * svtkScaledSOADataArrayTemplate is the counterpart of svtkSOADataArrayTemplate with a
 * scaling factor. Each component is stored in a separate array. The Scale value is
 * used to multiply the output of the stored value in the array. For example, if Scale
 * is 2 and the requested tuple value stored in memory is [1, 2, 3] then the returned tuple
 * values will actually be [2, 4, 6]. Similarly, if Scale is 2 and the tuple values
 * for SetTupleValue() is [2, 4, 6] then the stored values in memory will be
 * [1, 2, 3].
 *
 * @sa
 * svtkGenericDataArray svtkSOADataArrayTemplate
 */

#ifndef svtkScaledSOADataArrayTemplate_h
#define svtkScaledSOADataArrayTemplate_h

#include "svtkBuffer.h"
#include "svtkCommonCoreModule.h" // For export macro
#include "svtkGenericDataArray.h"

// The export macro below makes no sense, but is necessary for older compilers
// when we export instantiations of this class from svtkCommonCore.
template <class ValueTypeT>
class SVTKCOMMONCORE_EXPORT svtkScaledSOADataArrayTemplate
  : public svtkGenericDataArray<svtkScaledSOADataArrayTemplate<ValueTypeT>, ValueTypeT>
{
  typedef svtkGenericDataArray<svtkScaledSOADataArrayTemplate<ValueTypeT>, ValueTypeT>
    GenericDataArrayType;

public:
  typedef svtkScaledSOADataArrayTemplate<ValueTypeT> SelfType;
  svtkTemplateTypeMacro(SelfType, GenericDataArrayType);
  typedef typename Superclass::ValueType ValueType;

  enum DeleteMethod
  {
    SVTK_DATA_ARRAY_FREE = svtkAbstractArray::SVTK_DATA_ARRAY_FREE,
    SVTK_DATA_ARRAY_DELETE = svtkAbstractArray::SVTK_DATA_ARRAY_DELETE,
    SVTK_DATA_ARRAY_ALIGNED_FREE = svtkAbstractArray::SVTK_DATA_ARRAY_ALIGNED_FREE,
    SVTK_DATA_ARRAY_USER_DEFINED = svtkAbstractArray::SVTK_DATA_ARRAY_USER_DEFINED
  };

  static svtkScaledSOADataArrayTemplate* New();

  //@{
  /**
   * Set/Get the Scale value for the object. The default is 1.
   */
  void SetScale(ValueType scale)
  {
    if (scale != this->Scale)
    {
      if (scale == 0)
      {
        svtkErrorMacro("Cannot set Scale to 0");
      }
      else
      {
        this->Scale = scale;
        this->Modified();
      }
    }
  }
  ValueType GetScale() const { return this->Scale; }
  //@}

  //@{
  /**
   * Get the value at @a valueIdx. @a valueIdx assumes AOS ordering.
   */
  inline ValueType GetValue(svtkIdType valueIdx) const
  {
    svtkIdType tupleIdx;
    int comp;
    this->GetTupleIndexFromValueIndex(valueIdx, tupleIdx, comp);
    return this->GetTypedComponent(tupleIdx, comp);
  }
  //@}

  //@{
  /**
   * Set the value at @a valueIdx to @a value. @a valueIdx assumes AOS ordering.
   */
  inline void SetValue(svtkIdType valueIdx, ValueType value)
  {
    svtkIdType tupleIdx;
    int comp;
    this->GetTupleIndexFromValueIndex(valueIdx, tupleIdx, comp);
    this->SetTypedComponent(tupleIdx, comp, value);
  }
  //@}

  /**
   * Copy the tuple at @a tupleIdx into @a tuple.
   */
  inline void GetTypedTuple(svtkIdType tupleIdx, ValueType* tuple) const
  {
    for (size_t cc = 0; cc < this->Data.size(); cc++)
    {
      tuple[cc] = this->Data[cc]->GetBuffer()[tupleIdx] * this->Scale;
    }
  }

  /**
   * Set this array's tuple at @a tupleIdx to the values in @a tuple.
   */
  inline void SetTypedTuple(svtkIdType tupleIdx, const ValueType* tuple)
  {
    for (size_t cc = 0; cc < this->Data.size(); ++cc)
    {
      this->Data[cc]->GetBuffer()[tupleIdx] = tuple[cc] / this->Scale;
    }
  }

  /**
   * Get component @a comp of the tuple at @a tupleIdx.
   */
  inline ValueType GetTypedComponent(svtkIdType tupleIdx, int comp) const
  {
    return this->Data[comp]->GetBuffer()[tupleIdx] * this->Scale;
  }

  /**
   * Set component @a comp of the tuple at @a tupleIdx to @a value.
   */
  inline void SetTypedComponent(svtkIdType tupleIdx, int comp, ValueType value)
  {
    this->Data[comp]->GetBuffer()[tupleIdx] = value / this->Scale;
  }

  /**
   * Set component @a comp of all tuples to @a value.
   */
  void FillTypedComponent(int compIdx, ValueType value) override;

  /**
   * Use this API to pass externally allocated memory to this instance. Since
   * svtkScaledSOADataArrayTemplate uses separate contiguous regions for each
   * component, use this API to add arrays for each of the component.
   * \c save: When set to true, svtkScaledSOADataArrayTemplate will not release or
   * realloc the memory even when the AllocatorType is set to RESIZABLE. If
   * needed it will simply allow new memory buffers and "forget" the supplied
   * pointers. When save is set to false, this will be the \c deleteMethod
   * specified to release the array.
   * If updateMaxId is true, the array's MaxId will be updated, and assumes
   * that size is the number of tuples in the array.
   * \c size is specified in number of elements of ScalarType.
   */
  void SetArray(int comp, SVTK_ZEROCOPY ValueType* array, svtkIdType size, bool updateMaxId = false,
    bool save = false, int deleteMethod = SVTK_DATA_ARRAY_FREE);

  /**
   * This method allows the user to specify a custom free function to be
   * called when the array is deallocated. Calling this method will implicitly
   * mean that the given free function will be called when the class
   * cleans up or reallocates memory. This custom free function will be
   * used for all components.
   **/
  void SetArrayFreeFunction(void (*callback)(void*)) override;

  /**
   * This method allows the user to specify a custom free function to be
   * called when the array is deallocated. Calling this method will implicitly
   * mean that the given free function will be called when the class
   * cleans up or reallocates memory.
   **/
  void SetArrayFreeFunction(int comp, void (*callback)(void*));

  /**
   * Return a pointer to a contiguous block of memory containing all values for
   * a particular components (ie. a single array of the struct-of-arrays). Note
   * that this is to raw memory and no scaling of the data is done here.
   */
  ValueType* GetComponentArrayPointer(int comp);

  /**
   * Use of this method is discouraged, it creates a deep copy of the data into
   * a contiguous AoS-ordered buffer and prints a warning.
   */
  void* GetVoidPointer(svtkIdType valueIdx) override;

  /**
   * Export a copy of the data in AoS ordering to the preallocated memory
   * buffer.
   */
  void ExportToVoidPointer(void* ptr) override;

#ifndef __SVTK_WRAP__
  //@{
  /**
   * Perform a fast, safe cast from a svtkAbstractArray to a svtkDataArray.
   * This method checks if source->GetArrayType() returns DataArray
   * or a more derived type, and performs a static_cast to return
   * source as a svtkDataArray pointer. Otherwise, nullptr is returned.
   */
  static svtkScaledSOADataArrayTemplate<ValueType>* FastDownCast(svtkAbstractArray* source)
  {
    if (source)
    {
      switch (source->GetArrayType())
      {
        case svtkAbstractArray::ScaleSoADataArrayTemplate:
          if (svtkDataTypesCompare(source->GetDataType(), svtkTypeTraits<ValueType>::SVTK_TYPE_ID))
          {
            return static_cast<svtkScaledSOADataArrayTemplate<ValueType>*>(source);
          }
          break;
      }
    }
    return nullptr;
  }
  //@}
#endif

  int GetArrayType() const override { return svtkAbstractArray::ScaleSoADataArrayTemplate; }
  SVTK_NEWINSTANCE svtkArrayIterator* NewIterator() override;
  void SetNumberOfComponents(int numComps) override;
  void ShallowCopy(svtkDataArray* other) override;

  // Reimplemented for efficiency:
  void InsertTuples(
    svtkIdType dstStart, svtkIdType n, svtkIdType srcStart, svtkAbstractArray* source) override;
  // MSVC doesn't like 'using' here (error C2487). Just forward instead:
  // using Superclass::InsertTuples;
  void InsertTuples(svtkIdList* dstIds, svtkIdList* srcIds, svtkAbstractArray* source) override
  {
    this->Superclass::InsertTuples(dstIds, srcIds, source);
  }

protected:
  svtkScaledSOADataArrayTemplate();
  ~svtkScaledSOADataArrayTemplate() override;

  /**
   * Allocate space for numTuples. Old data is not preserved. If numTuples == 0,
   * all data is freed.
   */
  bool AllocateTuples(svtkIdType numTuples);

  /**
   * Allocate space for numTuples. Old data is preserved. If numTuples == 0,
   * all data is freed.
   */
  bool ReallocateTuples(svtkIdType numTuples);

  std::vector<svtkBuffer<ValueType>*> Data;
  svtkBuffer<ValueType>* AoSCopy;

private:
  svtkScaledSOADataArrayTemplate(const svtkScaledSOADataArrayTemplate&) = delete;
  void operator=(const svtkScaledSOADataArrayTemplate&) = delete;

  inline void GetTupleIndexFromValueIndex(svtkIdType valueIdx, svtkIdType& tupleIdx, int& comp) const
  {
    tupleIdx = valueIdx / this->NumberOfComponents;
    comp = valueIdx % this->NumberOfComponents;
  }

  friend class svtkGenericDataArray<svtkScaledSOADataArrayTemplate<ValueTypeT>, ValueTypeT>;
  /**
   * The value to scale the data stored in memory by.
   */
  ValueType Scale;
};

// Declare svtkArrayDownCast implementations for scale SoA containers:
svtkArrayDownCast_TemplateFastCastMacro(svtkScaledSOADataArrayTemplate);

#endif // header guard

// This portion must be OUTSIDE the include blockers. This is used to tell
// libraries other than svtkCommonCore that instantiations of
// svtkScaledSOADataArrayTemplate can be found externally. This prevents each library
// from instantiating these on their own.
#ifdef SVTK_SCALED_SOA_DATA_ARRAY_TEMPLATE_INSTANTIATING
#define SVTK_SCALED_SOA_DATA_ARRAY_TEMPLATE_INSTANTIATE(T)                                          \
  template class SVTKCOMMONCORE_EXPORT svtkScaledSOADataArrayTemplate<T>
#elif defined(SVTK_USE_EXTERN_TEMPLATE)
#ifndef SVTK_SCALED_SOA_DATA_ARRAY_TEMPLATE_EXTERN
#define SVTK_SCALED_SOA_DATA_ARRAY_TEMPLATE_EXTERN
#ifdef _MSC_VER
#pragma warning(push)
// The following is needed when the svtkScaledSOADataArrayTemplate is declared
// dllexport and is used from another class in svtkCommonCore
#pragma warning(disable : 4910) // extern and dllexport incompatible
#endif
svtkExternTemplateMacro(extern template class SVTKCOMMONCORE_EXPORT svtkScaledSOADataArrayTemplate);
#ifdef _MSC_VER
#pragma warning(pop)
#endif
#endif // SVTK_SCALED_SOA_DATA_ARRAY_TEMPLATE_EXTERN

// The following clause is only for MSVC 2008 and 2010
#elif defined(_MSC_VER) && !defined(SVTK_BUILD_SHARED_LIBS)
#pragma warning(push)

// C4091: 'extern ' : ignored on left of 'int' when no variable is declared
#pragma warning(disable : 4091)

// Compiler-specific extension warning.
#pragma warning(disable : 4231)

// We need to disable warning 4910 and do an extern dllexport
// anyway.  When deriving new arrays from an
// instantiation of this template the compiler does an explicit
// instantiation of the base class.  From outside the svtkCommon
// library we block this using an extern dllimport instantiation.
// For classes inside svtkCommon we should be able to just do an
// extern instantiation, but VS 2008 complains about missing
// definitions.  We cannot do an extern dllimport inside svtkCommon
// since the symbols are local to the dll.  An extern dllexport
// seems to be the only way to convince VS 2008 to do the right
// thing, so we just disable the warning.
#pragma warning(disable : 4910) // extern and dllexport incompatible

// Use an "extern explicit instantiation" to give the class a DLL
// interface.  This is a compiler-specific extension.
svtkInstantiateTemplateMacro(
  extern template class SVTKCOMMONCORE_EXPORT svtkScaledSOADataArrayTemplate);

#pragma warning(pop)

#endif

// SVTK-HeaderTest-Exclude: svtkScaledSOADataArrayTemplate.h
