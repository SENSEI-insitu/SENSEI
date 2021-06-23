/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkIdTypeArray.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkIdTypeArray
 * @brief   dynamic, self-adjusting array of svtkIdType
 *
 * svtkIdTypeArray is an array of values of type svtkIdType.
 * It provides methods for insertion and retrieval of values and will
 * automatically resize itself to hold new data.
 */

#ifndef svtkIdTypeArray_h
#define svtkIdTypeArray_h

#include "svtkAOSDataArrayTemplate.h" // Real Superclass
#include "svtkCommonCoreModule.h"     // For export macro
#include "svtkDataArray.h"

// Fake the superclass for the wrappers.
#ifndef __SVTK_WRAP__
#define svtkDataArray svtkAOSDataArrayTemplate<svtkIdType>
#endif
class SVTKCOMMONCORE_EXPORT svtkIdTypeArray : public svtkDataArray
{
public:
  svtkTypeMacro(svtkIdTypeArray, svtkDataArray);
#ifndef __SVTK_WRAP__
#undef svtkDataArray
#endif
  static svtkIdTypeArray* New();
  void PrintSelf(ostream& os, svtkIndent indent) override;

  // This macro expands to the set of method declarations that
  // make up the interface of svtkAOSDataArrayTemplate, which is ignored
  // by the wrappers.
#if defined(__SVTK_WRAP__) || defined(__WRAP_GCCXML__)
  svtkCreateWrappedArrayInterface(svtkIdType);
#else

  /**
   * Get the data type.
   */
  int GetDataType() const override
  {
    // This needs to overwritten from superclass because
    // the templated superclass is not able to differentiate
    // svtkIdType from a long long or an int since svtkIdType
    // is simply a typedef. This means that
    // svtkAOSDataArrayTemplate<svtkIdType> != svtkIdTypeArray.
    return SVTK_ID_TYPE;
  }
#endif

  /**
   * A faster alternative to SafeDownCast for downcasting svtkAbstractArrays.
   */
  static svtkIdTypeArray* FastDownCast(svtkAbstractArray* source)
  {
    return static_cast<svtkIdTypeArray*>(Superclass::FastDownCast(source));
  }

  /**
   * Get the minimum data value in its native type.
   */
  static svtkIdType GetDataTypeValueMin() { return SVTK_ID_MIN; }

  /**
   * Get the maximum data value in its native type.
   */
  static svtkIdType GetDataTypeValueMax() { return SVTK_ID_MAX; }

protected:
  svtkIdTypeArray();
  ~svtkIdTypeArray() override;

private:
  typedef svtkAOSDataArrayTemplate<svtkIdType> RealSuperclass;

  svtkIdTypeArray(const svtkIdTypeArray&) = delete;
  void operator=(const svtkIdTypeArray&) = delete;
};

// Define svtkArrayDownCast implementation:
svtkArrayDownCast_FastCastMacro(svtkIdTypeArray);

#endif
