/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkUnsignedIntArray.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkUnsignedIntArray
 * @brief   dynamic, self-adjusting array of unsigned int
 *
 * svtkUnsignedIntArray is an array of values of type unsigned int.  It
 * provides methods for insertion and retrieval of values and will
 * automatically resize itself to hold new data.
 *
 * The C++ standard does not define the exact size of the unsigned int type,
 * so use of this type directly is discouraged.  If an array of 32 bit unsigned
 * integers is needed, prefer svtkTypeUInt32Array to this class.
 */

#ifndef svtkUnsignedIntArray_h
#define svtkUnsignedIntArray_h

#include "svtkAOSDataArrayTemplate.h" // Real Superclass
#include "svtkCommonCoreModule.h"     // For export macro
#include "svtkDataArray.h"

// Fake the superclass for the wrappers.
#ifndef __SVTK_WRAP__
#define svtkDataArray svtkAOSDataArrayTemplate<unsigned int>
#endif
class SVTKCOMMONCORE_EXPORT svtkUnsignedIntArray : public svtkDataArray
{
public:
  svtkTypeMacro(svtkUnsignedIntArray, svtkDataArray);
#ifndef __SVTK_WRAP__
#undef svtkDataArray
#endif
  static svtkUnsignedIntArray* New();
  void PrintSelf(ostream& os, svtkIndent indent) override;

  // This macro expands to the set of method declarations that
  // make up the interface of svtkAOSDataArrayTemplate, which is ignored
  // by the wrappers.
#if defined(__SVTK_WRAP__) || defined(__WRAP_GCCXML__)
  svtkCreateWrappedArrayInterface(unsigned int);
#endif

  /**
   * A faster alternative to SafeDownCast for downcasting svtkAbstractArrays.
   */
  static svtkUnsignedIntArray* FastDownCast(svtkAbstractArray* source)
  {
    return static_cast<svtkUnsignedIntArray*>(Superclass::FastDownCast(source));
  }

  /**
   * Get the minimum data value in its native type.
   */
  static unsigned int GetDataTypeValueMin() { return SVTK_UNSIGNED_INT_MIN; }

  /**
   * Get the maximum data value in its native type.
   */
  static unsigned int GetDataTypeValueMax() { return SVTK_UNSIGNED_INT_MAX; }

protected:
  svtkUnsignedIntArray();
  ~svtkUnsignedIntArray() override;

private:
  typedef svtkAOSDataArrayTemplate<unsigned int> RealSuperclass;

  svtkUnsignedIntArray(const svtkUnsignedIntArray&) = delete;
  void operator=(const svtkUnsignedIntArray&) = delete;
};

// Define svtkArrayDownCast implementation:
svtkArrayDownCast_FastCastMacro(svtkUnsignedIntArray);

#endif
