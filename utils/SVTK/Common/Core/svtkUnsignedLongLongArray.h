/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkUnsignedLongLongArray.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkUnsignedLongLongArray
 * @brief   dynamic, self-adjusting array of unsigned long long
 *
 * svtkUnsignedLongLongArray is an array of values of type unsigned long long.
 * It provides methods for insertion and retrieval of values and will
 * automatically resize itself to hold new data.
 *
 * This class should not be used directly, as it only exists on systems
 * where the unsigned long long type is defined.  If you need an unsigned
 * 64 bit integer data array, use svtkTypeUInt64Array instead.
 */

#ifndef svtkUnsignedLongLongArray_h
#define svtkUnsignedLongLongArray_h

#include "svtkAOSDataArrayTemplate.h" // Real Superclass
#include "svtkCommonCoreModule.h"     // For export macro
#include "svtkDataArray.h"

// Fake the superclass for the wrappers.
#ifndef __SVTK_WRAP__
#define svtkDataArray svtkAOSDataArrayTemplate<unsigned long long>
#endif
class SVTKCOMMONCORE_EXPORT svtkUnsignedLongLongArray : public svtkDataArray
{
public:
  svtkTypeMacro(svtkUnsignedLongLongArray, svtkDataArray);
#ifndef __SVTK_WRAP__
#undef svtkDataArray
#endif
  static svtkUnsignedLongLongArray* New();
  void PrintSelf(ostream& os, svtkIndent indent) override;

  // This macro expands to the set of method declarations that
  // make up the interface of svtkAOSDataArrayTemplate, which is ignored
  // by the wrappers.
#if defined(__SVTK_WRAP__) || defined(__WRAP_GCCXML__)
  svtkCreateWrappedArrayInterface(unsigned long long);
#endif

  /**
   * A faster alternative to SafeDownCast for downcasting svtkAbstractArrays.
   */
  static svtkUnsignedLongLongArray* FastDownCast(svtkAbstractArray* source)
  {
    return static_cast<svtkUnsignedLongLongArray*>(Superclass::FastDownCast(source));
  }

  /**
   * Get the minimum data value in its native type.
   */
  static unsigned long long GetDataTypeValueMin() { return SVTK_UNSIGNED_LONG_LONG_MIN; }

  /**
   * Get the maximum data value in its native type.
   */
  static unsigned long long GetDataTypeValueMax() { return SVTK_UNSIGNED_LONG_LONG_MAX; }

protected:
  svtkUnsignedLongLongArray();
  ~svtkUnsignedLongLongArray() override;

private:
  typedef svtkAOSDataArrayTemplate<unsigned long long> RealSuperclass;

  svtkUnsignedLongLongArray(const svtkUnsignedLongLongArray&) = delete;
  void operator=(const svtkUnsignedLongLongArray&) = delete;
};

// Define svtkArrayDownCast implementation:
svtkArrayDownCast_FastCastMacro(svtkUnsignedLongLongArray);

#endif
