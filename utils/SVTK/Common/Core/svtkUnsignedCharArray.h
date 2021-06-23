/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkUnsignedCharArray.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkUnsignedCharArray
 * @brief   dynamic, self-adjusting array of unsigned char
 *
 * svtkUnsignedCharArray is an array of values of type unsigned char.
 * It provides methods for insertion and retrieval of values and will
 * automatically resize itself to hold new data.
 */

#ifndef svtkUnsignedCharArray_h
#define svtkUnsignedCharArray_h

#include "svtkAOSDataArrayTemplate.h" // Real Superclass
#include "svtkCommonCoreModule.h"     // For export macro
#include "svtkDataArray.h"

// Fake the superclass for the wrappers.
#ifndef __SVTK_WRAP__
#define svtkDataArray svtkAOSDataArrayTemplate<unsigned char>
#endif
class SVTKCOMMONCORE_EXPORT svtkUnsignedCharArray : public svtkDataArray
{
public:
  svtkTypeMacro(svtkUnsignedCharArray, svtkDataArray);
#ifndef __SVTK_WRAP__
#undef svtkDataArray
#endif
  static svtkUnsignedCharArray* New();
  void PrintSelf(ostream& os, svtkIndent indent) override;

  // This macro expands to the set of method declarations that
  // make up the interface of svtkAOSDataArrayTemplate, which is ignored
  // by the wrappers.
#if defined(__SVTK_WRAP__) || defined(__WRAP_GCCXML__)
  svtkCreateWrappedArrayInterface(unsigned char);
#endif

  /**
   * A faster alternative to SafeDownCast for downcasting svtkAbstractArrays.
   */
  static svtkUnsignedCharArray* FastDownCast(svtkAbstractArray* source)
  {
    return static_cast<svtkUnsignedCharArray*>(Superclass::FastDownCast(source));
  }

  /**
   * Get the minimum data value in its native type.
   */
  static unsigned char GetDataTypeValueMin() { return SVTK_UNSIGNED_CHAR_MIN; }

  /**
   * Get the maximum data value in its native type.
   */
  static unsigned char GetDataTypeValueMax() { return SVTK_UNSIGNED_CHAR_MAX; }

protected:
  svtkUnsignedCharArray();
  ~svtkUnsignedCharArray() override;

private:
  typedef svtkAOSDataArrayTemplate<unsigned char> RealSuperclass;

  svtkUnsignedCharArray(const svtkUnsignedCharArray&) = delete;
  void operator=(const svtkUnsignedCharArray&) = delete;
};

// Define svtkArrayDownCast implementation:
svtkArrayDownCast_FastCastMacro(svtkUnsignedCharArray);

#endif
