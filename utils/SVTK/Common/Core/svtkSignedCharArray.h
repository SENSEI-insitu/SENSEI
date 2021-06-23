/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkSignedCharArray.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkSignedCharArray
 * @brief   dynamic, self-adjusting array of signed char
 *
 * svtkSignedCharArray is an array of values of type signed char.
 * It provides methods for insertion and retrieval of values and will
 * automatically resize itself to hold new data.
 */

#ifndef svtkSignedCharArray_h
#define svtkSignedCharArray_h

#include "svtkAOSDataArrayTemplate.h" // Real Superclass
#include "svtkCommonCoreModule.h"     // For export macro
#include "svtkDataArray.h"

// Fake the superclass for the wrappers.
#ifndef __SVTK_WRAP__
#define svtkDataArray svtkAOSDataArrayTemplate<signed char>
#endif
class SVTKCOMMONCORE_EXPORT svtkSignedCharArray : public svtkDataArray
{
public:
  svtkTypeMacro(svtkSignedCharArray, svtkDataArray);
#ifndef __SVTK_WRAP__
#undef svtkDataArray
#endif
  static svtkSignedCharArray* New();
  void PrintSelf(ostream& os, svtkIndent indent) override;

  // This macro expands to the set of method declarations that
  // make up the interface of svtkAOSDataArrayTemplate, which is ignored
  // by the wrappers.
#if defined(__SVTK_WRAP__) || defined(__WRAP_GCCXML__)
  svtkCreateWrappedArrayInterface(signed char);
#endif

  /**
   * A faster alternative to SafeDownCast for downcasting svtkAbstractArrays.
   */
  static svtkSignedCharArray* FastDownCast(svtkAbstractArray* source)
  {
    return static_cast<svtkSignedCharArray*>(Superclass::FastDownCast(source));
  }

  /**
   * Get the minimum data value in its native type.
   */
  static signed char GetDataTypeValueMin() { return SVTK_SIGNED_CHAR_MIN; }

  /**
   * Get the maximum data value in its native type.
   */
  static signed char GetDataTypeValueMax() { return SVTK_SIGNED_CHAR_MAX; }

protected:
  svtkSignedCharArray();
  ~svtkSignedCharArray() override;

private:
  typedef svtkAOSDataArrayTemplate<signed char> RealSuperclass;

  svtkSignedCharArray(const svtkSignedCharArray&) = delete;
  void operator=(const svtkSignedCharArray&) = delete;
};

// Define svtkArrayDownCast implementation:
svtkArrayDownCast_FastCastMacro(svtkSignedCharArray);

#endif
