/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkCharArray.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkCharArray
 * @brief   dynamic, self-adjusting array of char
 *
 * svtkCharArray is an array of values of type char.  It provides
 * methods for insertion and retrieval of values and will
 * automatically resize itself to hold new data.
 */

#ifndef svtkCharArray_h
#define svtkCharArray_h

#include "svtkAOSDataArrayTemplate.h" // Real Superclass
#include "svtkCommonCoreModule.h"     // For export macro
#include "svtkDataArray.h"

// Fake the superclass for the wrappers.
#ifndef __SVTK_WRAP__
#define svtkDataArray svtkAOSDataArrayTemplate<char>
#endif
class SVTKCOMMONCORE_EXPORT svtkCharArray : public svtkDataArray
{
public:
  svtkTypeMacro(svtkCharArray, svtkDataArray);
#ifndef __SVTK_WRAP__
#undef svtkDataArray
#endif
  static svtkCharArray* New();
  void PrintSelf(ostream& os, svtkIndent indent) override;

  // This macro expands to the set of method declarations that
  // make up the interface of svtkAOSDataArrayTemplate, which is ignored
  // by the wrappers.
#if defined(__SVTK_WRAP__) || defined(__WRAP_GCCXML__)
  svtkCreateWrappedArrayInterface(char);
#endif

  /**
   * A faster alternative to SafeDownCast for downcasting svtkAbstractArrays.
   */
  static svtkCharArray* FastDownCast(svtkAbstractArray* source)
  {
    return static_cast<svtkCharArray*>(Superclass::FastDownCast(source));
  }

  /**
   * Get the minimum data value in its native type.
   */
  static char GetDataTypeValueMin() { return SVTK_CHAR_MIN; }

  /**
   * Get the maximum data value in its native type.
   */
  static char GetDataTypeValueMax() { return SVTK_CHAR_MAX; }

protected:
  svtkCharArray();
  ~svtkCharArray() override;

private:
  typedef svtkAOSDataArrayTemplate<char> RealSuperclass;

  svtkCharArray(const svtkCharArray&) = delete;
  void operator=(const svtkCharArray&) = delete;
};

// Define svtkArrayDownCast implementation:
svtkArrayDownCast_FastCastMacro(svtkCharArray);

#endif
