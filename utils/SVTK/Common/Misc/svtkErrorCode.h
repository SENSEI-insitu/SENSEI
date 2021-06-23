/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkErrorCode.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkErrorCode
 * @brief   superclass for error codes
 *
 * svtkErrorCode is an mechanism for (currently) reader object to
 * return errors during reading file.
 */

#ifndef svtkErrorCode_h
#define svtkErrorCode_h
#include "svtkCommonMiscModule.h" // For export macro
#include "svtkSystemIncludes.h"

// The superclass that all commands should be subclasses of
class SVTKCOMMONMISC_EXPORT svtkErrorCode
{
public:
  static const char* GetStringFromErrorCode(unsigned long event);
  static unsigned long GetErrorCodeFromString(const char* event);
  static unsigned long GetLastSystemError();

  // all the currently defined error codes
  // developers can use -- svtkErrorCode::UserError + int to
  // specify their own errors.
  // if this list is adjusted, be sure to adjust svtkErrorCodeErrorStrings
  // in svtkErrorCode.cxx to match.
  enum ErrorIds
  {
    NoError = 0,
    FirstSVTKErrorCode = 20000,
    FileNotFoundError,
    CannotOpenFileError,
    UnrecognizedFileTypeError,
    PrematureEndOfFileError,
    FileFormatError,
    NoFileNameError,
    OutOfDiskSpaceError,
    UnknownError,
    UserError = 40000
  };
};

#endif /* svtkErrorCode_h */

// SVTK-HeaderTest-Exclude: svtkErrorCode.h
