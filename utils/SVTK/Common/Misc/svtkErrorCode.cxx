/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkErrorCode.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkErrorCode.h"

#include <cctype>
#include <cerrno>
#include <cstring>

// this list should only contain the initial, contiguous
// set of error codes and should not include UserError
static const char* svtkErrorCodeErrorStrings[] = { "NoError", "FileNotFoundError",
  "CannotOpenFileError", "UnrecognizedFileTypeError", "PrematureEndOfFileError", "FileFormatError",
  "NoFileNameError", "OutOfDiskSpaceError", "UnknownError", "UserError", nullptr };

const char* svtkErrorCode::GetStringFromErrorCode(unsigned long error)
{
  static unsigned long numerrors = 0;
  if (error < FirstSVTKErrorCode)
  {
    return strerror(static_cast<int>(error));
  }
  else
  {
    error -= FirstSVTKErrorCode;
  }

  // find length of table
  if (!numerrors)
  {
    while (svtkErrorCodeErrorStrings[numerrors] != nullptr)
    {
      numerrors++;
    }
  }
  if (error < numerrors)
  {
    return svtkErrorCodeErrorStrings[error];
  }
  else if (error == svtkErrorCode::UserError)
  {
    return "UserError";
  }
  else
  {
    return "NoError";
  }
}

unsigned long svtkErrorCode::GetErrorCodeFromString(const char* error)
{
  unsigned long i;

  for (i = 0; svtkErrorCodeErrorStrings[i] != nullptr; i++)
  {
    if (!strcmp(svtkErrorCodeErrorStrings[i], error))
    {
      return i;
    }
  }
  if (!strcmp("UserError", error))
  {
    return svtkErrorCode::UserError;
  }
  return svtkErrorCode::NoError;
}

unsigned long svtkErrorCode::GetLastSystemError()
{
  return static_cast<unsigned long>(errno);
}
