/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkStdString.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkStdString
 * @brief   Wrapper around std::string to keep symbols short.
 *
 * svtkStdString derives from std::string to provide shorter symbol
 * names than basic_string<...> in namespace std given by the standard
 * STL string.
 */

#ifndef svtkStdString_h
#define svtkStdString_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkSystemIncludes.h"   // For SVTKCOMMONCORE_EXPORT.
#include <string>                // For the superclass.

class svtkStdString;
SVTKCOMMONCORE_EXPORT ostream& operator<<(ostream&, const svtkStdString&);

class svtkStdString : public std::string
{
public:
  typedef std::string StdString;
  typedef StdString::value_type value_type;
  typedef StdString::pointer pointer;
  typedef StdString::reference reference;
  typedef StdString::const_reference const_reference;
  typedef StdString::size_type size_type;
  typedef StdString::difference_type difference_type;
  typedef StdString::iterator iterator;
  typedef StdString::const_iterator const_iterator;
  typedef StdString::reverse_iterator reverse_iterator;
  typedef StdString::const_reverse_iterator const_reverse_iterator;

  svtkStdString()
    : StdString()
  {
  }
  svtkStdString(const value_type* s)
    : StdString(s)
  {
  }
  svtkStdString(const value_type* s, size_type n)
    : StdString(s, n)
  {
  }
  svtkStdString(const StdString& s, size_type pos = 0, size_type n = npos)
    : StdString(s, pos, n)
  {
  }

  operator const char*() { return this->c_str(); }
};

#endif
// SVTK-HeaderTest-Exclude: svtkStdString.h
