/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkHyperTreeGridTools.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright Nonice for more information.

=========================================================================*/

#ifndef svtkHyperTreeGridTools_h
#define svtkHyperTreeGridTools_h

namespace svtk
{
namespace hypertreegrid
{

template <class T>
bool HasTree(const T& e)
{
  return e.GetTree() != nullptr;
}

} // namespace hypertreegrid
} // namespace svtk

#endif // vtHyperTreeGridTools_h
// SVTK-HeaderTest-Exclude: svtkHyperTreeGridTools.h
