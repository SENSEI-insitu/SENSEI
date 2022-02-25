/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkAbstractCellLinks.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkAbstractCellLinks.h"

#include "svtkCellArray.h"
#include "svtkObjectFactory.h"

//----------------------------------------------------------------------------
svtkAbstractCellLinks::svtkAbstractCellLinks()
{
  this->SequentialProcessing = false;
  this->Type = svtkAbstractCellLinks::LINKS_NOT_DEFINED;
}

//----------------------------------------------------------------------------
svtkAbstractCellLinks::~svtkAbstractCellLinks() = default;

//----------------------------------------------------------------------------
int svtkAbstractCellLinks::ComputeType(svtkIdType maxPtId, svtkIdType maxCellId, svtkCellArray* ca)
{
  svtkIdType numEntries = ca->GetNumberOfConnectivityIds();
  svtkIdType max = maxPtId;
  max = (maxCellId > max ? maxCellId : max);
  max = (numEntries > max ? numEntries : max);

  if (max < SVTK_UNSIGNED_SHORT_MAX)
  {
    return svtkAbstractCellLinks::STATIC_CELL_LINKS_USHORT;
  }
  // for 64bit IDS we might be able to use a unsigned int instead
#if defined(SVTK_USE_64BIT_IDS) && SVTK_SIZEOF_INT == 4
  else if (max < static_cast<svtkIdType>(SVTK_UNSIGNED_INT_MAX))
  {
    return svtkAbstractCellLinks::STATIC_CELL_LINKS_UINT;
  }
#endif
  return svtkAbstractCellLinks::STATIC_CELL_LINKS_IDTYPE;
}

//----------------------------------------------------------------------------
void svtkAbstractCellLinks::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "Sequential Processing: " << (this->SequentialProcessing ? "true\n" : "false\n");
  os << indent << "Type: " << this->Type << "\n";
}
