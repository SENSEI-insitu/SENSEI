/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkFindCellStrategy.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkFindCellStrategy.h"

#include "svtkLogger.h"
#include "svtkPointSet.h"

//----------------------------------------------------------------------------
svtkFindCellStrategy::svtkFindCellStrategy()
{
  this->PointSet = nullptr;
}

//----------------------------------------------------------------------------
svtkFindCellStrategy::~svtkFindCellStrategy()
{
  // if ( this->PointSet != nullptr )
  // {
  //   svtkPointSet *ps = this->PointSet;
  //   this->PointSet = nullptr;
  //   ps->Delete();
  // }
}

//----------------------------------------------------------------------------
int svtkFindCellStrategy::Initialize(svtkPointSet* ps)
{
  // Make sure everything is up to snuff
  if (ps == nullptr || ps->GetPoints() == nullptr || ps->GetPoints()->GetNumberOfPoints() < 1)
  {
    svtkLog(ERROR, "Initialize must be called with non-NULL instance of svtkPointSet");
    return 0;
  }
  else
  {
    this->PointSet = ps;
    this->PointSet->GetBounds(this->Bounds);
    return 1;
  }
}

//----------------------------------------------------------------------------
void svtkFindCellStrategy::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "svtkPointSet: " << this->PointSet << "\n";
}
