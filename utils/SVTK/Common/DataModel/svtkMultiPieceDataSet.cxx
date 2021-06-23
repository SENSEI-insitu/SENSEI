/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkMultiPieceDataSet.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkMultiPieceDataSet.h"

#include "svtkInformation.h"
#include "svtkInformationVector.h"
#include "svtkObjectFactory.h"

svtkStandardNewMacro(svtkMultiPieceDataSet);
//----------------------------------------------------------------------------
svtkMultiPieceDataSet::svtkMultiPieceDataSet() = default;

//----------------------------------------------------------------------------
svtkMultiPieceDataSet::~svtkMultiPieceDataSet() = default;

//----------------------------------------------------------------------------
svtkMultiPieceDataSet* svtkMultiPieceDataSet::GetData(svtkInformation* info)
{
  return info ? svtkMultiPieceDataSet::SafeDownCast(info->Get(DATA_OBJECT())) : nullptr;
}

//----------------------------------------------------------------------------
svtkMultiPieceDataSet* svtkMultiPieceDataSet::GetData(svtkInformationVector* v, int i)
{
  return svtkMultiPieceDataSet::GetData(v->GetInformationObject(i));
}

//----------------------------------------------------------------------------
void svtkMultiPieceDataSet::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}
