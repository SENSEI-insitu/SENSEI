/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkMatrixToHomogeneousTransform.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkMatrixToHomogeneousTransform.h"

#include "svtkMatrix4x4.h"
#include "svtkObjectFactory.h"

svtkStandardNewMacro(svtkMatrixToHomogeneousTransform);
svtkCxxSetObjectMacro(svtkMatrixToHomogeneousTransform, Input, svtkMatrix4x4);

//----------------------------------------------------------------------------
svtkMatrixToHomogeneousTransform::svtkMatrixToHomogeneousTransform()
{
  this->Input = nullptr;
  this->InverseFlag = 0;
}

//----------------------------------------------------------------------------
svtkMatrixToHomogeneousTransform::~svtkMatrixToHomogeneousTransform()
{
  this->SetInput(nullptr);
}

//----------------------------------------------------------------------------
void svtkMatrixToHomogeneousTransform::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Update();

  this->Superclass::PrintSelf(os, indent);
  os << indent << "Input: " << this->Input << "\n";
  os << indent << "InverseFlag: " << this->InverseFlag << "\n";
}

//----------------------------------------------------------------------------
void svtkMatrixToHomogeneousTransform::Inverse()
{
  this->InverseFlag = !this->InverseFlag;
  this->Modified();
}

//----------------------------------------------------------------------------
void svtkMatrixToHomogeneousTransform::InternalUpdate()
{
  if (this->Input)
  {
    this->Matrix->DeepCopy(this->Input);
    if (this->InverseFlag)
    {
      this->Matrix->Invert();
    }
  }
  else
  {
    this->Matrix->Identity();
  }
}

//----------------------------------------------------------------------------
void svtkMatrixToHomogeneousTransform::InternalDeepCopy(svtkAbstractTransform* gtrans)
{
  svtkMatrixToHomogeneousTransform* transform =
    static_cast<svtkMatrixToHomogeneousTransform*>(gtrans);

  this->SetInput(transform->Input);

  if (this->InverseFlag != transform->InverseFlag)
  {
    this->Inverse();
  }
}

//----------------------------------------------------------------------------
svtkAbstractTransform* svtkMatrixToHomogeneousTransform::MakeTransform()
{
  return svtkMatrixToHomogeneousTransform::New();
}

//----------------------------------------------------------------------------
// Get the MTime
svtkMTimeType svtkMatrixToHomogeneousTransform::GetMTime()
{
  svtkMTimeType mtime = this->svtkHomogeneousTransform::GetMTime();

  if (this->Input)
  {
    svtkMTimeType matrixMTime = this->Input->GetMTime();
    if (matrixMTime > mtime)
    {
      return matrixMTime;
    }
  }
  return mtime;
}
