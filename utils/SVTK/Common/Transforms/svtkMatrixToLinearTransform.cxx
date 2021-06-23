/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkMatrixToLinearTransform.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkMatrixToLinearTransform.h"

#include "svtkMatrix4x4.h"
#include "svtkObjectFactory.h"

svtkStandardNewMacro(svtkMatrixToLinearTransform);
svtkCxxSetObjectMacro(svtkMatrixToLinearTransform, Input, svtkMatrix4x4);

//----------------------------------------------------------------------------
svtkMatrixToLinearTransform::svtkMatrixToLinearTransform()
{
  this->Input = nullptr;
  this->InverseFlag = 0;
}

//----------------------------------------------------------------------------
svtkMatrixToLinearTransform::~svtkMatrixToLinearTransform()
{
  this->SetInput(nullptr);
}

//----------------------------------------------------------------------------
void svtkMatrixToLinearTransform::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Update();

  this->Superclass::PrintSelf(os, indent);
  os << indent << "Input: " << this->Input << "\n";
  os << indent << "InverseFlag: " << this->InverseFlag << "\n";
}

//----------------------------------------------------------------------------
void svtkMatrixToLinearTransform::Inverse()
{
  this->InverseFlag = !this->InverseFlag;
  this->Modified();
}

//----------------------------------------------------------------------------
void svtkMatrixToLinearTransform::InternalUpdate()
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
void svtkMatrixToLinearTransform::InternalDeepCopy(svtkAbstractTransform* gtrans)
{
  svtkMatrixToLinearTransform* transform = static_cast<svtkMatrixToLinearTransform*>(gtrans);

  this->SetInput(transform->Input);

  if (this->InverseFlag != transform->InverseFlag)
  {
    this->Inverse();
  }
}

//----------------------------------------------------------------------------
svtkAbstractTransform* svtkMatrixToLinearTransform::MakeTransform()
{
  return svtkMatrixToLinearTransform::New();
}

//----------------------------------------------------------------------------
// Get the MTime
svtkMTimeType svtkMatrixToLinearTransform::GetMTime()
{
  svtkMTimeType mtime = this->svtkLinearTransform::GetMTime();

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
