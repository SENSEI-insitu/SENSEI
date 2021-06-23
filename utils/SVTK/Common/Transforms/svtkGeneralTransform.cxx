/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkGeneralTransform.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkGeneralTransform.h"
#include "svtkMath.h"
#include "svtkObjectFactory.h"

svtkStandardNewMacro(svtkGeneralTransform);

//----------------------------------------------------------------------------
svtkGeneralTransform::svtkGeneralTransform()
{
  this->Input = nullptr;

  // most of the functionality is provided by the concatenation
  this->Concatenation = svtkTransformConcatenation::New();

  // the stack will be allocated the first time Push is called
  this->Stack = nullptr;
}

//----------------------------------------------------------------------------
svtkGeneralTransform::~svtkGeneralTransform()
{
  this->SetInput(nullptr);

  if (this->Concatenation)
  {
    this->Concatenation->Delete();
  }
  if (this->Stack)
  {
    this->Stack->Delete();
  }
}

//----------------------------------------------------------------------------
void svtkGeneralTransform::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  os << indent << "Input: (" << this->Input << ")\n";
  os << indent << "InverseFlag: " << this->GetInverseFlag() << "\n";
  os << indent << "NumberOfConcatenatedTransforms: " << this->GetNumberOfConcatenatedTransforms()
     << "\n";
  if (this->GetNumberOfConcatenatedTransforms() != 0)
  {
    int n = this->GetNumberOfConcatenatedTransforms();
    for (int i = 0; i < n; i++)
    {
      svtkAbstractTransform* t = this->GetConcatenatedTransform(i);
      os << indent << "    " << i << ": " << t->GetClassName() << " at " << t << "\n";
    }
  }
}

//------------------------------------------------------------------------
// Pass the point through each transform in turn
template <class T2, class T3>
void svtkConcatenationTransformPoint(
  svtkAbstractTransform* input, svtkTransformConcatenation* concat, T2 point[3], T3 output[3])
{
  output[0] = point[0];
  output[1] = point[1];
  output[2] = point[2];

  int i = 0;
  int nTransforms = concat->GetNumberOfTransforms();
  int nPreTransforms = concat->GetNumberOfPreTransforms();

  // push point through the PreTransforms
  for (; i < nPreTransforms; i++)
  {
    concat->GetTransform(i)->InternalTransformPoint(output, output);
  }

  // push point though the Input, if present
  if (input)
  {
    if (concat->GetInverseFlag())
    {
      input = input->GetInverse();
    }
    input->InternalTransformPoint(output, output);
  }

  // push point through PostTransforms
  for (; i < nTransforms; i++)
  {
    concat->GetTransform(i)->InternalTransformPoint(output, output);
  }
}

//----------------------------------------------------------------------------
// Pass the point through each transform in turn,
// concatenate the derivatives.
template <class T2, class T3, class T4>
void svtkConcatenationTransformDerivative(svtkAbstractTransform* input,
  svtkTransformConcatenation* concat, T2 point[3], T3 output[3], T4 derivative[3][3])
{
  T4 matrix[3][3];

  output[0] = point[0];
  output[1] = point[1];
  output[2] = point[2];

  svtkMath::Identity3x3(derivative);

  int i = 0;
  int nTransforms = concat->GetNumberOfTransforms();
  int nPreTransforms = concat->GetNumberOfPreTransforms();

  // push point through the PreTransforms
  for (; i < nPreTransforms; i++)
  {
    concat->GetTransform(i)->InternalTransformDerivative(output, output, matrix);
    svtkMath::Multiply3x3(matrix, derivative, derivative);
  }

  // push point though the Input, if present
  if (input)
  {
    if (concat->GetInverseFlag())
    {
      input = input->GetInverse();
    }
    input->InternalTransformDerivative(output, output, matrix);
    svtkMath::Multiply3x3(matrix, derivative, derivative);
  }

  // push point through PostTransforms
  for (; i < nTransforms; i++)
  {
    concat->GetTransform(i)->InternalTransformDerivative(output, output, matrix);
    svtkMath::Multiply3x3(matrix, derivative, derivative);
  }
}

//------------------------------------------------------------------------
void svtkGeneralTransform::InternalTransformPoint(const float input[3], float output[3])
{
  svtkConcatenationTransformPoint(this->Input, this->Concatenation, input, output);
}

//----------------------------------------------------------------------------
void svtkGeneralTransform::InternalTransformPoint(const double input[3], double output[3])
{
  svtkConcatenationTransformPoint(this->Input, this->Concatenation, input, output);
}

//----------------------------------------------------------------------------
void svtkGeneralTransform::InternalTransformDerivative(
  const float input[3], float output[3], float derivative[3][3])
{
  svtkConcatenationTransformDerivative(this->Input, this->Concatenation, input, output, derivative);
}

//----------------------------------------------------------------------------
void svtkGeneralTransform::InternalTransformDerivative(
  const double input[3], double output[3], double derivative[3][3])
{
  svtkConcatenationTransformDerivative(this->Input, this->Concatenation, input, output, derivative);
}

//----------------------------------------------------------------------------
void svtkGeneralTransform::InternalDeepCopy(svtkAbstractTransform* gtrans)
{
  svtkGeneralTransform* transform = static_cast<svtkGeneralTransform*>(gtrans);

  // copy the input
  this->SetInput(transform->Input);

  // copy the concatenation
  this->Concatenation->DeepCopy(transform->Concatenation);

  // copy the stack
  if (transform->Stack)
  {
    if (this->Stack == nullptr)
    {
      this->Stack = svtkTransformConcatenationStack::New();
    }
    this->Stack->DeepCopy(transform->Stack);
  }
  else
  {
    if (this->Stack)
    {
      this->Stack->Delete();
      this->Stack = nullptr;
    }
  }
}

//----------------------------------------------------------------------------
void svtkGeneralTransform::InternalUpdate()
{
  // update the input
  if (this->Input)
  {
    if (this->Concatenation->GetInverseFlag())
    {
      this->Input->GetInverse()->Update();
    }
    else
    {
      this->Input->Update();
    }
  }

  // update the concatenation
  int nTransforms = this->Concatenation->GetNumberOfTransforms();
  for (int i = 0; i < nTransforms; i++)
  {
    this->Concatenation->GetTransform(i)->Update();
  }
}

//----------------------------------------------------------------------------
void svtkGeneralTransform::Concatenate(svtkAbstractTransform* transform)
{
  if (transform->CircuitCheck(this))
  {
    svtkErrorMacro("Concatenate: this would create a circular reference.");
    return;
  }
  this->Concatenation->Concatenate(transform);
  this->Modified();
};

//----------------------------------------------------------------------------
void svtkGeneralTransform::SetInput(svtkAbstractTransform* input)
{
  if (this->Input == input)
  {
    return;
  }
  if (input && input->CircuitCheck(this))
  {
    svtkErrorMacro("SetInput: this would create a circular reference.");
    return;
  }
  if (this->Input)
  {
    this->Input->Delete();
  }
  this->Input = input;
  if (this->Input)
  {
    this->Input->Register(this);
  }
  this->Modified();
}

//----------------------------------------------------------------------------
int svtkGeneralTransform::CircuitCheck(svtkAbstractTransform* transform)
{
  if (this->svtkAbstractTransform::CircuitCheck(transform) ||
    (this->Input && this->Input->CircuitCheck(transform)))
  {
    return 1;
  }

  int n = this->Concatenation->GetNumberOfTransforms();
  for (int i = 0; i < n; i++)
  {
    if (this->Concatenation->GetTransform(i)->CircuitCheck(transform))
    {
      return 1;
    }
  }

  return 0;
}

//----------------------------------------------------------------------------
svtkAbstractTransform* svtkGeneralTransform::MakeTransform()
{
  return svtkGeneralTransform::New();
}

//----------------------------------------------------------------------------
svtkMTimeType svtkGeneralTransform::GetMTime()
{
  svtkMTimeType mtime = this->svtkAbstractTransform::GetMTime();
  svtkMTimeType mtime2;

  if (this->Input)
  {
    mtime2 = this->Input->GetMTime();
    if (mtime2 > mtime)
    {
      mtime = mtime2;
    }
  }
  mtime2 = this->Concatenation->GetMaxMTime();
  if (mtime2 > mtime)
  {
    return mtime2;
  }
  return mtime;
}
