/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkTransform.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkTransform.h"
#include "svtkMath.h"
#include "svtkObjectFactory.h"

#include <cstdlib>

svtkStandardNewMacro(svtkTransform);

//----------------------------------------------------------------------------
svtkTransform::svtkTransform()
{
  this->Input = nullptr;

  // most of the functionality is provided by the concatenation
  this->Concatenation = svtkTransformConcatenation::New();

  // the stack will be allocated the first time Push is called
  this->Stack = nullptr;

  // initialize the legacy 'Point' info
  this->Point[0] = this->Point[1] = this->Point[2] = this->Point[3] = 0.0;
  this->DoublePoint[0] = this->DoublePoint[1] = this->DoublePoint[2] = this->DoublePoint[3] = 0.0;

  // save the original matrix MTime as part of a hack to support legacy code
  this->MatrixUpdateMTime = this->Matrix->GetMTime();
}

//----------------------------------------------------------------------------
svtkTransform::~svtkTransform()
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
void svtkTransform::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Update();

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
      svtkLinearTransform* t = this->GetConcatenatedTransform(i);
      os << indent << "    " << i << ": " << t->GetClassName() << " at " << t << "\n";
    }
  }

  os << indent << "DoublePoint: "
     << "( " << this->DoublePoint[0] << ", " << this->DoublePoint[1] << ", " << this->DoublePoint[2]
     << ", " << this->DoublePoint[3] << ")\n";

  os << indent << "Point: "
     << "( " << this->Point[0] << ", " << this->Point[1] << ", " << this->Point[2] << ", "
     << this->Point[3] << ")\n";
}

//----------------------------------------------------------------------------
void svtkTransform::Identity()
{
  this->Concatenation->Identity();

  // support for the legacy hack in InternalUpdate
  if (this->Matrix->GetMTime() > this->MatrixUpdateMTime)
  {
    this->Matrix->Identity();
  }

  this->Modified();
}

//----------------------------------------------------------------------------
void svtkTransform::Inverse()
{
  this->Concatenation->Inverse();

  // for the legacy hack in InternalUpdate
  if (this->Matrix->GetMTime() > this->MatrixUpdateMTime)
  {
    this->Matrix->Invert();
  }

  this->Modified();
}

//----------------------------------------------------------------------------
void svtkTransform::InternalDeepCopy(svtkAbstractTransform* gtrans)
{
  svtkTransform* transform = static_cast<svtkTransform*>(gtrans);

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

  // legacy stuff: copy Point and DoublePoint
  for (int j = 0; j < 3; j++)
  {
    this->Point[j] = transform->Point[j];
    this->DoublePoint[j] = transform->DoublePoint[j];
  }

  // to support the legacy hack in InternalUpdate
  this->Matrix->DeepCopy(transform->Matrix);
  this->MatrixUpdateMTime = this->Matrix->GetMTime();
}

//----------------------------------------------------------------------------
void svtkTransform::InternalUpdate()
{
  int i;
  int nTransforms = this->Concatenation->GetNumberOfTransforms();
  int nPreTransforms = this->Concatenation->GetNumberOfPreTransforms();

  // check to see whether someone has been fooling around with our matrix
  int doTheLegacyHack = 0;
  if (this->Matrix->GetMTime() > this->MatrixUpdateMTime)
  {
    svtkDebugMacro(<< "InternalUpdate: this->Matrix was modified by something other than 'this'");

    // check to see if we have any inputs or concatenated transforms
    int isPipelined = (this->Input != nullptr);
    for (i = 0; i < nTransforms && !isPipelined; i++)
    { // the svtkSimpleTransform is just a matrix placeholder,
      // it is not a real transform
      isPipelined = !this->Concatenation->GetTransform(i)->IsA("svtkSimpleTransform");
    }
    // do the legacy hack only if we have no input transforms
    doTheLegacyHack = !isPipelined;
  }

  // copy matrix from input
  if (this->Input)
  {
    this->Matrix->DeepCopy(this->Input->GetMatrix());
    // if inverse flag is set, invert the matrix
    if (this->Concatenation->GetInverseFlag())
    {
      this->Matrix->Invert();
    }
  }
  else if (doTheLegacyHack)
  {
    svtkWarningMacro("InternalUpdate: doing hack to support legacy code.  "
                    "This is deprecated in SVTK 4.2.  May be removed in a "
                    "future version.");
    // this heuristic works perfectly if GetMatrix() or GetMatrixPointer()
    // was called immediately prior to the matrix modifications
    // (fortunately, this is almost always the case)
    if (this->Matrix->GetMTime() > this->Concatenation->GetMaxMTime())
    { // don't apply operations that occurred after matrix modification
      nPreTransforms = nTransforms = 0;
    }
  }
  else
  { // otherwise, we start with the identity transform as our base
    this->Matrix->Identity();
  }

  // concatenate PreTransforms
  for (i = nPreTransforms - 1; i >= 0; i--)
  {
    svtkHomogeneousTransform* transform =
      static_cast<svtkHomogeneousTransform*>(this->Concatenation->GetTransform(i));
    svtkMatrix4x4::Multiply4x4(this->Matrix, transform->GetMatrix(), this->Matrix);
  }

  // concatenate PostTransforms
  for (i = nPreTransforms; i < nTransforms; i++)
  {
    svtkHomogeneousTransform* transform =
      static_cast<svtkHomogeneousTransform*>(this->Concatenation->GetTransform(i));
    svtkMatrix4x4::Multiply4x4(transform->GetMatrix(), this->Matrix, this->Matrix);
  }

  if (doTheLegacyHack)
  { // the transform operations have been incorporated into the matrix,
    // so delete them
    this->Concatenation->Identity();
  }
  else
  { // having this in the 'else' forces the legacy flag to be sticky
    this->MatrixUpdateMTime = this->Matrix->GetMTime();
  }
}

//----------------------------------------------------------------------------
void svtkTransform::Concatenate(svtkLinearTransform* transform)
{
  if (transform->CircuitCheck(this))
  {
    svtkErrorMacro("Concatenate: this would create a circular reference.");
    return;
  }
  this->Concatenation->Concatenate(transform);
  this->Modified();
}

//----------------------------------------------------------------------------
void svtkTransform::SetInput(svtkLinearTransform* input)
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
int svtkTransform::CircuitCheck(svtkAbstractTransform* transform)
{
  if (this->svtkLinearTransform::CircuitCheck(transform) ||
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
svtkAbstractTransform* svtkTransform::MakeTransform()
{
  return svtkTransform::New();
}

//----------------------------------------------------------------------------
svtkMTimeType svtkTransform::GetMTime()
{
  svtkMTimeType mtime = this->svtkLinearTransform::GetMTime();
  svtkMTimeType mtime2;

  // checking the matrix MTime is part of the legacy hack in InternalUpdate
  if ((mtime2 = this->Matrix->GetMTime()) > this->MatrixUpdateMTime)
  {
    if (mtime2 > mtime)
    {
      mtime = mtime2;
    }
  }

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

//----------------------------------------------------------------------------
// Get the x, y, z orientation angles from the transformation matrix as an
// array of three floating point values.
void svtkTransform::GetOrientation(double orientation[3], svtkMatrix4x4* amatrix)
{
#define SVTK_AXIS_EPSILON 0.001
#define SVTK_ORTHO_EPSILON 4e-16
  int i;

  // convenient access to matrix
  double(*matrix)[4] = amatrix->Element;
  double ortho[3][3];

  for (i = 0; i < 3; i++)
  {
    ortho[0][i] = matrix[0][i];
    ortho[1][i] = matrix[1][i];
    ortho[2][i] = matrix[2][i];
  }
  if (svtkMath::Determinant3x3(ortho) < 0)
  {
    ortho[0][2] = -ortho[0][2];
    ortho[1][2] = -ortho[1][2];
    ortho[2][2] = -ortho[2][2];
  }

  // Check whether matrix is orthogonal
  double r1 = svtkMath::Dot(ortho[0], ortho[1]);
  double r2 = svtkMath::Dot(ortho[0], ortho[2]);
  double r3 = svtkMath::Dot(ortho[1], ortho[2]);

  // Orthogonalize the matrix if it isn't already orthogonal
  if ((r1 * r1) + (r2 * r2) + (r3 * r3) > (SVTK_ORTHO_EPSILON * SVTK_ORTHO_EPSILON))
  {
    svtkMath::Orthogonalize3x3(ortho, ortho);
  }

  // compute the max scale as we need that for the epsilon test
  double scale0 = svtkMath::Norm(ortho[0]);
  double scale1 = svtkMath::Norm(ortho[1]);
  double maxScale = svtkMath::Norm(ortho[2]);
  maxScale = maxScale >= scale0 ? maxScale : scale0;
  maxScale = maxScale >= scale1 ? maxScale : scale1;
  if (maxScale == 0.0)
  {
    orientation[0] = 0.0;
    orientation[1] = 0.0;
    orientation[2] = 0.0;
    return;
  }

  // first rotate about y axis
  double x2 = ortho[2][0];
  double y2 = ortho[2][1];
  double z2 = ortho[2][2];

  double x3 = ortho[1][0];
  double y3 = ortho[1][1];
  double z3 = ortho[1][2];

  double d1 = sqrt(x2 * x2 + z2 * z2);

  double cosTheta, sinTheta;
  if (d1 < SVTK_AXIS_EPSILON * maxScale)
  {
    cosTheta = 1.0;
    sinTheta = 0.0;
  }
  else
  {
    cosTheta = z2 / d1;
    sinTheta = x2 / d1;
  }

  double theta = atan2(sinTheta, cosTheta);
  orientation[1] = -svtkMath::DegreesFromRadians(theta);

  // now rotate about x axis
  double d = sqrt(x2 * x2 + y2 * y2 + z2 * z2);

  double sinPhi, cosPhi;
  if (d < SVTK_AXIS_EPSILON * maxScale)
  {
    sinPhi = 0.0;
    cosPhi = 1.0;
  }
  else if (d1 < SVTK_AXIS_EPSILON * maxScale)
  {
    sinPhi = y2 / d;
    cosPhi = z2 / d;
  }
  else
  {
    sinPhi = y2 / d;
    cosPhi = (x2 * x2 + z2 * z2) / (d1 * d);
  }

  double phi = atan2(sinPhi, cosPhi);
  orientation[0] = svtkMath::DegreesFromRadians(phi);

  // finally, rotate about z
  double x3p = x3 * cosTheta - z3 * sinTheta;
  double y3p = -sinPhi * sinTheta * x3 + cosPhi * y3 - sinPhi * cosTheta * z3;
  double d2 = sqrt(x3p * x3p + y3p * y3p);

  double cosAlpha, sinAlpha;
  if (d2 < SVTK_AXIS_EPSILON * maxScale)
  {
    cosAlpha = 1.0;
    sinAlpha = 0.0;
  }
  else
  {
    cosAlpha = y3p / d2;
    sinAlpha = x3p / d2;
  }

  double alpha = atan2(sinAlpha, cosAlpha);
  orientation[2] = svtkMath::DegreesFromRadians(alpha);
}

//----------------------------------------------------------------------------
// Get the x, y, z orientation angles from the transformation matrix as an
// array of three floating point values.
void svtkTransform::GetOrientation(double orientation[3])
{
  this->Update();
  this->GetOrientation(orientation, this->Matrix);
}

//----------------------------------------------------------------------------
// svtkTransform::GetOrientationWXYZ
void svtkTransform::GetOrientationWXYZ(double wxyz[4])
{
  int i;

  this->Update();
  // convenient access to matrix
  double(*matrix)[4] = this->Matrix->Element;
  double ortho[3][3];

  for (i = 0; i < 3; i++)
  {
    ortho[0][i] = matrix[0][i];
    ortho[1][i] = matrix[1][i];
    ortho[2][i] = matrix[2][i];
  }
  if (svtkMath::Determinant3x3(ortho) < 0)
  {
    ortho[0][2] = -ortho[0][2];
    ortho[1][2] = -ortho[1][2];
    ortho[2][2] = -ortho[2][2];
  }

  svtkMath::Matrix3x3ToQuaternion(ortho, wxyz);

  // calc the return value wxyz
  double mag = sqrt(wxyz[1] * wxyz[1] + wxyz[2] * wxyz[2] + wxyz[3] * wxyz[3]);

  if (mag != 0.0)
  {
    wxyz[0] = 2.0 * svtkMath::DegreesFromRadians(atan2(mag, wxyz[0]));
    wxyz[1] /= mag;
    wxyz[2] /= mag;
    wxyz[3] /= mag;
  }
  else
  {
    wxyz[0] = 0.0;
    wxyz[1] = 0.0;
    wxyz[2] = 0.0;
    wxyz[3] = 1.0;
  }
}

//----------------------------------------------------------------------------
// Return the position from the current transformation matrix as an array
// of three floating point numbers. This is simply returning the translation
// component of the 4x4 matrix.
void svtkTransform::GetPosition(double position[3])
{
  this->Update();

  position[0] = this->Matrix->Element[0][3];
  position[1] = this->Matrix->Element[1][3];
  position[2] = this->Matrix->Element[2][3];
}

//----------------------------------------------------------------------------
// Return the x, y, z scale factors of the current transformation matrix as
// an array of three float numbers.
void svtkTransform::GetScale(double scale[3])
{
  this->Update();

  // convenient access to matrix
  double(*matrix)[4] = this->Matrix->Element;
  double U[3][3], VT[3][3];

  for (int i = 0; i < 3; i++)
  {
    U[0][i] = matrix[0][i];
    U[1][i] = matrix[1][i];
    U[2][i] = matrix[2][i];
  }

  svtkMath::SingularValueDecomposition3x3(U, U, scale, VT);
}

//----------------------------------------------------------------------------
// Return the inverse of the current transformation matrix.
void svtkTransform::GetInverse(svtkMatrix4x4* inverse)
{
  svtkMatrix4x4::Invert(this->GetMatrix(), inverse);
}

//----------------------------------------------------------------------------
// Obtain the transpose of the current transformation matrix.
void svtkTransform::GetTranspose(svtkMatrix4x4* transpose)
{
  svtkMatrix4x4::Transpose(this->GetMatrix(), transpose);
}
