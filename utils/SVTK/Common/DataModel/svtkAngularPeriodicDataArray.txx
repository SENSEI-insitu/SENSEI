/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkAngularPeriodicDataArray.txx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

    This software is distributed WITHOUT ANY WARRANTY; without even
    the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
    PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "svtkMath.h"
#include "svtkMatrix3x3.h"
#include "svtkObjectFactory.h"

#include <algorithm>

//------------------------------------------------------------------------------
// Can't use svtkStandardNewMacro on a templated class.
template <class Scalar>
svtkAngularPeriodicDataArray<Scalar>* svtkAngularPeriodicDataArray<Scalar>::New()
{
  SVTK_STANDARD_NEW_BODY(svtkAngularPeriodicDataArray<Scalar>);
}

//------------------------------------------------------------------------------
template <class Scalar>
void svtkAngularPeriodicDataArray<Scalar>::PrintSelf(ostream& os, svtkIndent indent)
{
  this->svtkAngularPeriodicDataArray<Scalar>::Superclass::PrintSelf(os, indent);
  os << indent << "Axis: " << this->Axis << "\n";
  os << indent << "Angle: " << this->Angle << "\n";
  os << indent << "Center: " << this->Center[0] << " " << this->Center[1] << " " << this->Center[2]
     << "\n";
}

//------------------------------------------------------------------------------
template <class Scalar>
void svtkAngularPeriodicDataArray<Scalar>::InitializeArray(svtkAOSDataArrayTemplate<Scalar>* data)
{
  this->Initialize();
  if (!data)
  {
    svtkErrorMacro(<< "No original data provided.");
    return;
  }

  if (data->GetNumberOfComponents() != 3 && data->GetNumberOfComponents() != 6 &&
    data->GetNumberOfComponents() != 9)
  {
    svtkWarningMacro(<< "Original data has " << data->GetNumberOfComponents()
                    << " components, Expecting 3 or 9.");
    return;
  }

  this->Superclass::InitializeArray(data);
}

//------------------------------------------------------------------------------
template <class Scalar>
void svtkAngularPeriodicDataArray<Scalar>::SetAngle(double angle)
{
  if (this->Angle != angle)
  {
    this->Angle = angle;
    this->AngleInRadians = svtkMath::RadiansFromDegrees(angle);
    this->InvalidateRange();
    this->UpdateRotationMatrix();
    this->Modified();
  }
}

//------------------------------------------------------------------------------
template <class Scalar>
void svtkAngularPeriodicDataArray<Scalar>::SetAxis(int axis)
{
  if (this->Axis != axis)
  {
    this->Axis = axis;
    this->InvalidateRange();
    this->UpdateRotationMatrix();
    this->Modified();
  }
}

//------------------------------------------------------------------------------
template <class Scalar>
void svtkAngularPeriodicDataArray<Scalar>::SetCenter(double* center)
{
  if (center)
  {
    bool diff = false;
    for (int i = 0; i < 3; i++)
    {
      if (this->Center[i] != center[i])
      {
        this->Center[i] = center[i];
        diff = true;
      }
    }
    if (diff)
    {
      this->InvalidateRange();
      this->Modified();
    }
  }
}

//------------------------------------------------------------------------------
template <class Scalar>
void svtkAngularPeriodicDataArray<Scalar>::Transform(Scalar* pos) const
{
  if (this->NumberOfComponents == 3)
  {
    // Axis rotation
    int axis0 = (this->Axis + 1) % this->NumberOfComponents;
    int axis1 = (this->Axis + 2) % this->NumberOfComponents;
    double posx = static_cast<double>(pos[axis0]) - this->Center[axis0];
    double posy = static_cast<double>(pos[axis1]) - this->Center[axis1];

    pos[axis0] = this->Center[axis0] +
      static_cast<Scalar>(cos(this->AngleInRadians) * posx - sin(this->AngleInRadians) * posy);
    pos[axis1] = this->Center[axis1] +
      static_cast<Scalar>(sin(this->AngleInRadians) * posx + cos(this->AngleInRadians) * posy);
    if (this->Normalize)
    {
      svtkMath::Normalize(pos);
    }
  }
  else if (this->NumberOfComponents == 9 || this->NumberOfComponents == 6)
  {
    // Template type force a copy to a double array for tensor
    double localPos[9];
    double tmpMat[9];
    double tmpMat2[9];
    std::copy(pos, pos + this->NumberOfComponents, localPos);
    if (this->NumberOfComponents == 6)
    {
      svtkMath::TensorFromSymmetricTensor(localPos);
    }

    svtkMatrix3x3::Transpose(this->RotationMatrix->GetData(), tmpMat);
    svtkMatrix3x3::Multiply3x3(this->RotationMatrix->GetData(), localPos, tmpMat2);
    svtkMatrix3x3::Multiply3x3(tmpMat2, tmpMat, localPos);
    std::copy(localPos, localPos + this->NumberOfComponents, pos);
  }
}

//------------------------------------------------------------------------------
template <class Scalar>
void svtkAngularPeriodicDataArray<Scalar>::UpdateRotationMatrix()
{
  int axis0 = (this->Axis + 1) % 3;
  int axis1 = (this->Axis + 2) % 3;
  this->RotationMatrix->Identity();
  this->RotationMatrix->SetElement(this->Axis, this->Axis, 1.);
  this->RotationMatrix->SetElement(axis0, axis0, cos(this->AngleInRadians));
  this->RotationMatrix->SetElement(axis0, axis1, -sin(this->AngleInRadians));
  this->RotationMatrix->SetElement(axis1, axis0, sin(this->AngleInRadians));
  this->RotationMatrix->SetElement(axis1, axis1, cos(this->AngleInRadians));
}

//------------------------------------------------------------------------------
template <class Scalar>
svtkAngularPeriodicDataArray<Scalar>::svtkAngularPeriodicDataArray()
{
  this->Axis = SVTK_PERIODIC_ARRAY_AXIS_X;
  this->Angle = 0.0;
  this->AngleInRadians = 0.0;
  this->Center[0] = 0.0;
  this->Center[1] = 0.0;
  this->Center[2] = 0.0;

  this->RotationMatrix = svtkMatrix3x3::New();
  this->RotationMatrix->Identity();
}

//------------------------------------------------------------------------------
template <class Scalar>
svtkAngularPeriodicDataArray<Scalar>::~svtkAngularPeriodicDataArray()
{
  this->RotationMatrix->Delete();
}
