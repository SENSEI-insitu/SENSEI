/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkImplicitVolume.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkImplicitVolume.h"

#include "svtkDoubleArray.h"
#include "svtkImageData.h"
#include "svtkObjectFactory.h"
#include "svtkPointData.h"
#include "svtkVoxel.h"

svtkStandardNewMacro(svtkImplicitVolume);
svtkCxxSetObjectMacro(svtkImplicitVolume, Volume, svtkImageData);

//----------------------------------------------------------------------------
// Construct an svtkImplicitVolume with no initial volume; the OutValue
// set to a large negative number; and the OutGradient set to (0,0,1).
svtkImplicitVolume::svtkImplicitVolume()
{
  this->Volume = nullptr;
  this->OutValue = SVTK_DOUBLE_MIN;

  this->OutGradient[0] = 0.0;
  this->OutGradient[1] = 0.0;
  this->OutGradient[2] = 1.0;

  this->PointIds = svtkIdList::New();
  this->PointIds->Allocate(8);
}

//----------------------------------------------------------------------------
svtkImplicitVolume::~svtkImplicitVolume()
{
  if (this->Volume)
  {
    this->Volume->Delete();
    this->Volume = nullptr;
  }
  this->PointIds->Delete();
}

//----------------------------------------------------------------------------
// Evaluate the ImplicitVolume. This returns the interpolated scalar value
// at x[3].
double svtkImplicitVolume::EvaluateFunction(double x[3])
{
  svtkDataArray* scalars;
  int ijk[3];
  svtkIdType numPts, i;
  double pcoords[3], weights[8], s;

  // See if a volume is defined
  if (!this->Volume || !(scalars = this->Volume->GetPointData()->GetScalars()))
  {
    svtkErrorMacro(
      << "Can't evaluate function: either volume is missing or volume has no point data");
    return this->OutValue;
  }

  // Find the cell that contains xyz and get it
  if (this->Volume->ComputeStructuredCoordinates(x, ijk, pcoords))
  {
    this->Volume->GetCellPoints(this->Volume->ComputeCellId(ijk), this->PointIds);
    svtkVoxel::InterpolationFunctions(pcoords, weights);

    numPts = this->PointIds->GetNumberOfIds();
    for (s = 0.0, i = 0; i < numPts; i++)
    {
      s += scalars->GetComponent(this->PointIds->GetId(i), 0) * weights[i];
    }
    return s;
  }
  else
  {
    return this->OutValue;
  }
}

//----------------------------------------------------------------------------
svtkMTimeType svtkImplicitVolume::GetMTime()
{
  svtkMTimeType mTime = this->svtkImplicitFunction::GetMTime();
  svtkMTimeType volumeMTime;

  if (this->Volume != nullptr)
  {
    volumeMTime = this->Volume->GetMTime();
    mTime = (volumeMTime > mTime ? volumeMTime : mTime);
  }

  return mTime;
}

//----------------------------------------------------------------------------
// Evaluate ImplicitVolume gradient.
void svtkImplicitVolume::EvaluateGradient(double x[3], double n[3])
{
  svtkDataArray* scalars;
  int i, ijk[3];
  double pcoords[3], weights[8], *v;
  svtkDoubleArray* gradient;

  // See if a volume is defined
  if (!this->Volume || !(scalars = this->Volume->GetPointData()->GetScalars()))
  {
    svtkErrorMacro(
      << "Can't evaluate gradient: either volume is missing or volume has no point data");
    for (i = 0; i < 3; i++)
    {
      n[i] = this->OutGradient[i];
    }
    return;
  }

  gradient = svtkDoubleArray::New();
  gradient->SetNumberOfComponents(3);
  gradient->SetNumberOfTuples(8);

  // Find the cell that contains xyz and get it
  if (this->Volume->ComputeStructuredCoordinates(x, ijk, pcoords))
  {
    svtkVoxel::InterpolationFunctions(pcoords, weights);
    this->Volume->GetVoxelGradient(ijk[0], ijk[1], ijk[2], scalars, gradient);

    n[0] = n[1] = n[2] = 0.0;
    for (i = 0; i < 8; i++)
    {
      v = gradient->GetTuple(i);
      n[0] += v[0] * weights[i];
      n[1] += v[1] * weights[i];
      n[2] += v[2] * weights[i];
    }
  }

  else
  { // use outside value
    for (i = 0; i < 3; i++)
    {
      n[i] = this->OutGradient[i];
    }
  }
  gradient->Delete();
}

//----------------------------------------------------------------------------
void svtkImplicitVolume::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "Out Value: " << this->GetOutValue() << "\n";
  os << indent << "Out Gradient: (" << this->GetOutGradient()[0] << ", "
     << this->GetOutGradient()[1] << ", " << this->GetOutGradient()[2] << ")\n";

  if (this->GetVolume())
  {
    os << indent << "Volume: " << this->GetVolume() << "\n";
  }
  else
  {
    os << indent << "Volume: (none)\n";
  }
}
