/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkIterativeClosestPointTransform.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkIterativeClosestPointTransform.h"

#include "svtkCellLocator.h"
#include "svtkDataSet.h"
#include "svtkLandmarkTransform.h"
#include "svtkMath.h"
#include "svtkObjectFactory.h"
#include "svtkPoints.h"
#include "svtkTransform.h"

svtkStandardNewMacro(svtkIterativeClosestPointTransform);

//----------------------------------------------------------------------------

svtkIterativeClosestPointTransform::svtkIterativeClosestPointTransform()
  : svtkLinearTransform()
{
  this->Source = nullptr;
  this->Target = nullptr;
  this->Locator = nullptr;
  this->LandmarkTransform = svtkLandmarkTransform::New();
  this->MaximumNumberOfIterations = 50;
  this->CheckMeanDistance = 0;
  this->MeanDistanceMode = SVTK_ICP_MODE_RMS;
  this->MaximumMeanDistance = 0.01;
  this->MaximumNumberOfLandmarks = 200;
  this->StartByMatchingCentroids = 0;

  this->NumberOfIterations = 0;
  this->MeanDistance = 0.0;
}

//----------------------------------------------------------------------------

const char* svtkIterativeClosestPointTransform::GetMeanDistanceModeAsString()
{
  if (this->MeanDistanceMode == SVTK_ICP_MODE_RMS)
  {
    return "RMS";
  }
  else
  {
    return "AbsoluteValue";
  }
}

//----------------------------------------------------------------------------

svtkIterativeClosestPointTransform::~svtkIterativeClosestPointTransform()
{
  ReleaseSource();
  ReleaseTarget();
  ReleaseLocator();
  this->LandmarkTransform->Delete();
}

//----------------------------------------------------------------------------

void svtkIterativeClosestPointTransform::SetSource(svtkDataSet* source)
{
  if (this->Source == source)
  {
    return;
  }

  if (this->Source)
  {
    this->ReleaseSource();
  }

  if (source)
  {
    source->Register(this);
  }

  this->Source = source;
  this->Modified();
}

//----------------------------------------------------------------------------

void svtkIterativeClosestPointTransform::ReleaseSource()
{
  if (this->Source)
  {
    this->Source->UnRegister(this);
    this->Source = nullptr;
  }
}

//----------------------------------------------------------------------------

void svtkIterativeClosestPointTransform::SetTarget(svtkDataSet* target)
{
  if (this->Target == target)
  {
    return;
  }

  if (this->Target)
  {
    this->ReleaseTarget();
  }

  if (target)
  {
    target->Register(this);
  }

  this->Target = target;
  this->Modified();
}

//----------------------------------------------------------------------------

void svtkIterativeClosestPointTransform::ReleaseTarget()
{
  if (this->Target)
  {
    this->Target->UnRegister(this);
    this->Target = nullptr;
  }
}

//----------------------------------------------------------------------------

void svtkIterativeClosestPointTransform::SetLocator(svtkCellLocator* locator)
{
  if (this->Locator == locator)
  {
    return;
  }

  if (this->Locator)
  {
    this->ReleaseLocator();
  }

  if (locator)
  {
    locator->Register(this);
  }

  this->Locator = locator;
  this->Modified();
}

//----------------------------------------------------------------------------

void svtkIterativeClosestPointTransform::ReleaseLocator()
{
  if (this->Locator)
  {
    this->Locator->UnRegister(this);
    this->Locator = nullptr;
  }
}

//----------------------------------------------------------------------------

void svtkIterativeClosestPointTransform::CreateDefaultLocator()
{
  if (this->Locator)
  {
    this->ReleaseLocator();
  }

  this->Locator = svtkCellLocator::New();
}

//------------------------------------------------------------------------

svtkMTimeType svtkIterativeClosestPointTransform::GetMTime()
{
  svtkMTimeType result = this->svtkLinearTransform::GetMTime();
  svtkMTimeType mtime;

  if (this->Source)
  {
    mtime = this->Source->GetMTime();
    if (mtime > result)
    {
      result = mtime;
    }
  }

  if (this->Target)
  {
    mtime = this->Target->GetMTime();
    if (mtime > result)
    {
      result = mtime;
    }
  }

  if (this->Locator)
  {
    mtime = this->Locator->GetMTime();
    if (mtime > result)
    {
      result = mtime;
    }
  }

  if (this->LandmarkTransform)
  {
    mtime = this->LandmarkTransform->GetMTime();
    if (mtime > result)
    {
      result = mtime;
    }
  }

  return result;
}

//----------------------------------------------------------------------------

void svtkIterativeClosestPointTransform::Inverse()
{
  svtkDataSet* tmp1 = this->Source;
  this->Source = this->Target;
  this->Target = tmp1;
  this->Modified();
}

//----------------------------------------------------------------------------

svtkAbstractTransform* svtkIterativeClosestPointTransform::MakeTransform()
{
  return svtkIterativeClosestPointTransform::New();
}

//----------------------------------------------------------------------------

void svtkIterativeClosestPointTransform::InternalDeepCopy(svtkAbstractTransform* transform)
{
  svtkIterativeClosestPointTransform* t = (svtkIterativeClosestPointTransform*)transform;

  this->SetSource(t->GetSource());
  this->SetTarget(t->GetTarget());
  this->SetLocator(t->GetLocator());
  this->SetMaximumNumberOfIterations(t->GetMaximumNumberOfIterations());
  this->SetCheckMeanDistance(t->GetCheckMeanDistance());
  this->SetMeanDistanceMode(t->GetMeanDistanceMode());
  this->SetMaximumMeanDistance(t->GetMaximumMeanDistance());
  this->SetMaximumNumberOfLandmarks(t->GetMaximumNumberOfLandmarks());

  this->Modified();
}

//----------------------------------------------------------------------------

void svtkIterativeClosestPointTransform::InternalUpdate()
{
  // Check source, target

  if (this->Source == nullptr || !this->Source->GetNumberOfPoints())
  {
    svtkErrorMacro(<< "Can't execute with nullptr or empty input");
    return;
  }

  if (this->Target == nullptr || !this->Target->GetNumberOfPoints())
  {
    svtkErrorMacro(<< "Can't execute with nullptr or empty target");
    return;
  }

  // Create locator

  this->CreateDefaultLocator();
  this->Locator->SetDataSet(this->Target);
  this->Locator->SetNumberOfCellsPerBucket(1);
  this->Locator->BuildLocator();

  // Create two sets of points to handle iteration

  int step = 1;
  if (this->Source->GetNumberOfPoints() > this->MaximumNumberOfLandmarks)
  {
    step = this->Source->GetNumberOfPoints() / this->MaximumNumberOfLandmarks;
    svtkDebugMacro(<< "Landmarks step is now : " << step);
  }

  svtkIdType nb_points = this->Source->GetNumberOfPoints() / step;

  // Allocate some points.
  // - closestp is used so that the internal state of LandmarkTransform remains
  //   correct whenever the iteration process is stopped (hence its source
  //   and landmark points might be used in a svtkThinPlateSplineTransform).
  // - points2 could have been avoided, but do not ask me why
  //   InternalTransformPoint is not working correctly on my computer when
  //   in and out are the same pointer.

  svtkPoints* points1 = svtkPoints::New();
  points1->SetNumberOfPoints(nb_points);

  svtkPoints* closestp = svtkPoints::New();
  closestp->SetNumberOfPoints(nb_points);

  svtkPoints* points2 = svtkPoints::New();
  points2->SetNumberOfPoints(nb_points);

  // Fill with initial positions (sample dataset using step)

  svtkTransform* accumulate = svtkTransform::New();
  accumulate->PostMultiply();

  svtkIdType i;
  int j;
  double p1[3], p2[3];

  if (StartByMatchingCentroids)
  {
    double source_centroid[3] = { 0, 0, 0 };
    for (i = 0; i < this->Source->GetNumberOfPoints(); i++)
    {
      this->Source->GetPoint(i, p1);
      source_centroid[0] += p1[0];
      source_centroid[1] += p1[1];
      source_centroid[2] += p1[2];
    }
    source_centroid[0] /= this->Source->GetNumberOfPoints();
    source_centroid[1] /= this->Source->GetNumberOfPoints();
    source_centroid[2] /= this->Source->GetNumberOfPoints();

    double target_centroid[3] = { 0, 0, 0 };
    for (i = 0; i < this->Target->GetNumberOfPoints(); i++)
    {
      this->Target->GetPoint(i, p2);
      target_centroid[0] += p2[0];
      target_centroid[1] += p2[1];
      target_centroid[2] += p2[2];
    }
    target_centroid[0] /= this->Target->GetNumberOfPoints();
    target_centroid[1] /= this->Target->GetNumberOfPoints();
    target_centroid[2] /= this->Target->GetNumberOfPoints();

    accumulate->Translate(target_centroid[0] - source_centroid[0],
      target_centroid[1] - source_centroid[1], target_centroid[2] - source_centroid[2]);
    accumulate->Update();

    for (i = 0, j = 0; i < nb_points; i++, j += step)
    {
      double outPoint[3];
      accumulate->InternalTransformPoint(this->Source->GetPoint(j), outPoint);
      points1->SetPoint(i, outPoint);
    }
  }
  else
  {
    for (i = 0, j = 0; i < nb_points; i++, j += step)
    {
      points1->SetPoint(i, this->Source->GetPoint(j));
    }
  }

  // Go

  svtkIdType cell_id;
  int sub_id;
  double dist2, totaldist = 0;
  double outPoint[3];

  svtkPoints *temp, *a = points1, *b = points2;

  this->NumberOfIterations = 0;

  do
  {
    // Fill points with the closest points to each vertex in input

    for (i = 0; i < nb_points; i++)
    {
      this->Locator->FindClosestPoint(a->GetPoint(i), outPoint, cell_id, sub_id, dist2);
      closestp->SetPoint(i, outPoint);
    }

    // Build the landmark transform

    this->LandmarkTransform->SetSourceLandmarks(a);
    this->LandmarkTransform->SetTargetLandmarks(closestp);
    this->LandmarkTransform->Update();

    // Concatenate (can't use this->Concatenate directly)

    accumulate->Concatenate(this->LandmarkTransform->GetMatrix());

    this->NumberOfIterations++;
    svtkDebugMacro(<< "Iteration: " << this->NumberOfIterations);
    if (this->NumberOfIterations >= this->MaximumNumberOfIterations)
    {
      break;
    }

    // Move mesh and compute mean distance if needed

    if (this->CheckMeanDistance)
    {
      totaldist = 0.0;
    }

    for (i = 0; i < nb_points; i++)
    {
      a->GetPoint(i, p1);
      this->LandmarkTransform->InternalTransformPoint(p1, p2);
      b->SetPoint(i, p2);
      if (this->CheckMeanDistance)
      {
        if (this->MeanDistanceMode == SVTK_ICP_MODE_RMS)
        {
          totaldist += svtkMath::Distance2BetweenPoints(p1, p2);
        }
        else
        {
          totaldist += sqrt(svtkMath::Distance2BetweenPoints(p1, p2));
        }
      }
    }

    if (this->CheckMeanDistance)
    {
      if (this->MeanDistanceMode == SVTK_ICP_MODE_RMS)
      {
        this->MeanDistance = sqrt(totaldist / (double)nb_points);
      }
      else
      {
        this->MeanDistance = totaldist / (double)nb_points;
      }
      svtkDebugMacro("Mean distance: " << this->MeanDistance);
      if (this->MeanDistance <= this->MaximumMeanDistance)
      {
        break;
      }
    }

    temp = a;
    a = b;
    b = temp;

  } while (1);

  // Now recover accumulated result

  this->Matrix->DeepCopy(accumulate->GetMatrix());

  accumulate->Delete();
  points1->Delete();
  closestp->Delete();
  points2->Delete();
}

//----------------------------------------------------------------------------

void svtkIterativeClosestPointTransform::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  if (this->Source)
  {
    os << indent << "Source: " << this->Source << "\n";
  }
  else
  {
    os << indent << "Source: (none)\n";
  }

  if (this->Target)
  {
    os << indent << "Target: " << this->Target << "\n";
  }
  else
  {
    os << indent << "Target: (none)\n";
  }

  if (this->Locator)
  {
    os << indent << "Locator: " << this->Locator << "\n";
  }
  else
  {
    os << indent << "Locator: (none)\n";
  }

  os << indent << "MaximumNumberOfIterations: " << this->MaximumNumberOfIterations << "\n";
  os << indent << "CheckMeanDistance: " << this->CheckMeanDistance << "\n";
  os << indent << "MeanDistanceMode: " << this->GetMeanDistanceModeAsString() << "\n";
  os << indent << "MaximumMeanDistance: " << this->MaximumMeanDistance << "\n";
  os << indent << "MaximumNumberOfLandmarks: " << this->MaximumNumberOfLandmarks << "\n";
  os << indent << "StartByMatchingCentroids: " << this->StartByMatchingCentroids << "\n";
  os << indent << "NumberOfIterations: " << this->NumberOfIterations << "\n";
  os << indent << "MeanDistance: " << this->MeanDistance << "\n";
  if (this->LandmarkTransform)
  {
    os << indent << "LandmarkTransform:\n";
    this->LandmarkTransform->PrintSelf(os, indent.GetNextIndent());
  }
}
