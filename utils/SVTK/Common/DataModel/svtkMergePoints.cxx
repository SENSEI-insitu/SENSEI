/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkMergePoints.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkMergePoints.h"

#include "svtkDataArray.h"
#include "svtkFloatArray.h"
#include "svtkIdList.h"
#include "svtkObjectFactory.h"
#include "svtkPoints.h"

svtkStandardNewMacro(svtkMergePoints);

//----------------------------------------------------------------------------
// Determine whether point given by x[3] has been inserted into points list.
// Return id of previously inserted point if this is true, otherwise return
// -1.
svtkIdType svtkMergePoints::IsInsertedPoint(const double x[3])
{
  //
  //  Locate bucket that point is in.
  //
  svtkIdType idx = this->GetBucketIndex(x);

  svtkIdList* bucket = this->HashTable[idx];

  if (!bucket)
  {
    return -1;
  }
  else // see whether we've got duplicate point
  {
    //
    // Check the list of points in that bucket.
    //
    svtkIdType ptId;
    svtkIdType nbOfIds = bucket->GetNumberOfIds();

    // For efficiency reasons, we break the data abstraction for points
    // and ids (we are assuming and svtkIdList
    // is storing ints).
    svtkDataArray* dataArray = this->Points->GetData();
    svtkIdType* idArray = bucket->GetPointer(0);
    if (dataArray->GetDataType() == SVTK_FLOAT)
    {
      float f[3];
      f[0] = static_cast<float>(x[0]);
      f[1] = static_cast<float>(x[1]);
      f[2] = static_cast<float>(x[2]);
      svtkFloatArray* floatArray = static_cast<svtkFloatArray*>(dataArray);
      float* pt;
      for (svtkIdType i = 0; i < nbOfIds; i++)
      {
        ptId = idArray[i];
        pt = floatArray->GetPointer(0) + 3 * ptId;
        if (f[0] == pt[0] && f[1] == pt[1] && f[2] == pt[2])
        {
          return ptId;
        }
      }
    }
    else
    {
      // Using the double interface
      double* pt;
      for (svtkIdType i = 0; i < nbOfIds; i++)
      {
        ptId = idArray[i];
        pt = dataArray->GetTuple(ptId);
        if (x[0] == pt[0] && x[1] == pt[1] && x[2] == pt[2])
        {
          return ptId;
        }
      }
    }
  }

  return -1;
}

//----------------------------------------------------------------------------
int svtkMergePoints::InsertUniquePoint(const double x[3], svtkIdType& id)
{
  //
  //  Locate bucket that point is in.
  //
  svtkIdType idx = this->GetBucketIndex(x);
  svtkIdList* bucket = this->HashTable[idx];

  if (bucket) // see whether we've got duplicate point
  {
    //
    // Check the list of points in that bucket.
    //
    svtkIdType ptId;
    svtkIdType nbOfIds = bucket->GetNumberOfIds();

    // For efficiency reasons, we break the data abstraction for points
    // and ids (we are assuming svtkPoints stores a svtkIdList
    // is storing ints).
    svtkDataArray* dataArray = this->Points->GetData();
    svtkIdType* idArray = bucket->GetPointer(0);

    if (dataArray->GetDataType() == SVTK_FLOAT)
    {
      float f[3];
      f[0] = static_cast<float>(x[0]);
      f[1] = static_cast<float>(x[1]);
      f[2] = static_cast<float>(x[2]);
      float* floatArray = static_cast<svtkFloatArray*>(dataArray)->GetPointer(0);
      float* pt;
      for (svtkIdType i = 0; i < nbOfIds; ++i)
      {
        ptId = idArray[i];
        pt = floatArray + 3 * ptId;
        if (f[0] == pt[0] && f[1] == pt[1] && f[2] == pt[2])
        {
          // point is already in the list, return 0 and set the id parameter
          id = ptId;
          return 0;
        }
      }
    }
    else
    {
      // Using the double interface
      double* pt;
      for (svtkIdType i = 0; i < nbOfIds; ++i)
      {
        ptId = idArray[i];
        pt = dataArray->GetTuple(ptId);
        if (x[0] == pt[0] && x[1] == pt[1] && x[2] == pt[2])
        {
          // point is already in the list, return 0 and set the id parameter
          id = ptId;
          return 0;
        }
      }
    }
  }
  else
  {
    // create a bucket point list and insert the point
    bucket = svtkIdList::New();
    bucket->Allocate(this->NumberOfPointsPerBucket / 2, this->NumberOfPointsPerBucket / 3);
    this->HashTable[idx] = bucket;
  }

  // point has to be added
  bucket->InsertNextId(this->InsertionPointId);
  this->Points->InsertPoint(this->InsertionPointId, x);
  id = this->InsertionPointId++;

  return 1;
}

//----------------------------------------------------------------------------
void svtkMergePoints::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}
