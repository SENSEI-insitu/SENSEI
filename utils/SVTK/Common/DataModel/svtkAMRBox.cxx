/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkAMRBox.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkAMRBox.h"

#include "svtkCellData.h"
#include "svtkImageData.h"
#include "svtkMath.h"
#include "svtkStructuredData.h"
#include "svtkType.h"
#include "svtkUnsignedCharArray.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <fstream>
#include <sstream>

//-----------------------------------------------------------------------------
svtkAMRBox::svtkAMRBox()
{
  this->Initialize();
}

//-----------------------------------------------------------------------------
svtkAMRBox::svtkAMRBox(const double* origin, const int* dimensions, const double* spacing,
  const double* globalOrigin, int gridDescription)
{
  int ndim[3];
  for (int d = 0; d < 3; ++d)
  {
    ndim[d] = dimensions[d] - 1;
  }
  int lo[3], hi[3];
  for (int d = 0; d < 3; ++d)
  {
    lo[d] = spacing[d] > 0.0
      ? static_cast<int>(std::round((origin[d] - globalOrigin[d]) / spacing[d]))
      : 0;
    hi[d] = lo[d] + ndim[d] - 1;
  }

  this->SetDimensions(lo, hi, gridDescription);
}

//-----------------------------------------------------------------------------
svtkAMRBox::svtkAMRBox(int ilo, int jlo, int klo, int ihi, int jhi, int khi)
{
  this->BuildAMRBox(ilo, jlo, klo, ihi, jhi, khi);
}

//-----------------------------------------------------------------------------
svtkAMRBox::svtkAMRBox(const int* lo, const int* hi)
{
  this->BuildAMRBox(lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]);
}

//-----------------------------------------------------------------------------
svtkAMRBox::svtkAMRBox(const int* dims)
{
  this->BuildAMRBox(dims[0], dims[2], dims[4], dims[1], dims[3], dims[5]);
}

//-----------------------------------------------------------------------------
void svtkAMRBox::BuildAMRBox(
  const int ilo, const int jlo, const int klo, const int ihi, const int jhi, const int khi)
{
  this->Initialize();
  this->SetDimensions(ilo, jlo, klo, ihi, jhi, khi);
}

//-----------------------------------------------------------------------------
svtkAMRBox::svtkAMRBox(const svtkAMRBox& other)
{
  *this = other;
}

//-----------------------------------------------------------------------------
svtkAMRBox& svtkAMRBox::operator=(const svtkAMRBox& other)
{
  assert("pre: AMR Box instance is invalid" && !other.IsInvalid());

  if (this == &other)
    return *this;
  for (int i = 0; i < 3; i++)
  {
    this->LoCorner[i] = other.LoCorner[i];
    this->HiCorner[i] = other.HiCorner[i];
  }
  return *this;
}

//-----------------------------------------------------------------------------
void svtkAMRBox::Initialize()
{
  for (int i = 0; i < 3; ++i)
  {
    this->LoCorner[i] = 0;
    this->HiCorner[i] = 0;
  }
}

//-----------------------------------------------------------------------------
void svtkAMRBox::SetDimensions(int ilo, int jlo, int klo, int ihi, int jhi, int khi, int desc)
{
  assert(ihi - ilo >= -1 && jhi - jlo >= -1 && khi - klo >= -1);
  this->LoCorner[0] = ilo;
  this->LoCorner[1] = jlo;
  this->LoCorner[2] = klo;
  this->HiCorner[0] = ihi;
  this->HiCorner[1] = jhi;
  this->HiCorner[2] = khi;

  switch (desc)
  {
    case SVTK_XY_PLANE:
      this->HiCorner[2] = this->LoCorner[2] - 1;
      break;
    case SVTK_XZ_PLANE:
      this->HiCorner[1] = this->LoCorner[1] - 1;
      break;
    case SVTK_YZ_PLANE:
      this->HiCorner[0] = this->LoCorner[0] - 1;
      break;
  }
}

//-----------------------------------------------------------------------------
void svtkAMRBox::SetDimensions(const int* lo, const int* hi, int desc)
{
  this->SetDimensions(lo[0], lo[1], lo[2], hi[0], hi[1], hi[2], desc);
}

//-----------------------------------------------------------------------------
void svtkAMRBox::SetDimensions(const int* dims, int desc)
{
  this->SetDimensions(dims[0], dims[2], dims[4], dims[1], dims[3], dims[5], desc);
}

//-----------------------------------------------------------------------------
void svtkAMRBox::GetDimensions(int* lo, int* hi) const
{
  assert("pre: AMR Box instance is invalid" && !this->IsInvalid());
  for (int q = 0; q < 3; ++q)
  {
    lo[q] = this->LoCorner[q];
    hi[q] = this->HiCorner[q];
  }
}

//-----------------------------------------------------------------------------
void svtkAMRBox::GetDimensions(int dims[6]) const
{
  assert("pre: AMR Box instance is invalid" && !this->IsInvalid());
  dims[0] = this->LoCorner[0];
  dims[1] = this->HiCorner[0];
  dims[2] = this->LoCorner[1];
  dims[3] = this->HiCorner[1];
  dims[4] = this->LoCorner[2];
  dims[5] = this->HiCorner[2];
}

//-----------------------------------------------------------------------------
void svtkAMRBox::GetValidHiCorner(int* hi) const
{
  assert("pre: AMR Box instance is invalid" && !this->IsInvalid());
  for (int q = 0; q < 3; ++q)
  {
    hi[q] = this->EmptyDimension(q) ? this->LoCorner[q] : this->HiCorner[q];
  }
}

//-----------------------------------------------------------------------------
void svtkAMRBox::GetBoxOrigin(
  const svtkAMRBox& extent, const double X0[3], const double spacing[3], double x0[3])
{
  assert("pre: input array is nullptr" && (x0 != nullptr));
  x0[0] = x0[1] = x0[2] = 0.0;

  for (int i = 0; i < 3; ++i)
  {
    x0[i] = X0[i] + extent.GetLoCorner()[i] * spacing[i];
  }
}

//-----------------------------------------------------------------------------
void svtkAMRBox::GetBounds(
  const svtkAMRBox& extent, const double origin[3], const double spacing[3], double bounds[6])
{
  int i, j;
  for (i = 0, j = 0; i < 3; ++i)
  {
    bounds[j++] = origin[i] + extent.LoCorner[i] * spacing[i];
    bounds[j++] = origin[i] + (extent.HiCorner[i] + 1) * spacing[i];
  }
}

//-----------------------------------------------------------------------------
bool svtkAMRBox::HasPoint(const svtkAMRBox& box, const double origin[3], const double spacing[3],
  double x, double y, double z)
{
  assert("pre: AMR Box instance is invalid" && !box.IsInvalid());

  double bb[6];
  svtkAMRBox::GetBounds(box, origin, spacing, bb);
  double min[3] = { bb[0], bb[2], bb[4] };
  double max[3] = { bb[1], bb[3], bb[5] };

  if (x >= min[0] && x <= max[0] && y >= min[1] && y <= max[1] && z >= min[2] && z <= max[2])
  {
    return true;
  }
  return false;
}

//-----------------------------------------------------------------------------
bool svtkAMRBox::operator==(const svtkAMRBox& other) const
{
  if ((this->Empty() && other.Empty()) ||
    (this->LoCorner[0] == other.LoCorner[0] && this->LoCorner[1] == other.LoCorner[1] &&
      this->LoCorner[2] == other.LoCorner[2] && this->HiCorner[0] == other.HiCorner[0] &&
      this->HiCorner[1] == other.HiCorner[1] && this->HiCorner[2] == other.HiCorner[2]))
  {
    return true;
  }
  return false;
}

//-----------------------------------------------------------------------------
ostream& svtkAMRBox::Print(ostream& os) const
{
  os << "-D AMR box => "
     << "Low: (" << this->LoCorner[0] << "," << this->LoCorner[1] << "," << this->LoCorner[2]
     << ") High: (" << this->HiCorner[0] << "," << this->HiCorner[1] << "," << this->HiCorner[2]
     << ")";
  return os;
}

//-----------------------------------------------------------------------------
void svtkAMRBox::Serialize(unsigned char*& buffer, svtkIdType& bytesize)
{
  assert("pre: input buffer is expected to be nullptr" && (buffer == nullptr));

  bytesize = svtkAMRBox::GetBytesize();
  buffer = new unsigned char[bytesize];
  assert(buffer != nullptr);

  // STEP 0: set pointer to traverse the buffer
  unsigned char* ptr = buffer;

  // STEP 7: serialize the low corner
  std::memcpy(ptr, &(this->LoCorner), 3 * sizeof(int));
  ptr += 3 * sizeof(int);

  // STEP 8: serialize the high corner
  std::memcpy(ptr, &(this->HiCorner), 3 * sizeof(int));
}

//-----------------------------------------------------------------------------
void svtkAMRBox::Serialize(int* buffer) const
{
  memcpy(buffer, this->LoCorner, 3 * sizeof(int));
  memcpy(buffer + 3, this->HiCorner, 3 * sizeof(int));
}

//-----------------------------------------------------------------------------
void svtkAMRBox::Deserialize(unsigned char* buffer, const svtkIdType& svtkNotUsed(bytesize))
{
  assert("pre: input buffer is nullptr" && (buffer != nullptr));

  // STEP 0: set pointer to traverse the buffer
  unsigned char* ptr = buffer;

  // STEP 7: de-serialize the low corner
  std::memcpy(&(this->LoCorner), ptr, 3 * sizeof(int));
  ptr += 3 * sizeof(int);

  // STEP 8: de-serialize the high corner
  std::memcpy(&(this->HiCorner), ptr, 3 * sizeof(int));
}

//-----------------------------------------------------------------------------
bool svtkAMRBox::IntersectBoxAlongDimension(const svtkAMRBox& other, const int q)
{
  assert("pre: dimension is out-of-bounds!" && (q >= 0) && (q <= 2));
  bool e1 = this->EmptyDimension(q);
  bool e2 = other.EmptyDimension(q);
  if (e1 && e2)
  {
    return true;
  }
  if (e1 || e2)
  {
    return false;
  }
  if (this->LoCorner[q] <= other.LoCorner[q])
  {
    this->LoCorner[q] = other.LoCorner[q];
  }
  if (this->HiCorner[q] >= other.HiCorner[q])
  {
    this->HiCorner[q] = other.HiCorner[q];
  }
  if (this->LoCorner[q] > this->HiCorner[q])
  {
    return false;
  }
  return true;
}

bool svtkAMRBox::Intersect(const svtkAMRBox& other)
{
  if (!this->IntersectBoxAlongDimension(other, 0) || !this->IntersectBoxAlongDimension(other, 1) ||
    !this->IntersectBoxAlongDimension(other, 2))
  {
    return false;
  }
  return true;
}

int svtkAMRBox::GetCellLinearIndex(
  const svtkAMRBox& box, const int i, const int j, const int k, int dim[3])
{
  // Convert to local numbering
  int I[3] = { i - box.GetLoCorner()[0], j - box.GetLoCorner()[1], k - box.GetLoCorner()[2] };

  // Get Cell sizes
  int N[3] = { dim[0] - 1, dim[1] - 1, dim[2] - 1 };

  // Reduce the sizes and indices to those
  // that correspond to the non-null dimensions
  int nd(0);
  for (int d = 0; d < 3; d++)
  {
    if (!box.EmptyDimension(d))
    {
      N[nd] = N[d];
      I[nd] = I[d];
      assert(I[nd] >= 0 && I[nd] < N[nd]);
      nd++;
    }
  }

  int idx = 0;
  for (int d = nd - 1; d >= 0; d--)
  {
    idx = idx * N[d] + I[d];
  }
  return idx;
}

void svtkAMRBox::Coarsen(int r)
{
  assert("pre: Input refinement ratio must be >= 2" && (r >= 2));
  assert("pre: AMR Box instance is invalid" && !this->IsInvalid());

  if (this->Empty())
  {
    std::cerr << "WARNING: tried refining an empty AMR box!\n";
    std::cerr << "FILE:" << __FILE__ << std::endl;
    std::cerr << "LINE:" << __LINE__ << std::endl;
    std::cerr.flush();
    return;
  }
  for (int q = 0; q < 3; ++q)
  {
    if (this->EmptyDimension(q))
    {
      continue;
    }
    this->LoCorner[q] =
      ((this->LoCorner[q] < 0) ? -abs(this->LoCorner[q] + 1) / r - 1 : this->LoCorner[q] / r);
    this->HiCorner[q] =
      (this->HiCorner[q] < 0 ? -abs(this->HiCorner[q] + 1) / r - 1 : this->HiCorner[q] / r);
  }
  assert("post: Coarsened AMR box should not be empty!" && !this->Empty());
  assert("post: Coarsened AMR Box instance is invalid" && !this->IsInvalid());
}

void svtkAMRBox::Refine(int r)
{
  assert("pre: Input refinement ratio must be >= 1" && (r >= 1));
  assert("pre: AMR Box instance is invalid" && !this->IsInvalid());

  if (this->Empty())
  {
    std::cerr << "WARNING: tried refining an empty AMR box!\n";
    std::cerr << "FILE:" << __FILE__ << std::endl;
    std::cerr << "LINE:" << __LINE__ << std::endl;
    std::cerr.flush();
    return;
  }
  for (int q = 0; q < 3; ++q)
  {
    if (!this->EmptyDimension(q))
    {
      this->LoCorner[q] = this->LoCorner[q] * r;
      this->HiCorner[q] = (this->HiCorner[q] + 1) * r - 1;
    }
  }
  assert("post: Refined AMR box should not be empty!" && !this->Empty());
  assert("post: Refined AMR Box instance is invalid" && !this->IsInvalid());
}

//-----------------------------------------------------------------------------
bool svtkAMRBox::DoesBoxIntersectAlongDimension(const svtkAMRBox& other, const int q) const
{
  if (this->EmptyDimension(q) && other.EmptyDimension(q))
  {
    return true;
  }
  int minVal = 0;
  int maxVal = 0;
  minVal = (this->LoCorner[q] < other.LoCorner[q]) ? other.LoCorner[q] : this->LoCorner[q];
  maxVal = (this->HiCorner[q] > other.HiCorner[q]) ? other.HiCorner[q] : this->HiCorner[q];

  if (minVal >= maxVal)
  {
    return false;
  }
  return true;
}

bool svtkAMRBox::DoesIntersect(const svtkAMRBox& other) const
{
  assert("pre: AMR Box instance is invalid" && !this->IsInvalid());
  if (this->Empty())
  {
    return false;
  }
  if (other.Empty())
  {
    return false;
  }
  if (!this->DoesBoxIntersectAlongDimension(other, 0) ||
    !this->DoesBoxIntersectAlongDimension(other, 1) ||
    !this->DoesBoxIntersectAlongDimension(other, 2))
  {
    return false;
  }
  return true;
}

int svtkAMRBox::ComputeStructuredCoordinates(const svtkAMRBox& box, const double dataOrigin[3],
  const double h[3], const double x[3], int ijk[3], double pcoords[3])
{
  double origin[3];
  svtkAMRBox::GetBoxOrigin(box, dataOrigin, h, origin);

  int num[3];
  box.GetNumberOfNodes(num);
  int extent[6] = { 0, num[0] - 1, 0, num[1] - 1, 0, num[2] - 1 };

  double bounds[6];
  svtkAMRBox::GetBounds(box, dataOrigin, h, bounds);

  // tolerance is needed for 2D data (this is squared tolerance)
  const double tol2 = 1e-12;

  //
  //  Compute the ijk location
  //
  int isInBounds = 1;
  for (int i = 0; i < 3; i++)
  {
    double d = x[i] - origin[i];
    double doubleLoc = d / h[i];
    // Floor for negative indexes.
    ijk[i] = svtkMath::Floor(doubleLoc);
    pcoords[i] = doubleLoc - static_cast<double>(ijk[i]);

    int tmpInBounds = 0;
    int minExt = extent[i * 2];
    int maxExt = extent[i * 2 + 1];

    // check if data is one pixel thick
    if (minExt == maxExt)
    {
      double dist = x[i] - bounds[2 * i];
      if (dist * dist <= h[i] * h[i] * tol2)
      {
        pcoords[i] = 0.0;
        ijk[i] = minExt;
        tmpInBounds = 1;
      }
    }

    // low boundary check
    else if (ijk[i] < minExt)
    {
      if ((h[i] >= 0 && x[i] >= bounds[i * 2]) || (h[i] < 0 && x[i] <= bounds[i * 2 + 1]))
      {
        pcoords[i] = 0.0;
        ijk[i] = minExt;
        tmpInBounds = 1;
      }
    }

    // high boundary check
    else if (ijk[i] >= maxExt)
    {
      if ((h[i] >= 0 && x[i] <= bounds[i * 2 + 1]) || (h[i] < 0 && x[i] >= bounds[i * 2]))
      {
        // make sure index is within the allowed cell index range
        pcoords[i] = 1.0;
        ijk[i] = maxExt - 1;
        tmpInBounds = 1;
      }
    }

    // else index is definitely within bounds
    else
    {
      tmpInBounds = 1;
    }

    // clear isInBounds if out of bounds for this dimension
    isInBounds = (isInBounds & tmpInBounds);
  }

  return isInBounds;
}

//------------------------------------------------------------------------------
void svtkAMRBox::GetGhostVector(int r, int nghost[6]) const
{
  // STEP 0: initialize nghost
  for (int i = 0; i < 3; ++i)
  {
    nghost[i * 2] = nghost[i * 2 + 1] = 0;
  }

  // STEP 1: compute number of ghost layers along each dimension's min and max.
  // Detecting partially overlapping boxes is based on the following:
  // Cell location k at level L-1 holds the range [k*r,k*r+(r-1)] of
  // level L, where r is the refinement ratio. Consequently, if the
  // min extent of the box is greater than k*r or if the max extent
  // of the box is less than k*r+(r-1), then the grid partially overlaps.

  svtkAMRBox coarsenedBox = *this;
  coarsenedBox.Coarsen(r);
  for (int i = 0; i < 3; ++i)
  {
    if (!this->EmptyDimension(i))
    {
      int minRange[2];
      minRange[0] = coarsenedBox.LoCorner[i] * r;
      minRange[1] = coarsenedBox.LoCorner[i] * r + (r - 1);
      if (this->LoCorner[i] > minRange[0])
      {
        nghost[i * 2] = (minRange[1] + 1) - this->LoCorner[i];
      }

      int maxRange[2];
      maxRange[0] = coarsenedBox.HiCorner[i] * r;
      maxRange[1] = coarsenedBox.HiCorner[i] * r + (r - 1);
      if (this->HiCorner[i] < maxRange[1])
      {
        nghost[i * 2 + 1] = this->HiCorner[i] - (maxRange[0] - 1);
      }
    }
  } // END for all dimensions
}

void svtkAMRBox::RemoveGhosts(int r)
{
  // Detecting partially overlapping boxes is based on the following:
  // Cell location k at level L-1 holds the range [k*r,k*r+(r-1)] of
  // level L, where r is the refinement ratio. Consequently, if the
  // min extent of the box is greater than k*r or if the max extent
  // of the box is less than k*r+(r-1), then the grid partially overlaps.
  svtkAMRBox coarsenedBox = *this;
  coarsenedBox.Coarsen(r);
  for (int i = 0; i < 3; ++i)
  {
    if (!this->EmptyDimension(i))
    {
      int minRange[2];
      minRange[0] = coarsenedBox.LoCorner[i] * r;
      minRange[1] = coarsenedBox.LoCorner[i] * r + (r - 1);
      if (this->LoCorner[i] > minRange[0])
      {
        this->LoCorner[i] = (minRange[1] + 1);
      }

      int maxRange[2];
      maxRange[0] = coarsenedBox.HiCorner[i] * r;
      maxRange[1] = coarsenedBox.HiCorner[i] * r + (r - 1);
      if (this->HiCorner[i] < maxRange[1])
      {
        this->HiCorner[i] = (maxRange[0] - 1);
      }
    }
  } // END for all dimensions
}

int svtkAMRBox::ComputeDimension() const
{
  int dim(3);
  for (int i = 0; i < 3; i++)
  {
    if (this->EmptyDimension(i))
    {
      dim--;
    }
  }
  return (dim);
}

void svtkAMRBox::GetNumberOfCells(int ext[3]) const
{
  ext[0] = this->HiCorner[0] - this->LoCorner[0] + 1;
  ext[1] = this->HiCorner[1] - this->LoCorner[1] + 1;
  ext[2] = this->HiCorner[2] - this->LoCorner[2] + 1;
}

svtkIdType svtkAMRBox::GetNumberOfCells() const
{
  int cellExtent[3];
  this->GetNumberOfCells(cellExtent);
  int numCells = 0;
  for (int i = 0; i < 3; ++i)
  {
    if (numCells == 0)
    {
      numCells = cellExtent[i];
    }
    else if (cellExtent[i] != 0)
    {
      numCells *= cellExtent[i];
    }
  }
  return (numCells);
}

//-----------------------------------------------------------------------------
bool svtkAMRBox::Contains(int i, int j, int k) const
{
  int ijk[3] = { i, j, k };
  return this->Contains(ijk);
}

//-----------------------------------------------------------------------------
bool svtkAMRBox::Contains(const int* I) const
{
  bool res(true);
  for (int i = 0; i < 3; i++)
  {
    if (!this->EmptyDimension(i) && (this->LoCorner[i] > I[i] || this->HiCorner[i] < I[i]))
    {
      res = false;
    }
  }
  return res;
}

//-----------------------------------------------------------------------------
bool svtkAMRBox::Contains(const svtkAMRBox& other) const
{
  const int* lo = other.LoCorner;
  const int* hi = other.HiCorner;
  return this->Contains(lo) && this->Contains(hi);
}

void svtkAMRBox::GetNumberOfNodes(int* ext) const
{
  ext[0] = this->HiCorner[0] - this->LoCorner[0] + 2;
  ext[1] = this->HiCorner[1] - this->LoCorner[1] + 2;
  ext[2] = this->HiCorner[2] - this->LoCorner[2] + 2;
  assert(ext[0] >= 1 && ext[1] >= 1 && ext[2] >= 1);
}

//-----------------------------------------------------------------------------
svtkIdType svtkAMRBox::GetNumberOfNodes() const
{
  assert("pre: AMR Box instance is invalid" && !this->IsInvalid());
  int ext[3];
  this->GetNumberOfNodes(ext);
  int numNodes = 0;
  for (int i = 0; i < 3; ++i)
  {
    if (numNodes == 0)
    {
      numNodes = ext[i];
    }
    else if (ext[i] != 0)
    {
      numNodes *= ext[i];
    }
  }
  return (numNodes);
}

//-----------------------------------------------------------------------------
void svtkAMRBox::Shift(int i, int j, int k)
{
  int ijk[3] = { i, j, k };
  this->Shift(ijk);
}

//-----------------------------------------------------------------------------
void svtkAMRBox::Shift(const int* ijk)
{
  for (int q = 0; q < 3; ++q)
  {
    this->LoCorner[q] = this->LoCorner[q] + ijk[q];
    this->HiCorner[q] = this->HiCorner[q] + ijk[q];
  }
}

//-----------------------------------------------------------------------------
void svtkAMRBox::Grow(int byN)
{
  assert("pre: AMR Box instance is invalid" && !this->IsInvalid());
  // TODO: One question here is, should we allow negative indices?
  //       Or should we otherwise, ensure that the box is grown with
  //       bounds.
  for (int q = 0; q < 3; ++q)
  {
    if (!this->EmptyDimension(q))
    {
      this->LoCorner[q] -= byN;
      this->HiCorner[q] += byN;
    }
  }
  assert("post: Grown AMR Box instance is invalid" && !this->IsInvalid());
}

//-----------------------------------------------------------------------------
void svtkAMRBox::Shrink(int byN)
{
  assert("pre: AMR Box instance is invalid" && !this->IsInvalid());
  // TODO: One question here is, should we allow negative indices?
  //       Or should we otherwise, ensure that the box is grown with
  //       bounds.
  for (int q = 0; q < 3; ++q)
  {
    if (!this->EmptyDimension(q))
    {
      this->LoCorner[q] += byN;
      this->HiCorner[q] -= byN;
    }
  }
  assert("post: Grown AMR Box instance is invalid" && !this->IsInvalid());
}
