/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkAMRBox.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkAMRBox
 * @brief   Encloses a rectangular region of voxel like cells.
 *
 *
 * svtkAMRBox stores information for an AMR block
 *
 * @sa
 * svtkAMRInformation
 */

#ifndef svtkAMRBox_h
#define svtkAMRBox_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkObject.h"
#include "svtkStructuredData.h" // For SVTK_XYZ_GRID definition

class SVTKCOMMONDATAMODEL_EXPORT svtkAMRBox
{
public:
  /**
   * Construct the empty box.
   */
  svtkAMRBox();

  /**
   * Copy construct this box from another.
   */
  svtkAMRBox(const svtkAMRBox& other);

  /**
   * Construct a specific 3D box.
   */
  svtkAMRBox(int ilo, int jlo, int klo, int ihi, int jhi, int khi);

  /**
   * Construct an AMR box from the description a svtkUniformGrid
   * Note that the dimensions specify the node dimensions, rather than the cell dimensions
   */
  svtkAMRBox(const double* origin, const int* dimensions, const double* spacing,
    const double* globalOrigin, int gridDescription = SVTK_XYZ_GRID);

  /**
   * Construct a specific box. (ilo,jlo,klo,)(ihi,jhi,khi)
   */
  svtkAMRBox(const int lo[3], const int hi[3]);

  svtkAMRBox(const int dims[6]);

  /**
   * Copy the other box to this box.
   */
  svtkAMRBox& operator=(const svtkAMRBox& other);

  virtual ~svtkAMRBox() {}

  //@{
  /**
   * Set the box to be invalid;
   */
  void Invalidate()
  {
    this->LoCorner[0] = this->LoCorner[1] = this->LoCorner[2] = 0;
    this->HiCorner[0] = this->HiCorner[1] = this->HiCorner[2] = -2;
  }
  //@}

  /**
   * Whether dimension i is empty, e.g. if the data set is type SVTK_XY_PLANE
   */
  bool EmptyDimension(int i) const { return HiCorner[i] <= LoCorner[i] - 1; }

  /**
   * Set the dimensions of the box. ilo,jlo,klo,ihi,jhi,khi
   */
  void SetDimensions(int ilo, int jlo, int klo, int ihi, int jhi, int khi, int desc = SVTK_XYZ_GRID);

  /**
   * Set the dimensions of the box. (ilo,jlo,klo),(ihi,jhi,khi)
   */
  void SetDimensions(const int lo[3], const int hi[3], int desc = SVTK_XYZ_GRID);

  /**
   * Set the dimensions of the box. (ilo,ihi,jlo,jhi,klo,khi)
   */
  void SetDimensions(const int dims[6], int desc = SVTK_XYZ_GRID);

  /**
   * Get the dimensions of this box. (ilo,jlo,jhi),(ihi,jhi,khi)
   */
  void GetDimensions(int lo[3], int hi[3]) const;

  /**
   * Get the dimensions of this box. (ilo,ihi, jlo,jhi, klo,khi)
   */
  void GetDimensions(int dims[6]) const;

  //@{
  /**
   * Gets the number of cells enclosed by the box.
   */
  svtkIdType GetNumberOfCells() const;
  void GetNumberOfCells(int num[3]) const;
  //@}

  //@{
  /**
   * Gets the number of nodes required to construct
   * a physical representation of the box.
   */
  void GetNumberOfNodes(int ext[3]) const;
  svtkIdType GetNumberOfNodes() const;
  //@}

  /**
   * Determines the dimension of the AMR box given the
   * box indices. Note, the AMR box can be on an arbitrary
   * axis-aligned plane, i.e., XZ or YZ.
   */
  int ComputeDimension() const;

  /**
   * Get the low corner index.
   */
  const int* GetLoCorner() const { return this->LoCorner; }
  const int* GetHiCorner() const { return this->HiCorner; }

  /**
   * Return a high corner. If dimension j is empty,
   * then hi[j] is set from lo[j]. This is convenient
   * For algorithm that must iterate over all cells
   */
  void GetValidHiCorner(int hi[3]) const;

  bool Empty() const { return this->IsInvalid(); }

  /**
   * Check to see if the AMR box instance is invalid.
   */
  bool IsInvalid() const
  {
    return ((this->HiCorner[0] < this->LoCorner[0] - 1) ||
      (this->HiCorner[1] < this->LoCorner[1] - 1) || (this->HiCorner[2] < this->LoCorner[2] - 1));
  }

  /**
   * Test if this box is equal with the box instance on the rhs.
   * Note: Two AMR boxes are equal if: (a) they have the same dimensionality
   * (b) they are at the same level and (c) they occupy the same index space.
   */
  bool operator==(const svtkAMRBox& other) const;

  /**
   * Test if this box is NOT equal with the box instance on the rhs.
   * Note: Two AMR boxes are equal if: (a) they have the same dimensionality
   * (b) they are at the same level and (c) they occupy the same index space.
   */
  bool operator!=(const svtkAMRBox& other) const { return (!(*this == other)); }

  /**
   * Send the box to a stream. "(ilo,jlo,jhi),(ihi,jhi,khi)"
   */
  ostream& Print(ostream& os) const;

  //@{
  /**
   * Serializes this object instance into a byte-stream.
   * buffer   -- user-supplied pointer where the serialized object is stored.
   * bytesize -- number of bytes, i.e., the size of the buffer.
   * NOTE: buffer is allocated internally by this method.
   * Pre-conditions:
   * buffer == nullptr
   * Post-conditions:
   * buffer   != nullptr
   * bytesize != 0
   */
  void Serialize(unsigned char*& buffer, svtkIdType& bytesize);
  void Serialize(int* buffer) const;
  //@}

  /**
   * Deserializes this object instance from the given byte-stream.
   * Pre-conditions:
   * buffer != nullptr
   * bytesize != 0
   */
  void Deserialize(unsigned char* buffer, const svtkIdType& bytesize);

  /**
   * Checks if this instance of svtkAMRBox intersects with the box passed through
   * the argument list along the given dimension q. True is returned iff the box
   * intersects successfully. Otherwise, there is no intersection along the
   * given dimension and false is returned.
   */
  bool DoesBoxIntersectAlongDimension(const svtkAMRBox& other, const int q) const;

  bool DoesIntersect(const svtkAMRBox& other) const;

  /**
   * Coarsen the box.
   */
  void Coarsen(int r);

  /**
   * Refine the box.
   */
  void Refine(int r);

  //@{
  /**
   * Grows the box in all directions.
   */
  void Grow(int byN);
  void Shrink(int byN);
  //@}

  //@{
  /**
   * Shifts the box in index space
   */
  void Shift(int i, int j, int k);
  void Shift(const int I[3]);
  //@}

  /**
   * Intersect this box with another box in place.  Returns
   * true if the boxes do intersect.  Note that the box is
   * modified to be the intersection or is made invalid.
   */
  bool Intersect(const svtkAMRBox& other);

  //@{
  /**
   * Test to see if a given cell index is inside this box.
   */
  bool Contains(int i, int j, int k) const;
  bool Contains(const int I[3]) const;
  //@}

  /**
   * Test to see if a given box is inside this box.
   */
  bool Contains(const svtkAMRBox&) const;

  /**
   * Given an AMR box and the refinement ratio, r, this method computes the
   * number of ghost layers in each of the 6 directions, i.e.,
   * [imin,imax,jmin,jmax,kmin,kmax]
   */
  void GetGhostVector(int r, int nghost[6]) const;

  /**
   * Given an AMR box and the refinement ratio, r, this shrinks
   * the AMRBox
   */
  void RemoveGhosts(int r);

public:
  /**
   * Returns the number of bytes allocated by this instance. In addition,
   * this number of bytes corresponds to the buffer size required to serialize
   * any svtkAMRBox instance.
   */
  static svtkIdType GetBytesize() { return 6 * sizeof(int); }

  /**
   * Returns the linear index of the given cell structured coordinates
   */
  static int GetCellLinearIndex(
    const svtkAMRBox& box, const int i, const int j, const int k, int imageDimension[3]);

  /**
   * Get the bounds of this box.
   */
  static void GetBounds(
    const svtkAMRBox& box, const double origin[3], const double spacing[3], double bounds[6]);

  /**
   * Get the world space origin of this box. The origin is the
   * location of the lower corner cell's lower corner node,
   */
  static void GetBoxOrigin(
    const svtkAMRBox& box, const double X0[3], const double spacing[3], double x0[3]);

  /**
   * Checks if the point is inside this AMRBox instance.
   * x,y,z the world point
   */
  static bool HasPoint(const svtkAMRBox& box, const double origin[3], const double spacing[3],
    double x, double y, double z);

  /**
   * Compute structured coordinates
   */
  static int ComputeStructuredCoordinates(const svtkAMRBox& box, const double dataOrigin[3],
    const double h[3], const double x[3], int ijk[3], double pcoords[3]);

protected:
  /**
   * Initializes this box instance.
   */
  void Initialize();

  /**
   * Intersects this instance of svtkAMRbox with box passed through the argument
   * list along the given dimension q. True is returned iff the box intersects
   * successfully. Otherwise, false is returned if there is no intersection at
   * the given dimension.
   */
  bool IntersectBoxAlongDimension(const svtkAMRBox& other, const int q);

private:
  int LoCorner[3]; // lo corner cell id.
  int HiCorner[3]; // hi corner cell id.

  //@{
  /**
   * This method builds the AMR box with the given dimensions.
   * Note: the dimension of the AMR box is automatically detected
   * within this method.
   */
  void BuildAMRBox(
    const int ilo, const int jlo, const int klo, const int ihi, const int jhi, const int khi);
  //@}
};

//*****************************************************************************
//@{
/**
 * Fill the region of "pArray" enclosed by "destRegion" with "fillValue"
 * "pArray" is defined on "arrayRegion".
 */
template <typename T>
void FillRegion(T* pArray, const svtkAMRBox& arrayRegion, const svtkAMRBox& destRegion, T fillValue)
{
  // Convert regions to array index space. SVTK arrays
  // always start with 0,0,0.
  int ofs[3];
  ofs[0] = -arrayRegion.GetLoCorner()[0];
  ofs[1] = -arrayRegion.GetLoCorner()[1];
  ofs[2] = -arrayRegion.GetLoCorner()[2];
  svtkAMRBox arrayDims(arrayRegion);
  arrayDims.Shift(ofs);
  svtkAMRBox destDims(destRegion);
  destDims.Shift(ofs);
  // Quick sanity check.
  if (!arrayRegion.Contains(destRegion))
  {
    svtkGenericWarningMacro(<< "ERROR: Array must enclose the destination region. "
                           << "Aborting the fill.");
  }
  // Get the bounds of the indices we fill.
  const int* destLo = destDims.GetLoCorner();
  int destHi[3];
  destDims.GetValidHiCorner(destHi);
  // Get the array dimensions.
  int arrayHi[3];
  arrayDims.GetNumberOfCells(arrayHi);
  // Fill.
  for (int k = destLo[2]; k <= destHi[2]; ++k)
  {
    svtkIdType kOfs = k * arrayHi[0] * arrayHi[1];
    for (int j = destLo[1]; j <= destHi[1]; ++j)
    {
      svtkIdType idx = kOfs + j * arrayHi[0] + destLo[0];
      for (int i = destLo[0]; i <= destHi[0]; ++i)
      {
        pArray[idx] = fillValue;
        ++idx;
      }
    }
  }
  //@}
}

#endif
// SVTK-HeaderTest-Exclude: svtkAMRBox.h
