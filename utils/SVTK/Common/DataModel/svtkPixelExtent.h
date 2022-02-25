/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkPixelExtenth.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkPixelExtent
 *
 * Representation of a cartesian pixel plane and common operations
 * on it. The implementation is intended to be fast and light
 * so that it may be used in place of int[4] with little or no
 * performance penalty.
 *
 * NOTE in most cases operation on an empty object produces
 * incorrect results. If it an issue query Empty() first.
 */

#ifndef svtkPixelExtent_h
#define svtkPixelExtent_h

#include "svtkCommonDataModelModule.h" // for export
#include "svtkSystemIncludes.h"        // for SVTK's system header config

#include <algorithm> // for inline impl
#include <climits>   // for inline impl
#include <deque>     // for inline impl
#include <iostream>  // for inline impl

class SVTKCOMMONDATAMODEL_EXPORT svtkPixelExtent
{
public:
  svtkPixelExtent();

  template <typename T>
  svtkPixelExtent(const T* ext);

  template <typename T>
  svtkPixelExtent(T ilo, T ihi, T jlo, T jhi);

  template <typename T>
  svtkPixelExtent(T width, T height)
  {
    this->SetData(T(0), width - T(1), T(0), height - T(1));
  }

  svtkPixelExtent(const svtkPixelExtent& other);

  svtkPixelExtent& operator=(const svtkPixelExtent& other);

  /**
   * Element access
   */
  int& operator[](int i) { return this->Data[i]; }
  const int& operator[](int i) const { return this->Data[i]; }

  /**
   * Set the extent.
   */
  void SetData(const svtkPixelExtent& ext);

  template <typename T>
  void SetData(const T* ext);

  template <typename T>
  void SetData(T ilo, T ihi, T jlo, T jhi);
  void Clear();

  /**
   * Direct access to internal data.
   */
  int* GetData() { return this->Data; }
  const int* GetData() const { return this->Data; }

  template <typename T>
  void GetData(T data[4]) const;

  unsigned int* GetDataU() { return reinterpret_cast<unsigned int*>(this->Data); }

  const unsigned int* GetDataU() const { return reinterpret_cast<const unsigned int*>(this->Data); }

  //@{
  /**
   * Get the start/end index.
   */
  void GetStartIndex(int first[2]) const;
  void GetStartIndex(int first[2], const int origin[2]) const;
  void GetEndIndex(int last[2]) const;
  //@}

  /**
   * Return true if empty.
   */
  int Empty() const;

  /**
   * Test for equivalence.
   */
  bool operator==(const svtkPixelExtent& other) const;

  //@{
  /**
   * Return non-zero if this extent contains the other.
   */
  int Contains(const svtkPixelExtent& other) const;
  int Contains(int i, int j) const;
  //@}

  /**
   * Return non-zero if the extent is disjoint from the other
   */
  int Disjoint(svtkPixelExtent other) const;

  /**
   * Get the number in each direction.
   */
  template <typename T>
  void Size(T nCells[2]) const;

  /**
   * Get the total number.
   */
  size_t Size() const;

  /**
   * In place intersection.
   */
  void operator&=(const svtkPixelExtent& other);

  /**
   * In place union
   */
  void operator|=(const svtkPixelExtent& other);

  //@{
  /**
   * Expand the extents by n.
   */
  void Grow(int n);
  void Grow(int q, int n);
  void GrowLow(int q, int n);
  void GrowHigh(int q, int n);
  //@}

  //@{
  /**
   * Shrink the extent by n.
   */
  void Shrink(int n);
  void Shrink(int q, int n);
  //@}

  /**
   * Shifts by low corner of this, moving to the origin.
   */
  void Shift();

  /**
   * Shift by low corner of the given extent.
   */
  void Shift(const svtkPixelExtent& ext);

  /**
   * Shift by the given amount.
   */
  void Shift(int* n);

  /**
   * Shift by the given amount in the given direction.
   */
  void Shift(int q, int n);

  /**
   * Divide the extent in half in the given direction. The
   * operation is done in-place the other half of the split
   * extent is returned. The return will be empty if the split
   * could not be made.
   */
  svtkPixelExtent Split(int dir);

  //@{
  /**
   * In-place conversion from cell based to node based extent, and vise-versa.
   */
  void CellToNode();
  void NodeToCell();
  //@}

  /**
   * Get the number in each direction.
   */
  template <typename T>
  static void Size(const svtkPixelExtent& ext, T nCells[2]);

  /**
   * Get the total number.
   */
  static size_t Size(const svtkPixelExtent& ext);

  /**
   * Add or remove ghost cells. If a problem domain is
   * provided then the result is clipled to be within the
   * problem domain.
   */
  static svtkPixelExtent Grow(const svtkPixelExtent& inputExt, int n);

  static svtkPixelExtent Grow(
    const svtkPixelExtent& inputExt, const svtkPixelExtent& problemDomain, int n);

  static svtkPixelExtent GrowLow(const svtkPixelExtent& ext, int q, int n);

  static svtkPixelExtent GrowHigh(const svtkPixelExtent& ext, int q, int n);

  /**
   * Remove ghost cells. If a problem domain is
   * provided the input is pinned at the domain.
   */
  static svtkPixelExtent Shrink(
    const svtkPixelExtent& inputExt, const svtkPixelExtent& problemDomain, int n);

  static svtkPixelExtent Shrink(const svtkPixelExtent& inputExt, int n);

  /**
   * Convert from point extent to cell extent
   * while respecting the dimensionality of the data.
   */
  static svtkPixelExtent NodeToCell(const svtkPixelExtent& inputExt);

  /**
   * Convert from cell extent to point extent
   * while respecting the dimensionality of the data.
   */
  static svtkPixelExtent CellToNode(const svtkPixelExtent& inputExt);

  //@{
  /**
   * Shift by the given amount while respecting mode.
   */
  static void Shift(int* ij, int n);
  static void Shift(int* ij, int* n);
  //@}

  /**
   * Split ext at i,j, resulting extents (up to 4) are appended
   * to newExts. If i,j is outside ext, ext is passed through
   * unmodified.
   */
  static void Split(int i, int j, const svtkPixelExtent& ext, std::deque<svtkPixelExtent>& newExts);

  /**
   * A - B = C
   * C is a set of disjoint extents such that the
   * intersection of B and C is empty and the intersection
   * of A and C is C.
   */
  static void Subtract(
    const svtkPixelExtent& A, const svtkPixelExtent& B, std::deque<svtkPixelExtent>& newExts);

  /**
   * Merge compatible extents in the list. Extents are compatible
   * if they are directly adjacent nad have the same extent along
   * the adjacent edge.
   */
  static void Merge(std::deque<svtkPixelExtent>& exts);

private:
  int Data[4];
};

/**
 * Stream insertion operator for formatted output of pixel extents.
 */
SVTKCOMMONDATAMODEL_EXPORT
std::ostream& operator<<(std::ostream& os, const svtkPixelExtent& ext);

//-----------------------------------------------------------------------------
template <typename T>
void svtkPixelExtent::SetData(const T* ext)
{
  Data[0] = static_cast<int>(ext[0]);
  Data[1] = static_cast<int>(ext[1]);
  Data[2] = static_cast<int>(ext[2]);
  Data[3] = static_cast<int>(ext[3]);
}

//-----------------------------------------------------------------------------
template <typename T>
void svtkPixelExtent::SetData(T ilo, T ihi, T jlo, T jhi)
{
  T ext[4] = { ilo, ihi, jlo, jhi };
  this->SetData(ext);
}

//-----------------------------------------------------------------------------
inline void svtkPixelExtent::SetData(const svtkPixelExtent& other)
{
  this->SetData(other.GetData());
}

//-----------------------------------------------------------------------------
template <typename T>
void svtkPixelExtent::GetData(T data[4]) const
{
  data[0] = static_cast<T>(this->Data[0]);
  data[1] = static_cast<T>(this->Data[1]);
  data[2] = static_cast<T>(this->Data[2]);
  data[3] = static_cast<T>(this->Data[3]);
}

//-----------------------------------------------------------------------------
inline void svtkPixelExtent::Clear()
{
  this->SetData<int>(INT_MAX, INT_MIN, INT_MAX, INT_MIN);
}

//-----------------------------------------------------------------------------
inline svtkPixelExtent::svtkPixelExtent()
{
  this->Clear();
}

//-----------------------------------------------------------------------------
template <typename T>
svtkPixelExtent::svtkPixelExtent(const T* ext)
{
  this->SetData(ext);
}

//-----------------------------------------------------------------------------
template <typename T>
svtkPixelExtent::svtkPixelExtent(T ilo, T ihi, T jlo, T jhi)
{
  this->SetData(ilo, ihi, jlo, jhi);
}

//-----------------------------------------------------------------------------
inline svtkPixelExtent& svtkPixelExtent::operator=(const svtkPixelExtent& other)
{
  if (&other != this)
  {
    this->Data[0] = other.Data[0];
    this->Data[1] = other.Data[1];
    this->Data[2] = other.Data[2];
    this->Data[3] = other.Data[3];
  }
  return *this;
}

//-----------------------------------------------------------------------------
inline svtkPixelExtent::svtkPixelExtent(const svtkPixelExtent& other)
{
  *this = other;
}

//-----------------------------------------------------------------------------
template <typename T>
void svtkPixelExtent::Size(const svtkPixelExtent& ext, T nCells[2])
{
  nCells[0] = ext[1] - ext[0] + 1;
  nCells[1] = ext[3] - ext[2] + 1;
}

//-----------------------------------------------------------------------------
inline size_t svtkPixelExtent::Size(const svtkPixelExtent& ext)
{
  return (ext[1] - ext[0] + 1) * (ext[3] - ext[2] + 1);
}

//-----------------------------------------------------------------------------
template <typename T>
void svtkPixelExtent::Size(T nCells[2]) const
{
  svtkPixelExtent::Size(*this, nCells);
}

//-----------------------------------------------------------------------------
inline size_t svtkPixelExtent::Size() const
{
  return svtkPixelExtent::Size(*this);
}

//-----------------------------------------------------------------------------
inline void svtkPixelExtent::GetStartIndex(int first[2]) const
{
  first[0] = this->Data[0];
  first[1] = this->Data[2];
}

//-----------------------------------------------------------------------------
inline void svtkPixelExtent::GetStartIndex(int first[2], const int origin[2]) const
{
  first[0] = this->Data[0] - origin[0];
  first[1] = this->Data[2] - origin[1];
}

//-----------------------------------------------------------------------------
inline void svtkPixelExtent::GetEndIndex(int last[2]) const
{
  last[0] = this->Data[1];
  last[1] = this->Data[3];
}

//-----------------------------------------------------------------------------
inline int svtkPixelExtent::Empty() const
{
  if (this->Data[0] > this->Data[1] || this->Data[2] > this->Data[3])
  {
    return 1;
  }
  return 0;
}

//-----------------------------------------------------------------------------
inline bool svtkPixelExtent::operator==(const svtkPixelExtent& other) const
{
  if ((this->Data[0] == other.Data[0]) && (this->Data[1] == other.Data[1]) &&
    (this->Data[2] == other.Data[2]) && (this->Data[3] == other.Data[3]))
  {
    return 1;
  }
  return 0;
}

//-----------------------------------------------------------------------------
inline int svtkPixelExtent::Contains(const svtkPixelExtent& other) const
{
  if ((this->Data[0] <= other.Data[0]) && (this->Data[1] >= other.Data[1]) &&
    (this->Data[2] <= other.Data[2]) && (this->Data[3] >= other.Data[3]))
  {
    return 1;
  }
  return 0;
}

//-----------------------------------------------------------------------------
inline int svtkPixelExtent::Contains(int i, int j) const
{
  if ((this->Data[0] <= i) && (this->Data[1] >= i) && (this->Data[2] <= j) && (this->Data[3] >= j))
  {
    return 1;
  }
  return 0;
}

//-----------------------------------------------------------------------------
inline void svtkPixelExtent::operator&=(const svtkPixelExtent& other)
{
  if (this->Empty())
  {
    return;
  }

  if (other.Empty())
  {
    this->Clear();
    return;
  }

  this->Data[0] = std::max(this->Data[0], other.Data[0]);
  this->Data[1] = std::min(this->Data[1], other.Data[1]);
  this->Data[2] = std::max(this->Data[2], other.Data[2]);
  this->Data[3] = std::min(this->Data[3], other.Data[3]);

  if (this->Empty())
  {
    this->Clear();
  }
}

//-----------------------------------------------------------------------------
inline void svtkPixelExtent::operator|=(const svtkPixelExtent& other)
{
  if (other.Empty())
  {
    return;
  }

  if (this->Empty())
  {
    this->SetData(other.GetData());
    return;
  }

  this->Data[0] = std::min(this->Data[0], other.Data[0]);
  this->Data[1] = std::max(this->Data[1], other.Data[1]);
  this->Data[2] = std::min(this->Data[2], other.Data[2]);
  this->Data[3] = std::max(this->Data[3], other.Data[3]);
}

//-----------------------------------------------------------------------------
inline int svtkPixelExtent::Disjoint(svtkPixelExtent other) const
{
  other &= *this;
  return other.Empty();
}

//-----------------------------------------------------------------------------
inline void svtkPixelExtent::Grow(int n)
{
  this->Data[0] -= n;
  this->Data[1] += n;
  this->Data[2] -= n;
  this->Data[3] += n;
}

//-----------------------------------------------------------------------------
inline void svtkPixelExtent::Grow(int q, int n)
{
  q *= 2;

  this->Data[q] -= n;
  this->Data[q + 1] += n;
}

//-----------------------------------------------------------------------------
inline void svtkPixelExtent::GrowLow(int q, int n)
{
  this->Data[2 * q] -= n;
}

//-----------------------------------------------------------------------------
inline void svtkPixelExtent::GrowHigh(int q, int n)
{
  this->Data[2 * q + 1] += n;
}

//-----------------------------------------------------------------------------
inline void svtkPixelExtent::Shrink(int n)
{
  this->Data[0] += n;
  this->Data[1] -= n;
  this->Data[2] += n;
  this->Data[3] -= n;
}

//-----------------------------------------------------------------------------
inline void svtkPixelExtent::Shrink(int q, int n)
{
  q *= 2;
  this->Data[q] += n;
  this->Data[q + 1] -= n;
}

//-----------------------------------------------------------------------------
inline void svtkPixelExtent::Shift(int* n)
{
  this->Data[0] += n[0];
  this->Data[1] += n[0];
  this->Data[2] += n[1];
  this->Data[3] += n[1];
}

//-----------------------------------------------------------------------------
inline void svtkPixelExtent::Shift(int q, int n)
{
  q *= 2;
  this->Data[q] += n;
  this->Data[q + 1] += n;
}

//-----------------------------------------------------------------------------
inline void svtkPixelExtent::Shift(const svtkPixelExtent& other)
{
  for (int q = 0; q < 2; ++q)
  {
    int qq = q * 2;
    int n = -other[qq];

    this->Data[qq] += n;
    this->Data[qq + 1] += n;
  }
}

//-----------------------------------------------------------------------------
inline void svtkPixelExtent::Shift()
{
  for (int q = 0; q < 2; ++q)
  {
    int qq = q * 2;
    int n = -this->Data[qq];

    this->Data[qq] += n;
    this->Data[qq + 1] += n;
  }
}

//-----------------------------------------------------------------------------
inline svtkPixelExtent svtkPixelExtent::Split(int dir)
{
  svtkPixelExtent half;

  int q = 2 * dir;
  int l = this->Data[q + 1] - this->Data[q] + 1;
  int s = l / 2;

  if (s)
  {
    s += this->Data[q];
    half = *this;
    half.Data[q] = s;
    this->Data[q + 1] = s - 1;
  }

  return half;
}

//-----------------------------------------------------------------------------
inline void svtkPixelExtent::CellToNode()
{
  ++this->Data[1];
  ++this->Data[3];
}

//-----------------------------------------------------------------------------
inline void svtkPixelExtent::NodeToCell()
{
  --this->Data[1];
  --this->Data[3];
}

//-----------------------------------------------------------------------------
inline bool operator<(const svtkPixelExtent& l, const svtkPixelExtent& r)
{
  return l.Size() < r.Size();
}

#endif
// SVTK-HeaderTest-Exclude: svtkPixelExtent.h
