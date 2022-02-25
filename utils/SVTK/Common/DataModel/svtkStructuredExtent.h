/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkStructuredExtent.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkStructuredExtent
 * @brief   helper class to aid working with structured
 *  extents.
 *
 *
 *  svtkStructuredExtent is an helper class that helps in arithmetic with
 *  structured extents. It defines a bunch of static methods (most of which are
 *  inlined) to aid in dealing with extents.
 */

#ifndef svtkStructuredExtent_h
#define svtkStructuredExtent_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkObject.h"

class SVTKCOMMONDATAMODEL_EXPORT svtkStructuredExtent : public svtkObject
{
public:
  static svtkStructuredExtent* New();
  svtkTypeMacro(svtkStructuredExtent, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Clamps \c ext to fit in \c wholeExt.
   */
  static void Clamp(int ext[6], const int wholeExt[]);

  /**
   * Returns true if \c ext is fits within \c wholeExt with at least 1 dimension
   * smaller than the \c wholeExt.
   */
  static bool StrictlySmaller(const int ext[6], const int wholeExt[6]);

  /**
   * Returns if \c ext fits within \c wholeExt. Unlike StrictlySmaller, this
   * method returns true even if \c ext == \c wholeExt.
   */
  static bool Smaller(const int ext[6], const int wholeExt[6]);

  /**
   * Grows the \c ext on each side by the given \c count.
   */
  static void Grow(int ext[6], int count);

  /**
   * Grows the \c ext on each side by the given \c count
   * while keeping it limited to the \c wholeExt.
   */
  static void Grow(int ext[6], int count, int wholeExt[6]);

  /**
   * Makes \c ext relative to \c wholeExt.
   */
  static void Transform(int ext[6], int wholeExt[6]);

  /**
   * Given the extents, computes the dimensions.
   */
  static void GetDimensions(const int ext[6], int dims[3]);

protected:
  svtkStructuredExtent();
  ~svtkStructuredExtent() override;

private:
  svtkStructuredExtent(const svtkStructuredExtent&) = delete;
  void operator=(const svtkStructuredExtent&) = delete;
};

//----------------------------------------------------------------------------
inline void svtkStructuredExtent::Clamp(int ext[6], const int wholeExt[6])
{
  ext[0] = (ext[0] < wholeExt[0]) ? wholeExt[0] : ext[0];
  ext[1] = (ext[1] > wholeExt[1]) ? wholeExt[1] : ext[1];

  ext[2] = (ext[2] < wholeExt[2]) ? wholeExt[2] : ext[2];
  ext[3] = (ext[3] > wholeExt[3]) ? wholeExt[3] : ext[3];

  ext[4] = (ext[4] < wholeExt[4]) ? wholeExt[4] : ext[4];
  ext[5] = (ext[5] > wholeExt[5]) ? wholeExt[5] : ext[5];
}

//----------------------------------------------------------------------------
inline bool svtkStructuredExtent::Smaller(const int ext[6], const int wholeExt[6])
{
  if (ext[0] < wholeExt[0] || ext[0] > wholeExt[0 + 1] || ext[0 + 1] < wholeExt[0] ||
    ext[0 + 1] > wholeExt[0 + 1])
  {
    return false;
  }

  if (ext[2] < wholeExt[2] || ext[2] > wholeExt[2 + 1] || ext[2 + 1] < wholeExt[2] ||
    ext[2 + 1] > wholeExt[2 + 1])
  {
    return false;
  }

  if (ext[4] < wholeExt[4] || ext[4] > wholeExt[4 + 1] || ext[4 + 1] < wholeExt[4] ||
    ext[4 + 1] > wholeExt[4 + 1])
  {
    return false;
  }

  return true;
}

//----------------------------------------------------------------------------
inline bool svtkStructuredExtent::StrictlySmaller(const int ext[6], const int wholeExt[6])
{
  if (!svtkStructuredExtent::Smaller(ext, wholeExt))
  {
    return false;
  }

  if (ext[0] > wholeExt[0] || ext[1] < wholeExt[1] || ext[2] > wholeExt[2] ||
    ext[3] < wholeExt[3] || ext[4] > wholeExt[4] || ext[5] < wholeExt[5])
  {
    return true;
  }

  return false;
}

//----------------------------------------------------------------------------
inline void svtkStructuredExtent::Grow(int ext[6], int count)
{
  ext[0] -= count;
  ext[2] -= count;
  ext[4] -= count;

  ext[1] += count;
  ext[3] += count;
  ext[5] += count;
}

//----------------------------------------------------------------------------
inline void svtkStructuredExtent::Grow(int ext[6], int count, int wholeExt[6])
{
  svtkStructuredExtent::Grow(ext, count);
  svtkStructuredExtent::Clamp(ext, wholeExt);
}

//----------------------------------------------------------------------------
inline void svtkStructuredExtent::Transform(int ext[6], int wholeExt[6])
{
  ext[0] -= wholeExt[0];
  ext[1] -= wholeExt[0];

  ext[2] -= wholeExt[2];
  ext[3] -= wholeExt[2];

  ext[4] -= wholeExt[4];
  ext[5] -= wholeExt[4];
}

//----------------------------------------------------------------------------
inline void svtkStructuredExtent::GetDimensions(const int ext[6], int dims[3])
{
  dims[0] = ext[1] - ext[0] + 1;
  dims[1] = ext[3] - ext[2] + 1;
  dims[2] = ext[5] - ext[4] + 1;
}

#endif
