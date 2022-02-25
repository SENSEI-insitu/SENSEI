/**
 * @class   svtkIntersectionCounter
 * @brief   Fast simple class for dealing with ray intersections
 *
 * svtkIntersectionCounter is used to intersect data and merge coincident
 * points along the intersect ray. It is light-weight and many of the member
 * functions are in-lined so its very fast. It is not derived from svtkObject
 * so it can be allocated on the stack.
 *
 * This class makes the finite ray intersection process more robust. It
 * merges intersections that are very close to one another (within a
 * tolerance). Such situations are common when intersection rays pass through
 * the edge or vertex of a mesh.
 *
 * @sa
 * svtkBoundingBox
 */

#ifndef svtkIntersectionCounter_h
#define svtkIntersectionCounter_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkSystemIncludes.h"
#include <algorithm> // for sorting
#include <vector>    // for implementation

// class SVTKCOMMONDATAMODEL_EXPORT svtkIntersectionCounter

class svtkIntersectionCounter
{
public:
  //@{
  /**
   * This tolerance must be converted to parametric space. Here tol is the
   * tolerance in world coordinates; length is the ray length.
   */
  svtkIntersectionCounter()
    : Tolerance(0.0001)
  {
  }
  svtkIntersectionCounter(double tol, double length)
  {
    this->Tolerance = (length > 0.0 ? (tol / length) : 0.0);
  }
  //@}

  /**
   * Set/Get the intersection tolerance.
   */
  void SetTolerance(double tol) { this->Tolerance = (tol < 0.0 ? 0.0001 : tol); }
  double GetTolerance() { return this->Tolerance; }

  /**
   * Add an intersection given by parametric coordinate t.
   */
  void AddIntersection(double t) { IntsArray.push_back(t); }

  /**
   * Reset the intersection process.
   */
  void Reset() { IntsArray.clear(); }

  /**
   * Returns number of intersections (even number of intersections, outside
   * or odd number of intersections, inside). This is done by considering
   * close intersections (within Tolerance) as being the same point.
   */
  int CountIntersections()
  {
    int size = static_cast<int>(IntsArray.size());

    // Fast check for trivial cases
    if (size <= 1)
    {
      return size; // 0 or 1
    }

    // Need to work harder: sort and then count the intersections
    std::sort(IntsArray.begin(), IntsArray.end());

    // If here, there is at least one intersection, and two inserted
    // intersection points
    int numInts = 1;
    std::vector<double>::iterator i0 = IntsArray.begin();
    std::vector<double>::iterator i1 = i0 + 1;

    // Now march through sorted array counting "separated" intersections
    while (i1 != IntsArray.end())
    {
      if ((*i1 - *i0) > this->Tolerance)
      {
        numInts++;
        i0 = i1;
      }
      i1++;
    }

    return numInts;
  }

protected:
  double Tolerance;
  std::vector<double> IntsArray;

}; // svtkIntersectionCounter

#endif
// SVTK-HeaderTest-Exclude: svtkIntersectionCounter.h
