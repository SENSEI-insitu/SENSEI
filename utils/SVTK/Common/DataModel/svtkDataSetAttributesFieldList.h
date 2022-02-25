/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkDataSetAttributesFieldList.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class svtkDataSetAttributesFieldList
 * @brief helps manage arrays from multiple svtkDataSetAttributes.
 *
 * svtkDataSetAttributesFieldList, also called svtkDataSetAttributes::FieldList,
 * is used to help with filters when dealing with arrays from multiple
 * svtkDataSetAttributes instances, potentially from multiple inputs.
 *
 * Consider a filter that appends multiple inputs, e.g. svtkAppendPolyData.
 * Besides appending mesh elements, such a filter also needs to combine field
 * arrays (point, and cell data) from inputs to pass on to the output.
 * Now if all the inputs had exactly the same set of arrays, we're all set.
 * However, more often than not, the inputs will have different sets of arrays.
 * The filter will need to match up from various inputs to combine together,
 * potentially dropping arrays not in all inputs. Furthermore, it needs to
 * ensure arrays in the output are flagged as attributes consistently. All of
 * this can be done using svtkDataSetAttributesFieldList.
 *
 * @section Usage Usage
 *
 * Typical usage is as follows:
 * 1. call `IntersectFieldList` or `UnionFieldList` for all input svtkDataSetAttributes
 *   instances,
 * 2. allocate arrays for the output svtkDataSetAttributes by using `CopyAllocate`,
 * 3. call `CopyData` per input (preserving the input order used in step 1) to
 *    copy tuple(s) from input to the output.
 *
 * `svtkDataSetAttributes::InitializeFieldList` is provided for API compatibility
 * with previous implementation of this class and it not required to be called.
 * Simply calling `UnionFieldList` or `IntersectFieldList` for the first
 * svtkDataSetAttributes instance is sufficient.
 *
 * `CopyAllocate, `CopyData`, and `InterpolatePoint` methods on this class
 * are called by similarly named variants on svtkDataSetAttributes that take in a
 * FieldList instance as an argument. Hence, either forms may be used.
 *
 * Calls to `UnionFieldList` and `IntersectFieldList` cannot be mixed. Use
 * `Reset` or `InitializeFieldList` to change mode and start reinitialization.
 */

#ifndef svtkDataSetAttributesFieldList_h
#define svtkDataSetAttributesFieldList_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkSmartPointer.h"          // for svtkSmartPointer
#include "svtkSystemIncludes.h"

#include <functional> // for std::function
#include <memory>     // for unique_ptr

class svtkAbstractArray;
class svtkDataSetAttributes;
class svtkIdList;

class SVTKCOMMONDATAMODEL_EXPORT svtkDataSetAttributesFieldList
{
public:
  /**
   * `number_of_inputs` parameter is not required and only provided for
   * backwards compatibility.
   */
  svtkDataSetAttributesFieldList(int number_of_inputs = 0);
  virtual ~svtkDataSetAttributesFieldList();
  void PrintSelf(ostream& os, svtkIndent indent);

  /**
   * Initializes the field list to empty.
   */
  void Reset();

  /**
   * Initialize the field list. This also adds the first input.
   * Calling this method is optional. The first call to `IntersectFieldList` or
   * `UnionFieldList` on a new instance or after calling `Reset()` will have the
   * same effect.
   */
  void InitializeFieldList(svtkDataSetAttributes* dsa);

  /**
   * Update the field list for an intersection of arrays registered so far and
   * those in `dsa`.
   */
  void IntersectFieldList(svtkDataSetAttributes* dsa);

  /**
   * Update the field list for an union of arrays registered so far and
   * those in `dsa`.
   */
  void UnionFieldList(svtkDataSetAttributes* dsa);

  //@{
  /**
   * These methods can called to generate and update the output
   * svtkDataSetAttributes. These match corresponding API on svtkDataSetAttributes
   * and can be called via the output svtkDataSetAttributes instance
   * instead as well.
   */
  void CopyAllocate(svtkDataSetAttributes* output, int ctype, svtkIdType sz, svtkIdType ext) const;
  void CopyData(int inputIndex, svtkDataSetAttributes* input, svtkIdType fromId,
    svtkDataSetAttributes* output, svtkIdType toId) const;
  void CopyData(int inputIdx, svtkDataSetAttributes* input, svtkIdType inputStart,
    svtkIdType numValues, svtkDataSetAttributes* output, svtkIdType outStart) const;
  void InterpolatePoint(int inputIdx, svtkDataSetAttributes* input, svtkIdList* inputIds,
    double* weights, svtkDataSetAttributes* output, svtkIdType toId) const;
  //@}

  /**
   * Use this method to provide a custom callback function to invoke for each
   * array in the input and corresponding array in the output.
   */
  void TransformData(int inputIndex, svtkDataSetAttributes* input, svtkDataSetAttributes* output,
    std::function<void(svtkAbstractArray*, svtkAbstractArray*)> op) const;

protected:
  /**
   * Called to create an output array for the given type.
   * Default implementation calls `svtkAbstractArray::CreateArray()`.
   */
  virtual svtkSmartPointer<svtkAbstractArray> CreateArray(int type) const;

private:
  class svtkInternals;
  std::unique_ptr<svtkInternals> Internals;

  svtkDataSetAttributesFieldList(const svtkDataSetAttributesFieldList&) = delete;
  void operator=(svtkDataSetAttributesFieldList&) = delete;
};

#endif
// SVTK-HeaderTest-Exclude: svtkDataSetAttributesFieldList.h
