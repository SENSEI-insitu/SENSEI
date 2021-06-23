/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkSortFieldData.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/**
 * @class   svtkSortFieldData
 * @brief   provides a method for sorting field data
 *
 *
 * svtkSortFieldData is used to sort data, based on its value, or with an
 * associated key, into either ascending or descending order. This is useful
 * for operations like selection, or analysis, when evaluating and processing
 * data.
 *
 * This class, which extends the base functionality of svtkSortDataArray,
 * is used to sort field data and its various subclasses (svtkFieldData,
 * svtkDataSetAttributes, svtkPointData, svtkCellData, etc.)
 *
 * @warning
 * This class has been threaded with svtkSMPTools. Using TBB or other
 * non-sequential type (set in the CMake variable
 * SVTK_SMP_IMPLEMENTATION_TYPE) may improve performance significantly on
 * multi-core machines.
 *
 * @warning
 * The sort methods below are static, hence the sorting methods can be
 * used without instantiating the class. All methods are thread safe.
 *
 * @sa
 * svtkSortDataArray
 */

#ifndef svtkSortFieldData_h
#define svtkSortFieldData_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkSortDataArray.h"

class svtkFieldData;

class SVTKCOMMONDATAMODEL_EXPORT svtkSortFieldData : public svtkSortDataArray
{
public:
  //@{
  /**
   * Standard SVTK methods for instantiating, managing type, and printing
   * information about this class.
   */
  static svtkSortFieldData* New();
  svtkTypeMacro(svtkSortFieldData, svtkSortDataArray);
  void PrintSelf(ostream& os, svtkIndent indent) override;
  //@}

  /**
   * Given field data (and derived classes such as point data and cell data),
   * sort all the arrays in the field data given an array and a component
   * number k from that array. In other words, if an array has n components,
   * the kth component is used to sort the array and all of the other arrays
   * in the field data.  Also note that the user can indicate whether the
   * function returns the sort indices (returnIndices=1). If the indices are
   * returned, then the user takes ownership of the data and must delete
   * it. Note that the indices are in sorted (ascending) order, and indicate
   * the final sorted position of the sort. So for example indices[0]=10
   * indicates that the original data in position 10 in the field, was moved
   * to position 0 after the sort. By default, returnIndices=0. (Other notes:
   * if any array is not the same length as the sorting array, then it will
   * be skipped and not sorted.)
   */
  static svtkIdType* Sort(svtkFieldData* fd, const char* arrayName, int k, int returnIndices)
  {
    return svtkSortFieldData::Sort(fd, arrayName, k, returnIndices, 0);
  }

  /**
   * Given field data (and derived classes such as point data and cell data),
   * sort all the arrays in the field data given an array and a component
   * number k from that array. In other words, if an array has n components,
   * the kth component is used to sort the array and all of the other arrays
   * in the field data.  The order of the sorted data is goven by dir: dir=0
   * means sort in ascending order; dir=1 means sort in descending
   * order. Also note that the user can indicate whether the function returns
   * the sort indices (returnIndices=1). If the indices are returned, then
   * the user takes ownership of the data and must delete it. Note that the
   * indices are always in sorted (ascending) order, and indicate the final
   * sorted position of the sort. So for example indices[0]=10 indicates that
   * the original data in position 10 in the field, was moved to position 0
   * after the sort (i.e., position 0 is the smallest value). However, if
   * sort direction dir=1, the indices do not change but the final shuffle of
   * the data is in reverse order (note idx[n-1] for n keys is the largest
   * value). By default, returnIndices=0. (Other notes: if any array is not
   * the same length as the sorting array, then it will be skipped and not
   * sorted.)
   */
  static svtkIdType* Sort(
    svtkFieldData* fd, const char* arrayName, int k, int returnIndices, int dir);

protected:
  svtkSortFieldData();
  ~svtkSortFieldData() override;

private:
  svtkSortFieldData(const svtkSortFieldData&) = delete;
  void operator=(const svtkSortFieldData&) = delete;
};

#endif // svtkSortFieldData_h
