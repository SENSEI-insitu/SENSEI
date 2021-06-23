/*=========================================================================

 Program:   Visualization Toolkit
 Module:    svtkAMRUtilities.h

 Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
 All rights reserved.
 See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notice for more information.

 =========================================================================*/
/**
 * @class   svtkAMRUtilities
 *
 *
 *  A concrete instance of svtkObject that employs a singleton design
 *  pattern and implements functionality for AMR specific operations.
 *
 * @sa
 *  svtkOverlappingAMR, svtkAMRBox
 */

#ifndef svtkAMRUtilities_h
#define svtkAMRUtilities_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkObject.h"
#include <vector> // For C++ vector

// Forward declarations
class svtkFieldData;
class svtkOverlappingAMR;
class svtkUniformGrid;

class SVTKCOMMONDATAMODEL_EXPORT svtkAMRUtilities : public svtkObject
{
public:
  // Standard Routines
  svtkTypeMacro(svtkAMRUtilities, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * This method detects and strips partially overlapping cells from a
   * given AMR dataset. If ghost layers are detected, they are removed and
   * new grid instances are created to represent the stripped
   * data-set otherwise, each block is shallow-copied.

   * .SECTION Assumptions
   * 1) The ghosted AMR data must have complete metadata information.
   */
  static void StripGhostLayers(
    svtkOverlappingAMR* ghostedAMRData, svtkOverlappingAMR* strippedAMRData);

  /**
   * A quick test of whether partially overlapping ghost cells exist. This test
   * starts from the highest-res boxes and checks if they have partially
   * overlapping cells. The code returns with true once partially overlapping
   * cells are detected. Otherwise, false is returned.
   */
  static bool HasPartiallyOverlappingGhostCells(svtkOverlappingAMR* amr);

  /**
   * Blank cells in overlapping AMR
   */
  static void BlankCells(svtkOverlappingAMR* amr);

protected:
  svtkAMRUtilities() {}
  ~svtkAMRUtilities() override {}

  /**
   * Given the real-extent w.r.t. the ghosted grid, this method copies the
   * field data (point/cell) data on the stripped grid.
   */
  static void CopyFieldsWithinRealExtent(
    int realExtent[6], svtkUniformGrid* ghostedGrid, svtkUniformGrid* strippedGrid);

  /**
   * Copies the fields from the given source to the given target.
   */
  static void CopyFieldData(
    svtkFieldData* target, svtkIdType targetIdx, svtkFieldData* source, svtkIdType sourceIdx);

  /**
   * Strips ghost layers from the given grid according to the given ghost
   * vector which encodes the number of cells to remote from each of the
   * 6 sides {imin,imax,jmin,jmax,kmin,kmax}. For example, a ghost vector
   * of {0,2,0,2,0,0} would indicate that there exist 2 ghost cells on the
   * imax and jmax side.
   */
  static svtkUniformGrid* StripGhostLayersFromGrid(svtkUniformGrid* grid, int ghost[6]);

  static void BlankGridsAtLevel(svtkOverlappingAMR* amr, int levelIdx,
    std::vector<std::vector<unsigned int> >& children, const std::vector<int>& processMap);

private:
  svtkAMRUtilities(const svtkAMRUtilities&) = delete;
  void operator=(const svtkAMRUtilities&) = delete;
};

#endif /* svtkAMRUtilities_h */
