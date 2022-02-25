/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkPath.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkPath
 * @brief   concrete dataset representing a path defined by Bezier
 * curves.
 *
 * svtkPath provides a container for paths composed of line segments,
 * 2nd-order (quadratic) and 3rd-order (cubic) Bezier curves.
 */

#ifndef svtkPath_h
#define svtkPath_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkPointSet.h"

class svtkIntArray;

class SVTKCOMMONDATAMODEL_EXPORT svtkPath : public svtkPointSet
{
public:
  static svtkPath* New();

  svtkTypeMacro(svtkPath, svtkPointSet);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Return what type of dataset this is.
   */
  int GetDataObjectType() override { return SVTK_PATH; }

  /**
   * Enumeration of recognized control point types:
   * - MOVE_TO: Point defining the origin of a new segment, not connected to
   * the previous point.
   * - LINE_TO: Draw a line from the previous point to the current one
   * - CONIC_CURVE: 2nd order (conic/quadratic) point. Must appear
   * in sets of 2, e.g. (0,0) MOVE_TO (0,1) CONIC_CURVE (1,2) CONIC_CURVE
   * defines a quadratic Bezier curve that passes through (0,0) and (1,2)
   * using (0,1) as a control (off) point.
   * - CUBIC_CURVE: 3rd order (cubic) control point. Must appear in sets of
   * 3, e.g. (0,0) MOVE_TO (0,1) CUBIC_CURVE (1,2) CUBIC_CURVE (4,0)
   * CUBIC_CURVE defines a cubic Bezier curve that passes through (0,0)
   * and (4,0), using (0,1) and (1,2) as control (off) points.
   */
  enum ControlPointType
  {
    MOVE_TO = 0,
    LINE_TO,
    CONIC_CURVE,
    CUBIC_CURVE
  };

  //@{
  /**
   * Insert the next control point in the path.
   */
  void InsertNextPoint(float pts[3], int code);
  void InsertNextPoint(double pts[3], int code);
  void InsertNextPoint(double x, double y, double z, int code);
  //@}

  //@{
  /**
   * Set/Get the array of control point codes:
   */
  void SetCodes(svtkIntArray*);
  svtkIntArray* GetCodes();
  //@}

  /**
   * svtkPath doesn't use cells. These methods return trivial values.
   */
  svtkIdType GetNumberOfCells() override { return 0; }
  using svtkDataSet::GetCell;
  svtkCell* GetCell(svtkIdType) override { return nullptr; }
  void GetCell(svtkIdType, svtkGenericCell*) override;
  int GetCellType(svtkIdType) override { return 0; }

  /**
   * svtkPath doesn't use cells, this method just clears ptIds.
   */
  void GetCellPoints(svtkIdType, svtkIdList* ptIds) override;

  /**
   * svtkPath doesn't use cells, this method just clears cellIds.
   */
  void GetPointCells(svtkIdType ptId, svtkIdList* cellIds) override;

  /**
   * Return the maximum cell size in this poly data.
   */
  int GetMaxCellSize() override { return 0; }

  /**
   * Method allocates initial storage for points. Use this method before the
   * method svtkPath::InsertNextPoint().
   */
  void Allocate(svtkIdType size = 1000, int extSize = 1000);

  /**
   * Begin inserting data all over again. Memory is not freed but otherwise
   * objects are returned to their initial state.
   */
  void Reset();

  //@{
  /**
   * Retrieve an instance of this class from an information object.
   */
  static svtkPath* GetData(svtkInformation* info);
  static svtkPath* GetData(svtkInformationVector* v, int i = 0);
  //@}

protected:
  svtkPath();
  ~svtkPath() override;

private:
  svtkPath(const svtkPath&) = delete;
  void operator=(const svtkPath&) = delete;
};

#endif
