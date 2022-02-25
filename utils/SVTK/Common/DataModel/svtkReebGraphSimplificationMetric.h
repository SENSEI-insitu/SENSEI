/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkReebGraphSimplificationMetric.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkReebGraphSimplificationMetric
 * @brief   abstract class for custom Reeb graph
 * simplification metric design.
 *
 * This class makes it possible to design customized simplification metric
 * evaluation algorithms, enabling the user to control the definition of what
 * should be considered as noise or signal in the topological filtering process.
 *
 * References:
 * "Topological persistence and simplification",
 * H. Edelsbrunner, D. Letscher, and A. Zomorodian,
 * Discrete Computational Geometry, 28:511-533, 2002.
 *
 * "Extreme elevation on a 2-manifold",
 * P.K. Agarwal, H. Edelsbrunner, J. Harer, and Y. Wang,
 * ACM Symposium on Computational Geometry, pp. 357-365, 2004.
 *
 * "Simplifying flexible isosurfaces using local geometric measures",
 * H. Carr, J. Snoeyink, M van de Panne,
 * IEEE Visualization, 497-504, 2004
 *
 * "Loop surgery for volumetric meshes: Reeb graphs reduced to contour trees",
 * J. Tierny, A. Gyulassy, E. Simon, V. Pascucci,
 * IEEE Trans. on Vis. and Comp. Graph. (Proc of IEEE VIS), 15:1177-1184, 2009.
 *
 *
 * See Graphics/Testing/Cxx/TestReebGraph.cxx for an example of concrete
 * implementation.
 */

#ifndef svtkReebGraphSimplificationMetric_h
#define svtkReebGraphSimplificationMetric_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkObject.h"

class svtkDataSet;
class svtkDataArray;
class svtkAbstractArray;

class SVTKCOMMONDATAMODEL_EXPORT svtkReebGraphSimplificationMetric : public svtkObject
{
public:
  static svtkReebGraphSimplificationMetric* New();
  svtkTypeMacro(svtkReebGraphSimplificationMetric, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  //@{
  /**
   * Set the lowest possible value for the custom metric space.
   * This value can be set prior to launching the Reeb graph simplification and
   * then used inside the ComputeMetric call to make sure the returned value of
   * ComputeMetric call is indeed between 0 and 1.
   */
  svtkSetMacro(LowerBound, double);
  svtkGetMacro(LowerBound, double);
  //@}

  //@{
  /**
   * Set the highest possible value for the custom metric space.
   * This value can be set prior to launching the Reeb graph simplification and
   * then used inside the ComputeMetric call to make sure the returned value of
   * ComputeMetric call is indeed between 0 and 1.
   */
  svtkSetMacro(UpperBound, double);
  svtkGetMacro(UpperBound, double);
  //@}

  /**
   * Function to implement in your simplification metric algorithm.
   * Given the input mesh and the Ids of the vertices living on the Reeb graph
   * arc to consider for removal, you should return a value between 0 and 1 (the
   * smallest the more likely the arc will be removed, depending on the
   * user-defined simplification threshold).
   */
  virtual double ComputeMetric(svtkDataSet* mesh, svtkDataArray* field, svtkIdType startCriticalPoint,
    svtkAbstractArray* vertexList, svtkIdType endCriticalPoint);

protected:
  svtkReebGraphSimplificationMetric();
  ~svtkReebGraphSimplificationMetric() override;

  double LowerBound, UpperBound;

private:
  svtkReebGraphSimplificationMetric(const svtkReebGraphSimplificationMetric&) = delete;
  void operator=(const svtkReebGraphSimplificationMetric&) = delete;
};

#endif
