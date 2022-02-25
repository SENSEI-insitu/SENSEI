/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkIterativeClosestPointTransform.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/**
 * @class   svtkIterativeClosestPointTransform
 * @brief   Implementation of the ICP algorithm.
 *
 * Match two surfaces using the iterative closest point (ICP) algorithm.
 * The core of the algorithm is to match each vertex in one surface with
 * the closest surface point on the other, then apply the transformation
 * that modify one surface to best match the other (in a least square sense).
 * This has to be iterated to get proper convergence of the surfaces.
 * @attention
 * Use svtkTransformPolyDataFilter to apply the resulting ICP transform to
 * your data. You might also set it to your actor's user transform.
 * @attention
 * This class makes use of svtkLandmarkTransform internally to compute the
 * best fit. Use the GetLandmarkTransform member to get a pointer to that
 * transform and set its parameters. You might, for example, constrain the
 * number of degrees of freedom of the solution (i.e. rigid body, similarity,
 * etc.) by checking the svtkLandmarkTransform documentation for its SetMode
 * member.
 * @sa
 * svtkLandmarkTransform
 */

#ifndef svtkIterativeClosestPointTransform_h
#define svtkIterativeClosestPointTransform_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkLinearTransform.h"

#define SVTK_ICP_MODE_RMS 0
#define SVTK_ICP_MODE_AV 1

class svtkCellLocator;
class svtkLandmarkTransform;
class svtkDataSet;

class SVTKCOMMONDATAMODEL_EXPORT svtkIterativeClosestPointTransform : public svtkLinearTransform
{
public:
  static svtkIterativeClosestPointTransform* New();
  svtkTypeMacro(svtkIterativeClosestPointTransform, svtkLinearTransform);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  //@{
  /**
   * Specify the source and target data sets.
   */
  void SetSource(svtkDataSet* source);
  void SetTarget(svtkDataSet* target);
  svtkGetObjectMacro(Source, svtkDataSet);
  svtkGetObjectMacro(Target, svtkDataSet);
  //@}

  //@{
  /**
   * Set/Get a spatial locator for speeding up the search process.
   * An instance of svtkCellLocator is used by default.
   */
  void SetLocator(svtkCellLocator* locator);
  svtkGetObjectMacro(Locator, svtkCellLocator);
  //@}

  //@{
  /**
   * Set/Get the maximum number of iterations. Default is 50.
   */
  svtkSetMacro(MaximumNumberOfIterations, int);
  svtkGetMacro(MaximumNumberOfIterations, int);
  //@}

  //@{
  /**
   * Get the number of iterations since the last update
   */
  svtkGetMacro(NumberOfIterations, int);
  //@}

  //@{
  /**
   * Force the algorithm to check the mean distance between two iterations.
   * Default is Off.
   */
  svtkSetMacro(CheckMeanDistance, svtkTypeBool);
  svtkGetMacro(CheckMeanDistance, svtkTypeBool);
  svtkBooleanMacro(CheckMeanDistance, svtkTypeBool);
  //@}

  //@{
  /**
   * Specify the mean distance mode. This mode expresses how the mean
   * distance is computed. The RMS mode is the square root of the average
   * of the sum of squares of the closest point distances. The Absolute
   * Value mode is the mean of the sum of absolute values of the closest
   * point distances. The default is SVTK_ICP_MODE_RMS
   */
  svtkSetClampMacro(MeanDistanceMode, int, SVTK_ICP_MODE_RMS, SVTK_ICP_MODE_AV);
  svtkGetMacro(MeanDistanceMode, int);
  void SetMeanDistanceModeToRMS() { this->SetMeanDistanceMode(SVTK_ICP_MODE_RMS); }
  void SetMeanDistanceModeToAbsoluteValue() { this->SetMeanDistanceMode(SVTK_ICP_MODE_AV); }
  const char* GetMeanDistanceModeAsString();
  //@}

  //@{
  /**
   * Set/Get the maximum mean distance between two iteration. If the mean
   * distance is lower than this, the convergence stops. The default
   * is 0.01.
   */
  svtkSetMacro(MaximumMeanDistance, double);
  svtkGetMacro(MaximumMeanDistance, double);
  //@}

  //@{
  /**
   * Get the mean distance between the last two iterations.
   */
  svtkGetMacro(MeanDistance, double);
  //@}

  //@{
  /**
   * Set/Get the maximum number of landmarks sampled in your dataset.
   * If your dataset is dense, then you will typically not need all the
   * points to compute the ICP transform. The default is 200.
   */
  svtkSetMacro(MaximumNumberOfLandmarks, int);
  svtkGetMacro(MaximumNumberOfLandmarks, int);
  //@}

  //@{
  /**
   * Starts the process by translating source centroid to target centroid.
   * The default is Off.
   */
  svtkSetMacro(StartByMatchingCentroids, svtkTypeBool);
  svtkGetMacro(StartByMatchingCentroids, svtkTypeBool);
  svtkBooleanMacro(StartByMatchingCentroids, svtkTypeBool);
  //@}

  //@{
  /**
   * Get the internal landmark transform. Use it to constrain the number of
   * degrees of freedom of the solution (i.e. rigid body, similarity, etc.).
   */
  svtkGetObjectMacro(LandmarkTransform, svtkLandmarkTransform);
  //@}

  /**
   * Invert the transformation.  This is done by switching the
   * source and target.
   */
  void Inverse() override;

  /**
   * Make another transform of the same type.
   */
  svtkAbstractTransform* MakeTransform() override;

protected:
  //@{
  /**
   * Release source and target
   */
  void ReleaseSource(void);
  void ReleaseTarget(void);
  //@}

  /**
   * Release locator
   */
  void ReleaseLocator(void);

  /**
   * Create default locator. Used to create one when none is specified.
   */
  void CreateDefaultLocator(void);

  /**
   * Get the MTime of this object also considering the locator.
   */
  svtkMTimeType GetMTime() override;

  svtkIterativeClosestPointTransform();
  ~svtkIterativeClosestPointTransform() override;

  void InternalUpdate() override;

  /**
   * This method does no type checking, use DeepCopy instead.
   */
  void InternalDeepCopy(svtkAbstractTransform* transform) override;

  svtkDataSet* Source;
  svtkDataSet* Target;
  svtkCellLocator* Locator;
  int MaximumNumberOfIterations;
  svtkTypeBool CheckMeanDistance;
  int MeanDistanceMode;
  double MaximumMeanDistance;
  int MaximumNumberOfLandmarks;
  svtkTypeBool StartByMatchingCentroids;

  int NumberOfIterations;
  double MeanDistance;
  svtkLandmarkTransform* LandmarkTransform;

private:
  svtkIterativeClosestPointTransform(const svtkIterativeClosestPointTransform&) = delete;
  void operator=(const svtkIterativeClosestPointTransform&) = delete;
};

#endif
