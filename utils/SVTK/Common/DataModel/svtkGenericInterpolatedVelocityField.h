/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkGenericInterpolatedVelocityField.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkGenericInterpolatedVelocityField
 * @brief   Interface for obtaining
 * interpolated velocity values
 *
 * svtkGenericInterpolatedVelocityField acts as a continuous velocity field
 * by performing cell interpolation on the underlying svtkDataSet.
 * This is a concrete sub-class of svtkFunctionSet with
 * NumberOfIndependentVariables = 4 (x,y,z,t) and
 * NumberOfFunctions = 3 (u,v,w). Normally, every time an evaluation
 * is performed, the cell which contains the point (x,y,z) has to
 * be found by calling FindCell. This is a computationally expansive
 * operation. In certain cases, the cell search can be avoided or shortened
 * by providing a guess for the cell iterator. For example, in streamline
 * integration, the next evaluation is usually in the same or a neighbour
 * cell. For this reason, svtkGenericInterpolatedVelocityField stores the last
 * cell iterator. If caching is turned on, it uses this iterator as the
 * starting point.
 *
 * @warning
 * svtkGenericInterpolatedVelocityField is not thread safe. A new instance
 * should be created by each thread.
 *
 * @sa
 * svtkFunctionSet svtkGenericStreamTracer
 */

#ifndef svtkGenericInterpolatedVelocityField_h
#define svtkGenericInterpolatedVelocityField_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkFunctionSet.h"

class svtkGenericDataSet;
class svtkGenericCellIterator;
class svtkGenericAdaptorCell;

class svtkGenericInterpolatedVelocityFieldDataSetsType;

class SVTKCOMMONDATAMODEL_EXPORT svtkGenericInterpolatedVelocityField : public svtkFunctionSet
{
public:
  svtkTypeMacro(svtkGenericInterpolatedVelocityField, svtkFunctionSet);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Construct a svtkGenericInterpolatedVelocityField with no initial data set.
   * Caching is on. LastCellId is set to -1.
   */
  static svtkGenericInterpolatedVelocityField* New();

  using Superclass::FunctionValues;
  /**
   * Evaluate the velocity field, f, at (x, y, z, t).
   * For now, t is ignored.
   */
  int FunctionValues(double* x, double* f) override;

  /**
   * Add a dataset used for the implicit function evaluation.
   * If more than one dataset is added, the evaluation point is
   * searched in all until a match is found. THIS FUNCTION
   * DOES NOT CHANGE THE REFERENCE COUNT OF dataset FOR THREAD
   * SAFETY REASONS.
   */
  virtual void AddDataSet(svtkGenericDataSet* dataset);

  /**
   * Set the last cell id to -1 so that the next search does not
   * start from the previous cell
   */
  void ClearLastCell();

  /**
   * Return the cell cached from last evaluation.
   */
  svtkGenericAdaptorCell* GetLastCell();

  /**
   * Returns the interpolation weights cached from last evaluation
   * if the cached cell is valid (returns 1). Otherwise, it does not
   * change w and returns 0.
   */
  int GetLastLocalCoordinates(double pcoords[3]);

  //@{
  /**
   * Turn caching on/off.
   */
  svtkGetMacro(Caching, svtkTypeBool);
  svtkSetMacro(Caching, svtkTypeBool);
  svtkBooleanMacro(Caching, svtkTypeBool);
  //@}

  //@{
  /**
   * Caching statistics.
   */
  svtkGetMacro(CacheHit, int);
  svtkGetMacro(CacheMiss, int);
  //@}

  //@{
  /**
   * If you want to work with an arbitrary vector array, then set its name
   * here. By default this in nullptr and the filter will use the active vector
   * array.
   */
  svtkGetStringMacro(VectorsSelection);
  void SelectVectors(const char* fieldName) { this->SetVectorsSelection(fieldName); }
  //@}

  //@{
  /**
   * Returns the last dataset that was visited. Can be used
   * as a first guess as to where the next point will be as
   * well as to avoid searching through all datasets to get
   * more information about the point.
   */
  svtkGetObjectMacro(LastDataSet, svtkGenericDataSet);
  //@}

  /**
   * Copy the user set parameters from source. This copies
   * the Caching parameters. Sub-classes can add more after
   * chaining.
   */
  virtual void CopyParameters(svtkGenericInterpolatedVelocityField* from);

protected:
  svtkGenericInterpolatedVelocityField();
  ~svtkGenericInterpolatedVelocityField() override;

  svtkGenericCellIterator* GenCell; // last cell

  double LastPCoords[3]; // last local coordinates
  int CacheHit;
  int CacheMiss;
  svtkTypeBool Caching;

  svtkGenericDataSet* LastDataSet;

  svtkSetStringMacro(VectorsSelection);
  char* VectorsSelection;

  svtkGenericInterpolatedVelocityFieldDataSetsType* DataSets;

  int FunctionValues(svtkGenericDataSet* ds, double* x, double* f);

  static const double TOLERANCE_SCALE;

private:
  svtkGenericInterpolatedVelocityField(const svtkGenericInterpolatedVelocityField&) = delete;
  void operator=(const svtkGenericInterpolatedVelocityField&) = delete;
};

#endif
