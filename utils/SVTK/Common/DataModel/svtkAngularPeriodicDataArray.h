/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkAngularPeriodicDataArray.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

    This software is distributed WITHOUT ANY WARRANTY; without even
    the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
    PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/**
 * @class   svtkAngularPeriodicDataArray
 * @brief   Map native an Array into an angulat
 * periodic array
 *
 *
 * Map an array into a periodic array. Data from the original array are
 * rotated (on the fly) by the specified angle along the specified axis
 * around the specified point. Lookup is not implemented.
 * Creating the array is virtually free, accessing a tuple require some
 * computation.
 */

#ifndef svtkAngularPeriodicDataArray_h
#define svtkAngularPeriodicDataArray_h

#include "svtkPeriodicDataArray.h" // Parent

#define SVTK_PERIODIC_ARRAY_AXIS_X 0
#define SVTK_PERIODIC_ARRAY_AXIS_Y 1
#define SVTK_PERIODIC_ARRAY_AXIS_Z 2

class svtkMatrix3x3;

template <class Scalar>
class svtkAngularPeriodicDataArray : public svtkPeriodicDataArray<Scalar>
{
public:
  svtkAbstractTemplateTypeMacro(svtkAngularPeriodicDataArray<Scalar>, svtkPeriodicDataArray<Scalar>);
  svtkAOSArrayNewInstanceMacro(svtkAngularPeriodicDataArray<Scalar>);
  static svtkAngularPeriodicDataArray* New();
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Initialize the mapped array with the original input data array.
   */
  void InitializeArray(svtkAOSDataArrayTemplate<Scalar>* inputData);

  //@{
  /**
   * Set/Get the rotation angle in degrees. Default is 0.
   */
  void SetAngle(double angle);
  svtkGetMacro(Angle, double);
  //@}

  //@{
  /**
   * Set/Get the rotation center. Default is 0,0,0.
   */
  void SetCenter(double* center);
  svtkGetVector3Macro(Center, double);
  //@}

  //@{
  /**
   * Set/Get the rotation axis. Default is SVTK_PERIODIC_ARRAY_AXIS_X axis.
   */
  void SetAxis(int axis);
  svtkGetMacro(Axis, int);
  void SetAxisToX(void) { this->SetAxisType(SVTK_PERIODIC_ARRAY_AXIS_X); }
  void SetAxisToY(void) { this->SetAxisType(SVTK_PERIODIC_ARRAY_AXIS_Y); }
  void SetAxisToZ(void) { this->SetAxisType(SVTK_PERIODIC_ARRAY_AXIS_Z); }
  //@}

protected:
  svtkAngularPeriodicDataArray();
  ~svtkAngularPeriodicDataArray() override;

  /**
   * Transform the provided tuple
   */
  void Transform(Scalar* tuple) const override;

  /**
   * Update rotation matrix from Axis, Angle and Center
   */
  void UpdateRotationMatrix();

private:
  svtkAngularPeriodicDataArray(const svtkAngularPeriodicDataArray&) = delete;
  void operator=(const svtkAngularPeriodicDataArray&) = delete;

  double Angle;          // Rotation angle in degrees
  double AngleInRadians; // Rotation angle in radians
  double Center[3];      // Rotation center
  int Axis;              // Rotation Axis

  svtkMatrix3x3* RotationMatrix;
};

#include "svtkAngularPeriodicDataArray.txx"

#endif // svtkAngularPeriodicDataArray_h
// SVTK-HeaderTest-Exclude: svtkAngularPeriodicDataArray.h
