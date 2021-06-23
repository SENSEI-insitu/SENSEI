/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkPerlinNoise.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkPerlinNoise
 * @brief   an implicit function that implements Perlin noise
 *
 * svtkPerlinNoise computes a Perlin noise field as an implicit function.
 * svtkPerlinNoise is a concrete implementation of svtkImplicitFunction.
 * Perlin noise, originally described by Ken Perlin, is a non-periodic and
 * continuous noise function useful for modeling real-world objects.
 *
 * The amplitude and frequency of the noise pattern are adjustable.  This
 * implementation of Perlin noise is derived closely from Greg Ward's version
 * in Graphics Gems II.
 *
 * @sa
 * svtkImplicitFunction
 */

#ifndef svtkPerlinNoise_h
#define svtkPerlinNoise_h

#include "svtkCommonDataModelModule.h" // For export macro
#include "svtkImplicitFunction.h"

class SVTKCOMMONDATAMODEL_EXPORT svtkPerlinNoise : public svtkImplicitFunction
{
public:
  svtkTypeMacro(svtkPerlinNoise, svtkImplicitFunction);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Instantiate the class.
   */
  static svtkPerlinNoise* New();

  //@{
  /**
   * Evaluate PerlinNoise function.
   */
  using svtkImplicitFunction::EvaluateFunction;
  double EvaluateFunction(double x[3]) override;
  //@}

  /**
   * Evaluate PerlinNoise gradient.  Currently, the method returns a 0
   * gradient.
   */
  void EvaluateGradient(double x[3], double n[3]) override;

  //@{
  /**
   * Set/get the frequency, or physical scale,  of the noise function
   * (higher is finer scale).  The frequency can be adjusted per axis, or
   * the same for all axes.
   */
  svtkSetVector3Macro(Frequency, double);
  svtkGetVectorMacro(Frequency, double, 3);
  //@}

  //@{
  /**
   * Set/get the phase of the noise function.  This parameter can be used to
   * shift the noise function within space (perhaps to avoid a beat with a
   * noise pattern at another scale).  Phase tends to repeat about every
   * unit, so a phase of 0.5 is a half-cycle shift.
   */
  svtkSetVector3Macro(Phase, double);
  svtkGetVectorMacro(Phase, double, 3);
  //@}

  //@{
  /**
   * Set/get the amplitude of the noise function. Amplitude can be negative.
   * The noise function varies randomly between -|Amplitude| and |Amplitude|.
   * Therefore the range of values is 2*|Amplitude| large.
   * The initial amplitude is 1.
   */
  svtkSetMacro(Amplitude, double);
  svtkGetMacro(Amplitude, double);
  //@}

protected:
  svtkPerlinNoise();
  ~svtkPerlinNoise() override {}

  double Frequency[3];
  double Phase[3];
  double Amplitude;

private:
  svtkPerlinNoise(const svtkPerlinNoise&) = delete;
  void operator=(const svtkPerlinNoise&) = delete;
};

#endif
